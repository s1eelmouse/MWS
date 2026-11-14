"""Metric semantics, split isolation, failed calls and journal durability without downloads."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from benchmark import run_benchmark
from compare_runs import comparison_rows
from measurements import percentile, distribution, write_json, model_weights_bytes, model_artifacts, ERROR_TYPES
from run_experiment import stage
from schema import to_answer
from test_case_schema import CALL, FACT


class FakeBackend:
    device = "cpu"

    def __init__(self):
        self.batches = 0

    def extract(self, calls, batch_size, measurements):
        self.batches += 1
        results = []
        for call in calls:
            assert not hasattr(call, "facts"), "Gold reached backend"
            measurements.append({"call_id": call.call_id, "input_tokens": 100,
                                 "output_tokens": 20, "generation_limit_reached": False})
            results.append({"error": "invalid_json"} if call.call_id.endswith("2")
                           else json.loads(to_answer([FACT])))
        return results


class MeasurementTest(unittest.TestCase):
    def test_disk_weights_exclude_metadata_and_unknown_is_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text("{}")
            self.assertIsNone(model_weights_bytes(model_artifacts(root)))
            (root / "model-1.safetensors").write_bytes(b"a" * 7)
            (root / "model-2.safetensors").write_bytes(b"b" * 9)
            self.assertEqual(model_weights_bytes(model_artifacts(root)), 16)
        self.assertIsNone(model_weights_bytes(None))

    def test_failure_counts_preserve_all_types_and_exclude_warmup(self):
        kinds = [*ERROR_TYPES, "custom_failure"]
        backend = FakeBackend()
        rows = [{**copy.deepcopy(CALL), "call_id": f"synthetic:failure{i}"} for i in range(len(kinds))]
        with tempfile.TemporaryDirectory() as temp, patch.object(backend, "extract", side_effect=[
                [{"error": "invalid_json"}], *[[{"error": kind}] for kind in kinds]]):
            report = run_benchmark(rows, backend, Path(temp), 1, 1)
            self.assertEqual(report["error_counts"], dict.fromkeys(kinds, 1))
            self.assertEqual(sum(report["error_counts"].values()), report["failed_calls"])
            self.assertEqual(report["failed_calls"], len(kinds))

    def test_rank_and_missing_lengths_are_explicit(self):
        self.assertEqual(percentile(list(range(1, 21)), .95), 19)
        self.assertEqual(distribution([None, 3, 7]),
                         {"count": 2, "unknown": 1, "p50": 3, "p95": 7, "max": 7})

    def test_gold_isolation_failed_calls_batch_latency_and_warmup(self):
        rows = [{**copy.deepcopy(CALL), "call_id": f"synthetic:call{i}"} for i in range(3)]
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = run_benchmark(rows, backend, root, 2, 1)
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
            batches = [json.loads(line) for line in (root / "batches.jsonl").read_text().splitlines()]
            self.assertEqual(backend.batches, 3)  # one warmup + two measured batches
            self.assertEqual(report["calls"], 3)
            self.assertEqual(report["failed_calls"], 1)
            self.assertEqual(report["error_counts"]["invalid_json"], 1)
            self.assertIsNone(report["gpu_seconds_per_1000_calls"])
            self.assertEqual(report["quality"]["diagnostics"]["format_valid_rate"], 2 / 3)
            self.assertEqual(calls[0]["completion_latency_seconds"], batches[0]["seconds"])
            self.assertEqual(calls[1]["completion_latency_seconds"], batches[0]["seconds"])
            self.assertAlmostEqual(report["calls_per_hour"], 3 * 3600 / sum(b["seconds"] for b in batches))
            self.assertLess(report["successful_calls_per_hour"], report["calls_per_hour"])
            self.assertTrue((root / "memory.csv").is_file())

    def test_prediction_normalization_counts_reach_the_report(self):
        class Backend:
            device = "cpu"

            def extract(self, calls, batch_size, measurements):
                for call in calls:
                    measurements.append({
                        "call_id": call.call_id, "input_tokens": 1, "output_tokens": 1,
                        "generation_limit_reached": False,
                        "normalization": {"speaker_from_turn": 1}})
                return [json.loads(to_answer([FACT])) for _ in calls]

        with tempfile.TemporaryDirectory() as temp:
            report = run_benchmark([CALL], Backend(), Path(temp), 1, 1)
        self.assertEqual(report["normalization_counts"], {"speaker_from_turn": 1})
        self.assertEqual(report["failed_calls"], 0)

    def test_invalid_warmup_answers_are_recorded_without_hiding_bad_models(self):
        with tempfile.TemporaryDirectory() as temp:
            row = {**CALL, "call_id": "synthetic:call2"}
            report = run_benchmark([row], FakeBackend(), Path(temp), 1, 1)
            self.assertEqual(report["warmup_errors"], [{"warmup_batch": 0,
                             "call_id": row["call_id"], "error": "invalid_json"}])
            self.assertEqual(report["failed_calls"], 1)
            self.assertEqual(report["quality"]["diagnostics"]["format_valid_rate"], 0)

    def test_history_is_reset_between_warmup_and_measurement(self):
        class Backend(FakeBackend):
            warmed = False
            resets = 0

            def extract(self, calls, batch_size, measurements):
                if self.batches == 1:
                    assert not self.warmed, 'Warmup history reached measurement'
                self.warmed = True
                return super().extract(calls, batch_size, measurements)

            def reset_series_history(self):
                self.warmed = False
                self.resets += 1

        backend = Backend()
        with tempfile.TemporaryDirectory() as temp:
            run_benchmark([CALL], backend, Path(temp), 1, 1)
        self.assertEqual(backend.resets, 1)

    def test_warmup_runtime_failure_remains_explicit(self):
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp, patch.object(
                backend, "extract", side_effect=RuntimeError("out of memory")):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                run_benchmark([CALL], backend, Path(temp), 1, 1)
            self.assertFalse((Path(temp) / "metrics.json").exists())

    def test_saving_does_not_overwrite_and_stage_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "report.json", {"preserve": True})
            with self.assertRaises(FileExistsError):
                write_json(root / "report.json", {})
            with self.assertRaisesRegex(RuntimeError, "exit 7"):
                stage(root, "failed-test", [sys.executable, "-c", "raise SystemExit(7)"])
            event = json.loads((root / "stages.jsonl").read_text())
            self.assertEqual(event["exit_code"], 7)
            self.assertEqual(json.loads((root / "report.json").read_text()), {"preserve": True})

    def test_comparison_refuses_incompatible_or_duplicate_results(self):
        base = {"status": "complete", "protocol": "v1", "dataset_sha256": "data",
                "selected_calls_sha256": "ids", "contract_signature": "schema", "limits": {},
                "warmup_batches": 1, "quality": {"scorer_sha256": "scorer"},
                "settings": {"model": "test", "device": "cpu", "batch_size": 1},
                "dtype": "float32", "prompt_lookup_tokens": 0}
        with self.assertRaisesRegex(ValueError, "distinct"):
            comparison_rows([base, base, base])
        changed = {**base, "selected_calls_sha256": "other"}
        with self.assertRaisesRegex(ValueError, "Different data"):
            comparison_rows([base, base, changed])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            comparison_rows([base, base, {**base, "status": "failed"}])

    def test_comparison_exports_three_measurements_and_same_subset_baseline(self):
        from compare_runs import main
        from case_io import write_jsonl
        from training_data import file_digest
        import csv
        import xml.etree.ElementTree as ET
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gold_path = root / "gold.jsonl"
            write_jsonl(gold_path, [CALL])
            reports = []
            for size in (1, 2, 3):
                destination = root / f"run{size}"
                destination.mkdir()
                report = run_benchmark([CALL], FakeBackend(), destination, size, 1)
                report.update(status="complete", protocol="local-inference-v1", dtype="float32",
                              dataset_sha256=file_digest(gold_path), selected_calls_sha256="same",
                              contract_signature="same", limits={}, prompt_lookup_tokens=0,
                              settings={"model": "fake", "device": "cpu", "batch_size": size, "limit": 1})
                if size == 1:
                    # Earlier reports cannot supply failure breakdowns; preserve blanks.
                    del report["error_counts"]
                else:
                    report.update(model_parameters=123, model_load_seconds=.5,
                                  model_artifacts={"model.safetensors": {"bytes": 456},
                                                   "config.json": {"bytes": 9}})
                    report["quality"]["official"]["fact_precision"] = .75
                path = destination / "report.json"
                write_json(path, report)
                reports.append(str(path))
            output = root / "comparison"
            with patch.object(sys, "argv", ["compare_runs.py", "--reports", *reports,
                     "--gold", str(gold_path), "--baseline-pred", str(gold_path),
                     "--output-dir", str(output)]):
                main()
            with (output / "comparison.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(float(rows[0]["fact_recall"]), 1)
            self.assertEqual(float(rows[0]["temporal_recall_delta_from_baseline"]), 0)
            self.assertEqual(rows[0]["model_parameters"], "")
            self.assertEqual(rows[0]["errors_invalid_json"], "")
            self.assertEqual(rows[1]["model_parameters"], "123")
            self.assertEqual(rows[1]["model_weights_bytes"], "456")
            self.assertEqual(rows[1]["model_load_seconds"], "0.5")
            self.assertEqual(rows[1]["errors_invalid_json"], "0")
            self.assertEqual(float(rows[1]["fact_precision_delta_from_baseline"]), -.25)
            self.assertEqual(float(rows[1]["format_valid_rate_delta_from_baseline"]), 0)
            self.assertEqual(rows[1]["state_accuracy_delta_from_baseline"], "")
            for key in ("fact_precision", "fact_recall", "temporal_recall", "state_accuracy", "grounding_turn"):
                self.assertIn(key + "_delta_from_baseline", rows[1])
            with (output / "per_attribute.csv").open() as stream:
                attributes = list(csv.DictReader(stream))
            self.assertEqual(len(attributes), 20)  # five types for baseline + three modes
            action = next(row for row in attributes if row["attribute"] == "action")
            self.assertEqual(action["attribute_kind"], "static")
            self.assertEqual(action["n_gold_facts"], "1")
            self.assertEqual(action["fact_recall"], "1.0")
            self.assertTrue(any(row["attribute_kind"] == "temporal" for row in attributes))
            ET.parse(output / "resources-quality.svg")


if __name__ == "__main__":
    unittest.main()
