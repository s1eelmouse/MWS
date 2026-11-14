"""The wrapper keeps official numbers but rejects ambiguous inputs and reports failures."""
import copy
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from case_evaluation import evaluate_predictions
from test_case_schema import CALL, FACT


class EvaluationTest(unittest.TestCase):
    def test_official_scorer_bytes_are_pinned(self):
        path = Path(__file__).resolve().parents[1] / "src/official_case_metrics.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         path.with_suffix(".sha256").read_text().strip())

    def test_perfect_single_object_prediction(self):
        report, _ = evaluate_predictions([CALL], [{"call_id": CALL["call_id"], "facts": [FACT]}])
        self.assertEqual(report["official"]["fact_recall"], 1)
        self.assertEqual(report["diagnostics"]["format_valid_rate"], 1)

    def test_predictions_align_by_id_not_row_order(self):
        other = {**copy.deepcopy(CALL), "call_id": "synthetic:call0001"}
        report, _ = evaluate_predictions([CALL, other], [other, CALL])
        self.assertEqual(report["official"]["fact_recall"], 1)

    def test_missing_call_is_penalized(self):
        report, errors = evaluate_predictions([CALL], [])
        self.assertEqual(report["official"]["fact_recall"], 0)
        self.assertEqual(report["diagnostics"]["missing_calls"], 1)
        self.assertEqual(errors[0]["error"], "missing_prediction")

    def test_failure_is_not_counted_as_valid_empty_json(self):
        report, _ = evaluate_predictions([CALL], [{"call_id": CALL["call_id"], "facts": [], "error": "invalid_json"}])
        self.assertEqual(report["diagnostics"]["format_valid_rate"], 0)

    def test_unknown_and_duplicate_ids_are_errors(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([CALL], [{**CALL, "call_id": "unknown"}])
        with self.assertRaises(ValueError):
            evaluate_predictions([CALL], [CALL, CALL])

    def test_invalid_fact_schema_is_reported(self):
        report, _ = evaluate_predictions([CALL], [{"call_id": CALL["call_id"], "facts": [{"bad": 1}]}])
        self.assertEqual(report["diagnostics"]["format_valid_rate"], 0)

    def test_source_errors_remain_visible(self):
        pred = copy.deepcopy(CALL)
        pred["facts"][0]["source"]["turn_idx"] = 999
        report, _ = evaluate_predictions([CALL], [pred])
        self.assertEqual(report["diagnostics"]["calls_with_invalid_sources"], 1)

    def test_bad_interval_is_reported_without_changing_official_fact_counts(self):
        pred = copy.deepcopy(CALL)
        pred['facts'][0]['valid_to'] = '2026-11-11T00:00:00'
        report, _ = evaluate_predictions([CALL], [pred])
        self.assertEqual(report['official']['n_pred_facts'], 1)
        self.assertEqual(report['official']['fact_recall'], 1)
        self.assertEqual(report['official']['temporal_recall'], 0)
        self.assertEqual(report['diagnostics']['invalid_or_failed_calls'], 1)

    def test_known_official_state_bug_is_not_silently_fixed(self):
        row = copy.deepcopy(CALL)
        row["facts"][0]["attribute"] = "service_state"
        row["facts"][0]["value"] = "работает нормально"
        second = copy.deepcopy(row["facts"][0])
        second["object"] = {"id": "synthetic:dev1", "name": "модем", "type": "device"}
        second["value"] = "красный индикатор"
        row["facts"].append(second)
        report, _ = evaluate_predictions([row], [row])
        self.assertEqual(report["official"]["state_accuracy"], .5)
        self.assertTrue(report["known_scorer_issues"])

    def test_per_attribute_error_types_explain_official_metrics(self):
        state = {**copy.deepcopy(FACT), "attribute": "service_state", "value": "периодические обрывы",
                 "object": {"id": "synthetic:svc0", "type": "service", "name": "домашний интернет"}}
        gold = {**copy.deepcopy(CALL), "facts": [copy.deepcopy(FACT), state]}
        wrong_date = {**copy.deepcopy(FACT), "valid_from": "2026-11-10T00:00:00"}
        wrong_turn = {**copy.deepcopy(state), "source": {"turn_idx": 0}}
        extra = {**copy.deepcopy(FACT), "value": "смена тарифа", "object": {"name": "тариф"}}
        report, _ = evaluate_predictions([gold], [{"call_id": CALL["call_id"],
                                                   "facts": [wrong_date, wrong_turn, extra]}])
        action, service = report["per_attribute"]["action"], report["per_attribute"]["service_state"]
        self.assertEqual(action["attribute_kind"], "static")
        self.assertEqual(service["attribute_kind"], "temporal")
        self.assertEqual((action["matched_facts"], action["missed_facts"], action["extra_facts"]), (1, 0, 1))
        self.assertEqual((action["valid_from_errors"], action["valid_to_errors"], action["date_errors"]), (1, 0, 1))
        self.assertEqual(action["source_turn_errors"], 0)
        self.assertEqual(action["temporal_recall"], 0)
        self.assertEqual((service["matched_facts"], service["date_errors"], service["source_turn_errors"]), (1, 0, 1))
        self.assertEqual(service["grounding_turn"], 0)
        self.assertEqual(report["per_attribute"]["tariff"]["missed_facts"], 0)
        # Official numbers are untouched by the diagnostics.
        self.assertEqual(report["official"]["n_pred_facts"], 3)
        self.assertEqual(report["official"]["fact_recall"], 1)

    def test_failed_call_counts_every_gold_fact_as_missed(self):
        report, _ = evaluate_predictions([CALL], [{"call_id": CALL["call_id"], "facts": [], "error": "invalid_json"}])
        action = report["per_attribute"]["action"]
        self.assertEqual((action["matched_facts"], action["missed_facts"], action["extra_facts"]), (0, 1, 0))


if __name__ == "__main__":
    unittest.main()
