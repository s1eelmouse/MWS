"""Measure local inference and official quality on identical calls (PLAN.md section 6)."""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from case_io import prediction_row, read_jsonl, validate_gold, write_jsonl
from case_evaluation import evaluate_predictions
from config import (MAX_INPUT_TOKENS, MAX_NEW_TOKENS, PROMPT_LOOKUP_TOKENS,
                    RETRY_ON_TOKEN_LIMIT, RETRY_REPETITION_PENALTY)
from measurements import (MemorySampler, distribution, environment, write_json, model_artifacts,
                          model_weights_bytes, ERROR_TYPES)
from schema import as_call, contract_signature
from training_data import file_digest


def run_benchmark(rows, backend, output, batch_size, warmup_batches, sync=lambda: None):
    """Backend receives CallInput only; labels stay exclusively in the evaluator."""
    calls = [as_call(row) for row in rows]
    predictions, records, batches = [], [], []
    warmup_errors = []
    warmup_start = time.perf_counter()
    for warmup_batch in range(warmup_batches):
        # Invalid answers are quality outcomes, not a reason to skip measuring a model.
        warmup = backend.extract(calls[:batch_size], batch_size, measurements=[])
        if len(warmup) != len(calls[:batch_size]):
            raise ValueError("Warmup returned a different number of calls")
        warmup_errors.extend({"warmup_batch": warmup_batch, "call_id": call.call_id, "error": result["error"]}
                             for call, result in zip(calls[:batch_size], warmup) if "error" in result)
    sync()
    warmup_seconds = time.perf_counter() - warmup_start
    torch = getattr(backend, "torch", None)
    if backend.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with MemorySampler(output / "memory.csv") as memory, \
            (output / "predictions.jsonl").open("x", encoding="utf-8") as pred_file, \
            (output / "calls.jsonl").open("x", encoding="utf-8") as call_file, \
            (output / "batches.jsonl").open("x", encoding="utf-8") as batch_file, \
            (output / "failures.jsonl").open("x", encoding="utf-8") as failure_file:
        for start in range(0, len(calls), batch_size):
            batch = calls[start:start + batch_size]
            details = []
            sync()
            begin = time.perf_counter()
            try:
                results = backend.extract(batch, batch_size, measurements=details)
                sync()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if backend.device == "cuda":
                    torch.cuda.empty_cache()
                results = [{"error": "out_of_memory"} for _ in batch]
            elapsed = time.perf_counter() - begin
            if len(results) != len(batch):
                raise ValueError("Backend returned a different number of calls")
            batch_id = len(batches)
            batch_record = {"batch_id": batch_id, "calls": len(batch), "seconds": elapsed}
            batches.append(batch_record)
            batch_file.write(json.dumps(batch_record) + "\n")
            by_id = {d["call_id"]: d for d in details}
            for call, result in zip(batch, results):
                prediction = prediction_row(call.model_dump(mode="json"), result)
                prediction["call_id"] = call.call_id
                record = {"call_id": call.call_id, "batch_id": batch_id,
                          "input_tokens": None, "output_tokens": None,
                          **by_id.get(call.call_id, {}),
                          "error": prediction.get("error"),
                          "completion_latency_seconds": elapsed}
                diagnostic = record.pop("diagnostic", None)
                if "error" in prediction:
                    failure_file.write(json.dumps({**record, **(diagnostic or {})}, ensure_ascii=False) + "\n")
                records.append(record)
                predictions.append(prediction)
                pred_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                call_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            for stream in (pred_file, call_file, batch_file, failure_file):
                stream.flush()
            print(f"Measured {len(predictions)}/{len(calls)} calls", flush=True)
    seconds = sum(batch["seconds"] for batch in batches)
    if memory.error:
        raise RuntimeError(f"Memory sampling failed: {memory.error}")
    quality, errors = evaluate_predictions(rows, predictions)
    write_json(output / "metrics.json", quality)
    write_jsonl(output / "errors.jsonl", errors)
    failed = sum("error" in p for p in predictions)
    # Only explicit execution/schema failures; fact mismatches are quality diagnostics.
    error_counts = dict.fromkeys(ERROR_TYPES, 0)
    error_counts.update(Counter(e["error"] for e in errors if e.get("error")))
    normalization_counts = Counter()
    for record in records:
        for key, value in (record.get("normalization") or {}).items():
            normalization_counts[key] += value
    return {"calls": len(calls), "failed_calls": failed,
            "error_counts": error_counts,
            "normalization_counts": dict(normalization_counts),
            "warmup_batches": warmup_batches, "warmup_seconds": warmup_seconds,
            "warmup_errors": warmup_errors,
            "measurement_seconds": seconds,
            "calls_per_hour": len(calls) * 3600 / seconds,
            "successful_calls_per_hour": (len(calls) - failed) * 3600 / seconds,
            "gpu_seconds_per_1000_calls": seconds * 1000 / len(calls) if backend.device == "cuda" else None,
            "gpu_seconds_definition": "one allocated GPU * measured local processing seconds; not CUDA kernel time",
            "latency_definition": "local end-to-end batch completion, shared by calls in batch; excludes queue, network and file I/O",
            "percentile_method": "nearest_rank",
            "call_completion_seconds": distribution([r["completion_latency_seconds"] for r in records]),
            "batch_seconds": distribution([b["seconds"] for b in batches]),
            "input_tokens": distribution([r["input_tokens"] for r in records]),
            "output_tokens": distribution([r["output_tokens"] for r in records]),
            "output_token_definition": "generated token IDs through first EOS inclusive; excludes padding",
            "memory": {**memory.report(),
                       "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if backend.device == "cuda" else None,
                       "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if backend.device == "cuda" else None},
            "quality": quality}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/organizer/val.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--failures", help="diagnostic subset: JSONL containing call_id and error")
    parser.add_argument("--with-schema", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup_batches < 1:
        parser.error("batch size and warmup batches must be positive")
    rows = read_jsonl(args.input)
    if args.failures:
        failure_rows = read_jsonl(args.failures)
        ids = {r["call_id"] for r in failure_rows if r.get("error")}
        unknown = ids - {r["call_id"] for r in rows}
        if unknown:
            parser.error("Failure file contains unknown call IDs")
        rows = [r for r in rows if r["call_id"] in ids]
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        rows = rows[:args.limit]
    if not rows:
        parser.error("Empty input")
    for row in rows:
        validate_gold(row)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = {"protocol": "local-inference-v2", "settings": vars(args),
              "dataset_sha256": file_digest(args.input),
              "selected_calls_sha256": hashlib.sha256(json.dumps([r["call_id"] for r in rows]).encode()).hexdigest(),
              "contract_signature": contract_signature(),
              "diagnostic_subset": bool(args.failures),
              "failures_sha256": file_digest(args.failures) if args.failures else None,
              "limits": {"input": MAX_INPUT_TOKENS, "output": MAX_NEW_TOKENS},
              "prompt_lookup_tokens": PROMPT_LOOKUP_TOKENS,
              "retry_on_token_limit": RETRY_ON_TOKEN_LIMIT,
              "retry_repetition_penalty": RETRY_REPETITION_PENALTY}
    try:
        report["hardware"] = environment()
        write_json(output / "environment.json", report["hardware"])
        report["model_artifacts"] = model_artifacts(args.model)
        report["model_weights_bytes"] = model_weights_bytes(report["model_artifacts"])
        from extractor import Extractor
        load_start = time.perf_counter()
        backend = Extractor(args.model, args.with_schema, args.device)
        sync = backend.torch.cuda.synchronize if args.device == "cuda" else lambda: None
        sync()
        report["model_load_seconds"] = time.perf_counter() - load_start
        report["constrained_json"] = bool(getattr(backend, "constrained_json", False))
        report["grammar_compile_seconds"] = getattr(backend, "grammar_compile_seconds", None)
        report["model_parameters"] = backend.model.num_parameters()
        report["dtype"] = str(backend.model.dtype)
        report["torch_threads"] = backend.torch.get_num_threads()
        report["model_revision"] = getattr(backend.model.config, "_commit_hash", None)
        report.update(run_benchmark(rows, backend, output, args.batch_size, args.warmup_batches, sync))
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
