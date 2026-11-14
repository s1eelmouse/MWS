"""Evaluate case predictions using the exact organizer scorer, aligned by call_id."""
import argparse
from config import ORGANIZER_DIR
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from case_evaluation import evaluate_predictions
from case_io import prediction_row, read_jsonl, write_jsonl
from schema import as_call


def ask(endpoint: str, row: dict, timeout=300):
    start = time.perf_counter()
    try:
        reply = requests.post(endpoint, json=as_call(row).model_dump(mode="json"), timeout=timeout)
        reply.raise_for_status()
        result = reply.json()
        if result.get("call_id") != row["call_id"]:
            raise ValueError("Service returned a different call_id")
    except (requests.RequestException, TypeError, ValueError) as exc:
        result = {"call_id": row["call_id"], "facts": [], "error": "request_failed", "details": str(exc)}
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(ORGANIZER_DIR / "val.jsonl"))
    parser.add_argument("--pred")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--json", help="new JSON report file")
    parser.add_argument("--errors", help="new per-call error report")
    parser.add_argument("--reference-metrics", help="optional published organizer baseline")
    args = parser.parse_args()
    if args.workers <= 0 or args.timeout <= 0:
        parser.error("--workers and --timeout must be positive")
    for output in (args.json, args.errors):
        if output and Path(output).exists():
            parser.error(f"Output already exists: {output}; choose a new experiment filename")
    gold = read_jsonl(args.gold, args.limit)
    latency = None
    if args.pred:
        preds = read_jsonl(args.pred)
        if args.limit is not None:
            # A full prediction file may be scored on an explicitly selected gold subset.
            all_ids = {row["call_id"] for row in read_jsonl(args.gold)}
            unknown = {row["call_id"] for row in preds} - all_ids
            if unknown:
                parser.error("Predictions contain call IDs absent from the complete gold file")
            selected = {row["call_id"] for row in gold}
            preds = [row for row in preds if row["call_id"] in selected]
    else:
        if not gold:
            parser.error("Gold file is empty")
        endpoint = args.url.rstrip("/") + "/extract"
        warmup, _ = ask(endpoint, gold[0], args.timeout)
        if warmup.get("error"):
            raise SystemExit(f"Service warmup failed: {warmup}")
        start = time.perf_counter()
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(lambda row: ask(endpoint, row, args.timeout), gold))
        elapsed = time.perf_counter() - start
        preds = [row for row, _ in results]
        durations = sorted(duration for _, duration in results)
        latency = {
            "calls_per_hour": len(gold) / elapsed * 3600,
            "wall_seconds": elapsed, "workers": args.workers,
            "p50_seconds": statistics.median(durations),
            "p95_seconds": durations[min(len(durations)-1, int(len(durations)*.95))],
        }
    report, errors = evaluate_predictions(gold, preds, catalog_path=args.gold)
    if latency is not None:
        report["latency"] = latency
    if args.reference_metrics:
        reference = json.loads(Path(args.reference_metrics).read_text(encoding="utf-8"))
        report["published_reference_delta"] = {
            key: round(value-reference[key], 6)
            for key, value in report["official"].items()
            if type(value) in (int, float) and type(reference.get(key)) in (int, float)
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    if args.errors:
        write_jsonl(args.errors, errors)


if __name__ == "__main__":
    main()
