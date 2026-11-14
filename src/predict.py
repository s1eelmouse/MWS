"""Generate case predictions with stable call IDs and incremental, non-overwriting output."""
import argparse
import json
import os
import time
from pathlib import Path

from case_io import prediction_row, read_jsonl
from config import MERGED_DIR
from schema import as_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/organizer/val.jsonl")
    parser.add_argument("--output", required=True, help="new file; each completed batch is flushed")
    parser.add_argument("--model", default=os.getenv("MODEL_DIR", MERGED_DIR))
    parser.add_argument("--with-schema", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    output = Path(args.output)
    if output.exists() or output.with_suffix(".run.json").exists():
        parser.error("Output or run report exists; choose a fresh experiment filename")
    rows = read_jsonl(args.input, args.limit)
    if not rows:
        parser.error("Input contains no calls")
    calls = [as_call(row) for row in rows]
    from extractor import Extractor
    extractor = Extractor(args.model, args.with_schema)
    extractor.extract(calls[:1])  # warmup, excluded from timing
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = failed = 0
    start = time.perf_counter()
    with output.open("x", encoding="utf-8") as stream:
        for begin in range(0, len(calls), args.batch_size):
            batch = calls[begin:begin + args.batch_size]
            results = extractor.extract(batch, args.batch_size)
            if len(results) != len(batch):
                raise RuntimeError("Backend returned an unexpected number of results")
            for call, result in zip(batch, results):
                row = prediction_row(call.model_dump(mode="json"), result)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                failed += "error" in row
                completed += 1
            stream.flush()
            print(f"Completed {completed}/{len(calls)}; failed: {failed}")
    elapsed = time.perf_counter() - start
    report = {
        "model": args.model, "calls": completed, "failed": failed,
        "wall_seconds": elapsed, "calls_per_hour": completed / elapsed * 3600,
        "batch_size": args.batch_size, "with_schema": args.with_schema,
        "constrained_json": bool(getattr(extractor, "constrained_json", False)),
        "temporal_normalization": getattr(extractor, "temporal_normalization", "off"),
        "catalog_sha256": getattr(extractor, "catalog_sha256", None),
        "grammar_compile_seconds": getattr(extractor, "grammar_compile_seconds", None),
    }
    with output.with_suffix(".run.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
