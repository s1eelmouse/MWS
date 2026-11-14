"""Compare compatible benchmarks; emit CSV and an offline resources/quality SVG."""
import argparse
import csv
import html
import json
from pathlib import Path
from measurements import write_json


def comparison_rows(reports):
    if len(reports) < 3:
        raise ValueError("PLAN.md requires at least three measured configurations")
    keys = ("protocol", "dataset_sha256", "selected_calls_sha256", "contract_signature", "limits")
    for report in reports:
        if report.get("status") != "complete":
            raise ValueError("Incomplete/failed benchmark cannot enter comparison")
        if any(report[k] != reports[0][k] for k in keys):
            raise ValueError("Different data, subset, contract, limits or measurement protocol")
        if report["quality"]["scorer_sha256"] != reports[0]["quality"]["scorer_sha256"]:
            raise ValueError("Scorer differs")
        if report["warmup_batches"] != reports[0]["warmup_batches"]:
            raise ValueError("Warmup protocol differs")
    variants = {(r["settings"]["model"], r["settings"]["device"], r["settings"]["batch_size"],
                 r["dtype"], r["prompt_lookup_tokens"]) for r in reports}
    if len(variants) < 3:
        raise ValueError("Need three distinct configurations, not three copies of a report")
    result = []
    for report in reports:
        cfg, quality = report["settings"], report["quality"]
        result.append({"model": cfg["model"], "device": cfg["device"], "dtype": report["dtype"], "batch_size": cfg["batch_size"],
                       "calls": report["calls"], "failed_calls": report["failed_calls"],
                       "p50_seconds": report["call_completion_seconds"]["p50"],
                       "p95_seconds": report["call_completion_seconds"]["p95"],
                       "seconds_per_call": report["measurement_seconds"] / report["calls"],
                       "calls_per_hour": report["calls_per_hour"],
                       "gpu_seconds_per_1000_calls": report["gpu_seconds_per_1000_calls"],
                       "rss_peak_bytes": report["memory"]["process_rss_sampled_peak_bytes"],
                       "cuda_peak_allocated_bytes": report["memory"]["cuda_peak_allocated_bytes"],
                       "cuda_peak_reserved_bytes": report["memory"]["cuda_peak_reserved_bytes"],
                       "format_valid_rate": quality["diagnostics"]["format_valid_rate"],
                       **{k: quality["official"][k] for k in ("fact_precision", "fact_recall", "temporal_recall", "state_accuracy", "grounding_turn")}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--baseline-pred", required=True)
    parser.add_argument("--gold", default="data/organizer/val.jsonl")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reports]
    rows = comparison_rows(reports)
    from case_io import read_jsonl
    from case_evaluation import evaluate_predictions
    from training_data import file_digest
    if file_digest(args.gold) != reports[0]["dataset_sha256"]:
        parser.error("Baseline gold differs from benchmark dataset")
    gold = read_jsonl(args.gold, reports[0]["settings"].get("limit"))
    ids = {r["call_id"] for r in gold}
    all_pred = read_jsonl(args.baseline_pred)
    all_ids = {g["call_id"] for g in read_jsonl(args.gold)}
    if any(r["call_id"] not in all_ids for r in all_pred):
        parser.error("Unknown baseline call IDs")
    baseline, _ = evaluate_predictions(gold, [r for r in all_pred if r["call_id"] in ids])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "baseline.json", baseline)
    for row in rows:
        for metric in ("temporal_recall", "state_accuracy"):
            reference = baseline["official"][metric]
            row[metric + "_delta_from_baseline"] = (row[metric] - reference
                if row[metric] is not None and reference is not None else None)
    with (output / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "comparison.json", {"runs": rows, "baseline_resource_metrics": None,
               "note": "Baseline predictions provide quality only; runtime cannot be reconstructed.",
               "reports": args.reports})
    # Static export: x is amortized processing time, not per-call latency for batches.
    height = max(700, 640 + len(rows) * 20)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}">',
           f'<rect width="1000" height="{height}" fill="white"/>',
           '<g font-family="sans-serif" font-size="14" fill="#182638">']
    xmax = max(r["seconds_per_call"] for r in rows) * 1.1
    for offset, metric in ((0, "temporal_recall"), (300, "state_accuracy")):
        svg.append(f'<text x="70" y="{25+offset}">{metric} vs seconds / call (amortized)</text>')
        svg.append(f'<path d="M70,{50+offset} V{250+offset} H930" fill="none" stroke="#777"/>')
        for fraction in (0, .5, 1):
            svg.append(f'<text x="20" y="{255+offset-200*fraction}">{fraction}</text>')
            svg.append(f'<text x="{70+860*fraction}" y="{275+offset}">{xmax*fraction:.2f}</text>')
        if baseline["official"][metric] is not None:
            ybase = 250 + offset - 200 * baseline["official"][metric]
            svg.append(f'<path d="M70,{ybase} H930" stroke="#888" stroke-dasharray="5 5"/>')
        for i, row in enumerate(rows):
            if row[metric] is None:
                svg.append(f'<text x="{80+i*120}" y="{45+offset}">{i+1}: N/A</text>')
                continue
            x, y = 70 + 860 * row["seconds_per_call"] / xmax, 250 + offset - 200 * row[metric]
            svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#245acc"/><text x="{x+7}" y="{y-6}">{i+1}</text>')
    for i, row in enumerate(rows):
        label = html.escape(f'{i+1}: {row["model"]} / {row["device"]} / batch {row["batch_size"]}')
        svg.append(f'<text x="70" y="{620+i*20}">{label}</text>')
    svg.append('</g></svg>')
    (output / "resources-quality.svg").write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    main()
