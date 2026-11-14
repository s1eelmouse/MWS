"""Compare compatible benchmarks; emit CSV and an offline resources/quality SVG."""
import argparse
import csv
import html
import json
from pathlib import Path
from measurements import write_json, model_weights_bytes, ERROR_TYPES
from config import CASE_DATA_DIR
from run_context import catalog_for

QUALITY_METRICS = ("fact_precision", "fact_recall", "temporal_recall", "state_accuracy", "grounding_turn")


def delta(value, reference):
    return value - reference if value is not None and reference is not None else None


def attribute_rows(reports, baseline, report_paths):
    rows = []
    sources = [("baseline", {}, baseline)] + [
        (path, report["settings"], report["quality"]) for path, report in zip(report_paths, reports)]
    for run, settings, quality in sources:
        for attribute, metrics in quality.get("per_attribute", {}).items():
            reference = baseline["per_attribute"].get(attribute, {})
            rows.append({"run": run, "model": settings.get("model"),
                         "device": settings.get("device"), "batch_size": settings.get("batch_size"),
                         "attribute": attribute,
                         "attribute_kind": "temporal" if attribute in
                             ("service_state", "tariff", "ticket_status", "agreement") else "static",
                         **metrics,
                         **{key + "_delta_from_baseline": delta(metrics.get(key), reference.get(key))
                            for key in ("fact_precision", "fact_recall")}})
    return rows


def write_csv(path, rows):
    # Union preserves custom failure/attribute fields rather than silently dropping them.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(reports):
    if len(reports) < 3:
        raise ValueError("PLAN.md requires at least three measured configurations")
    keys = ("protocol", "dataset_sha256", "selected_calls_sha256", "contract_signature", "limits")
    pipeline_keys = ("constrained_json", "temporal_normalization", "catalog_sha256", "grammar_backend", "grammar_threads",
                     "retry_on_token_limit", "retry_repetition_penalty", "series_history_enabled",
                     "series_history_max_entries", "history_settings", 'fact_recovery',
                     'fact_recovery_policy', 'temporal_catalog_sha256')
    for report in reports:
        if report.get("diagnostic_subset"):
            raise ValueError("Diagnostic failure subset cannot enter the resource-quality curve")
        if report.get("status") != "complete":
            raise ValueError("Incomplete/failed benchmark cannot enter comparison")
        if any(report[k] != reports[0][k] for k in keys):
            raise ValueError("Different data, subset, contract, limits or measurement protocol")
        if any(report.get(k) != reports[0].get(k) for k in pipeline_keys):
            raise ValueError("Different decoding, date rules, catalog, retry or history settings")
        if report["quality"]["scorer_sha256"] != reports[0]["quality"]["scorer_sha256"]:
            raise ValueError("Scorer differs")
        if any(report['quality'].get(k) != reports[0]['quality'].get(k) for k in
               ('scorer_version', 'patch_adapter_sha256', 'catalog_sha256')):
            raise ValueError('Scorer version, local patch or scoring catalog differs')
        if report["warmup_batches"] != reports[0]["warmup_batches"]:
            raise ValueError("Warmup protocol differs")
    for device in ("cpu", "cuda"):
        same_device = [r for r in reports if r["settings"]["device"] == device]
        hardware = [r.get("hardware") for r in same_device]
        if hardware and all(hardware):
            fields = ("cpu", "logical_cpus", "ram_bytes", "nvidia_smi")
            if any(any(h.get(k) != hardware[0].get(k) for k in fields) for h in hardware):
                raise ValueError("Hardware differs within the same device group")
        if len({r.get("torch_threads") for r in same_device}) > 1:
            raise ValueError("CPU thread settings differ within the same device group")
    variants = {(r["settings"]["model"], r["settings"]["device"], r["settings"]["batch_size"],
                 r["dtype"], r["prompt_lookup_tokens"]) for r in reports}
    if len(variants) < 3:
        raise ValueError("Need three distinct configurations, not three copies of a report")
    result = []
    for report in reports:
        cfg, quality = report["settings"], report["quality"]
        result.append({"model": cfg["model"], "device": cfg["device"], "dtype": report["dtype"], "batch_size": cfg["batch_size"],
                       'fact_recovery': report.get('fact_recovery'),
                       'fact_recovery_policy': report.get('fact_recovery_policy'),
                       'temporal_normalization': report.get('temporal_normalization'),
                       'temporal_catalog_sha256': report.get('temporal_catalog_sha256'),
                       'scorer_version': quality.get('scorer_version', '1'),
                       'scorer_sha256': quality.get('scorer_sha256'),
                       'scorer_patch_sha256': quality.get('patch_adapter_sha256'),
                       'scoring_catalog_sha256': quality.get('catalog_sha256'),
                       'contract_signature': report.get('contract_signature'),
                       'model_source_contract_signature': (report.get('model_contract') or {}).get('source_signature'),
                       "model_parameters": report.get("model_parameters"),
                       "model_weights_bytes": report.get("model_weights_bytes",
                           model_weights_bytes(report.get("model_artifacts"))),
                       "model_load_seconds": report.get("model_load_seconds"),
                       "calls": report["calls"], "failed_calls": report["failed_calls"],
                       "successful_calls_per_hour": report.get("successful_calls_per_hour"),
                       **{f"{kind}_tokens_{p}": report.get(f"{kind}_tokens", {}).get(p)
                          for kind in ("input", "output") for p in ("p50", "p95")},
                       **{"errors_" + kind: report["error_counts"].get(kind, 0)
                          if report.get("error_counts") is not None else None for kind in ERROR_TYPES},
                       "error_counts": report.get("error_counts"),
                       "p50_seconds": report["call_completion_seconds"]["p50"],
                       "p95_seconds": report["call_completion_seconds"]["p95"],
                       "seconds_per_call": report["measurement_seconds"] / report["calls"],
                       "calls_per_hour": report["calls_per_hour"],
                       "gpu_seconds_per_1000_calls": report["gpu_seconds_per_1000_calls"],
                       "rss_peak_bytes": report["memory"]["process_rss_sampled_peak_bytes"],
                       "cuda_peak_allocated_bytes": report["memory"]["cuda_peak_allocated_bytes"],
                       "cuda_peak_reserved_bytes": report["memory"]["cuda_peak_reserved_bytes"],
                       "format_valid_rate": quality["diagnostics"]["format_valid_rate"],
                       **{k: quality["official"][k] for k in QUALITY_METRICS}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--baseline-pred", required=True)
    parser.add_argument("--gold", default=str(Path(CASE_DATA_DIR) / 'val.jsonl'))
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
    baseline, _ = evaluate_predictions(gold, [r for r in all_pred if r["call_id"] in ids],
        scorer_version=reports[0]['quality'].get('scorer_version', '1'), catalog_path=catalog_for(args.gold))
    reference_quality = reports[0]['quality']
    reference_identity = {**reference_quality, 'scorer_version': reference_quality.get('scorer_version', '1')}
    if any(baseline.get(key) != reference_identity.get(key) for key in
           ('scorer_sha256', 'scorer_version', 'patch_adapter_sha256', 'catalog_sha256')):
        parser.error('Baseline scorer or catalog differs from measured reports')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "baseline.json", baseline)
    for row in rows:
        for metric in QUALITY_METRICS:
            reference = baseline["official"][metric]
            row[metric + "_delta_from_baseline"] = delta(row[metric], reference)
        row["format_valid_rate_delta_from_baseline"] = delta(
            row["format_valid_rate"], baseline["diagnostics"]["format_valid_rate"])
    write_csv(output / "comparison.csv", [
        {**row, "error_counts": json.dumps(row["error_counts"], sort_keys=True)
         if row["error_counts"] is not None else None} for row in rows])
    write_csv(output / "per_attribute.csv", attribute_rows(reports, baseline, args.reports))
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
