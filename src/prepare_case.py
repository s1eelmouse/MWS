"""Copy the case package locally, preserving original splits and file bytes."""
import argparse
from config import ORGANIZER_DIR
import hashlib
import json
import shutil
from pathlib import Path

from case_io import read_jsonl
from check_data import inspect_rows

FILES = (
    "train.jsonl", "val.jsonl", "baseline_pred.jsonl", "baseline_metrics.json",
    "catalog.json", "schema.md", "case_metrics.py", "README.md",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source: Path, target: Path) -> dict:
    source, target = source.resolve(), target.resolve()
    if source == target:
        raise ValueError("Source and destination must differ; originals stay untouched")
    for name in FILES:
        if not (source / name).is_file():
            raise ValueError(f"Missing organizer file: {name}")
        if (target / name).exists() and digest(source / name) != digest(target / name):
            raise ValueError(f"Destination differs: {target / name}; choose a fresh directory")
    train, val = read_jsonl(source / "train.jsonl"), read_jsonl(source / "val.jsonl")
    reports = {name: inspect_rows(rows) for name, rows in [("train", train), ("val", val)]}
    if any(report["invalid_calls"] for report in reports.values()):
        raise ValueError("Invalid organizer calls: inspect with check_data.py before preparing")
    train_ids, val_ids = {r["call_id"] for r in train}, {r["call_id"] for r in val}
    train_accounts = {r["call_id"].split(":")[0] for r in train}
    val_accounts = {r["call_id"].split(":")[0] for r in val}
    if train_ids & val_ids or train_accounts & val_accounts:
        raise ValueError("Train/validation call or account overlap")
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if not (target / name).exists():
            shutil.copyfile(source / name, target / name)
    report = {
        "sha256": {name: digest(source / name) for name in FILES},
        "validation": reports, "split_policy": "organizer train/val preserved; never re-split val",
    }
    manifest = target / "package_manifest.json"
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if manifest.exists() and manifest.read_text(encoding="utf-8") != content:
        raise ValueError("Existing package manifest differs; choose a fresh directory")
    if not manifest.exists():
        manifest.write_text(content, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--out-dir", default=ORGANIZER_DIR, type=Path)
    args = parser.parse_args()
    report = prepare(args.source_dir, args.out_dir)
    print(json.dumps(report["validation"], ensure_ascii=False, indent=2))
    print(f"Prepared {args.out_dir}; source files and organizer splits unchanged")


if __name__ == "__main__":
    main()
