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
OPTIONAL_FILES = ("ANSWERS_v2.md", "CHECKSUMS.txt")  # present since package v2


def verify_checksums(source: Path) -> int:
    """Compare files with the organizer CHECKSUMS.txt, if the package has one.

    The list cannot contain a valid hash of itself, so CHECKSUMS.txt is skipped.
    """
    listing = source / "CHECKSUMS.txt"
    if not listing.is_file():
        return 0
    checked = 0
    for line in listing.read_text(encoding="utf-8").splitlines():
        expected, _, name = line.strip().partition("  ")
        if not name or name == "CHECKSUMS.txt":
            continue
        if not (source / name).is_file() or digest(source / name) != expected:
            raise ValueError(f"{name} does not match organizer CHECKSUMS.txt; re-download the package")
        checked += 1
    return checked


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source: Path, target: Path) -> dict:
    source, target = source.resolve(), target.resolve()
    if source == target:
        raise ValueError("Source and destination must differ; originals stay untouched")
    for name in FILES:
        if not (source / name).is_file():
            raise ValueError(f"Missing organizer file: {name}")
    checked = verify_checksums(source)
    names = FILES + tuple(name for name in OPTIONAL_FILES if (source / name).is_file())
    for name in names:
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
    for name in names:
        if not (target / name).exists():
            shutil.copyfile(source / name, target / name)
    report = {
        "sha256": {name: digest(source / name) for name in names},
        "organizer_checksums_verified": checked,
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
