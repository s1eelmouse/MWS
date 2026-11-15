"""Copy the case package locally, preserving original splits and file bytes."""
import argparse
import hashlib
import json
import shutil
import io
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from case_io import read_jsonl
from check_data import inspect_rows
from config import CASE_DATA_DIR

FILES = (
    "train.jsonl", "val.jsonl", "baseline_pred.jsonl", "baseline_metrics.json",
    "catalog.json", "schema.md", "case_metrics.py", "README.md",
)
OPTIONAL_FILES = ('ANSWERS_v2.md', 'CHECKSUMS.txt')
SCORER_SHA256 = '863f87737dd1108e772250967f43697fa2411b3394c190d192b7ff66666259c3'


def checksum_entries(text):
    entries = {}
    for line in text.lstrip('\ufeff').splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'([0-9a-fA-F]{64})\s+\*?(.+)', line.strip())
        if not match:
            raise ValueError('Malformed organizer checksum line')
        expected, name = match.groups()
        if name not in FILES + OPTIONAL_FILES or PurePosixPath(name).name != name or name in entries:
            raise ValueError('Unexpected or duplicate checksum filename: ' + name)
        entries[name] = expected.lower()
    return entries


def verify_checksums(source):
    """Verify supplied bytes before copying; the self-referential checksum is excluded."""
    source = Path(source)
    listing = source / 'CHECKSUMS.txt'
    if not listing.is_file():
        return 0
    entries = checksum_entries(listing.read_text(encoding='utf-8-sig'))
    checked = 0
    for name, expected in entries.items():
        if name == 'CHECKSUMS.txt':
            continue
        if not (source / name).is_file() or digest(source / name) != expected:
            raise ValueError(f'{name} does not match organizer CHECKSUMS.txt; re-download the package')
        checked += 1
    return checked


def package_files(archive):
    """Read only allowlisted members of direct/nested v2.1 ZIPs; execute no source."""
    required = FILES + OPTIONAL_FILES
    def read_bundle(bundle):
        members = {}
        for info in bundle.infolist():
            name = PurePosixPath(info.filename).name
            if not info.is_dir() and name in required:
                if name in members:
                    raise ValueError('Ambiguous archive member: ' + name)
                if info.file_size > 128 * 2**20:
                    raise ValueError('Organizer file exceeds 128 MiB safety limit')
                members[name] = info
        if set(members) != set(required):
            return None
        return {name: bundle.read(info) for name, info in members.items()}
    with zipfile.ZipFile(archive) as outer:
        data = read_bundle(outer)
        if data is None:
            nested = [info for info in outer.infolist()
                      if PurePosixPath(info.filename).name == 'CompactFactExtraction_participants.zip']
            if len(nested) != 1 or nested[0].file_size > 256 * 2**20:
                raise ValueError('Archive lacks the complete organizer v2.1 package')
            with zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) as inner:
                data = read_bundle(inner)
    if data is None:
        raise ValueError('Archive lacks the complete organizer v2.1 package')
    if hashlib.sha256(data['case_metrics.py']).hexdigest() != SCORER_SHA256:
        raise ValueError('Unexpected scorer version; review the organizer update first')
    checks = checksum_entries(data['CHECKSUMS.txt'].decode('utf-8-sig'))
    warnings = []
    for name, value in data.items():
        if name not in checks:
            raise ValueError('Missing checksum: ' + name)
        if hashlib.sha256(value).hexdigest() != checks[name]:
            if name == 'CHECKSUMS.txt':
                warnings.append('Organizer CHECKSUMS.txt self-checksum differs; other files verified')
            else:
                raise ValueError('Checksum mismatch: ' + name)
    return data, warnings


def prepare_archive(archive, target):
    data, warnings = package_files(archive)
    target = Path(target)
    for name, value in data.items():
        if (target / name).exists() and (target / name).read_bytes() != value:
            raise ValueError(f'Destination differs: {target / name}; choose a fresh directory')
    with tempfile.TemporaryDirectory(prefix='mws-v21-source-') as temp:
        source = Path(temp)
        for name, value in data.items():
            (source / name).write_bytes(value)
        report = prepare(source, target)
    return {'data_directory': str(target.resolve()), 'validation': report['validation'],
            'sha256': report['sha256'], 'warnings': warnings, 'metrics_version': '2.1'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source: Path, target: Path) -> dict:
    source, target = source.resolve(), target.resolve()
    if source == target:
        raise ValueError("Source and destination must differ; originals stay untouched")
    checked = verify_checksums(source)
    names = FILES + tuple(name for name in OPTIONAL_FILES if (source / name).is_file())
    for name in names:
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
    for name in names:
        if not (target / name).exists():
            shutil.copyfile(source / name, target / name)
    report = {
        "sha256": {name: digest(source / name) for name in names},
        "validation": reports, "split_policy": "organizer train/val preserved; never re-split val",
        'organizer_checksums_verified': checked,
        'validation_contract_version': '2.1',
        'organizer_scorer_version': {'863f87737dd1108e772250967f43697fa2411b3394c190d192b7ff66666259c3': '2.1',
                                    'dffec9b8b20c5a8a101a33a27b1672f2b6195f00ab60f021a1c14b31eeeacabd': '1'}.get(digest(source / 'case_metrics.py'), 'unrecognized'),
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir", type=Path)
    source.add_argument('--archive', type=Path, help='direct or nested organizer v2.1 ZIP')
    parser.add_argument("--out-dir", default=Path(CASE_DATA_DIR), type=Path)
    args = parser.parse_args()
    report = prepare_archive(args.archive, args.out_dir) if args.archive else prepare(args.source_dir, args.out_dir)
    for warning in report.get('warnings', []):
        print('WARNING:', warning)
    print(json.dumps(report["validation"], ensure_ascii=False, indent=2))
    print(f"Prepared {args.out_dir}; source files and organizer splits unchanged")


if __name__ == "__main__":
    main()
