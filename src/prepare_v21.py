"""Validate and prepare the organizer 2.1 archive without replacing legacy data."""
import argparse
import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from config import ORGANIZER_DIR
from prepare_case import FILES, prepare

SCORER_SHA256 = "863f87737dd1108e772250967f43697fa2411b3394c190d192b7ff66666259c3"


def package_files(archive):
    with zipfile.ZipFile(archive) as outer:
        if "CompactFactExtraction_participants.zip" in outer.namelist():
            payload = outer.read("CompactFactExtraction_participants.zip")
        else:
            payload = Path(archive).read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        required = (*FILES, "ANSWERS_v2.md", "CHECKSUMS.txt")
        names = {}
        for name in bundle.namelist():
            base = Path(name).name
            if not name.endswith('/') and base in required:
                if base in names:
                    raise ValueError(f"Ambiguous archive member: {base}")
                names[base] = name
        if any(name not in names for name in required):
            raise ValueError("Archive lacks the complete organizer v2.1 package")
        # Only named data files are read; no archive paths or Python code are executed.
        data = {name: bundle.read(names[name]) for name in required}
    if hashlib.sha256(data['case_metrics.py']).hexdigest() != SCORER_SHA256:
        raise ValueError("Unexpected scorer version; review the organizer update first")
    warnings = []
    checks = {}
    for line in data['CHECKSUMS.txt'].decode('utf-8-sig').splitlines():
        parts = line.split()
        if len(parts) == 2:
            digest, name = parts
            name = name.lstrip('*')
            checks[name] = digest.lower()
    for name, content in data.items():
        if name not in checks:
            raise ValueError(f"Missing checksum: {name}")
        if hashlib.sha256(content).hexdigest() != checks[name]:
            if name == 'CHECKSUMS.txt':
                warnings.append('Organizer CHECKSUMS.txt self-checksum differs; other files verified')
            else:
                raise ValueError(f"Checksum mismatch: {name}")
    return data, warnings


def prepare_archive(archive, target):
    data, warnings = package_files(archive)
    target = Path(target)
    for name, content in data.items():
        existing = target / name
        if existing.exists() and existing.read_bytes() != content:
            raise ValueError(f"Destination differs: {existing}; choose a fresh directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v21-source-', dir=target.parent) as temp:
        source = Path(temp)
        for name, content in data.items():
            (source / name).write_bytes(content)
        report = prepare(source, target)
        for name in ('ANSWERS_v2.md', 'CHECKSUMS.txt'):
            if not (target / name).exists():
                shutil.copyfile(source / name, target / name)
    return {"data_directory": str(target.resolve()), "validation": report['validation'],
            "sha256": report['sha256'], "warnings": warnings, "metrics_version": "2.1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--out-dir', type=Path, default=ORGANIZER_DIR)
    args = parser.parse_args()
    print(json.dumps(prepare_archive(args.archive, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
