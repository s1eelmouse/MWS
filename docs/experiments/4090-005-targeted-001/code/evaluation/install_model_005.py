"""Install only a verified merged 005 model from an existing server archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

EXPECTED = 'cef9fcde9f1f2a812e42c6122a0f5094fbaa456f6077596cf03ebb86e4d552f6'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(archive):
    files = {}
    for member in archive.getmembers():
        name = PurePosixPath(member.name)
        if name.is_absolute() or '..' in name.parts or '\\' in member.name or ':' in member.name:
            raise ValueError('Unsafe archive path: ' + member.name)
        if member.isdir():
            continue
        if not member.isfile() or str(name) in files:
            raise ValueError('Links/special files/duplicate names are not accepted')
        files[str(name)] = member
    candidates = []
    for name in files:
        p = PurePosixPath(name)
        if p.name == 'model.safetensors' and all(str(p.parent / n) in files
                for n in ('config.json', 'tokenizer.json', 'tokenizer_config.json')):
            candidates.append(name)
    return files, sorted(candidates)


def check_assets(directory, expected_files):
    for name, checksum in (expected_files or {}).items():
        if digest(directory / name) != checksum:
            raise ValueError('Model asset checksum mismatch: ' + name)


def install(archive_path, destination, member_name=None, expected=EXPECTED, expected_files=None):
    destination = Path(destination).resolve()
    if destination.exists():
        manifest = json.loads((destination / 'INSTALL_005.json').read_text())
        for name, checksum in manifest['files'].items():
            if digest(destination / name) != checksum:
                raise ValueError('Previously installed model changed: ' + name)
        if manifest['files']['model.safetensors'] != expected:
            raise ValueError('Wrong installed weights')
        check_assets(destination, expected_files)
        print('005 ALREADY INSTALLED AND VERIFIED:', destination)
        return
    with tarfile.open(archive_path) as archive:
        files, candidates = inventory(archive)
        if member_name:
            member_name = str(PurePosixPath(member_name))
            if member_name not in candidates:
                raise ValueError('--member must name a complete merged model: ' + repr(candidates))
        elif len(candidates) == 1:
            member_name = candidates[0]
        else:
            raise ValueError('Need a unique merged model. Inspect candidates and pass --member: ' + repr(candidates))
        parent = PurePosixPath(member_name).parent
        selected = {PurePosixPath(n).name: m for n, m in files.items()
                    if PurePosixPath(n).parent == parent}
        destination.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destination.parent).free < sum(m.size for m in selected.values()) + 256 * 2**20:
            raise RuntimeError('Not enough free disk space')
        staging = Path(tempfile.mkdtemp(prefix='.005-staging-', dir=destination.parent))
        print('Installing merged files from:', str(parent), flush=True)
        for name, member in selected.items():
            with archive.extractfile(member) as source, (staging / name).open('xb') as target:
                shutil.copyfileobj(source, target, length=2**20)
        checksums = {n: digest(staging / n) for n in selected}
        if checksums['model.safetensors'] != expected:
            raise ValueError('005 weight checksum mismatch; staging preserved at ' + str(staging))
        check_assets(staging, expected_files)
        (staging / 'INSTALL_005.json').write_text(json.dumps({
            'archive': str(Path(archive_path).resolve()), 'archive_model_member': member_name,
            'files': checksums, 'expected_weights_sha256': expected}, indent=2) + '\n')
        staging.rename(destination)
    print('005 INSTALLED AND VERIFIED:', destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path.home() / '005-noise-2026-11-15.tar.gz')
    parser.add_argument('--output', type=Path, default=Path.home() / 'mws-v21/models/005-noise-20261115')
    parser.add_argument('--member', help='Exact model.safetensors member if the archive has several models')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    if args.list:
        with tarfile.open(args.archive) as archive:
            print(json.dumps(inventory(archive)[1], indent=2))
    else:
        expected_files = json.loads((Path(__file__).parent / 'expected_models.json').read_text())['005']
        install(args.archive, args.output, args.member, expected_files=expected_files)


if __name__ == '__main__':
    main()
