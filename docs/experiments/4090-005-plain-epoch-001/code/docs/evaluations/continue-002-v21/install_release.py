"""Verified additive release install; never replaces the old runtime or model."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

RELEASE = 'case-v21-002'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(root):
    manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text())
    if manifest['release_id'] != RELEASE:
        raise ValueError('Wrong release ID')
    for name, expected in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or digest(root.joinpath(*relative.parts)) != expected:
            raise ValueError(f'Release file checksum mismatch: {name}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--team-root', default=str(Path.home() / 'team_shpana'))
    args = parser.parse_args()
    os.umask(0o077)
    archive_path, team = Path(args.archive), Path(args.team_root).resolve()
    if digest(archive_path) != args.expected_sha256:
        raise ValueError('Uploaded package checksum mismatch')
    parent = team / 'releases'
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = parent / RELEASE
    if destination.exists():
        check(destination)
        # Compare manifest bytes, so a different package cannot reuse this name.
        with tarfile.open(archive_path) as archive:
            expected = archive.extractfile(RELEASE + '/RELEASE_MANIFEST.json').read()
        if (destination / 'RELEASE_MANIFEST.json').read_bytes() != expected:
            raise FileExistsError('Different release already exists; preserve it and choose a new release name')
        print('ALREADY INSTALLED AND VERIFIED:', destination)
        return
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError('Duplicate archive entries')
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or name.parts[0] != RELEASE or not member.isfile():
                raise ValueError('Only regular relative files in the release directory are allowed')
        if shutil.disk_usage(parent).free < sum(m.size for m in members) + 256 * 2**20:
            raise RuntimeError('Insufficient free space for new release')
        staging = Path(tempfile.mkdtemp(prefix='.case-v21-staging-', dir=parent))
        archive.extractall(staging, filter='data')
    staged = staging / RELEASE
    manifest = check(staged)
    expected_files = {str(PurePosixPath(RELEASE) / name) for name in manifest['files']} | {RELEASE + '/RELEASE_MANIFEST.json'}
    if {m.name for m in members} != expected_files:
        raise ValueError('Archive inventory differs from the manifest')
    staged.rename(destination)
    staging.rmdir()
    print('RELEASE INSTALLED AND VERIFIED:', destination)
    print('Existing .venv-v100, models/continue-002, data and prior results preserved.')


if __name__ == '__main__':
    main()
