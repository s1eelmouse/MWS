"""Pack this comparison's reports, predictions and logs; never model weights."""
import argparse
import hashlib
from pathlib import Path
import re
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home() / 'mws-v21')
    parser.add_argument('--run-id', default='4090-unified-005-001')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.run_id):
        parser.error('Invalid run ID')
    result = args.root.resolve() / 'out' / args.run_id
    if not result.is_dir():
        raise FileNotFoundError(result)
    archive = Path.home() / (args.run_id + '-results.tar.gz')
    with tarfile.open(archive, 'x:gz') as bundle:
        for path in sorted(result.rglob('*')):
            if path.is_symlink():
                raise ValueError('Unexpected symlink: ' + str(path))
            if path.is_file():
                bundle.add(path, arcname=args.run_id + '/' + path.relative_to(result).as_posix())
        log = Path(str(result) + '.log')
        if log.is_file():
            bundle.add(log, arcname=args.run_id + '.log')
    with archive.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    Path(str(archive) + '.sha256').write_text(checksum + '\n')
    print(archive, checksum)


if __name__ == '__main__':
    main()
