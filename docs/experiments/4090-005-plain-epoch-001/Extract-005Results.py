"""Extract a downloaded results archive without overwriting older experiments."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',required=True,type=Path)
    ap.add_argument('--sha256',required=True)
    ap.add_argument('--destination',required=True,type=Path)
    args=ap.parse_args()
    destination=args.destination.resolve()
    if destination.exists():raise FileExistsError('Extraction directory already exists; preserve previous results.')
    actual=digest(args.archive)
    if actual!=args.sha256:raise ValueError('Archive checksum mismatch')
    with tarfile.open(args.archive) as archive:
        members=archive.getmembers();files=set()
        for member in members:
            name=member.name.replace('\\','/')
            rel=PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts or ':' in name:
                raise ValueError('Unsafe archive path: '+member.name)
            if not member.isfile() and not member.isdir():
                raise ValueError('Only regular files/directories are accepted: '+member.name)
            target=destination.joinpath(*rel.parts).resolve()
            if not target.is_relative_to(destination):raise ValueError('Archive path leaves results directory')
            if member.isfile():
                key=str(target).casefold()
                if key in files:raise ValueError('Duplicate file in archive: '+member.name)
                files.add(key)
        required=sum(m.size for m in members if m.isfile())
        if shutil.disk_usage(destination.parent).free<required+64*2**20:
            raise RuntimeError('Insufficient space to extract archive')
        destination.mkdir()
        archive.extractall(destination,filter='data')
    inventory=[{'path':p.relative_to(destination).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)}
               for p in sorted(destination.rglob('*')) if p.is_file()]
    manifest={'downloaded_utc':datetime.now(timezone.utc).isoformat(),'archive':str(args.archive.resolve()),
              'archive_sha256':actual,'server_checksum_matches':True,'extracted_files':len(inventory),
              'destination':str(destination),'files':inventory}
    (destination.parent/'download-verification.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('DOWNLOAD AND EXTRACTION COMPLETE')
    print('Files:',len(inventory))
    print('Results:',destination)
    for item in inventory:
        if Path(item['path']).name in ('raw-answers.jsonl','predictions.jsonl','predictions-before-series.jsonl','report.json'):
            print('Found:',item['path'])


if __name__=='__main__':main()
