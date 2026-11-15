"""Package actual local runtime bytes; no Git mutations, models or raw run data."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

RELEASE='case-v21-status-001'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',required=True,type=Path)
    ap.add_argument('--output-dir',required=True,type=Path)
    args=ap.parse_args()
    root=args.repo.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    files={}
    def add(path):
        if path.is_symlink():raise ValueError('No symlinks in release')
        if set(path.parts)&{'__pycache__','download'} or path.suffix=='.pyc':return
        files[path.relative_to(root).as_posix()]=path.read_bytes()
    folders=('src','tests','data/organizer-v2.1','docs/evaluations','docs/migrations/4090-to-v100',
             'docs/experiments/4090-continue-v21-001/evaluations/status-recovery-001',
             'docs/experiments/4090-continue-v21-001/comparisons/state-audit-001')
    for folder in folders:
        for path in (root/folder).rglob('*'):
            if path.is_file():add(path)
    for path in root.iterdir():
        if path.is_file() and (path.name in ('AGENTS.md','README.md','.gitignore','.gitattributes','.dockerignore')
                              or path.name.startswith(('requirements','Dockerfile','docker-compose'))):add(path)
    add(root/'data/sample_case.jsonl')
    for name in files:
        if set(Path(name).parts)&{'.git','.venv','out','download'} or name.endswith(('.pem','.pub','.safetensors')):
            raise ValueError('Unexpected artifact in runtime: '+name)
    for name in ('official_case_metrics.py','official_case_metrics_v21.py'):
        assert hashlib.sha256(files['src/'+name]).hexdigest()==files['src/'+name.replace('.py','.sha256')].decode().strip()
    assert files['src/official_case_metrics_v21.py']==files['data/organizer-v2.1/case_metrics.py']
    manifest={'release_id':RELEASE,'created_utc':datetime.now(timezone.utc).isoformat(),
              'base_git_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
              'working_tree_status':subprocess.check_output(['git','-C',str(root),'status','--short'],text=True),
              'purpose':'Evaluation-only deployment: intervals plus ticket assertion recovery. Existing trained weights unchanged.',
              'files':{name:hashlib.sha256(value).hexdigest() for name,value in sorted(files.items())}}
    files['RELEASE_MANIFEST.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    target=out/('team-shpana-'+RELEASE+'.tar.gz')
    with target.open('xb') as stream,gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name,value in sorted(files.items()):
                item=tarfile.TarInfo(RELEASE+'/'+name);item.size=len(value);item.mtime=0;item.mode=0o644
                archive.addfile(item,io.BytesIO(value))
    sha=hashlib.sha256(target.read_bytes()).hexdigest()
    Path(str(target)+'.sha256').write_text(sha+'\n',encoding='ascii')
    with tarfile.open(target) as archive:
        assert len(archive.getmembers())==len(files)
        for member in archive:
            assert archive.extractfile(member).read()==files[member.name.removeprefix(RELEASE+'/')]
    (out/'status-release-manifest.json').write_bytes(files['RELEASE_MANIFEST.json'])
    print(json.dumps({'archive':str(target),'sha256':sha,'bytes':target.stat().st_size,'files':len(files)},indent=2))


if __name__=='__main__':main()
