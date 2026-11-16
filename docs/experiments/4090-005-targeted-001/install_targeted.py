"""Install a pinned train/code bundle into a NEW directory; never overwrite models."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile
import tempfile

RUN='005-targeted-001'
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def install(archive,expected,root):
    archive=Path(archive).resolve();root=Path(root).resolve()
    if len(expected)!=64 or digest(archive)!=expected:raise ValueError('Wrong bundle SHA256')
    target=root/'targeted'/(RUN+'-trainonly')
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive path')
        for member in members:
            p=PurePosixPath(member.name)
            if not member.isfile() or p.is_absolute() or '..' in p.parts or '\\' in member.name or ':' in member.name:
                raise ValueError('Unsafe archive entry')
        mf=next((m for m in members if m.name=='MANIFEST.json'),None)
        if mf is None:raise ValueError('Manifest missing')
        manifest=json.load(bundle.extractfile(mf))
        if manifest.get('run')!=RUN or set(names)!=set(manifest['files'])|{'MANIFEST.json'}:raise ValueError('Unexpected bundle files')
        if sum(m.size for m in members)>150*2**20:raise ValueError('Oversized bundle')
        if target.exists():
            for name,sha in manifest['files'].items():
                if not (target/name).is_file() or digest(target/name)!=sha:raise ValueError('Existing installation differs; preserve it')
            if (target/'MANIFEST.json').read_bytes()!=bundle.extractfile(mf).read():raise ValueError('Existing manifest differs')
            print('ALREADY INSTALLED AND VERIFIED:',target);return
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.targeted-',dir=target.parent) as temp:
            staging=Path(temp)/'package';staging.mkdir()
            for member in members:
                dest=staging/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
                with bundle.extractfile(member) as source,dest.open('xb') as out:shutil.copyfileobj(source,out)
                if member.name!='MANIFEST.json' and digest(dest)!=manifest['files'][member.name]:raise ValueError('Corrupt bundle file')
            if target.exists():raise FileExistsError('Concurrent install; preserve it')
            staging.rename(target)
    print('INSTALLED AND VERIFIED:',target)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--sha256',required=True);ap.add_argument('--root',type=Path,default=Path.home()/'mws-v21')
    args=ap.parse_args();install(args.archive,args.sha256,args.root)
if __name__=='__main__':main()
