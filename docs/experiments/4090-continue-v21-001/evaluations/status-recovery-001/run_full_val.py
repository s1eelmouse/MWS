"""Evaluation only: existing v2.1-trained merged model, immutable status release."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

RELEASE = 'case-v21-status-001'
MODEL_SHA256 = '193691230bba18d7c772d70e94651c425ca494081a42570a8b948cbc6175c044'
VAL_SHA256 = 'e12610354aa69dff83e1d7c1e8c94736c97d5761d20af58790435f4be1941808'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id',default='4090-v21-status-001')
    ap.add_argument('--batch-size',type=int,choices=range(1,9),default=2)
    ap.add_argument('--root',type=Path,default=Path.home()/'mws-v21')
    ap.add_argument('--model',type=Path)
    ap.add_argument('--check-only',action='store_true',help='verify files and show command; no model loading')
    args=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.run_id):ap.error('Use a simple unique run ID')
    team=args.root.resolve();code=team/'releases'/RELEASE
    model=(args.model or team/'out/4090-continue-v21-001/merged').resolve()
    data=code/'data/organizer-v2.1';root=team/'out'/args.run_id
    manifest=json.loads((code/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest['release_id']!=RELEASE:raise ValueError('Wrong release')
    for name,expected in manifest['files'].items():
        if digest(code/name)!=expected:raise ValueError('Release checksum mismatch: '+name)
    if digest(data/'val.jsonl')!=VAL_SHA256:raise ValueError('Wrong val; must be complete v2.1')
    if len((data/'val.jsonl').read_text(encoding='utf-8').splitlines())!=383:raise ValueError('Expected 383 calls')
    if digest(model/'model.safetensors')!=MODEL_SHA256:raise ValueError('Wrong model weights; expected the v2.1-trained candidate')
    benchmark=[sys.executable,'-u',str(code/'src/benchmark.py'),'--input',str(data/'val.jsonl'),
               '--model',str(model),'--device','cuda','--batch-size',str(args.batch_size),
               '--warmup-batches','1','--output-dir',str(root/'validation')]
    if args.check_only:
        print(json.dumps({'status':'files_verified','release':str(code),'model':str(model),
                          'evaluation_command':benchmark},ensure_ascii=False,indent=2))
        return
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CASE_DATA_DIR=str(data),ORGANIZER_DIR=str(data),
        SCORER_VERSION='2.1',ALLOW_LEGACY_CONTRACT='0',
        CATALOG_PATH=str(data/'catalog.json'),TEMPORAL_CATALOG_PATH=str(data/'catalog.json'),
        MAX_INPUT_TOKENS='12288',MAX_SEQUENCE_TOKENS='16384',MAX_NEW_TOKENS='4096',MAX_TEXT_CHARS='0',
        JSON_CONSTRAINED='1',JSON_GRAMMAR_BACKEND='torch_native',JSON_GRAMMAR_THREADS='4',
        TEMPORAL_NORMALIZATION='case_v2',TEMPORAL_SERIES_HISTORY='1',FACT_RECOVERY='1',
        TEMPORAL_SERIES_MAX_ENTRIES='10000',HISTORY_MAX_ACCOUNTS='512',HISTORY_MAX_BYTES='8388608',
        HISTORY_TTL_SECONDS='3600',SAVE_RAW_RESPONSES='1',PROMPT_LOOKUP_TOKENS='0',
        RETRY_ON_TOKEN_LIMIT='1',RETRY_REPETITION_PENALTY='1.2',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',MWS_CODE_REVISION=RELEASE)
    sys.path.insert(0,str(code/'src'))
    from run_experiment import stage
    import fcntl
    root.parent.mkdir(parents=True,exist_ok=True)
    with (root.parent/'v21-validation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        root.mkdir(exist_ok=False)
        settings={'created_utc':datetime.now(timezone.utc).isoformat(),'release':RELEASE,
                  'model':str(model),'model_sha256':MODEL_SHA256,'val_sha256':VAL_SHA256,
                  'training':False,'batch_size':args.batch_size,'calls':383,
                  'scorer':'2.1-unmodified','environment':{k:os.environ[k] for k in (
                    'FACT_RECOVERY','TEMPORAL_NORMALIZATION','TEMPORAL_SERIES_HISTORY','JSON_CONSTRAINED',
                    'JSON_GRAMMAR_BACKEND','JSON_GRAMMAR_THREADS','RETRY_ON_TOKEN_LIMIT',
                    'MAX_INPUT_TOKENS','MAX_SEQUENCE_TOKENS','MAX_NEW_TOKENS')},
                  'runner_sha256':digest(Path(__file__))}
        (root/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (root/'release-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        try:
            stage(root,'tests',[sys.executable,'-m','unittest','discover','-s',str(code/'tests'),'-v'])
            stage(root,'validation',benchmark)
        except BaseException as exc:
            (root/'pipeline-status.json').write_text(json.dumps({'status':'failed','error':str(exc)})+'\n',encoding='utf-8')
            raise
        (root/'pipeline-status.json').write_text('{"status":"complete"}\n',encoding='utf-8')
        print('DONE:',root/'validation/report.json',flush=True)


if __name__=='__main__':main()
