"""One plain organizer 2.1 epoch from merged 005, then one full clean validation."""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

TRAIN_SHA='0878bfd983e14c75b51f32f04d961c5b82b10fa243845a25fe1c811cb4882466'
RUN='005-plain-epoch-001'

def digest(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def install_train(root):
 destination=root/'data/organizer-v2.1-verified/train.jsonl'
 if destination.exists():
  if digest(destination)!=TRAIN_SHA:raise ValueError('Existing train differs; preserve it')
  return destination
 archive=root/'train-v21-original.jsonl.gz'
 destination.parent.mkdir(parents=True,exist_ok=True)
 partial=destination.with_suffix('.jsonl.partial')
 with gzip.open(archive,'rb') as source,partial.open('xb') as target:
  total=0
  while chunk:=source.read(2**20):
   total+=len(chunk)
   if total>64*2**20:raise ValueError('Unexpected train size')
   target.write(chunk)
 if digest(partial)!=TRAIN_SHA:raise ValueError('Train checksum mismatch; partial preserved')
 partial.rename(destination)
 return destination

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--root',type=Path,default=Path.home()/'mws-v21')
 ap.add_argument('--install-train',action='store_true')
 ap.add_argument('--check-only',action='store_true')
 args=ap.parse_args();team=args.root.resolve()
 if args.install_train:
  print('ORIGINAL TRAIN VERIFIED:',install_train(team));return
 code=team/'releases/case-v21-unified-001'
 spec=importlib.util.spec_from_file_location('release',code/'evaluation/run_comparison.py')
 release=importlib.util.module_from_spec(spec);spec.loader.exec_module(release)
 manifest=release.verify_release(code)
 model=team/'models/005-noise-20261115'
 assets=json.loads((code/'evaluation/expected_models.json').read_text())['005']
 for name,expected in assets.items():
  if digest(model/name)!=expected:raise ValueError('Wrong 005 asset: '+name)
 train=team/'data/organizer-v2.1-verified/train.jsonl'
 if digest(train)!=TRAIN_SHA:raise ValueError('Wrong training data')
 if len(train.read_text(encoding='utf-8').splitlines())!=3926:raise ValueError('Expected 3926 training calls')
 release.configure(code)
 # All old 005 updates are already inside these merged weights. A new LoRA and
 # optimizer are created; never load the original 005 adapter on top a second time.
 os.environ.update(BASE_MODEL=str(model),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 val=code/'data/organizer-v2.1/val.jsonl';catalog=val.parent/'catalog.json'
 root=team/'out'/RUN
 train_command=[sys.executable,'-u',str(code/'src/train.py'),'--train',str(train),
  '--val',str(val),'--catalog',str(catalog),'--train-variant','plain','--epochs','1',
  '--batch-size','1','--learning-rate','0.00003','--seed','42','--save-steps','50',
  '--output-dir',str(root/'adapter')]
 merge_command=[sys.executable,'-u',str(code/'src/merge.py'),'--adapter-dir',str(root/'adapter'),
  '--output',str(root/'merged')]
 benchmark=[sys.executable,'-u',str(code/'src/benchmark.py'),'--model',str(root/'merged'),
  '--input',str(val),'--device','cuda','--batch-size','2','--warmup-batches','1',
  '--output-dir',str(root/'validation')]
 settings={'run':RUN,'created_utc':datetime.now(timezone.utc).isoformat(),'epochs':1,
  'train_variant':'plain','source_model':str(model),'source_model_assets':assets,
  'source_model_policy':'fresh LoRA on merged 005; no init-adapter and no optimizer resume',
  'learning_rate':3e-5,'seed':42,'microbatch':1,'effective_batch':16,'expected_steps':246,
  'train_sha256':TRAIN_SHA,'val_sha256':digest(val),'catalog_sha256':digest(catalog),
  'source_release':'case-v21-unified-001','services_candidate_enabled':False,
  'runner_sha256':digest(Path(__file__)),'commands':{'train':train_command,'merge':merge_command,'validation':benchmark},
  'inference':{k:os.environ[k] for k in ('TEMPORAL_NORMALIZATION','FACT_RECOVERY',
   'TEMPORAL_SERIES_HISTORY','JSON_CONSTRAINED','JSON_GRAMMAR_BACKEND','JSON_GRAMMAR_THREADS',
   'MAX_INPUT_TOKENS','MAX_NEW_TOKENS','TARIFF_BOUNDARY_POLICY')},
  'clean_baseline':{'fact_precision':.5415,'fact_recall':.5805,'temporal_recall':.5061,
   'state_accuracy':.5602,'grounding_turn':.7824},
  'baseline_kind':'full clean CPU replay of saved 005 generations; historical inference stack differs'}
 if args.check_only:
  if root.exists():raise FileExistsError('Run already exists; do not launch twice: '+str(root))
  print(json.dumps(settings,indent=2));return
 sys.path.insert(0,str(code/'src'))
 from run_experiment import stage
 import fcntl
 root.parent.mkdir(parents=True,exist_ok=True)
 with (root.parent/'v21-validation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  root.mkdir(exist_ok=False)
  (root/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
  (root/'release-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  status={'status':'failed'}
  try:
   stage(root,'tests',[sys.executable,'-m','unittest','discover','-s',str(code/'tests'),'-v'])
   stage(root,'data-check',[sys.executable,str(code/'src/train.py'),'--train',str(train),'--val',str(val),
    '--catalog',str(catalog),'--train-variant','plain','--check-data'])
   stage(root,'train',train_command)
   stage(root,'merge',merge_command)
   stage(root,'validation',benchmark)
   report=json.loads((root/'validation/report.json').read_text())
   current=report['quality']['official']
   comparison={'before':settings['clean_baseline'],'after':current,
    'delta_percentage_points':{k:round(100*(current[k]-v),4) for k,v in settings['clean_baseline'].items()},
    'failed_calls':report['failed_calls'],'automatic_replacement':False,
    'noisy_validation':'not run; required before claiming noise robustness'}
   (root/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
   status={'status':'complete','automatic_replacement':False}
   print('DONE:',root/'comparison.json',flush=True)
  finally:(root/'pipeline-status.json').write_text(json.dumps(status)+'\n')

if __name__=='__main__':main()
