"""One conservative focus epoch, fresh LoRA, full noisy and clean evaluations."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

RUN='005-targeted-001'
PARENT_SHA='232db122df3e8666e4da818290741d1a55d5ad38dc2892657d525116e7e3b4ce'
DATA_SHA='2005ae5b24ba61ce73287dadbfce20077be8a65913e725322399d49f8f55d563'
INPUT_SHA={'clean':'e12610354aa69dff83e1d7c1e8c94736c97d5761d20af58790435f4be1941808',
           'noisy':'c76823bc32fdcfc6e62410c2d0f8222043220130f3a0c31345cd7025b12e85da'}

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def verify_package(package):
    manifest=json.loads((package/'MANIFEST.json').read_text())
    for name,expected in manifest['files'].items():
        if digest(package/name)!=expected:raise ValueError('Changed package file: '+name)
    return manifest

def prepare(team,package):
    manifest=verify_package(package)
    old=team/'releases/case-v21-unified-001'
    spec=importlib.util.spec_from_file_location('reference_profile',old/'evaluation/run_comparison.py')
    profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
    profile.verify_release(old);profile.configure(old)
    code=package/'code';model=team/'out/005-plain-epoch-001/merged'
    prior=json.loads((team/'out/005-plain-epoch-001/validation/report.json').read_text())
    if prior['status']!='complete' or prior['calls']!=383:raise ValueError('Parent run incomplete')
    if digest(model/'model.safetensors')!=PARENT_SHA:raise ValueError('Wrong parent weights')
    for name,info in prior['model_artifacts'].items():
        if digest(model/name)!=info['sha256']:raise ValueError('Changed parent asset: '+name)
    train=package/'data/train_targeted.jsonl'
    if digest(train)!=DATA_SHA:raise ValueError('Wrong targeted train')
    inputs={'clean':old/'data/organizer-v2.1/val.jsonl','noisy':old/'data/noisy-015/val-noisy-015.jsonl'}
    for label,path in inputs.items():
        if digest(path)!=INPUT_SHA[label]:raise ValueError('Wrong evaluation data: '+label)
    os.environ.update(BASE_MODEL=str(model),AGREEMENT_CLEANUP='0',NOISE_ROBUST_RECOVERY='0',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MWS_CODE_REVISION=RUN)
    sys.path.insert(0,str(code/'src'))
    from case_io import read_jsonl,validate_gold
    from training_data import ensure_training_splits
    data=read_jsonl(train);validation=read_jsonl(inputs['clean'])
    ensure_training_splits(data,validation)
    for row in data+validation:validate_gold(row)
    if len(data)!=5962:raise ValueError('Unexpected training count')
    context={'run':RUN,'created_utc':datetime.now(timezone.utc).isoformat(),'parent_model':str(model),
      'parent_weights_sha256':PARENT_SHA,'policy':'fresh LoRA on merged parent, fresh optimizer; no init-adapter/resume',
      'epochs':1,'learning_rate':1e-5,'seed':42,'microbatch':1,'effective_batch':16,'expected_steps':math.ceil(len(data)/16),
      'train_sha256':DATA_SHA,'input_sha256':INPUT_SHA,'source_release':'case-v21-unified-001',
      'package':str(package),'package_manifest_sha256':digest(package/'MANIFEST.json'),
      'dataset_provenance':json.loads((package/'data/manifest.json').read_text()),
      'prediction_rules_changed':False,'loss_algorithm_changed':False,
      'checkpoint_save_steps':100,'checkpoint_retention':4,'automatic_promotion':False,
      'environment':{k:os.environ[k] for k in ('BASE_MODEL','SCORER_VERSION','TEMPORAL_NORMALIZATION','FACT_RECOVERY',
        'TEMPORAL_SERIES_HISTORY','JSON_CONSTRAINED','JSON_GRAMMAR_BACKEND','JSON_GRAMMAR_THREADS',
        'MAX_INPUT_TOKENS','MAX_NEW_TOKENS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
    return code,model,train,inputs,context,manifest

def token_preflight(model,train,val):
    from transformers import AutoTokenizer
    from case_io import read_jsonl
    from training_data import encode_row
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    maximum=0;records=0
    for path in ((train,) if val is None else (train,val)):
        for row in read_jsonl(path):
            encoded=encode_row(tokenizer,row,16384)
            maximum=max(maximum,len(encoded['input_ids']));records+=1
    print(f'TOKEN BUDGETS OK: {records} rows; maximum sequence {maximum}',flush=True)
    return {'rows':records,'max_sequence_tokens_observed':maximum,'no_truncation':True}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.home()/'mws-v21')
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--train-only',action='store_true',default=True,help='train and merge only; no validation loss or inference (default)')
    args=ap.parse_args();team=args.root.resolve();package=Path(__file__).resolve().parent
    code,model,train,inputs,settings,manifest=prepare(team,package)
    settings['training_only']=args.train_only
    output=team/'out'/RUN
    if output.exists():raise FileExistsError('Existing run is protected: '+str(output))
    if args.check_only:
        print(json.dumps({'status':'FILES AND DATA VERIFIED','run':RUN,'rows':5962,'steps':373,
                          'training_started':False,'model':str(model),'learning_rate':1e-5},indent=2));return
    import fcntl
    from run_experiment import stage
    output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/'v21-validation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output.mkdir(exist_ok=False)
        (output/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2)+'\n')
        (output/'release-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        status={'status':'failed','automatic_promotion':False}
        try:
            if not args.train_only:
                stage(output,'tests',[sys.executable,'-m','unittest','discover','-s',str(code/'tests'),'-v'])
            token_info=token_preflight(model,train,None if args.train_only else inputs['clean'])
            (output/'token-budget.json').write_text(json.dumps(token_info,indent=2)+'\n')
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            if not torch.cuda.is_bf16_supported():raise RuntimeError('Expected native BF16-capable RTX 4090')
            free,total=torch.cuda.mem_get_info()
            if free<16*2**30:raise RuntimeError('Less than 16 GiB VRAM free; do not start concurrent training')
            (output/'gpu-preflight.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(0),
                'free_bytes':free,'total_bytes':total,'torch':torch.__version__},indent=2)+'\n')
            common=[sys.executable,'-u',str(code/'src/train.py'),'--train',str(train),'--val',str(inputs['clean']),
                    '--catalog',str(inputs['clean'].parent/'catalog.json'),'--train-variant','plain',
                    '--epochs','1','--batch-size','1','--learning-rate','0.00001','--seed','42']
            if args.train_only:common+=['--no-eval']
            stage(output,'smoke-train',common+['--limit','16','--max-steps','1','--save-steps','1',
                  '--output-dir',str(output/'smoke-adapter')])
            stage(output,'train',common+['--save-steps','100','--save-total-limit','4','--output-dir',str(output/'adapter')])
            stage(output,'merge',[sys.executable,'-u',str(code/'src/merge.py'),'--adapter-dir',str(output/'adapter'),
                                  '--output',str(output/'merged')])
            if args.train_only:
                status={'status':'complete','training_only':True,'validation':'not_run','automatic_promotion':False}
                print('TRAINING AND MERGE COMPLETE. NO VALIDATION RUN:',output,flush=True)
                return
            comparisons={}
            # Noise is the current limiting condition, so measure it first.
            for label in ('noisy','clean'):
                dest=output/('validation-'+label)
                stage(output,'validation-'+label,[sys.executable,'-u',str(code/'src/benchmark.py'),
                    '--model',str(output/'merged'),'--input',str(inputs[label]),'--device','cuda',
                    '--batch-size','2','--warmup-batches','1','--output-dir',str(dest)])
                report=json.loads((dest/'report.json').read_text())
                if report['status']!='complete' or report['calls']!=383:raise RuntimeError('Incomplete '+label+' evaluation')
                before=json.loads((package/f'baseline_{label}.json').read_text())['official']
                current=report['quality']['official']
                keys=('fact_precision','fact_recall','temporal_recall','state_accuracy','grounding_turn')
                comparisons[label]={'before':before,'after':current,'failed_calls':report['failed_calls'],
                    'delta_percentage_points':{k:round(100*(current[k]-before[k]),4) for k in keys},
                    'desired_T_passed':current['temporal_recall']>=.4506,
                    'desired_S_passed':current['state_accuracy']>=.5145}
                (output/'comparison.json').write_text(json.dumps({'automatic_promotion':False,'results':comparisons},indent=2)+'\n')
            status={'status':'complete','automatic_promotion':False,'calls_per_dataset':383}
            print('DONE:',output/'comparison.json',flush=True)
        finally:(output/'pipeline-status.json').write_text(json.dumps(status)+'\n')

if __name__=='__main__':main()
