"""Wait for training, probe longest prompts, then evaluate ONE full val at batch 4."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

PARENT_SHA='232db122df3e8666e4da818290741d1a55d5ad38dc2892657d525116e7e3b4ce'
TRAIN_SHA='2005ae5b24ba61ce73287dadbfce20077be8a65913e725322399d49f8f55d563'
DATA_SHA={'clean':'e12610354aa69dff83e1d7c1e8c94736c97d5761d20af58790435f4be1941808',
          'noisy':'c76823bc32fdcfc6e62410c2d0f8222043220130f3a0c31345cd7025b12e85da'}

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def training_state(parent):
    path=parent/'pipeline-status.json'
    if not path.exists():return 'waiting'
    state=json.loads(path.read_text())
    if state.get('status')=='failed':raise RuntimeError('Training failed; inspect its log before evaluation')
    if state.get('status')!='complete':return 'waiting'
    if not (parent/'merged/model.safetensors').is_file():raise RuntimeError('Merged weights missing')
    return 'ready'

def longest_calls(rows,tokenizer,prompt_for,as_call,count=8):
    ranked=sorted(((len(tokenizer(prompt_for(tokenizer,as_call(row)),add_special_tokens=False)['input_ids']),i)
                   for i,row in enumerate(rows)),reverse=True)
    selected=ranked[:count]
    return [rows[i] for _,i in selected], [{'call_id':rows[i]['call_id'],'input_tokens':n} for n,i in selected]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.home()/'mws-v21')
    ap.add_argument('--dataset',choices=('noisy','clean'),default='noisy')
    ap.add_argument('--wait-minutes',type=int,default=150)
    args=ap.parse_args()
    if args.wait_minutes<1:ap.error('wait-minutes must be positive')
    team=args.root.resolve();parent=team/'out/005-targeted-001'
    run=f'005-targeted-001-val-{args.dataset}-b4-001';output=team/'out'/run
    if output.exists():raise FileExistsError('Previous evaluation is protected: '+str(output))
    deadline=time.monotonic()+args.wait_minutes*60
    while training_state(parent)!='ready':
        if time.monotonic()>deadline:raise TimeoutError('Training wait expired; no GPU evaluation started')
        print(datetime.now(timezone.utc).isoformat(),'WAITING FOR TRAINING; NO GPU WORK',flush=True)
        time.sleep(30)
    import fcntl
    with (team/'out/v21-validation.lock').open('a') as lock:
        # Training may have written complete just before releasing its lock.
        deadline=time.monotonic()+args.wait_minutes*60
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>deadline:raise TimeoutError('Another job still owns the GPU pipeline lock')
                print('WAITING FOR PIPELINE LOCK; NO GPU WORK',flush=True);time.sleep(15)
        settings=json.loads((parent/'settings.json').read_text())
        if settings['parent_weights_sha256']!=PARENT_SHA or settings['train_sha256']!=TRAIN_SHA:
            raise ValueError('Unexpected training experiment')
        package=team/'targeted/005-targeted-001-trainonly'
        spec=importlib.util.spec_from_file_location('training_profile',package/'run_targeted.py')
        profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
        profile.verify_package(package)
        reference=team/'releases/case-v21-unified-001'
        spec=importlib.util.spec_from_file_location('inference_profile',reference/'evaluation/run_comparison.py')
        inference=importlib.util.module_from_spec(spec);spec.loader.exec_module(inference)
        inference.verify_release(reference);inference.configure(reference)
        model=parent/'merged';code=package/'code'
        os.environ.update(BASE_MODEL=str(model),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                          AGREEMENT_CLEANUP='0',NOISE_ROBUST_RECOVERY='0',MWS_CODE_REVISION=run)
        sys.path.insert(0,str(code/'src'))
        from case_io import read_jsonl,write_jsonl
        from training_data import prompt_for
        from schema import as_call
        from run_experiment import stage
        from transformers import AutoTokenizer
        data=(reference/'data/noisy-015/val-noisy-015.jsonl' if args.dataset=='noisy'
              else reference/'data/organizer-v2.1/val.jsonl')
        if digest(data)!=DATA_SHA[args.dataset]:raise ValueError('Wrong evaluation data')
        rows=read_jsonl(data)
        if len(rows)!=383:raise ValueError('Expected full val, 383 calls')
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        free,total=torch.cuda.mem_get_info()
        if free<16*2**30:raise RuntimeError('Less than 16 GiB free; do not overlap other GPU jobs')
        tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
        stress,selected=longest_calls(rows,tokenizer,prompt_for,as_call)
        output.mkdir(exist_ok=False)
        provenance={'run':run,'dataset':args.dataset,'input_sha256':DATA_SHA[args.dataset],
                    'model':str(model),'weights_sha256':digest(model/'model.safetensors'),
                    'batch_size':4,'gpu_free_bytes':free,'gpu_total_bytes':total,
                    'stress_selection':selected,'stress_is_diagnostic_only':True,
                    'full_input_order_preserved':True,'requested_batch_not_silently_reduced':True,
                    'script_sha256':digest(Path(__file__)),'training_unchanged':True}
        (output/'settings.json').write_text(json.dumps(provenance,indent=2)+'\n')
        status={'status':'failed'}
        try:
            write_jsonl(output/'stress-calls.jsonl',stress)
            write_jsonl(output/'stress-selection.jsonl',[{'call_id':r['call_id'],'error':'memory_probe'} for r in stress])
            base=[sys.executable,'-u',str(code/'src/benchmark.py'),'--model',str(model),'--device','cuda',
                  '--batch-size','4','--warmup-batches','1']
            print('TRAINING COMPLETE. TESTING BATCH 4 ON 8 LONGEST PROMPTS.',flush=True)
            stage(output,'memory-probe',base+['--input',str(output/'stress-calls.jsonl'),
                  '--failures',str(output/'stress-selection.jsonl'),'--output-dir',str(output/'memory-probe')])
            probe=json.loads((output/'memory-probe/report.json').read_text())
            if probe['status']!='complete' or probe['failed_calls'] or probe.get('warmup_errors'):
                raise RuntimeError('Batch-4 probe failed; full validation NOT started. Preserve the probe report.')
            if probe['memory']['cuda_peak_reserved_bytes']>.9*total:
                raise RuntimeError('Insufficient measured VRAM headroom for batch 4; full validation NOT started')
            print('BATCH-4 PROBE PASSED. STARTING ONE FULL VAL, 383 CALLS.',flush=True)
            stage(output,'validation',base+['--input',str(data),'--output-dir',str(output/'validation')])
            report=json.loads((output/'validation/report.json').read_text())
            if report['status']!='complete' or report['calls']!=383:raise RuntimeError('Incomplete full validation')
            if report['model_artifacts']['model.safetensors']['sha256']!=provenance['weights_sha256']:
                raise RuntimeError('Model changed during evaluation')
            old=json.loads((package/f'baseline_{args.dataset}.json').read_text())['official']
            current=report['quality']['official']
            keys=('fact_precision','fact_recall','temporal_recall','state_accuracy','grounding_turn')
            comparison={'parent_model_batch2':old,'targeted_model_batch4':current,
                        'delta_percentage_points':{k:round(100*(current[k]-old[k]),4) for k in keys},
                        'comparison_caveat':'Weights and batch size both differ; not a controlled speedup measurement.',
                        'failed_calls':report['failed_calls'],'automatic_promotion':False}
            (output/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
            status={'status':'complete','calls':383,'failed_calls':report['failed_calls'],'batch_size':4}
            print('DONE:',output/'validation/report.json',flush=True)
        finally:(output/'pipeline-status.json').write_text(json.dumps(status)+'\n')

if __name__=='__main__':main()
