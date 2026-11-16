"""Conservative train-only adaptation of the team's current-state focus dataset."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

POLICY = 'focus-plus-source-replay-v1'
ACK = re.compile(r'\b(?:да|угу|ага|ладно|хорошо|верно|правильно|подходит|договорились|соглас[а-я]*|устраивает|подтвержда[а-я]*)\b')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rank(cid,seed,tag):return hashlib.sha256(f'{seed}:{tag}:{cid}'.encode()).hexdigest()
def original_id(cid):return cid.split('~')[0]

def source_reasons(row):
    turns={t['idx']:t for t in row['turns']}
    agreements=[f for f in row['facts'] if f['attribute']=='agreement']
    short=any(len(re.findall(r'\w+',turns[f['source']['turn_idx']]['text']))<=12
              and ACK.search(turns[f['source']['turn_idx']]['text'].lower().replace('ё','е'))
              for f in agreements)
    return agreements,short

def build(original,focus,val,*,seed=42,source_replay=400):
    from case_io import validate_gold
    from schema import to_answer
    from training_data import ensure_training_splits
    if source_replay<0:raise ValueError('Negative replay budget')
    ensure_training_splits(original,val)
    byid={r['call_id']:r for r in original}
    if len(byid)!=len(original):raise ValueError('Duplicate original IDs')
    for row in original+focus:validate_gold(row)
    seen=set();represented=set();result=[];changed=0;protected=0
    for row in focus:
        if row['call_id'] in seen:raise ValueError('Duplicate focus ID')
        seen.add(row['call_id']);cid=original_id(row['call_id'])
        if cid not in byid:raise ValueError('Focus call not derived from supplied train')
        base=byid[cid];represented.add(cid)
        if to_answer(row['facts'])!=to_answer(base['facts']):raise ValueError('Focus changed training targets')
        if row['started_at']!=base['started_at'] or [(t['idx'],t['speaker']) for t in row['turns']]!=[(t['idx'],t['speaker']) for t in base['turns']]:
            raise ValueError('Focus changed timestamp, turn IDs or speakers')
        item=copy.deepcopy(row)
        positions={t['idx']:i for i,t in enumerate(base['turns'])};indices=set()
        for fact in base['facts']:
            pos=positions[fact['source']['turn_idx']]
            indices.update(range(max(0,pos-1),min(len(base['turns']),pos+2)))
        for pos in sorted(indices):
            protected+=1
            changed+=item['turns'][pos]['text']!=base['turns'][pos]['text']
            item['turns'][pos]['text']=base['turns'][pos]['text']
        # Preserve the supplied sample metadata; record text protection separately.
        validate_gold(item);result.append(item)
    missing=[r for r in original if r['call_id'] not in represented]
    for row in missing:
        item=copy.deepcopy(row);item['call_id']+='~target-bg';result.append(item)
    short=[];explicit=[]
    for row in original:
        agreements,is_short=source_reasons(row)
        if agreements:(short if is_short else explicit).append(row)
    short.sort(key=lambda r:rank(r['call_id'],seed,'short'))
    explicit.sort(key=lambda r:rank(r['call_id'],seed,'explicit'))
    half=source_replay//2
    chosen=short[:half]+explicit[:source_replay-half]
    used={r['call_id'] for r in chosen}
    if len(chosen)<source_replay:
        extra=sorted([r for r in short+explicit if r['call_id'] not in used],key=lambda r:rank(r['call_id'],seed,'fill'))
        chosen+=extra[:source_replay-len(chosen)]
    for row in chosen:
        item=copy.deepcopy(row);item['call_id']+='~target-src';result.append(item)
    result.sort(key=lambda r:rank(r['call_id'],seed,'shuffle'))
    if len({r['call_id'] for r in result})!=len(result):raise ValueError('Duplicate result IDs')
    ensure_training_splits(result,val)
    for row in result:
        validate_gold(row)
        if to_answer(row['facts'])!=to_answer(byid[original_id(row['call_id'])]['facts']):raise ValueError('Target changed')
    return result,{'policy':POLICY,'seed':seed,'focus_rows':len(focus),'restored_background_rows':len(missing),
        'source_replay_rows':len(chosen),'source_replay_short':sum(source_reasons(r)[1] for r in chosen),
        'source_replay_explicit':sum(not source_reasons(r)[1] for r in chosen),
        'source_neighborhood_turns_protected':protected,'turn_texts_restored':changed,
        'result_rows':len(result),'unique_original_train_calls':len({original_id(r['call_id']) for r in result}),
        'all_original_calls_represented':{original_id(r['call_id']) for r in result}==set(byid),
        'full_call_targets_preserved':True,'val_used_for_selection':False,
        'attribute_facts':dict(Counter(f['attribute'] for r in result for f in r['facts'])),
        'source_replay_ids':[r['call_id'] for r in chosen]}

def main():
    from case_io import read_jsonl
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('train','focus','val','output-dir'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--source-replay',type=int,default=400)
    args=ap.parse_args();out=Path(args.output_dir)
    if out.exists():ap.error('Output exists; preserve it and choose a new path')
    result,info=build(read_jsonl(args.train),read_jsonl(args.focus),read_jsonl(args.val),seed=args.seed,source_replay=args.source_replay)
    info['input_sha256']={n:digest(getattr(args,n)) for n in ('train','focus','val')}
    out.mkdir(parents=True)
    path=out/'train_targeted.jsonl'
    path.write_bytes(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in result).encode('utf-8'))
    info['output_sha256']=digest(path)
    (out/'manifest.json').write_bytes((json.dumps(info,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({k:v for k,v in info.items() if k!='source_replay_ids'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
