"""Paired evaluation from all 383 saved raw generations, without a model.

Control 1 reproduces the saved server result; control 2 isolates interval rules;
candidate changes only ticket recovery after intervals. Gold enters scoring only.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def save_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    item=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(item)
    return item


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',required=True,type=Path)
    ap.add_argument('--old-release',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    repo, output=args.repo.resolve(),args.output.resolve()
    if output.exists():ap.error('Output exists; preserve previous experiments')
    data=repo/'data/organizer-v2.1'
    os.environ.update(SCORER_VERSION='2.1',CATALOG_PATH=str(data/'catalog.json'),
        TEMPORAL_CATALOG_PATH=str(data/'catalog.json'),TEMPORAL_NORMALIZATION='case_v2',
        FACT_RECOVERY='1',TEMPORAL_SERIES_HISTORY='1')
    sys.path.insert(0,str(repo/'src'))
    from schema import as_call
    from prediction import parse_prediction
    from series_history import new_series_memory
    from case_evaluation import evaluate_predictions
    from scoring import Scorer
    import fact_recovery as new_recovery
    import temporal_case_v2 as new_dates
    # Reuse the diagnostic selection with an exact count assertion against scorer.
    audit_module=module(repo/'docs/experiments/4090-continue-v21-001/comparisons/state-audit-001/audit_states.py','state_diagnostic')
    old_recovery=module(args.old_release/'src/fact_recovery.py','control_recovery')
    old_dates=module(args.old_release/'src/temporal_case_v2.py','control_dates')
    run=repo/'docs/experiments/4090-continue-v21-001/attachments/download/4090-continue-v21-001'
    sources=[run/'validation/raw-answers.jsonl',run/'validation-recovery-001/tail/raw-answers.jsonl']
    raw_rows=[r for p in sources for r in load_rows(p)]
    raw={r['call_id']:r['raw_response'] for r in raw_rows}
    gold=load_rows(data/'val.jsonl')
    calls=[as_call(r) for r in gold]
    assert len(raw)==len(raw_rows)==len(calls)==383
    assert set(raw)=={c.call_id for c in calls}
    output.mkdir(parents=True)
    scorer=Scorer('2.1',data/'catalog.json')
    reports={'fresh_gpu_inference':False,'gold_passed_to_rules':False,'calls':len(calls),
             'scoring':scorer.metadata(),'val_sha256':hashlib.sha256((data/'val.jsonl').read_bytes()).hexdigest(),
             'raw_sha256':{str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
             'code_sha256':{name:hashlib.sha256((repo/'src'/name).read_bytes()).hexdigest() for name in
                    ('ticket_recovery.py','fact_recovery.py','prediction.py','temporal_case_v2.py','series_history.py')},
             'variants':{}}
    previous_states=None
    for name, dates, recovery in [('server-control',old_dates,old_recovery),
                                  ('interval-control',new_dates,old_recovery),
                                  ('status-candidate',new_dates,new_recovery)]:
        sys.modules['temporal_case_v2']=dates
        sys.modules['fact_recovery']=recovery
        before,audit=[],[]
        started=time.perf_counter()
        for call in calls:
            decisions=[]
            parsed,error,_=parse_prediction(raw[call.call_id],call,temporal_mode='case_v2',
                temporal_decisions=decisions,catalog_path=str(data/'catalog.json'),
                date_catalog_path=str(data/'catalog.json'),fact_recovery=True)
            assert error is None,(call.call_id,error)
            before.append({'call_id':call.call_id,**parsed.model_dump(mode='json')})
            audit.extend({'call_id':call.call_id,**r} for r in decisions)
        memory=new_series_memory();memory._clock=lambda:0
        predictions,history_events,counts=memory.apply(calls,before)
        elapsed=time.perf_counter()-started
        if name=='server-control':
            assert before==load_rows(run/'validation-recovery-001/predictions-before-series.jsonl')
            assert predictions==load_rows(run/'validation-recovery-001/predictions.jsonl')
        rerule_equal=None
        if name=='interval-control':
            # Prior date-only replay started AFTER the old calendar had changed
            # model dates. Raw replay tests the actual new production order.
            rerule_equal=predictions==load_rows(repo/'docs/experiments/4090-continue-v21-001/attachments/download/interval-rules-state-audit-001/predictions.jsonl')
        metrics,errors=evaluate_predictions(gold,predictions,scorer_version='2.1',catalog_path=data/'catalog.json')
        states=audit_module.units(gold,predictions,scorer.module)
        assert round(sum(r['correct'] for r in states)/len(states),4)==metrics['official']['state_accuracy']
        transitions=Counter()
        state_changes=[]
        if previous_states is not None:
            for a,b in zip(previous_states,states):
                assert (a['call_id'],a['object_id'],a['attribute'])==(b['call_id'],b['object_id'],b['attribute'])
                if a['correct']!=b['correct']:
                    transitions[b['attribute']+(':+fix' if b['correct'] else ':-regression')]+=1
                    state_changes.append({'before':a,'after':b})
        previous_states=states
        directory=output/name;directory.mkdir()
        save_rows(directory/'predictions.jsonl',predictions)
        save_rows(directory/'predictions-before-series.jsonl',before)
        save_rows(directory/'audit.jsonl',audit+history_events)
        save_rows(directory/'errors.jsonl',errors)
        save_rows(directory/'state-changes.jsonl',state_changes)
        actions=Counter(r.get('action','legacy_add') for r in audit if r.get('component')=='fact_recovery')
        reports['variants'][name]={'metrics':metrics,'cpu_rules_seconds':elapsed,
                                  'equals_prior_date_only_replay':rerule_equal,
                                  'cpu_rules_ms_per_call':elapsed*1000/len(calls),
                                  'state_transitions_vs_previous_variant':dict(transitions),
                                  'recovery_actions':dict(actions),
                                  'state_breakdown':audit_module.aggregate(states)}
        print(name,json.dumps(metrics['official']),dict(transitions),'rule_ms_per_call',round(elapsed*1000/len(calls),3),flush=True)
        save(output/'comparison.json',reports)
    sys.modules['temporal_case_v2']=new_dates
    sys.modules['fact_recovery']=new_recovery


if __name__=='__main__':main()
