"""Measure the actual unified inference policy on frozen cycle-2 generations."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys
import time

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / 'repository'
sys.path[:0] = [str(ROOT / 'src'), 'C:/Users/Tripozik/Documents/Codex/2026-11-13/qwen-3-1-7b-tesla-v100-6/work/test-deps']
from case_io import read_jsonl, write_jsonl
from case_evaluation import evaluate_predictions
from schema import as_call, Facts
from postprocess_dates import process_predictions
from prediction import parse_prediction
from catalog_normalization import load_catalog
from scoring import make_scorer
from temporal_case_v2 import load_layers
from series_history import new_series_memory

data = ROOT / 'data/organizer-v2.1'
gold = read_jsonl(data / 'val.jsonl')
calls = [as_call(row) for row in gold]
by_id = {c.call_id: c for c in calls}
old = {row['call_id']: as_call(row).model_dump(mode='json') for row in read_jsonl(ROOT / 'case_data/val.jsonl')}
changed = [c.call_id for c in calls if old[c.call_id] != c.model_dump(mode='json')]
assert len(changed) == 24
raw_path = BASE / 'server-results/continue-002-full-val-003/continue-002-full-val-003/raw-answers.jsonl'
raw = {}
for row in read_jsonl(raw_path):
    parsed, error, _ = parse_prediction(row['raw_response'], by_id[row['call_id']], temporal_mode='off',
                                        catalog_path='', fact_recovery=False)
    raw[row['call_id']] = {'call_id': row['call_id'], **(error or parsed.model_dump(mode='json'))}
raw = [raw[c.call_id] for c in calls]
catalog = data / 'catalog.json'
layers = load_layers(catalog)
resolver = load_catalog(catalog)
scorer = make_scorer('2.1', catalog)
baseline_predictions = read_jsonl(data / 'baseline_pred.jsonl')
def official_subset(gold, predictions):
    ids = {r['call_id'] for r in gold}
    return scorer.evaluate({r['call_id']: r for r in gold},
                           {r['call_id']: r for r in predictions if r['call_id'] in ids})
target = ROOT / 'out/team-v21-unified-002'
target.mkdir(exist_ok=False)
summary = {'kind': 'frozen_cycle2_generations_CPU_reprocessing_not_new_model_run',
           'source_branch_commit': '9cb2e93', 'raw_answers_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
           'changed_input_call_ids': changed, 'unchanged_input_calls': len(calls) - len(changed),
           'catalog_sha256': hashlib.sha256(catalog.read_bytes()).hexdigest(), 'variants': {},
           'note': 'Rule timing excludes scorer, generation and model memory; not a GPU resource-curve point.'}
splits = {'all_383': gold, 'unchanged_359': [r for r in gold if r['call_id'] not in changed],
          'first_half_191': gold[:191], 'second_half_192': gold[191:]}
summary['organizer_baseline_by_split'] = {name: official_subset(rows, baseline_predictions) for name, rows in splits.items()}
for mode, recovery in [('task', False), ('task', True), ('case_v2', False), ('case_v2', True)]:
    begin = time.perf_counter()
    final, audit, counts = process_predictions(calls, raw, mode=mode, catalog=resolver,
                          date_catalog=layers, fact_recovery=recovery, recovery_catalog_path=str(catalog),
                          allow_series_history=True)
    seconds = time.perf_counter() - begin
    for row in final:
        if not row.get('error'):
            Facts.model_validate({'facts': row['facts']})
    metrics, errors = evaluate_predictions(gold, final, catalog_path=catalog)
    name = f'{mode}-strict-recovery-{int(recovery)}'
    summary['variants'][name] = {'metrics': metrics, 'counts': counts, 'cpu_rules_seconds': seconds,
        'cpu_rules_ms_per_call': seconds * 1000 / len(calls),
        'metrics_by_split': {name: official_subset(rows, final) for name, rows in splits.items()}}
    write_jsonl(target / (name + '.jsonl'), final)
    write_jsonl(target / (name + '-audit.jsonl'), audit)
    print(name, metrics['official'], 'recovery', counts.get('recovered_facts', 0), 'CPU ms/call', round(seconds*1000/len(calls), 2), flush=True)
# Original teammate recovery in the same causal position as production.
def module(filename):
    spec = importlib.util.spec_from_file_location('original_' + filename, BASE / 'work/team-v21-9cb2e93/src' / (filename + '.py'))
    item = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(item)
    return item
original_dates, original_recovery = module('temporal_case_v2'), module('fact_recovery')
final, _, _ = process_predictions(calls, raw, mode='off', catalog=resolver)
for call, row in zip(calls, final):
    if not row.get('error'):
        row['facts'], _ = original_dates.repair_case_v2_dates(row['facts'], call, original_dates.load_layers(str(catalog)))
        row['facts'], _ = original_recovery.recover_facts(row['facts'], call, str(catalog))
final, _, _ = new_series_memory().apply(calls, final)
metrics, _ = evaluate_predictions(gold, final, catalog_path=catalog)
summary['teammate_original_rules_with_our_history'] = metrics
(target / 'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('SAVED', target / 'comparison.json')
