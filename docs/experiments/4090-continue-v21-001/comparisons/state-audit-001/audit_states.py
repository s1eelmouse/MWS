"""Replay integrated intervals and explain the exact official state units.

Evaluation-only: gold is never passed to production date/history functions.
Run with project dependencies; --output must be a new directory. Full transcripts
and predictions belong in ignored attachments/download, not in Git.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rows(path, values):
    path.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values), encoding='utf-8')


def units(gold, predictions, module):
    """Mirror VERSION=2.1 selection; diagnostic categories do not change score."""
    pred = {row['call_id']: row for row in predictions}
    result = []
    for call in gold:
        facts = pred.get(call['call_id'], {}).get('facts') or []
        targets = {}
        for fact in call['facts']:
            if fact['attribute'] in module.SINGLE_VALUED and fact.get('valid_to') is None:
                key = (fact['object']['id'], fact['attribute'])
                if key not in targets or fact['valid_from'] > targets[key]['valid_from']:
                    targets[key] = fact
        for (oid, attr), target in targets.items():
            same_attr = [f for f in facts if f.get('attribute') == attr]
            related = [f for f in same_attr if module._object_sim(target, f) > 0]
            opened = [f for f in related if f.get('valid_to') is None]

            def matches(f):
                need = module.VALUE_SIM if min(len(module._tokens(target['value'])),
                                               len(module._tokens(f.get('value', '')))) <= 2 else module.VALUE_SIM_LONG
                return module._value_sim(target['value'], f.get('value', '')) >= need

            selected = max(opened, key=lambda f: (f.get('valid_from') or '',
                            module._value_sim(target['value'], f.get('value', '')))) if opened else None
            ok = selected is not None and matches(selected)
            matching = [f for f in related if matches(f)]
            category = ('correct' if ok else 'missing_attribute' if not same_attr else
                        'object_binding' if not related else 'all_compatible_facts_closed' if not opened else
                        'correct_open_overshadowed_by_date' if any(matches(f) for f in opened) else
                        'correct_value_only_closed' if matching else 'wrong_value')
            evidence_idx = {target['source']['turn_idx']}
            evidence_idx.update(f['source']['turn_idx'] for f in related)
            context = [t for t in call['turns'] if any(abs(t['idx'] - i) <= 1 for i in evidence_idx)]
            result.append({'call_id': call['call_id'], 'object_id': oid, 'attribute': attr,
                           'correct': ok, 'category': category, 'gold': target, 'selected': selected,
                           'compatible_predictions': related, 'other_object_predictions':
                           [f for f in same_attr if f not in related],
                           'matching_value_any_object': any(matches(f) for f in same_attr),
                           'matching_value_compatible_object': bool(matching),
                           'gold_source_context': context})
    return result


def aggregate(values):
    out = {}
    for attr in sorted({r['attribute'] for r in values}):
        group = [r for r in values if r['attribute'] == attr]
        errors = [r for r in group if not r['correct']]
        out[attr] = {'n': len(group), 'correct': sum(r['correct'] for r in group),
                     'errors': len(errors), 'categories': dict(Counter(r['category'] for r in errors)),
                     'error_gold_values': dict(Counter(r['gold']['value'] for r in errors).most_common()),
                     'correct_gold_values': dict(Counter(r['gold']['value'] for r in group if r['correct']).most_common()),
                     'errors_with_matching_value_any_object': sum(r['matching_value_any_object'] for r in errors),
                     'errors_with_matching_value_compatible_object': sum(r['matching_value_compatible_object'] for r in errors)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    if output.exists():
        ap.error('Output already exists; choose a new audit directory')
    data = repo / 'data/organizer-v2.1'
    os.environ.update(SCORER_VERSION='2.1', CATALOG_PATH=str(data/'catalog.json'),
                      TEMPORAL_CATALOG_PATH=str(data/'catalog.json'), TEMPORAL_NORMALIZATION='case_v2',
                      TEMPORAL_SERIES_HISTORY='1', FACT_RECOVERY='1')
    sys.path.insert(0, str(repo/'src'))
    from schema import as_call, Facts
    from scoring import Scorer
    from case_evaluation import evaluate_predictions
    from series_history import new_series_memory
    from temporal_case_v2 import repair_case_v2_dates, ValueLayers
    from compare_temporal import compare_date_predictions

    exp = repo/'docs/experiments/4090-continue-v21-001'
    source = exp/'attachments/download/4090-continue-v21-001/validation-recovery-001'
    gold_path = data/'val.jsonl'
    before_path, final_path = source/'predictions-before-series.jsonl', source/'predictions.jsonl'
    gold, before, final = read(gold_path), read(before_path), read(final_path)
    assert len(gold) == len(before) == len(final) == 383
    assert [r['call_id'] for r in gold] == [r['call_id'] for r in before] == [r['call_id'] for r in final]
    calls = [as_call(r) for r in gold]
    control = new_series_memory()
    control._clock = lambda: 0  # The saved active inference interval is shorter than TTL.
    assert control.apply(calls, before)[0] == final
    layers = ValueLayers(json.loads((data/'catalog.json').read_text(encoding='utf-8')))
    new_before, decisions = [], []
    for call, row in zip(calls, before):
        facts, audit = repair_case_v2_dates(row['facts'], call, layers)
        Facts.model_validate({'facts': facts})
        assert len(facts) == len(row['facts'])
        for a, b in zip(facts, row['facts']):
            assert {k:v for k,v in a.items() if k not in ('valid_from','valid_to')} == {
                k:v for k,v in b.items() if k not in ('valid_from','valid_to')}
        assert repair_case_v2_dates(facts, call, layers)[0] == facts
        new_before.append({**row, 'facts': facts})
        decisions.extend({'call_id': call.call_id, **r} for r in audit if r['changes'])
    memory = new_series_memory()
    memory._clock = lambda: 0
    integrated = memory.apply(calls, new_before)[0]
    # Agreement with the already measured date-only candidate is an independent
    # check that the live integration is the implementation we evaluated.
    previous_candidate = repo/'out/compare-volodya-state-001/volodya_rule.jsonl'
    candidate_equal = integrated == read(previous_candidate) if previous_candidate.exists() else None
    assert candidate_equal is not False
    scorer = Scorer('2.1', data/'catalog.json')
    official = scorer.module
    metrics, _ = evaluate_predictions(gold, integrated, scorer_version='2.1', catalog_path=data/'catalog.json')
    old_metrics, _ = evaluate_predictions(gold, final, scorer_version='2.1', catalog_path=data/'catalog.json')
    date_comparison, date_details = compare_date_predictions(gold, final, integrated,
                                 scorer_version='2.1', catalog_path=data/'catalog.json')
    values = units(gold, integrated, official)
    old_values = units(gold, final, official)
    assert sum(r['correct'] for r in values) == 315 and len(values) == 623
    assert round(sum(r['correct'] for r in values)/len(values), 4) == metrics['official']['state_accuracy']
    assert [(r['correct'], r['category']) for r in values] == [(r['correct'], r['category']) for r in old_values]
    gold_self = official.evaluate({r['call_id']:r for r in gold}, {r['call_id']:r for r in gold})
    summary = {'audit_version': 1, 'scorer': scorer.metadata(), 'gold_sha256': sha(gold_path),
               'before_sha256': sha(before_path), 'original_final_sha256': sha(final_path),
               'code_sha256': {name: sha(repo/'src'/name) for name in (
                   'temporal_case_v2.py', 'state_intervals.py', 'ticket_intervals.py', 'series_history.py')},
               'gold_given_to_production_rules': False, 'fresh_gpu_inference': False,
               'same_as_previously_evaluated_candidate': candidate_equal,
               'before': old_metrics, 'after': metrics, 'date_transitions': date_comparison['transitions'],
               'state_units': len(values), 'state_correct': sum(r['correct'] for r in values),
               'gold_vs_gold_state_accuracy': gold_self['state_accuracy'],
               'by_attribute': aggregate(values)}
    old_runs = list((repo/'docs/experiments/v100-continue-002-full-val-003/attachments/download').rglob('predictions.jsonl'))
    if len(old_runs) == 1:
        older = units(gold, read(old_runs[0]), official)
        assert len(older) == len(values)
        transitions = defaultdict(Counter)
        for a, b in zip(older, values):
            assert (a['call_id'],a['object_id'],a['attribute']) == (b['call_id'],b['object_id'],b['attribute'])
            transitions[b['attribute']][('correct' if a['correct'] else 'wrong') + '_to_' +
                                       ('correct' if b['correct'] else 'wrong')] += 1
        summary['previous_model_prediction_sha256'] = sha(old_runs[0])
        summary['training_state_transitions'] = {a:dict(c) for a,c in transitions.items()}
        rows_to_save = [{'before': a, 'after': b} for a,b in zip(older,values) if a['correct'] != b['correct']]
    else:
        rows_to_save = []
    output.mkdir(parents=True)
    save(output/'summary.json', summary)
    rows(output/'predictions.jsonl', integrated)
    rows(output/'interval-decisions.jsonl', decisions)
    rows(output/'state-units.jsonl', values)
    rows(output/'state-errors.jsonl', [r for r in values if not r['correct']])
    rows(output/'training-state-transitions.jsonl', rows_to_save)
    save(output/'date-transitions.json', date_comparison)
    # Full call input context supports human review of ambiguity outside source ±1.
    failed_ids = {r['call_id'] for r in values if not r['correct']}
    rows(output/'error-calls.jsonl', [r for r in gold if r['call_id'] in failed_ids])
    summary['integrated_prediction_sha256'] = sha(output/'predictions.jsonl')
    save(output/'summary.json', summary)
    print(json.dumps({k:summary[k] for k in ('state_units','state_correct','gold_vs_gold_state_accuracy',
                     'by_attribute','training_state_transitions') if k in summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
