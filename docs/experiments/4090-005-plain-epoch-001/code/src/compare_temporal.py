"""Evaluate date-only changes on identical ordered facts with the pinned scorer."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from case_evaluation import evaluate_predictions, index_calls
from case_io import read_jsonl, write_jsonl
from scoring import make_scorer
from config import SCORER_VERSION
from run_context import catalog_for


def _day(value):
    return value[:10] if value else None


def _without_dates(row):
    row = copy.deepcopy(row)
    for fact in row.get('facts', []):
        fact.pop('valid_from', None)
        fact.pop('valid_to', None)
    return row


def compare_date_predictions(gold_rows, before_rows, after_rows, *, scorer_version=None, catalog_path=None):
    scorer = make_scorer(scorer_version, catalog_path)
    before, after = index_calls(before_rows), index_calls(after_rows)
    if set(before) != set(after):
        raise ValueError('Call IDs changed')
    for cid in before:
        if _without_dates(before[cid]) != _without_dates(after[cid]):
            raise ValueError(f'{cid}: facts, order or non-date fields changed; use a separate experiment')
    baseline, _ = evaluate_predictions(gold_rows, before_rows, scorer=scorer)
    result, _ = evaluate_predictions(gold_rows, after_rows, scorer=scorer)
    counts, all_counts, details = Counter(), Counter(), []
    for call in gold_rows:
        cid = call['call_id']
        old_facts, new_facts = before.get(cid, {}).get('facts', []), after.get(cid, {}).get('facts', [])
        matches = scorer.match_facts(call['facts'], old_facts)
        positions = {id(f): i for i, f in enumerate(old_facts)}
        for gold, old, _ in matches:
            index = positions[id(old)]
            new = new_facts[index]
            old_ok = all(_day(old.get(field)) == _day(gold.get(field)) for field in ['valid_from', 'valid_to'])
            new_ok = all(_day(new.get(field)) == _day(gold.get(field)) for field in ['valid_from', 'valid_to'])
            status = 'fixed' if not old_ok and new_ok else 'regressed' if old_ok and not new_ok else 'still_correct' if new_ok else 'still_wrong'
            eligible = scorer.temporal_eligible(gold, call)
            all_counts[status] += 1
            if eligible:
                counts[status] += 1
            changed = any(old.get(field) != new.get(field) for field in ['valid_from', 'valid_to'])
            all_counts['changed_matched_facts'] += changed
            counts['changed_matched_facts'] += changed and eligible
            if changed:
                details.append({'call_id': cid, 'fact_index': index, 'attribute': old['attribute'],
                                'value': old['value'], 'status': status, 'temporal_eligible': eligible,
                                'before': {f: old.get(f) for f in ['valid_from', 'valid_to']},
                                'after': {f: new.get(f) for f in ['valid_from', 'valid_to']},
                                'gold': {f: gold.get(f) for f in ['valid_from', 'valid_to']}})
    assert baseline['official']['n_temporal'] == result['official']['n_temporal']
    assert baseline['official']['fact_recall'] == result['official']['fact_recall']
    assert baseline['official']['fact_precision'] == result['official']['fact_precision']
    return {'before': baseline, 'after': result, 'transitions': dict(counts),
            'transitions_all_matched_facts': dict(all_counts), 'scoring': scorer.metadata(),
            'net_correct_pairs': counts['fixed'] - counts['regressed']}, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', required=True)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--scorer-version', choices=('1', '2.1', '2.1-datefix'), default=SCORER_VERSION)
    parser.add_argument('--catalog')
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        parser.error('Output exists; choose a new experiment name')
    report, details = compare_date_predictions(read_jsonl(args.gold), read_jsonl(args.before), read_jsonl(args.after),
        scorer_version=args.scorer_version, catalog_path=catalog_for(args.gold, args.catalog))
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'comparison.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    write_jsonl(output / 'changed-matches.jsonl', details)
    print(json.dumps({'before': report['before']['official'], 'after': report['after']['official'],
                      'transitions': report['transitions'], 'net_correct_pairs': report['net_correct_pairs']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
