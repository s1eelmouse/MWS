
"""Agreement of two checkpoints as a label filter for self-training.

Two models extract facts from the same calls. A fact both of them produce is more likely
correct than a fact only one produces. This tool measures how much more, and writes the
agreed facts as pseudo-labels in the train.jsonl format.

Agreement levels (each includes the previous one):
  value   same attribute, polarity, object and value by the organizer matching rule
  dates   + valid_from and valid_to equal by day
  strict  + the same source turn

A training row lists ALL facts of a call, so a call where the checkpoints agree on half of
the facts would teach the model to skip the other half. Pseudo-labelled calls are therefore
kept only when at least --min-call-agreement of both checkpoints' facts are agreed.

Gold is optional and used only to MEASURE the filter (precision of each bucket, how many
correct facts survive). Pseudo-labels are built from the two prediction files alone; gold
never reaches them. On unlabeled calls run without --gold: the report then has counts only.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from case_io import read_jsonl, validate_gold
from scoring import Scorer

LEVELS = ('value', 'dates', 'strict')
BUCKETS = ('strict', 'dates_only', 'value_only', 'a_only', 'b_only')


def _day(value):
    return value[:10] if isinstance(value, str) and value else None


def _usable(fact):
    return (isinstance(fact, dict) and isinstance(fact.get('attribute'), str)
            and isinstance(fact.get('value'), str) and isinstance(fact.get('object'), dict))


def agreement_buckets(facts_a, facts_b, scorer):
    """Split the facts of one call: index lists into facts_a (and facts_b for b_only)."""
    position_b = {id(fact): index for index, fact in enumerate(facts_b)}
    buckets = {name: [] for name in BUCKETS}
    matched_a, matched_b = set(), set()
    for fact_a, fact_b, _score in scorer.match_facts(facts_a, facts_b):
        index_a = next(i for i, fact in enumerate(facts_a) if fact is fact_a)
        matched_a.add(index_a)
        matched_b.add(position_b[id(fact_b)])
        same_dates = (_day(fact_a.get('valid_from')) == _day(fact_b.get('valid_from'))
                      and _day(fact_a.get('valid_to')) == _day(fact_b.get('valid_to')))
        same_turn = ((fact_a.get('source') or {}).get('turn_idx')
                     == (fact_b.get('source') or {}).get('turn_idx'))
        name = 'strict' if same_dates and same_turn else 'dates_only' if same_dates else 'value_only'
        buckets[name].append(index_a)
    buckets['a_only'] = [i for i in range(len(facts_a)) if i not in matched_a]
    buckets['b_only'] = [i for i in range(len(facts_b)) if i not in matched_b]
    return buckets


def _level_members(buckets, level):
    names = {'value': ('strict', 'dates_only', 'value_only'), 'dates': ('strict', 'dates_only'),
             'strict': ('strict',)}[level]
    return sorted(index for name in names for index in buckets[name])


def analyse(rows_a, rows_b, scorer, gold_rows=None, level='strict', min_call_agreement=0.8):
    """Return (report, pseudo_label_rows). Calls are joined by call_id; A supplies the facts."""
    by_b = {row['call_id']: row for row in rows_b}
    gold = {row['call_id']: row for row in gold_rows} if gold_rows else {}
    count = Counter()
    correct = Counter()
    per_attribute = {}
    pseudo = []
    for row_a in rows_a:
        cid = row_a['call_id']
        if cid not in by_b:
            count['calls_missing_in_b'] += 1
            continue
        facts_a = [f for f in (row_a.get('facts') or []) if _usable(f)]
        facts_b = [f for f in (by_b[cid].get('facts') or []) if _usable(f)]
        count['calls'] += 1
        count['facts_a'] += len(facts_a)
        count['facts_b'] += len(facts_b)
        buckets = agreement_buckets(facts_a, facts_b, scorer)
        right_a = right_b = set()
        if cid in gold:
            right_a = {id(pred) for _g, pred, _s in scorer.match_facts(gold[cid]['facts'], facts_a)}
            right_b = {id(pred) for _g, pred, _s in scorer.match_facts(gold[cid]['facts'], facts_b)}
            count['gold_facts'] += len(gold[cid]['facts'])
            count['correct_a'] += len(right_a)
        for name, indices in buckets.items():
            source, right = (facts_b, right_b) if name == 'b_only' else (facts_a, right_a)
            for index in indices:
                fact = source[index]
                slot = per_attribute.setdefault(fact['attribute'], {'count': Counter(), 'correct': Counter()})
                count[name] += 1
                slot['count'][name] += 1
                if id(fact) in right:
                    correct[name] += 1
                    slot['correct'][name] += 1
        kept = _level_members(buckets, level)
        if cid in gold:
            count['correct_kept'] += sum(id(facts_a[i]) in right_a for i in kept)
        count['kept'] += len(kept)
        larger = max(len(facts_a), len(facts_b))
        if larger and len(kept) / larger >= min_call_agreement:
            count['calls_kept'] += 1
            count['facts_in_kept_calls'] += len(kept)
            if cid in gold:
                count['correct_in_kept_calls'] += sum(id(facts_a[i]) in right_a for i in kept)
                count['gold_in_kept_calls'] += len(gold[cid]['facts'])
            pseudo.append({'call_id': cid, 'facts': [facts_a[i] for i in kept]})

    def table(counts, rights):
        result = {}
        for name in BUCKETS:
            entry = {'facts': counts[name]}
            if gold:
                entry['correct'] = rights[name]
                entry['precision'] = round(rights[name] / counts[name], 4) if counts[name] else None
            result[name] = entry
        return result

    report = {
        'level_used_for_pseudo_labels': level, 'gold_used_for_measurement': bool(gold),
        'gold_used_for_pseudo_labels': False,
        'calls': count['calls'], 'calls_missing_in_b': count['calls_missing_in_b'],
        'facts_a': count['facts_a'], 'facts_b': count['facts_b'],
        'buckets': table(count, correct),
        'per_attribute': {a: table(v['count'], v['correct']) for a, v in sorted(per_attribute.items())},
        'pseudo_labels': {'facts': count['kept'],
                          'share_of_a': round(count['kept'] / count['facts_a'], 4) if count['facts_a'] else None},
    }
    report['pseudo_label_calls'] = {'min_call_agreement': min_call_agreement, 'calls': count['calls_kept'],
                                    'share_of_calls': round(count['calls_kept'] / count['calls'], 4) if count['calls'] else None,
                                    'facts': count['facts_in_kept_calls']}
    if gold and count['calls_kept']:
        report['pseudo_label_calls'].update(
            precision=round(count['correct_in_kept_calls'] / count['facts_in_kept_calls'], 4),
            recall_within_these_calls=round(count['correct_in_kept_calls'] / count['gold_in_kept_calls'], 4)
            if count['gold_in_kept_calls'] else None)
    if gold:
        report['model_a'] = {'precision': round(count['correct_a'] / count['facts_a'], 4) if count['facts_a'] else None,
                             'recall': round(count['correct_a'] / count['gold_facts'], 4) if count['gold_facts'] else None}
        report['pseudo_labels'].update(
            precision=round(count['correct_kept'] / count['kept'], 4) if count['kept'] else None,
            recall=round(count['correct_kept'] / count['gold_facts'], 4) if count['gold_facts'] else None,
            correct_facts_of_a_kept=round(count['correct_kept'] / count['correct_a'], 4) if count['correct_a'] else None)
    return report, pseudo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pred-a', required=True, help='predictions of the main checkpoint; its facts are kept')
    parser.add_argument('--pred-b', required=True, help='predictions of the second checkpoint on the same calls')
    parser.add_argument('--catalog', required=True, help='organizer catalog.json for value matching')
    parser.add_argument('--gold', help='labelled calls; only to measure the filter')
    parser.add_argument('--calls', help='transcripts (jsonl with turns) to write pseudo-labels as training rows')
    parser.add_argument('--level', choices=LEVELS, default='strict')
    parser.add_argument('--min-call-agreement', type=float, default=0.8,
                        help='keep a call as a training row only if this share of facts is agreed')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    scorer = Scorer('2.1', args.catalog)
    report, pseudo = analyse(read_jsonl(args.pred_a), read_jsonl(args.pred_b), scorer,
                             read_jsonl(args.gold) if args.gold else None, args.level,
                             args.min_call_agreement)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if args.calls:
        # Training rows: transcript from --calls, facts from agreement. Gold facts are not copied.
        calls = {row['call_id']: row for row in read_jsonl(args.calls)}
        rows, rejected = [], 0
        for row in pseudo:
            if row['call_id'] not in calls:
                continue
            item = {**{k: v for k, v in calls[row['call_id']].items() if k != 'facts'}, 'facts': row['facts']}
            try:
                validate_gold(item)  # the same contract check train.py applies
            except ValueError:
                rejected += 1
                continue
            rows.append(item)
        pseudo = rows
        report['pseudo_label_calls'].update(written=len(rows), rejected_by_contract=rejected)
    with (output / 'pseudo_labels.jsonl').open('x', encoding='utf-8') as stream:
        for row in pseudo:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    report['inputs_sha256'] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                               for name, path in (('pred_a', args.pred_a), ('pred_b', args.pred_b),
                                                  ('catalog', args.catalog), ('gold', args.gold),
                                                  ('calls', args.calls)) if path}
    report['scorer_sha256'] = scorer.sha256
    with (output / 'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
