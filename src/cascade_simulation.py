"""Cascade on paper: the small model answers, flagged calls take the large model's answer.

No model is called. Both answers already exist: predictions of the small model and of the
large one (the organizer baseline file). For each routing signal the tool mixes the two
prediction files call by call and scores the mix with the official scorer, reporting the
share of calls sent to the large model next to the resulting metrics.

Signals are computed from the transcript, the catalog and the small model's own output,
never from gold, so each of them could run in production:
  tariff_named_no_fact   a catalog tariff is named next to «тариф», the model gave no tariff
  symptom_named_no_state a catalog symptom is recognisable in the call, no service_state fact
  checkpoints_disagree   two checkpoints agree on less than --agreement-threshold of the facts
                         (needs --pred-small-b)
  any                    at least one of the above

Reference rows: `small_only`, `large_only`, `random` (same share as `any`, mean of seeded
draws: a signal must beat it) and `oracle` (per call the answer with more gold matches).
The oracle reads gold and is an upper bound, not a usable policy.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from case_io import read_jsonl
from catalog_hints import build_hints, find_hints
from checkpoint_agreement import agreement_buckets
from scoring import Scorer

METRICS = ('fact_precision', 'fact_recall', 'temporal_recall', 'state_accuracy', 'grounding_turn')
RANDOM_DRAWS = 20


def _words(text):
    return set(re.findall(r'[а-яa-z0-9+]+', text.lower().replace('ё', 'е')))


def _facts(row):
    return [f for f in (row or {}).get('facts') or []
            if isinstance(f, dict) and isinstance(f.get('attribute'), str)
            and isinstance(f.get('value'), str) and isinstance(f.get('object'), dict)]


def call_signals(call, small, small_b, catalog, hints, scorer, agreement_threshold):
    """Gold-free reasons to distrust the small model's answer for one call."""
    facts = _facts(small)
    attributes = {fact['attribute'] for fact in facts}
    signals = set()
    if 'tariff' not in attributes:
        turns = [_words(turn['text']) for turn in call['turns']]
        turns = [spoken for spoken in turns if any(word.startswith('тариф') for word in spoken)]
        if any(_words(name) and _words(name) <= spoken for name in catalog.get('tariffs', ()) for spoken in turns):
            signals.add('tariff_named_no_fact')
    if 'service_state' not in attributes:
        named = [value for value in find_hints(call, {'service_state': hints['service_state']})['service_state']
                 if value != 'работает нормально']
        if named:
            signals.add('symptom_named_no_state')
    if small_b is not None:
        other = _facts(small_b)
        buckets = agreement_buckets(facts, other, scorer)
        agreed = len(buckets['strict']) + len(buckets['dates_only']) + len(buckets['value_only'])
        larger = max(len(facts), len(other))
        if larger and agreed / larger < agreement_threshold:
            signals.add('checkpoints_disagree')
    return signals


def score_mix(gold, small, large, to_large, scorer):
    mixed = {cid: (large if cid in to_large else small).get(cid, {'call_id': cid, 'facts': []}) for cid in gold}
    result = scorer.evaluate(gold, mixed)
    return {name: result.get(name) for name in METRICS}


def simulate(gold_rows, small_rows, large_rows, catalog, scorer, small_b_rows=None,
             agreement_threshold=0.5, seed=42):
    gold = {row['call_id']: row for row in gold_rows}
    small = {row['call_id']: row for row in small_rows}
    large = {row['call_id']: row for row in large_rows}
    small_b = {row['call_id']: row for row in small_b_rows} if small_b_rows else None
    hints = build_hints(catalog)
    flagged = {}
    for cid, call in gold.items():
        flagged[cid] = call_signals(call, small.get(cid), small_b.get(cid, {}) if small_b else None,
                                    catalog, hints, scorer, agreement_threshold)
    names = ['tariff_named_no_fact', 'symptom_named_no_state'] + (['checkpoints_disagree'] if small_b else [])
    routes = {name: {cid for cid, found in flagged.items() if name in found} for name in names}
    routes['any'] = {cid for cid, found in flagged.items() if found}
    rows = {'small_only': {'uses_gold': False, 'calls_to_large': 0, **score_mix(gold, small, large, set(), scorer)}}
    for name, chosen in routes.items():
        rows[name] = {'uses_gold': False, 'calls_to_large': len(chosen),
                      **score_mix(gold, small, large, chosen, scorer)}
    # control: the same number of calls chosen at random
    size, ids = len(routes['any']), sorted(gold)
    draws = [score_mix(gold, small, large, set(random.Random(f'{seed}:{i}').sample(ids, size)), scorer)
             for i in range(RANDOM_DRAWS)] if size else []
    if draws:
        rows['random'] = {'uses_gold': False, 'calls_to_large': size, 'draws': RANDOM_DRAWS,
                          **{m: round(sum(d[m] for d in draws if d[m] is not None) / len(draws), 4) for m in METRICS}}
    better = set()
    for cid, call in gold.items():
        hits = [len(scorer.match_facts(call['facts'], _facts(source.get(cid)))) for source in (small, large)]
        if hits[1] > hits[0]:
            better.add(cid)
    rows['oracle'] = {'uses_gold': True, 'calls_to_large': len(better), **score_mix(gold, small, large, better, scorer)}
    rows['large_only'] = {'uses_gold': False, 'calls_to_large': len(gold),
                          **score_mix(gold, small, large, set(gold), scorer)}
    for row in rows.values():
        row['share_to_large'] = round(row['calls_to_large'] / len(gold), 4)
    return {'calls': len(gold), 'agreement_threshold': agreement_threshold if small_b else None,
            'second_checkpoint_used': bool(small_b), 'policies': rows}


def as_markdown(report):
    lines = ['| Политика | Доля звонков большой модели | T | S | P | R | G | Читает эталон |', '|---|---:|---:|---:|---:|---:|---:|---|']
    for name, row in report['policies'].items():
        cells = [f"{row[m]:.4f}" if row[m] is not None else '—' for m in
                 ('temporal_recall', 'state_accuracy', 'fact_precision', 'fact_recall', 'grounding_turn')]
        lines.append(f"| `{name}` | {row['share_to_large']:.1%} | " + ' | '.join(cells)
                     + f" | {'да' if row['uses_gold'] else 'нет'} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gold', required=True, help='labelled calls with transcripts (val.jsonl)')
    parser.add_argument('--pred-small', required=True)
    parser.add_argument('--pred-large', required=True, help='e.g. the organizer baseline_pred.jsonl')
    parser.add_argument('--pred-small-b', help='second checkpoint of the small model, enables checkpoints_disagree')
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--agreement-threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    scorer = Scorer('2.1', args.catalog)
    catalog = json.loads(Path(args.catalog).read_text(encoding='utf-8'))
    report = simulate(read_jsonl(args.gold), read_jsonl(args.pred_small), read_jsonl(args.pred_large), catalog, scorer,
                      read_jsonl(args.pred_small_b) if args.pred_small_b else None,
                      args.agreement_threshold, args.seed)
    report['inputs_sha256'] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                               for name, path in (('gold', args.gold), ('pred_small', args.pred_small),
                                                  ('pred_large', args.pred_large), ('pred_small_b', args.pred_small_b),
                                                  ('catalog', args.catalog)) if path}
    report['scorer_sha256'] = scorer.sha256
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    (output / 'table.md').write_text(as_markdown(report), encoding='utf-8')
    print(as_markdown(report))


if __name__ == '__main__':
    main()
