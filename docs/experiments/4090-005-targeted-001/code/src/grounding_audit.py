"""Explain source-turn errors using the unchanged official fact matcher.

Evaluation only: labels are read here and never passed to prediction rules.
Counts concern matched facts; unmatched predictions do not enter grounding_turn.
No source, value, date or prediction is rewritten by this module.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

ATTRIBUTES = ('service_state', 'tariff', 'ticket_status', 'action', 'agreement')
CONFIRMATION = re.compile(
    r'\b(?:да|угу|ага|ладно|хорошо|верно|правильно|подходит|договорились|'
    r'соглас[а-я]*|устраивает|подтвержда[а-я]*)\b')


def _confirmation(turn):
    if turn is None:
        return False
    text = turn['text'].lower().replace('ё', 'е')
    return len(re.findall(r'\w+', text)) <= 12 and bool(CONFIRMATION.search(text))


def _index(rows, label):
    result = {}
    for row in rows:
        if row['call_id'] in result:
            raise ValueError(f'Duplicate {label} call_id: {row["call_id"]}')
        result[row['call_id']] = row
    return result


def diagnose_grounding(gold_rows, prediction_rows, *, scorer, details=False):
    """Return additive counters and optional errors; keep full official matches intact."""
    gold = _index(gold_rows, 'gold')
    predictions = _index(prediction_rows, 'prediction')
    if predictions.keys() - gold.keys():
        raise ValueError('Predictions contain unknown call IDs')
    counts = {attr: Counter() for attr in ATTRIBUTES}
    errors = []
    for cid, call in gold.items():
        prediction = predictions.get(cid, {})
        predicted = [] if prediction.get('error') else prediction.get('facts') or []
        gfacts = call['facts']
        pairs = scorer.match_facts(gfacts, predicted)
        matched_pred = {id(p) for _, p, _ in pairs}
        for fact in predicted:
            if fact.get('attribute') in counts and id(fact) not in matched_pred:
                counts[fact['attribute']]['unmatched_predictions'] += 1
        turns = {turn['idx']: turn for turn in call.get('turns', [])}
        positions = {turn['idx']: i for i, turn in enumerate(call.get('turns', []))}
        for gfact, pfact, _ in pairs:
            attr = gfact['attribute']
            if attr not in counts:
                continue
            item = counts[attr]
            item['matched_facts'] += 1
            gi = gfact['source']['turn_idx']
            pi = (pfact.get('source') or {}).get('turn_idx')
            correct = pi == gi  # preserve the official scorer's equality semantics
            item['correct_source_turn'] += correct
            if correct:
                continue
            item['wrong_source_turn'] += 1
            gt, pt = turns.get(gi), turns.get(pi) if type(pi) is int else None
            if gt is None or pt is None:
                bucket = 'missing_or_invalid_source_turn'
            else:
                distance = abs(positions[gi] - positions[pi])
                bucket = ('adjacent_turn' if distance == 1 else 'two_turns_apart'
                          if distance == 2 else 'farther_than_two_turns')
                item['different_speaker'] += gt['speaker'] != pt['speaker']
                item['pred_source_is_short_confirmation'] += _confirmation(pt)
                item['gold_source_is_short_confirmation'] += _confirmation(gt)
            item[bucket] += 1
            if details:
                errors.append({'call_id': cid, 'attribute': attr,
                               'gold_fact_index': next(i for i,f in enumerate(gfacts) if f is gfact),
                               'pred_fact_index': next(i for i,f in enumerate(predicted) if f is pfact),
                               'value': pfact['value'], 'pred_source_turn': pi,
                               'gold_source_turn': gi, 'distance_bucket': bucket,
                               'pred_source_is_short_confirmation': _confirmation(pt),
                               'gold_source_is_short_confirmation': _confirmation(gt)})
    fields = ('matched_facts', 'correct_source_turn', 'wrong_source_turn', 'unmatched_predictions',
              'adjacent_turn', 'two_turns_apart', 'farther_than_two_turns',
              'missing_or_invalid_source_turn', 'different_speaker',
              'pred_source_is_short_confirmation', 'gold_source_is_short_confirmation')
    total_errors = sum(item['wrong_source_turn'] for item in counts.values())
    result = {}
    for attr, item in counts.items():
        result[attr] = {key: item[key] for key in fields}
        result[attr]['grounding_turn'] = (round(item['correct_source_turn']/item['matched_facts'], 4)
                                          if item['matched_facts'] else None)
        result[attr]['share_of_all_source_errors'] = (round(item['wrong_source_turn']/total_errors, 4)
                                                      if total_errors else None)
    matched = sum(item['matched_facts'] for item in counts.values())
    correct = sum(item['correct_source_turn'] for item in counts.values())
    return {'scope': 'officially_matched_facts_only', 'matched_facts': matched,
            'correct_source_turn': correct, 'wrong_source_turn': total_errors,
            'grounding_turn': round(correct/matched, 4) if matched else None,
            'per_attribute': result}, errors


def compare_grounding(gold_rows, before, after, *, scorer):
    """Check individual regressions and source-only invariants, not just percentages."""
    old, new = _index(before, 'before'), _index(after, 'after')
    if old.keys() != new.keys():
        raise ValueError('Candidate and reference must contain the same call IDs')
    fixes, regressions = [], []
    source_only = True
    for call in gold_rows:
        cid = call['call_id']
        left, right = old.get(cid, {}).get('facts') or [], new.get(cid, {}).get('facts') or []
        def content(fact):
            return {key:value for key,value in fact.items() if key not in ('source', 'speaker')}
        source_only &= [content(f) for f in left] == [content(f) for f in right]
        a = {id(g): (g['source']['turn_idx'] == (p.get('source') or {}).get('turn_idx'))
             for g,p,_ in scorer.match_facts(call['facts'], left)}
        b = {id(g): (g['source']['turn_idx'] == (p.get('source') or {}).get('turn_idx'))
             for g,p,_ in scorer.match_facts(call['facts'], right)}
        # Losing a matched fact with a formerly correct source is a regression too.
        for index, fact in enumerate(call['facts']):
            was, now = a.get(id(fact), False), b.get(id(fact), False)
            entry = {'call_id':cid, 'gold_fact_index':index, 'attribute':fact['attribute']}
            if now and not was:
                fixes.append(entry)
            if was and not now:
                regressions.append(entry)
    return {'only_sources_and_speakers_changed': bool(source_only),
            'fixes':len(fixes), 'regressions':len(regressions),
            'fixed_facts':fixes, 'regressed_facts':regressions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', required=True)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--candidate', help='optional saved answers after a proposed fix')
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        parser.error('Output directory already exists; choose a new experiment name')
    from case_io import read_jsonl, write_jsonl
    from case_evaluation import evaluate_predictions
    from scoring import Scorer
    scorer = Scorer('2.1', args.catalog)
    gold, pred = read_jsonl(args.gold), read_jsonl(args.pred)
    start = time.perf_counter()
    metrics, _ = evaluate_predictions(gold, pred, scorer=scorer)
    baseline, errors = diagnose_grounding(gold, pred, scorer=scorer, details=True)
    report = {'baseline':baseline, 'baseline_metrics':metrics, 'settings':vars(args),
              'gold_passed_to_runtime_rules':False, **scorer.metadata()}
    if args.candidate:
        candidate = read_jsonl(args.candidate)
        candidate_metrics, _ = evaluate_predictions(gold, candidate, scorer=scorer)
        report['candidate_metrics'] = candidate_metrics
        report['comparison'] = compare_grounding(gold, pred, candidate, scorer=scorer)
        keys = ('fact_precision','fact_recall','temporal_recall','state_accuracy','grounding_turn')
        report['metric_deltas'] = {key:round(candidate_metrics['official'][key]-metrics['official'][key], 4)
                                  if metrics['official'][key] is not None and candidate_metrics['official'][key] is not None
                                  else None for key in keys}
        report['no_aggregate_metric_regressions'] = all(v is None or v >= 0 for v in report['metric_deltas'].values())
    report['cpu_analysis_seconds'] = time.perf_counter()-start
    report['inputs_sha256'] = {key:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                               for key,path in [('gold',args.gold),('pred',args.pred),('candidate',args.candidate)] if path}
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output/'source-errors.jsonl', errors)
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output':str(output), 'grounding':baseline['grounding_turn'],
                      'wrong_sources':baseline['wrong_source_turn']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
