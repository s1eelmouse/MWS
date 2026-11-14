"""Apply deterministic rules to saved predictions without a model or gold access."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time

from case_io import read_jsonl, write_jsonl
from catalog_normalization import load_catalog, normalize_catalog_facts
from schema import Facts, as_call, contract_signature, source_problems
from temporal import normalize_fact_dates
from series_history import SeriesDateMemory


def process_predictions(calls, predictions, *, mode='conservative', catalog=None,
                        deduplicate=False, speaker_from_source=False, allow_series_history=False):
    if mode not in ('off', 'conservative', 'task', 'binding'):
        raise ValueError('Unknown temporal mode')
    if allow_series_history and mode not in ('task', 'binding'):
        raise ValueError('Series history requires task or binding mode')
    by_id = {row['call_id']: row for row in predictions}
    if len(by_id) != len(predictions) or len({call.call_id for call in calls}) != len(calls):
        raise ValueError('Duplicate call IDs')
    if set(by_id) != {call.call_id for call in calls}:
        raise ValueError('Inputs and predictions must contain the same call IDs')
    output, audit, counts = [], [], Counter()
    processed = {}
    for call in calls:
        row = copy.deepcopy(by_id[call.call_id])
        if row.get('error'):
            processed[call.call_id] = row
            counts['failed_calls_preserved'] += 1
            continue
        items = row['facts']
        if catalog is not None:
            items, decisions = normalize_catalog_facts(items, catalog)
            audit.extend({'call_id': call.call_id, 'component': 'catalog', **d} for d in decisions)
            counts['catalog_fields_normalized'] += len(decisions)
        if mode in ('conservative', 'task', 'binding'):
            items, decisions = normalize_fact_dates(items, call, mode=mode)
            audit.extend({'call_id': call.call_id, 'component': 'temporal', **d} for d in decisions)
            for d in decisions:
                counts['temporal_decisions:' + d['reason']] += 1
                counts['dates_changed'] += len(d['changes'])
        turns = {turn.idx: turn for turn in call.turns}
        seen, final = set(), []
        for index, item in enumerate(items):
            turn = turns.get(item['source']['turn_idx'])
            if speaker_from_source and turn is not None and item.get('speaker') != turn.speaker:
                audit.append({'call_id': call.call_id, 'component': 'diarization',
                              'fact_index': index, 'field': 'speaker',
                              'before': item.get('speaker'), 'after': turn.speaker})
                item['speaker'] = turn.speaker
                counts['speaker_from_turn'] += 1
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if deduplicate and key in seen:
                counts['exact_duplicates_removed'] += 1
                audit.append({'call_id': call.call_id, 'component': 'deduplication',
                              'fact_index': index, 'reason': 'exact_duplicate'})
                continue
            seen.add(key)
            final.append(item)
        parsed = Facts.model_validate({'facts': final})
        problems = source_problems(parsed, call)
        if problems:
            raise ValueError(f'{call.call_id}: invalid sources: {problems}')
        row['facts'] = final
        processed[call.call_id] = row
    output = [processed[call.call_id] for call in calls]
    if allow_series_history:
        output, history_events, history_counts = SeriesDateMemory().apply(calls, output)
        audit.extend(history_events)
        counts.update(history_counts)
        for call, row in zip(calls, output):
            if not row.get('error'):
                Facts.model_validate({'facts': row['facts']})
    return output, audit, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='calls; gold fields are stripped at the boundary')
    parser.add_argument('--pred', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', choices=['off', 'conservative', 'task', 'binding'], default='task')
    parser.add_argument('--catalog', help='optional exact catalog normalization; separate from date-only A/B')
    parser.add_argument('--deduplicate', action='store_true')
    parser.add_argument('--speaker-from-source', action='store_true')
    parser.add_argument('--allow-series-history', action='store_true',
                        help='use earlier calls of the same identified ticket/account; default matches stateless HTTP')
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        parser.error('Output directory already exists; choose a new experiment name')
    calls = [as_call(row) for row in read_jsonl(args.input)]
    if not calls:
        parser.error('Input contains no calls')
    predictions = read_jsonl(args.pred)
    resolver = load_catalog(args.catalog) if args.catalog else None
    start = time.perf_counter()
    results, audit, counts = process_predictions(
        calls, predictions, mode=args.mode, catalog=resolver,
        deduplicate=args.deduplicate, speaker_from_source=args.speaker_from_source,
        allow_series_history=args.allow_series_history)
    elapsed = time.perf_counter() - start
    manifest = {
        'settings': vars(args), 'calls': len(calls), 'counts': counts,
        'processing_seconds': elapsed, 'contract_signature': contract_signature(),
        'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        'prediction_sha256': hashlib.sha256(Path(args.pred).read_bytes()).hexdigest(),
        'catalog_sha256': hashlib.sha256(Path(args.catalog).read_bytes()).hexdigest() if args.catalog else None,
        'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ['temporal.py', 'temporal_binding.py', 'series_history.py',
                                       'catalog_normalization.py', 'postprocess_dates.py']},
        'gold_fields_passed_to_rules': False,
        'history_scope': 'earlier_same_account_ticket' if args.allow_series_history else 'single_call',
    }
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / 'predictions.jsonl', results)
    write_jsonl(output / 'audit.jsonl', audit)
    with (output / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
