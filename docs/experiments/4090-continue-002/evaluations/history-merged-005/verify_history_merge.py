"""CPU-only comparison on frozen predictions; generation and training are not run."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / 'src'))
from case_io import read_jsonl, write_jsonl
from case_evaluation import evaluate_predictions
from compare_temporal import compare_date_predictions
from postprocess_dates import process_predictions
from schema import as_call, contract_signature
from series_history import SeriesDateMemory, _account, _time_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', required=True)
    parser.add_argument('--pred-before-history', required=True)
    parser.add_argument('--old-final', required=True)
    parser.add_argument('--reference-ref', default='992b439')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    gold = read_jsonl(args.gold)
    calls = [as_call(row) for row in gold]  # strips labels at the rules boundary
    original = read_jsonl(args.pred_before_history)
    previous_final = read_jsonl(args.old_final)
    before, _, _ = process_predictions(calls, original, mode='task')
    source = subprocess.check_output(
        ['git', '-C', str(REPO), 'show', args.reference_ref + ':src/series_history.py'])
    namespace = {'__name__': 'pinned_history_reference'}
    exec(compile(source, 'pinned_series_history.py', 'exec'), namespace)
    reference, _, reference_counts = namespace['SeriesDateMemory']().apply(calls, before)
    memory = SeriesDateMemory()
    after, audit, counts = memory.apply(calls, before)
    comparisons = {}
    for label, predictions in [('without_history', before), ('vova_992b439', reference),
                               ('old_full_val_003', previous_final)]:
        comparison, changes = compare_date_predictions(gold, predictions, after)
        comparisons[label] = {'before': comparison['before']['official'],
                              'after': comparison['after']['official'],
                              'transitions': comparison['transitions'],
                              'net_correct_pairs': comparison['net_correct_pairs']}
        write_jsonl(output / (label + '-changed-matches.jsonl'), changes)
    lookup = {row['call_id']: row for row in before}
    ordered = sorted(calls, key=lambda c: (_account(c) or c.call_id, *_time_key(c.started_at), c.call_id))
    equality = {}
    for batch_size in (1, 2):
        stream, predictions = SeriesDateMemory(), {}
        for offset in range(0, len(ordered), batch_size):
            batch = ordered[offset:offset + batch_size]
            rows, _, _ = stream.apply(batch, [lookup[c.call_id] for c in batch])
            predictions.update({c.call_id: row for c, row in zip(batch, rows)})
        equality[str(batch_size)] = [predictions[c.call_id] for c in calls] == after
        assert equality[str(batch_size)], 'Chronological streaming differs from full batch'
    durations = []
    for _ in range(5):
        fresh = SeriesDateMemory()
        start = time.perf_counter()
        fresh.apply(calls, before)
        durations.append(time.perf_counter() - start)
    traced = SeriesDateMemory()
    tracemalloc.start()
    traced.apply(calls, before)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    quality, _ = evaluate_predictions(gold, after)
    verification = {'protocol': 'frozen-predictions-cpu-date-only-v1',
                    'calls': len(calls), 'reference_ref': args.reference_ref,
                    'contract_signature': contract_signature(),
                    'file_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                    for name, path in [('gold', args.gold),
                                                       ('pred_before_history', args.pred_before_history),
                                                       ('old_final', args.old_final)]},
                    'source_sha256': {name: hashlib.sha256((REPO / 'src' / name).read_bytes()).hexdigest()
                                      for name in ['series_history.py', 'temporal.py', 'temporal_binding.py']},
                    'quality': quality, 'comparisons': comparisons, 'history': memory.stats(),
                    'counts': counts, 'reference_counts': reference_counts,
                    'stream_equals_batch': equality, 'history_cpu_seconds_runs': durations,
                    'history_cpu_seconds_median': statistics.median(durations),
                    'history_python_traced_peak_bytes': peak,
                    'gold_passed_to_rules': False, 'training_or_generation_run': False,
                    'resource_note': 'Local Windows CPU, copying/sorting/history only; no GPU speed claim. '
                                     'tracemalloc measures Python allocations, not process RSS.'}
    write_jsonl(output / 'predictions.jsonl', after)
    write_jsonl(output / 'temporal-audit.jsonl', audit)
    (output / 'verification.json').write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'official': quality['official'], 'comparisons': comparisons,
                      'memory': memory.stats(), 'cpu_median': statistics.median(durations)},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
