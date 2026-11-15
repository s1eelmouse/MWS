"""Recover quality after an interrupted full val; never train or alter original files."""
import argparse
from collections import Counter
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import sys
import time


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


class ReplayClock:
    def __init__(self):
        self.elapsed = 0.0
        self.started = None

    def __call__(self):
        return self.elapsed + (time.perf_counter() - self.started if self.started is not None else 0.0)


def restore_history(calls, before, final, batches, records):
    """Replay completed batch observations, verifying every saved final answer.

    Replay excludes the server outage. Stored batch durations approximate active
    elapsed time for TTL. Any observable mismatch refuses recovery instead of
    silently changing completed answers.
    """
    from series_history import new_series_memory
    clock = ReplayClock()
    history = new_series_memory()
    history._clock = clock
    offset = 0
    for batch in batches:
        count = batch['calls']
        chunk_calls = calls[offset:offset + count]
        if len(chunk_calls) != count or any(r['batch_id'] != batch['batch_id'] for r in records[offset:offset + count]):
            raise ValueError('Saved batch boundaries disagree with call records')
        clock.elapsed += batch['seconds']
        replayed, _, _ = history.apply(chunk_calls, before[offset:offset + count])
        if replayed != final[offset:offset + count]:
            raise ValueError(f'History replay differs at batch {batch["batch_id"]}. '
                             'Original answers preserved; inspect TTL/settings before proceeding.')
        offset += count
    if offset != len(final):
        raise ValueError('Saved batches do not cover all completed calls')
    return history, clock


class Progress:
    def __init__(self, target, completed, total, old_failed):
        self.target, self.completed, self.total, self.old_failed = target, completed, total, old_failed

    def write(self, text):
        text = re.sub(r'^Completed (\d+)/(\d+); failed: (\d+)',
            lambda m: f'Completed {self.completed + int(m[1])}/{self.total}; failed: {self.old_failed + int(m[3])}', text)
        return self.target.write(text)

    def flush(self):
        self.target.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='4090-continue-v21-001')
    parser.add_argument('--recovery-id', default='validation-recovery-001')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    for value in (args.run_id, args.recovery_id):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value):
            parser.error('Simple run and recovery IDs required')
    home = Path.home()
    code = home / 'mws-v21/releases/case-v21-002'
    root = home / 'mws-v21/out' / args.run_id
    original, model = root / 'validation', root / 'merged'
    output = root / args.recovery_id
    data = code / 'data/organizer-v2.1'
    os.chdir(code)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', CASE_DATA_DIR=str(data), ORGANIZER_DIR=str(data),
        SCORER_VERSION='2.1', ALLOW_LEGACY_CONTRACT='1', CATALOG_PATH=str(data / 'catalog.json'),
        TEMPORAL_CATALOG_PATH=str(data / 'catalog.json'), MAX_INPUT_TOKENS='12288', MAX_SEQUENCE_TOKENS='16384',
        MAX_NEW_TOKENS='4096', MAX_TEXT_CHARS='0', JSON_CONSTRAINED='1', JSON_GRAMMAR_BACKEND='torch_native',
        JSON_GRAMMAR_THREADS='4', TEMPORAL_NORMALIZATION='case_v2', TEMPORAL_SERIES_HISTORY='1', FACT_RECOVERY='1',
        TEMPORAL_SERIES_MAX_ENTRIES='10000', HISTORY_MAX_ACCOUNTS='512', HISTORY_MAX_BYTES='8388608',
        HISTORY_TTL_SECONDS='3600', SAVE_RAW_RESPONSES='1', PROMPT_LOOKUP_TOKENS='0', RETRY_ON_TOKEN_LIMIT='1',
        RETRY_REPETITION_PENALTY='1.2', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MWS_CODE_REVISION='case-v21-002')
    sys.path.insert(0, str(code / 'src'))
    from schema import as_call
    from case_evaluation import evaluate_predictions
    from case_io import write_jsonl
    from measurements import distribution
    import fcntl
    with (root.parent / 'v21-training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Original training/evaluation pipeline is still running; do not start another copy')
        if (original / 'metrics.json').exists():
            raise SystemExit('Original metrics already exist; inspect them before recovery')
        manifest = json.loads((root / 'release-manifest.json').read_text())
        for name, expected in manifest['files'].items():
            if digest(code / name) != expected:
                raise ValueError('Release differs from original run: ' + name)
        stages = rows(root / 'stages.jsonl')
        if any(not any(r['stage'] == stage and r['exit_code'] == 0 for r in stages) for stage in ('train', 'merge')):
            raise ValueError('Original successful train/merge stage records not found')
        gold = rows(data / 'val.jsonl')
        calls = [as_call(row) for row in gold]
        saved = rows(original / 'predictions.jsonl')
        before = rows(original / 'predictions-before-series.jsonl')
        records = rows(original / 'calls.jsonl')
        batches = rows(original / 'batches.jsonl')
        count = len(saved)
        if not 0 < count < len(gold):
            raise ValueError('Expected an incomplete nonempty saved prefix')
        expected_ids = [c.call_id for c in calls[:count]]
        if any([r['call_id'] for r in part] != expected_ids for part in (saved, before, records)):
            raise ValueError('Saved files are not the identical ordered prefix of the new val')
        original_hashes = {str(p.relative_to(root)): digest(p) for p in original.rglob('*') if p.is_file()}
        # CPU-only preflight proves the saved history can actually be reconstructed.
        history, clock = restore_history(calls[:count], before, saved, batches, records)
        print(f'CHECK OK: {count}/{len(gold)} saved; {len(gold)-count} remain. '
              'History replay exactly matches every completed answer.', flush=True)
        if args.check_only:
            return
        output.mkdir(exist_ok=False)
        status, error, started = 'failed', None, time.perf_counter()
        try:
            shutil.copyfile(__file__, output / 'recovery-script.py')
            write(output / 'provenance.json', {'original_run': str(root), 'completed_calls': count,
                'remaining_calls': len(gold)-count, 'original_files_sha256': original_hashes,
                'val_sha256': digest(data / 'val.jsonl'), 'model_sha256': digest(model / 'model.safetensors'),
                'history_prefix_replay_exact': True,
                'history_clock_policy': 'Replay completed batch durations; exclude outage/load/warmup; resume active clock',
                'limitation': 'Historical TTL expiry during unmeasured I/O gaps cannot be reconstructed exactly.'})
            write_jsonl(output / 'remaining.jsonl', gold[count:])
            import extractor
            import torch
            if not torch.cuda.is_available():
                raise ValueError('CUDA unavailable')
            free, _ = torch.cuda.mem_get_info()
            if free < 10 * 2**30:
                raise ValueError('Less than 10 GiB free VRAM; inspect active GPU jobs')
            real_class = extractor.Extractor
            backend = real_class(str(model), False, 'cuda')
            # benchmark resets history after warmup. Hydrate precisely at that boundary.
            def reset():
                restored, active_clock = restore_history(calls[:count], before, saved, batches, records)
                active_clock.started = time.perf_counter()
                backend.series_memory = restored
            backend.reset_series_history = reset
            extractor.Extractor = lambda *a, **k: backend
            sys.argv = ['benchmark.py', '--input', str(output / 'remaining.jsonl'), '--model', str(model),
                        '--device', 'cuda', '--batch-size', '2', '--warmup-batches', '1',
                        '--output-dir', str(output / 'tail')]
            try:
                with contextlib.redirect_stdout(Progress(sys.stdout, count, len(gold), sum(bool(r.get('error')) for r in saved))):
                    runpy.run_path(str(code / 'src/benchmark.py'), run_name='__main__')
            finally:
                extractor.Extractor = real_class
            final = saved + rows(output / 'tail/predictions.jsonl')
            full_before = before + rows(output / 'tail/predictions-before-series.jsonl')
            if [r['call_id'] for r in final] != [r['call_id'] for r in gold]:
                raise ValueError('Recovered predictions do not cover full ordered val')
            quality, errors = evaluate_predictions(gold, final, scorer_version='2.1', catalog_path=data / 'catalog.json')
            old_count = len(batches)
            tail_records = rows(output / 'tail/calls.jsonl')
            tail_batches = rows(output / 'tail/batches.jsonl')
            for row in tail_records + tail_batches:
                row['batch_id'] += old_count
            all_records, all_batches = records + tail_records, batches + tail_batches
            for name, items in [('predictions.jsonl', final), ('predictions-before-series.jsonl', full_before),
                                ('calls.jsonl', all_records), ('batches.jsonl', all_batches), ('errors.jsonl', errors)]:
                write_jsonl(output / name, items)
            raw_complete = False
            try:
                raw = rows(original / 'raw-answers.jsonl') + rows(output / 'tail/raw-answers.jsonl')
                if [r['call_id'] for r in raw] == [r['call_id'] for r in gold]:
                    write_jsonl(output / 'raw-answers.jsonl', raw)
                    raw_complete = True
            except (OSError, ValueError):
                pass  # Retain original raw files; incomplete optional raw capture does not invalidate quality.
            audit = []
            normalizations = Counter()
            for record in all_records:
                normalizations.update(record.get('normalization') or {})
                audit.extend({'call_id': record['call_id'], 'component': 'temporal', **d}
                             for d in record.get('temporal_decisions', []))
                audit.extend(record.get('series_decisions', []))
            write_jsonl(output / 'temporal-audit.jsonl', audit)
            write(output / 'metrics.json', quality)
            base, _ = evaluate_predictions(gold, rows(data / 'baseline_pred.jsonl'), scorer_version='2.1', catalog_path=data / 'catalog.json')
            write(output / 'baseline-metrics.json', base)
            seconds = sum(r['seconds'] for r in all_batches)
            tail_report = json.loads((output / 'tail/report.json').read_text())
            write(output / 'report.json', {'protocol': 'local-inference-v2-recovered', 'status': 'quality_complete',
                'calls': len(final), 'failed_calls': sum(bool(r.get('error')) for r in final), 'quality': quality,
                'recovery': {'completed_before_outage': count, 'generated_after_outage': len(final)-count,
                             'segments': 2, 'outage_included_in_measurement': False, 'full_raw_capture': raw_complete},
                'normalization_counts': dict(normalizations),
                'error_counts': dict(Counter(r['error'] for r in final if r.get('error'))),
                'measurement_seconds_segments_sum': seconds, 'calls_per_hour_segments_sum': len(final)*3600/seconds,
                'gpu_seconds_per_1000_calls_segments_sum': seconds*1000/len(final),
                'call_completion_seconds': distribution([r['completion_latency_seconds'] for r in all_records]),
                'input_tokens': distribution([r.get('input_tokens') for r in all_records]),
                'output_tokens': distribution([r.get('output_tokens') for r in all_records]),
                'full_run_cuda_peak_allocated_bytes': None, 'full_run_cuda_peak_reserved_bytes': None,
                'tail_resources': tail_report['memory'],
                'resource_limitation': 'Interrupted run. First-segment CUDA peaks were not saved; no continuous full-run resource claim.',
                'model_artifacts': tail_report['model_artifacts'], 'run_context': tail_report['run_context'],
                'note': 'run_context.val in tail describes remaining subset; full-val SHA is in provenance.json.'})
            from compare_temporal import compare_date_predictions
            comparison, details = compare_date_predictions(gold, full_before, final, scorer_version='2.1', catalog_path=data / 'catalog.json')
            write(output / 'history-comparison.json', comparison)
            write_jsonl(output / 'history-changed-matches.jsonl', details)
            ref = {'fact_precision': .4792, 'fact_recall': .5117, 'temporal_recall': .4709,
                   'state_accuracy': .4767, 'grounding_turn': .7150}
            write(output / 'comparison.json', {'official': quality['official'], 'baseline': base['official'],
                'second_adapter_v100_reference': ref,
                'delta_from_second_adapter': {k: round(quality['official'][k]-v, 6) if quality['official'][k] is not None else None for k,v in ref.items()},
                'thresholds': {'temporal_recall': .4506, 'state_accuracy': .5145}})
            for name, expected in original_hashes.items():
                if digest(root / name) != expected:
                    raise ValueError('Original file changed during recovery: ' + name)
            status = 'complete'
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            write(output / 'recovery-status.json', {'status': status, 'error': error,
                'elapsed_seconds': time.perf_counter()-started, 'training_rerun': False})
        print('DONE:', output / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
