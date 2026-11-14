"""Evaluate the unchanged cycle-2 weights on the entire organizer v2.1 val."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
EXPECTED_WEIGHTS = 'ef0bc5ae0646f6a4bda4cee69fe64b292d710e5b54119ac5ae9868720af584f7'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--team-root', default=str(Path.home() / 'team_shpana'))
    parser.add_argument('--run-id', default='continue-002-v21-full-val-002')
    parser.add_argument('--budget-gib', type=float, default=12)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.run_id):
        parser.error('Use a simple unique run ID')
    if not math.isfinite(args.budget_gib) or args.budget_gib <= 0 or args.batch_size <= 0:
        parser.error('Positive memory budget and batch size required')
    team = Path(args.team_root).resolve()
    model = team / 'models/continue-002'
    data = ROOT / 'data/organizer-v2.1'
    output = team / 'out' / args.run_id
    preflight = team / 'out' / (args.run_id + '-preflight')
    if output.exists() or preflight.exists():
        raise FileExistsError('Run exists. Preserve it and select a new --run-id.')
    os.chdir(ROOT)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, CASE_DATA_DIR=str(data), SCORER_VERSION='2.1',
                      ALLOW_LEGACY_CONTRACT='1', CATALOG_PATH=str(data / 'catalog.json'),
                      MAX_INPUT_TOKENS='12288', MAX_SEQUENCE_TOKENS='16384', MAX_NEW_TOKENS='4096',
                      MAX_TEXT_CHARS='0', PROMPT_LOOKUP_TOKENS='0', JSON_CONSTRAINED='1',
                      JSON_GRAMMAR_BACKEND='torch_native', JSON_GRAMMAR_THREADS='4',
                      TEMPORAL_NORMALIZATION='case_v2', FACT_RECOVERY='1',
                      TEMPORAL_CATALOG_PATH=str(data / 'catalog.json'), TEMPORAL_SERIES_HISTORY='1',
                      TEMPORAL_SERIES_MAX_ENTRIES='10000', HISTORY_MAX_ACCOUNTS='512',
                      HISTORY_MAX_BYTES='8388608', HISTORY_TTL_SECONDS='3600', SAVE_RAW_RESPONSES='1',
                      RETRY_ON_TOKEN_LIMIT='1', RETRY_REPETITION_PENALTY='1.2',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MWS_CODE_REVISION='case-v21-002')
    sys.path.insert(0, str(ROOT / 'src'))
    from case_io import read_jsonl
    from check_data import inspect_rows
    from case_evaluation import evaluate_predictions
    from measurements import write_json
    from schema import check_model_contract, contract_manifest
    from training_data import ensure_training_splits
    from run_context import context
    import fcntl
    (team / 'out').mkdir(parents=True, exist_ok=True)
    with (team / 'out/v21-evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another v2.1 evaluation from this team directory is running')
        preflight.mkdir()
        begin, status, error = time.perf_counter(), 'failed', None
        try:
            package = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())
            for name, expected in package['files'].items():
                with (ROOT / name).open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != expected:
                    raise ValueError(f'Release checksum mismatch: {name}')
            shutil.copyfile(ROOT / 'RELEASE_MANIFEST.json', preflight / 'release-manifest.json')
            shutil.copyfile(__file__, preflight / 'run_full_val.py')
            with (preflight / 'tests.log').open('x', encoding='utf-8') as log:
                test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'],
                                      cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if test.returncode:
                raise RuntimeError(f'Server tests failed; inspect {preflight / "tests.log"}')
            subprocess.run([sys.executable, '-m', 'compileall', '-q', str(ROOT / 'src'), str(ROOT / 'tests')], check=True)
            train, val = read_jsonl(data / 'train.jsonl'), read_jsonl(data / 'val.jsonl')
            if len(train) != 3926 or len(val) != 383:
                raise ValueError('Wrong organizer split sizes')
            ensure_training_splits(train, val)
            validation = {'train': inspect_rows(train), 'val': inspect_rows(val)}
            write_json(preflight / 'data-validation.json', validation)
            if any(r['invalid_calls'] for r in validation.values()):
                raise ValueError('New dataset validation failed')
            provenance = context(train=data / 'train.jsonl', val=data / 'val.jsonl', catalog=data / 'catalog.json')
            compatibility = check_model_contract(model, allow_legacy=True)
            with (model / 'model.safetensors').open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != EXPECTED_WEIGHTS:
                    raise ValueError('This is not the verified second-cycle merged model')
            write_json(preflight / 'run-settings.json', {'settings': vars(args), 'run_context': provenance,
                       'model_contract': compatibility, 'target_contract': contract_manifest(),
                       'model': str(model), 'evaluation_only': True, 'model_weights_modified': False})
            baseline, errors = evaluate_predictions(val, read_jsonl(data / 'baseline_pred.jsonl'),
                                                   scorer_version='2.1', catalog_path=data / 'catalog.json')
            published = json.loads((data / 'baseline_metrics.json').read_text())
            if baseline['official'] != published:
                raise ValueError('Official v2.1 baseline does not reproduce exactly')
            write_json(preflight / 'baseline-metrics.json', baseline)
            print('PREFLIGHT OK: release, v2.1 data, legacy contract, cycle-2 weights and published baseline.', flush=True)
            import torch
            if not torch.cuda.is_available() or not (torch.version.cuda or '').startswith('12.'):
                raise ValueError('Expected the existing CUDA 12.x environment')
            free, total = torch.cuda.mem_get_info()
            budget = int(args.budget_gib * 2**30)
            if budget > total or free < budget + 2 * 2**30:
                raise ValueError('Insufficient free VRAM for budget plus 2 GiB headroom; coordinate/wait')
            torch.cuda.set_per_process_memory_fraction(budget / total, 0)
            torch.set_num_threads(4)
            write_json(preflight / 'gpu-policy.json', {'gpu': torch.cuda.get_device_name(0),
                'free_bytes_before': free, 'total_bytes': total, 'allocator_budget_bytes': budget,
                'threads': 4, 'note': 'Not an exclusive reservation; other teams can allocate concurrently.'})
            sys.argv = ['benchmark.py', '--input', str(data / 'val.jsonl'), '--model', str(model),
                        '--device', 'cuda', '--batch-size', str(args.batch_size),
                        '--warmup-batches', '1', '--output-dir', str(output)]
            runpy.run_path(str(ROOT / 'src/benchmark.py'), run_name='__main__')
            sys.argv = ['compare_temporal.py', '--gold', str(data / 'val.jsonl'), '--scorer-version', '2.1',
                        '--catalog', str(data / 'catalog.json'),
                        '--before', str(output / 'predictions-before-series.jsonl'),
                        '--after', str(output / 'predictions.jsonl'), '--output-dir', str(output / 'history-date-comparison')]
            runpy.run_path(str(ROOT / 'src/compare_temporal.py'), run_name='__main__')
            actual = json.loads((output / 'metrics.json').read_text())['official']
            comparison = {'scorer_version': '2.1', 'baseline': published, 'model': actual,
                          'delta': {k: round(actual[k] - published[k], 6) if actual.get(k) is not None else None
                                    for k in ('fact_precision', 'fact_recall', 'temporal_recall', 'state_accuracy', 'grounding_turn')},
                          'target_minimum': {'temporal_recall': 0.4506, 'state_accuracy': 0.5145}}
            write_json(output / 'quality-vs-baseline.json', comparison)
            status = 'complete'
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            write_json(preflight / 'pipeline-status.json', {'status': status, 'error': error,
                       'elapsed_seconds': time.perf_counter() - begin, 'evaluation_only': True})
            if output.exists():
                shutil.copytree(preflight, output / 'preflight')
        print('DONE:', output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
