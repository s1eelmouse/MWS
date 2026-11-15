"""One fresh-optimizer epoch from verified cycle-2 adapter, merge and full v2.1 val."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

EXPECTED_ADAPTER = '8ffa31fb6dd06ef5020a20126dfa32c101ab5023b1a1e8e2af92055dd7c90c84'
V100_REFERENCE = {'fact_precision': .4792, 'fact_recall': .5117, 'temporal_recall': .4709,
                  'state_accuracy': .4767, 'grounding_turn': .7150}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def stage(root, code, name, arguments):
    begin = time.perf_counter()
    command = [sys.executable, '-u', *map(str, arguments)]
    print('START:', name, flush=True)
    result, error = None, None
    try:
        with (root / (name + '.log')).open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=code, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                result = process.wait()
            finally:
                process.stdout.close()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if result:
            raise RuntimeError(f'{name} failed, exit {result}; inspect {root / (name + ".log")}')
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        with (root / 'stages.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'stage': name, 'command': command, 'exit_code': result,
                                    'seconds': time.perf_counter() - begin, 'error': error}) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='4090-continue-v21-001')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.run_id):
        parser.error('Use a simple unique run ID')
    home = Path.home()
    code = home / 'mws-v21/releases/case-v21-002'
    initial = home / 'mws/out/4090-continue-002'
    root = home / 'mws-v21/out' / args.run_id
    data = code / 'data/organizer-v2.1'
    os.chdir(code)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', BASE_MODEL='Qwen/Qwen3-0.6B',
        CASE_DATA_DIR=str(data), ORGANIZER_DIR=str(data), SCORER_VERSION='2.1', ALLOW_LEGACY_CONTRACT='1',
        CATALOG_PATH=str(data / 'catalog.json'), TEMPORAL_CATALOG_PATH=str(data / 'catalog.json'),
        MAX_INPUT_TOKENS='12288', MAX_SEQUENCE_TOKENS='16384', MAX_NEW_TOKENS='4096', MAX_TEXT_CHARS='0',
        JSON_CONSTRAINED='1', JSON_GRAMMAR_BACKEND='torch_native', JSON_GRAMMAR_THREADS='4',
        TEMPORAL_NORMALIZATION='case_v2', TEMPORAL_SERIES_HISTORY='1', FACT_RECOVERY='1',
        TEMPORAL_SERIES_MAX_ENTRIES='10000', HISTORY_MAX_ACCOUNTS='512', HISTORY_MAX_BYTES='8388608',
        HISTORY_TTL_SECONDS='3600', SAVE_RAW_RESPONSES='1', PROMPT_LOOKUP_TOKENS='0',
        RETRY_ON_TOKEN_LIMIT='1', RETRY_REPETITION_PENALTY='1.2', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', MWS_CODE_REVISION='case-v21-002')
    sys.path.insert(0, str(code / 'src'))
    import fcntl
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / 'v21-training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another v2.1 training from this directory is running')
        root.mkdir(exist_ok=False)
        start, status, error = time.perf_counter(), 'failed', None
        try:
            manifest = json.loads((code / 'RELEASE_MANIFEST.json').read_text())
            if manifest['release_id'] != 'case-v21-002':
                raise ValueError('Unexpected code release')
            for name, expected in manifest['files'].items():
                if digest(code / name) != expected:
                    raise ValueError('Release file changed: ' + name)
            if digest(initial / 'adapter_model.safetensors') != EXPECTED_ADAPTER:
                raise ValueError('Initial adapter differs from the verified second-generation adapter')
            from training_data import adapter_origin
            origin = adapter_origin(initial, root / 'adapter', 'Qwen/Qwen3-0.6B', allow_legacy=True)
            import torch
            if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
                raise ValueError('Expected the RTX 4090 CUDA environment')
            free, total = torch.cuda.mem_get_info()
            if free < 14 * 2**30:
                raise ValueError('Less than 14 GiB free VRAM; inspect other GPU processes first')
            from measurements import environment
            write(root / 'environment.json', environment())
            write(root / 'settings.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
                'run_id': args.run_id, 'code': str(code), 'initial_adapter': origin,
                'epochs': 1, 'learning_rate': .0001, 'seed': 42, 'train_batch': 1,
                'gradient_accumulation': 16, 'effective_train_batch': 16, 'validation_batch': 2,
                'checkpoint_every_steps': 50, 'rolling_checkpoint_limit': 2,
                'optimizer': 'fresh, no checkpoint resume', 'train_sha256': digest(data / 'train.jsonl'),
                'val_sha256': digest(data / 'val.jsonl'), 'free_vram_bytes_before': free,
                'total_vram_bytes': total, 'inference_environment': {k: os.environ[k]
                    for k in ('SCORER_VERSION','JSON_CONSTRAINED','JSON_GRAMMAR_BACKEND','TEMPORAL_NORMALIZATION',
                              'TEMPORAL_SERIES_HISTORY','FACT_RECOVERY','CATALOG_PATH','TEMPORAL_CATALOG_PATH')}})
            shutil.copyfile(__file__, root / 'run-cycle.py')
            shutil.copyfile(code / 'RELEASE_MANIFEST.json', root / 'release-manifest.json')
            stage(root, code, 'tests', ['-X', 'utf8', '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
            stage(root, code, 'check-data', ['src/train.py', '--check-data'])
            stage(root, code, 'train', ['src/train.py', '--train', data / 'train.jsonl', '--val', data / 'val.jsonl',
                '--catalog', data / 'catalog.json', '--init-adapter-dir', initial, '--allow-legacy-contract',
                '--epochs', '1', '--batch-size', '1', '--learning-rate', '0.0001', '--seed', '42',
                '--save-steps', '50', '--output-dir', root / 'adapter'])
            stage(root, code, 'merge', ['src/merge.py', '--adapter-dir', root / 'adapter', '--output', root / 'merged'])
            stage(root, code, 'full-val', ['src/benchmark.py', '--input', data / 'val.jsonl', '--model', root / 'merged',
                '--device', 'cuda', '--batch-size', '2', '--warmup-batches', '1', '--output-dir', root / 'validation'])
            stage(root, code, 'history-comparison', ['src/compare_temporal.py', '--gold', data / 'val.jsonl',
                '--before', root / 'validation/predictions-before-series.jsonl', '--after', root / 'validation/predictions.jsonl',
                '--scorer-version', '2.1', '--catalog', data / 'catalog.json', '--output-dir', root / 'history-comparison'])
            stage(root, code, 'baseline', ['src/eval.py', '--gold', data / 'val.jsonl', '--pred', data / 'baseline_pred.jsonl',
                '--scorer-version', '2.1', '--catalog', data / 'catalog.json', '--reference-metrics', data / 'baseline_metrics.json',
                '--json', root / 'baseline-metrics.json'])
            actual = json.loads((root / 'validation/metrics.json').read_text())['official']
            baseline = json.loads((data / 'baseline_metrics.json').read_text())
            write(root / 'comparison.json', {'official': actual, 'baseline': baseline,
                'second_adapter_v100_reference': V100_REFERENCE,
                'delta_from_second_adapter': {k: round(actual[k] - value, 6) if actual[k] is not None else None
                                              for k, value in V100_REFERENCE.items()},
                'thresholds': {'temporal_recall': .4506, 'state_accuracy': .5145},
                'note': 'Quality reference is V100 run 003; different hardware/load cannot establish speedup.'})
            status = 'complete'
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            write(root / 'pipeline-status.json', {'status': status, 'error': error,
                'elapsed_seconds': time.perf_counter() - start, 'initial_adapter_preserved': True})
        print('DONE:', root / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
