"""Four comparable full-val GPU measurements. Evaluation only, never training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

RELEASE = 'case-v21-unified-001'
WEIGHTS = {'own': '193691230bba18d7c772d70e94651c425ca494081a42570a8b948cbc6175c044',
           '005': 'cef9fcde9f1f2a812e42c6122a0f5094fbaa456f6077596cf03ebb86e4d552f6'}
DATASETS = {'clean': 'e12610354aa69dff83e1d7c1e8c94736c97d5761d20af58790435f4be1941808',
            'noisy': 'c76823bc32fdcfc6e62410c2d0f8222043220130f3a0c31345cd7025b12e85da'}
STAGES = ('own-noisy', '005-noisy', '005-clean', 'own-clean')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_release(code):
    manifest = json.loads((code / 'RELEASE_MANIFEST.json').read_text())
    if manifest['release_id'] != RELEASE:
        raise ValueError('Wrong release')
    for name, expected in manifest['files'].items():
        if digest(code / name) != expected:
            raise ValueError('Release checksum mismatch: ' + name)
    return manifest


def configure(code):
    data = code / 'data/organizer-v2.1'
    os.environ.update(CUDA_VISIBLE_DEVICES='0', CASE_DATA_DIR=str(data), ORGANIZER_DIR=str(data),
        SCORER_VERSION='2.1', ALLOW_LEGACY_CONTRACT='0',
        CATALOG_PATH=str(data / 'catalog.json'), TEMPORAL_CATALOG_PATH=str(data / 'catalog.json'),
        MAX_INPUT_TOKENS='12288', MAX_SEQUENCE_TOKENS='16384', MAX_NEW_TOKENS='4096', MAX_TEXT_CHARS='0',
        JSON_CONSTRAINED='1', JSON_GRAMMAR_BACKEND='torch_native', JSON_GRAMMAR_THREADS='4',
        TEMPORAL_NORMALIZATION='case_v2', TEMPORAL_SERIES_HISTORY='1', FACT_RECOVERY='1',
        TARIFF_BOUNDARY_POLICY='explicit', TEMPORAL_SERIES_MAX_ENTRIES='10000',
        HISTORY_MAX_ACCOUNTS='512', HISTORY_MAX_BYTES='8388608', HISTORY_TTL_SECONDS='3600',
        SAVE_RAW_RESPONSES='1', PROMPT_LOOKUP_TOKENS='0', RETRY_ON_TOKEN_LIMIT='1',
        RETRY_REPETITION_PENALTY='1.2', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', MWS_CODE_REVISION=RELEASE)


def compact_results(root):
    results = {}
    for name in STAGES:
        path = root / name / 'report.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        results[name] = {'status': report['status'], 'quality': report.get('quality'),
            **{k: report.get(k) for k in ('measurement_seconds', 'failed_calls', 'calls_per_hour',
                'gpu_seconds_per_1000_calls', 'call_completion_seconds', 'memory', 'history',
                'model_weights_bytes', 'dtype', 'fact_recovery_policy', 'dataset_sha256')}}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home() / 'mws-v21')
    parser.add_argument('--run-id', default='4090-unified-005-001')
    parser.add_argument('--batch-size', type=int, choices=range(1, 9), default=2)
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--own-model', type=Path)
    parser.add_argument('--model-005', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.run_id):
        parser.error('Use a simple unique run ID')
    if len(set(args.stages)) != len(args.stages):
        parser.error('Repeated stages are not allowed')
    team = args.root.resolve()
    code = team / 'releases' / RELEASE
    output = team / 'out' / args.run_id
    manifest = verify_release(code)
    models = {'own': (args.own_model or team / 'out/4090-continue-v21-001/merged').resolve(),
              '005': (args.model_005 or team / 'models/005-noise-20261115').resolve()}
    assets = json.loads((code / 'evaluation/expected_models.json').read_text())
    inputs = {'clean': code / 'data/organizer-v2.1/val.jsonl',
              'noisy': code / 'data/noisy-015/val-noisy-015.jsonl'}
    for kind in {s.rsplit('-', 1)[0] for s in args.stages}:
        if digest(models[kind] / 'model.safetensors') != WEIGHTS[kind]:
            raise ValueError('Wrong weights: ' + kind)
        for name, expected in assets[kind].items():
            if digest(models[kind] / name) != expected:
                raise ValueError('Model asset mismatch: ' + kind + '/' + name)
    for kind, path in inputs.items():
        if digest(path) != DATASETS[kind] or len(path.read_text(encoding='utf-8').splitlines()) != 383:
            raise ValueError('Wrong dataset: ' + kind)
    commands = {name: [sys.executable, '-u', str(code / 'src/benchmark.py'),
        '--model', str(models[name.rsplit('-', 1)[0]]),
        '--input', str(inputs[name.rsplit('-', 1)[1]]), '--device', 'cuda',
        '--batch-size', str(args.batch_size), '--warmup-batches', '1',
        '--output-dir', str(output / name)] for name in args.stages}
    if args.check_only:
        print(json.dumps({'status': 'verified', 'training': False, 'commands': commands}, indent=2))
        return
    configure(code)
    sys.path.insert(0, str(code / 'src'))
    from run_experiment import stage
    import fcntl
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / 'v21-validation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(exist_ok=False)
        public_env = {k: os.environ[k] for k in (
            'SCORER_VERSION', 'ALLOW_LEGACY_CONTRACT', 'JSON_CONSTRAINED', 'JSON_GRAMMAR_BACKEND',
            'JSON_GRAMMAR_THREADS', 'TEMPORAL_NORMALIZATION', 'TEMPORAL_SERIES_HISTORY',
            'FACT_RECOVERY', 'TARIFF_BOUNDARY_POLICY', 'SAVE_RAW_RESPONSES', 'OMP_NUM_THREADS',
            'MKL_NUM_THREADS', 'RETRY_ON_TOKEN_LIMIT', 'RETRY_REPETITION_PENALTY',
            'MAX_INPUT_TOKENS', 'MAX_SEQUENCE_TOKENS', 'MAX_NEW_TOKENS')}
        (output / 'settings.json').write_text(json.dumps({
            'created_utc': datetime.now(timezone.utc).isoformat(), 'training': False,
            'release': RELEASE, 'stages': args.stages, 'batch_size': args.batch_size,
            'models': {k: str(v) for k, v in models.items()}, 'weight_sha256': WEIGHTS,
            'dataset_sha256': DATASETS, 'environment': public_env,
            'runner_sha256': digest(Path(__file__))}, indent=2) + '\n')
        (output / 'release-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        status = 'failed'
        try:
            stage(output, 'tests', [sys.executable, '-m', 'unittest', 'discover', '-s', str(code / 'tests'), '-v'])
            for name, command in commands.items():
                stage(output, name, command)
                (output / 'comparison.json').write_text(json.dumps(compact_results(output), indent=2) + '\n')
            status = 'complete'
        finally:
            (output / 'pipeline-status.json').write_text(json.dumps({'status': status}) + '\n')
        print('DONE:', output / 'comparison.json', flush=True)


if __name__ == '__main__':
    main()
