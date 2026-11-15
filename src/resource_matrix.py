"""Plan or execute three compatible FP16 GPU benchmarks (batch 1/2/4).

Default is plan-only. No training, quantization or model download is performed
by this wrapper. Actual inference may download dependencies/model resources
according to the existing backend; point --model at installed local weights.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def plan(model, gold, output, python=sys.executable):
    model, gold, output = Path(model).resolve(), Path(gold).resolve(), Path(output).resolve()
    catalog = gold.with_name('catalog.json')
    env = {'CASE_DATA_DIR': str(gold.parent), 'CATALOG_PATH': str(catalog),
           'TEMPORAL_CATALOG_PATH': str(catalog), 'SCORER_VERSION': '2.1',
           'TEMPORAL_NORMALIZATION': 'case_v2', 'FACT_RECOVERY': '1',
           'TEMPORAL_SERIES_HISTORY': '0', 'JSON_CONSTRAINED': '1',
           'JSON_GRAMMAR_BACKEND': 'torch_native', 'JSON_GRAMMAR_THREADS': '8',
           'SAVE_RAW_RESPONSES': '1', 'MAX_INPUT_TOKENS': '12288',
           'MAX_SEQUENCE_TOKENS': '16384', 'MAX_NEW_TOKENS': '4096',
           'MAX_TEXT_CHARS': '0', 'PROMPT_LOOKUP_TOKENS': '0',
           'RETRY_ON_TOKEN_LIMIT': '1', 'RETRY_REPETITION_PENALTY': '1.2',
           'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
           'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1'}
    commands = [[python, '-u', str(ROOT / 'src/benchmark.py'), '--input', str(gold),
                 '--model', str(model), '--device', 'cuda', '--batch-size', str(batch),
                 '--warmup-batches', '1', '--output-dir', str(output / f'batch-{batch}')]
                for batch in (1, 2, 4)]
    return {'environment': env, 'commands': commands,
            'notes': ['FP16 only: current backend has no 4-bit inference switch.',
                      'No teacher timing is available: 5x acceleration is not proven.',
                      'Full val in all modes, no history, no silent truncation.',
                      'A failed or incompatible run must not enter a resource-quality curve.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--gold', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--execute', action='store_true', help='actually run the three GPU evaluations')
    args = p.parse_args()
    matrix = plan(args.model, args.gold, args.output_dir)
    if not args.execute:
        print(json.dumps(matrix, ensure_ascii=False, indent=2))
        return
    if args.output_dir.exists():
        p.error('Output exists; choose a new directory')
    if not args.model.is_dir() or not args.gold.is_file() or not args.gold.with_name('catalog.json').is_file():
        p.error('Local model, gold and adjacent catalog must exist')
    import torch
    if not torch.cuda.is_available():
        p.error('CUDA GPU is unavailable; no benchmark started')
    env = {**os.environ, **matrix['environment']}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'plan.json').write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + '\n')
    results = []
    for batch, command in zip((1, 2, 4), matrix['commands']):
        print('Starting batch', batch, flush=True)
        with (args.output_dir / f'batch-{batch}.log').open('x') as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        results.append({'batch_size': batch, 'exit_code': completed.returncode})
        (args.output_dir / 'execution.json').write_text(json.dumps(results, indent=2) + '\n')
        if completed.returncode:
            raise SystemExit('Evaluation failed; log and previous results preserved. Remaining runs not started.')
    print('All runs finished; use compare_runs.py to validate compatibility before plotting.')


if __name__ == '__main__':
    main()
