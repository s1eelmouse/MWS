"""Explicit validation launcher for the shared V100; never starts training."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', default=str(ROOT / 'out/v100-assets.json'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--mode', choices=('smoke', 'long', 'full'), default='smoke')
    parser.add_argument('--history', choices=('0', '1'), default='1')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--budget-gib', type=float, default=12)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.run_id):
        parser.error('Use a simple unique run ID')
    if args.batch_size < 1 or not math.isfinite(args.budget_gib) or args.budget_gib <= 0:
        parser.error('Batch and memory budget must be positive')
    os.chdir(ROOT)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, MAX_INPUT_TOKENS='12288', MAX_SEQUENCE_TOKENS='16384',
                      MAX_NEW_TOKENS='4096', MAX_TEXT_CHARS='0', PROMPT_LOOKUP_TOKENS='0',
                      JSON_CONSTRAINED='1', JSON_GRAMMAR_BACKEND='torch_native', JSON_GRAMMAR_THREADS='4',
                      TEMPORAL_NORMALIZATION='task', TEMPORAL_SERIES_HISTORY=args.history,
                      TEMPORAL_SERIES_MAX_ENTRIES='10000', HISTORY_MAX_ACCOUNTS='512',
                      HISTORY_MAX_BYTES='8388608', HISTORY_TTL_SECONDS='3600',
                      SAVE_RAW_RESPONSES='1', RETRY_ON_TOKEN_LIMIT='1', RETRY_REPETITION_PENALTY='1.2',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MWS_CODE_REVISION='18fa824-plus-v100-migration-001')
    assets = json.loads(Path(args.assets).read_text())
    os.environ['CATALOG_PATH'] = assets['catalog']
    if Path(assets['restored_hf_cache']).is_dir():
        os.environ['HF_HOME'] = assets['restored_hf_cache']
    sys.path.insert(0, str(ROOT / 'src'))
    ROOT.joinpath('out').mkdir(exist_ok=True)
    output = ROOT / 'out' / args.run_id
    policy_file = ROOT / 'out' / (args.run_id + '-gpu-policy.json')
    if output.exists() or policy_file.exists():
        raise FileExistsError('Results exist. Choose a new run ID.')
    with (ROOT / 'out/v100-validation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another validation from this checkout is running')
        import torch
        if not torch.cuda.is_available() or not (torch.version.cuda or '').startswith('12.'):
            raise SystemExit('Need the verified CUDA 12.x environment')
        free, total = torch.cuda.mem_get_info()
        budget = int(args.budget_gib * 2**30)
        if budget > total or free < budget + 2 * 2**30:
            raise SystemExit('Insufficient free VRAM for requested budget plus 2 GiB headroom. Coordinate/wait.')
        torch.cuda.set_per_process_memory_fraction(budget / total, 0)
        torch.set_num_threads(4)
        policy = {'settings': vars(args), 'free_vram_before_bytes': free, 'total_vram_bytes': total,
                  'pytorch_allocator_budget_bytes': budget, 'torch_threads': torch.get_num_threads(),
                  'note': 'Allocator limit, not an exclusive GPU reservation; other teams can allocate concurrently.'}
        policy_file.write_text(json.dumps(policy, indent=2) + '\n')
        source = assets['val']
        extra = ['--limit', '5'] if args.mode == 'smoke' else []
        if args.mode == 'long':
            from case_io import read_jsonl, write_jsonl
            from schema import as_call
            from training_data import prompt_for
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(assets['model'], local_files_only=True)
            rows = read_jsonl(source)
            longest = sorted(rows, key=lambda row: len(tokenizer(
                prompt_for(tokenizer, as_call(row), False), add_special_tokens=False)['input_ids']), reverse=True)[:4]
            subset = ROOT / 'out' / (args.run_id + '-long-inputs.jsonl')
            write_jsonl(subset, longest)
            source = str(subset)
        sys.argv = ['benchmark.py', '--input', source, '--model', assets['model'], '--device', 'cuda',
                    '--batch-size', str(args.batch_size), '--warmup-batches', '1', '--output-dir', str(output), *extra]
        runpy.run_path(str(ROOT / 'src/benchmark.py'), run_name='__main__')
        if args.mode == 'full':
            sys.argv = ['compare_temporal.py', '--gold', assets['val'],
                        '--before', str(output / 'predictions-before-series.jsonl'),
                        '--after', str(output / 'predictions.jsonl'),
                        '--output-dir', str(output / 'history-date-comparison')]
            runpy.run_path(str(ROOT / 'src/compare_temporal.py'), run_name='__main__')
        print('DONE:', output, flush=True)


if __name__ == '__main__':
    main()
