"""Validate installed 005 on clean/noisy val. Plan by default; never trains."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from release_profile import environment, preflight

ROOT = Path(__file__).resolve().parents[1]


def plan(model, gold, output, mode='clean', device='cuda'):
    model, gold, output = [Path(p).resolve() for p in (model, gold, output)]
    commands = []
    selected = gold
    if mode == 'noisy':
        selected = output / 'input/val.jsonl'
        commands.append([sys.executable, str(ROOT/'src/make_noisy_val.py'), '--input', str(gold),
                         '--output', str(selected), '--level', '0.15', '--seed', '42'])
    commands.append([sys.executable, '-u', str(ROOT/'src/benchmark.py'), '--input', str(selected),
                     '--model', str(model), '--device', device, '--batch-size', '2',
                     '--warmup-batches', '1', '--output-dir', str(output/'benchmark')])
    return {'mode': mode, 'environment': environment(gold), 'commands': commands}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--gold', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--mode', choices=('clean', 'noisy'), default='clean')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    run = plan(args.model, args.gold, args.output_dir, args.mode, args.device)
    if not args.execute:
        print(json.dumps(run, ensure_ascii=False, indent=2))
        return
    if args.output_dir.exists():
        p.error('Output exists; choose a fresh directory')
    verified = preflight(args.model, args.gold)
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            p.error('CUDA unavailable; refusing CPU fallback')
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'preflight.json').write_text(json.dumps(verified, indent=2)+'\n')
    (args.output_dir/'plan.json').write_text(json.dumps(run, indent=2)+'\n')
    with (args.output_dir/'execution.log').open('x') as log:
        for command in run['commands']:
            subprocess.run(command, cwd=ROOT, env={**os.environ, **run['environment']},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Complete:', args.output_dir/'benchmark/report.json')


if __name__ == '__main__':
    main()
