"""Training-set variants from the supplied train split: ASR noise and attribute resampling.

Writes three files in the train.jsonl format, so each is usable as `train.py --train`:

  train_noise.jsonl            every call clean + one noisy copy (level drawn per call)
  train_resampled.jsonl        every call clean + exact extra copies of selected calls
  train_noise_resampled.jsonl  both: clean + noisy copy, and the extra copies are noisy too

Only turn texts change in noisy copies; facts, dates and source turns stay as in gold
(source quote/span become null, as in make_noisy_val). Copies keep the account prefix of
call_id, so the train/val account check still separates them. val is never read except to
verify that no account is shared. Nothing is truncated or dropped.
"""
import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from case_io import validate_gold
from make_noisy_val import make_noisy

NOISE_LEVELS = (0.10, 0.20, 0.30)
MAX_EXTRA_COPIES = 2
NORMAL_STATE = 'работает нормально'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _words(text):
    return set(re.findall(r'[а-яa-z0-9+]+', text.lower().replace('ё', 'е')))


def resampling_reasons(row, tariffs):
    """Why a call is shown more often. Each reason targets a measured weakness of the model."""
    count = Counter(fact['attribute'] for fact in row['facts'])
    states = [f for f in row['facts'] if f['attribute'] == 'service_state']
    reasons = []
    if count['service_state'] >= 3:
        reasons.append('many_service_states')       # recall 0.33 on the V100 run
    if any(f['value'].lower() == NORMAL_STATE and f.get('valid_to') for f in states) \
            and any(f.get('valid_to') is None for f in states):
        reasons.append('state_change')               # closed normal state + current symptom
    if count['action'] >= 2:
        reasons.append('many_actions')               # recall 0.28
    if count['tariff'] >= 2:
        reasons.append('tariff_change')              # current vs previous tariff
    labelled = ' '.join(f['value'].lower().replace('ё', 'е') for f in row['facts'] if f['attribute'] == 'tariff')
    tariff_turns = [_words(turn['text']) for turn in row['turns']]
    tariff_turns = [spoken for spoken in tariff_turns if any(word.startswith('тариф') for word in spoken)]
    for tariff in tariffs:
        name = _words(tariff)
        # the name next to the word «тариф» in one turn, yet no fact carries it: offered or discussed
        if name and tariff.lower().replace('ё', 'е') not in labelled \
                and any(name <= spoken for spoken in tariff_turns):
            reasons.append('tariff_named_but_not_a_fact')
            break
    return reasons


def _copy(row, suffix):
    item = copy.deepcopy(row)
    item['call_id'] = f"{row['call_id']}~{suffix}"
    return item


def noisy_copies(rows, seed, tag):
    """One noisy copy per row; the level is drawn per call from NOISE_LEVELS."""
    by_level = {level: [] for level in NOISE_LEVELS}
    for row in rows:
        level = random.Random(f'{seed}:{tag}:{row["call_id"]}').choice(NOISE_LEVELS)
        by_level[level].append(row)
    result = {}
    for level, group in by_level.items():
        for original, noisy in zip(group, make_noisy(group, level, f'{seed}:{tag}')):
            noisy['call_id'] = f"{original['call_id']}~{tag}{int(level * 100):02d}"
            result[original['call_id']] = noisy
    return [result[row['call_id']] for row in rows]


def build_variants(rows, tariffs, seed=42):
    reasons = {row['call_id']: resampling_reasons(row, tariffs) for row in rows}
    extra = {cid: min(MAX_EXTRA_COPIES, len(found)) for cid, found in reasons.items()}
    noise = noisy_copies(rows, seed, 'n')
    resampled, noisy_extra = [], []
    for index in range(1, MAX_EXTRA_COPIES + 1):
        chosen = [row for row in rows if extra[row['call_id']] >= index]
        resampled += [_copy(row, f'r{index}') for row in chosen]
        noisy_extra += noisy_copies(chosen, seed, f'r{index}n')
    variants = {
        'train_noise.jsonl': rows + noise,
        'train_resampled.jsonl': rows + resampled,
        'train_noise_resampled.jsonl': rows + noise + noisy_extra,
    }
    return variants, reasons


def describe(rows):
    facts = Counter(fact['attribute'] for row in rows for fact in row['facts'])
    total = sum(facts.values())
    return {'calls': len(rows), 'facts': total,
            'attribute_share': {a: round(n / total, 4) for a, n in sorted(facts.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--val', help='only to verify that no account is shared with train')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rows = read_jsonl(args.train)
    tariffs = json.loads(Path(args.catalog).read_text(encoding='utf-8')).get('tariffs', [])
    variants, reasons = build_variants(rows, tariffs, args.seed)
    val_accounts = {r['call_id'].split(':')[0] for r in read_jsonl(args.val)} if args.val else set()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'source': args.train, 'source_sha256': hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
                'seed': args.seed, 'noise_levels': NOISE_LEVELS, 'max_extra_copies': MAX_EXTRA_COPIES,
                'source_stats': describe(rows),
                'resampling_reasons': dict(Counter(r for found in reasons.values() for r in found)),
                'calls_resampled': sum(bool(found) for found in reasons.values()), 'files': {}}
    for name, items in variants.items():
        ids = [item['call_id'] for item in items]
        if len(set(ids)) != len(ids):
            raise ValueError(f'{name}: duplicate call_id')
        if {cid.split(':')[0] for cid in ids} & val_accounts:
            raise ValueError(f'{name}: shares an account with val')
        for item in items:
            validate_gold(item)
        path = output / name
        with path.open('x', encoding='utf-8') as stream:
            for item in items:
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
        manifest['files'][name] = {**describe(items), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    with (output / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
