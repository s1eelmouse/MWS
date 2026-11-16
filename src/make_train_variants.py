"""Training-set variants from the supplied train split: ASR noise and attribute resampling.

Writes four files in the train.jsonl format, so each is usable as `train.py --train`:

  train_noise.jsonl            every call clean + one noisy copy (level drawn per call)
  train_resampled.jsonl        every call clean + exact extra copies of selected calls
  train_noise_resampled.jsonl  both: clean + noisy copy, and the extra copies are noisy too
  train_focus.jsonl            5 000 calls for one short epoch: problem calls for current states
                               (service state, tariff, ticket status) as noisy copies that keep
                               the textual basis of every gold fact, a replay share of ordinary
                               calls, clean copies of the calls with most problems

Only turn texts change in noisy copies; facts, dates and source turns stay as in gold
(source quote/span become null, as in make_noisy_val). Copies keep the account prefix of
call_id, so the train/val account check still separates them. val is never read except to
verify that no account is shared. Nothing is truncated or dropped.
"""
import argparse
import copy
import difflib
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from case_io import validate_gold
from make_noisy_val import corrupt_text, make_noisy

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


FOCUS_TARGET_CALLS = 5000   # ~45-50 min of one epoch on RTX 4090 at the measured 0.52-0.60 s per call
FOCUS_REPLAY_SHARE = 0.35   # ordinary calls kept so the model does not drift to the problem types only
CORRECTION = re.compile(r'\bне\s+[^,.!?;]{1,30}[,—-]\s*(?:а|у\s+меня|у\s+нас)\b')
STOP_WORDS = {'в', 'и', 'на', 'по', 'не', 'с', 'для', 'у', 'к', 'за'}


def focus_reasons(row, tariffs, symptoms=()):
    """Problem types seen in the 005 error reports, detected in train (val is never read)."""
    # state_accuracy and temporal_recall do not score actions: that reason is not a focus here
    reasons = [r for r in resampling_reasons(row, tariffs) if r != 'many_actions']
    statuses = Counter(f['object'].get('id') or f['object'].get('name') for f in row['facts']
                       if f['attribute'] == 'ticket_status' and f.get('polarity', 1) == 1)
    if statuses and max(statuses.values()) >= 2:
        reasons.append('ticket_status_chain')           # several statuses of one ticket: which is current
    has_tariff = any(f['attribute'] == 'tariff' for f in row['facts'])
    if has_tariff and any(CORRECTION.search(t['text'].lower().replace('ё', 'е'))
                          and ('тариф' in t['text'].lower() or any(_words(n) and _words(n) <= _words(t['text'])
                                                                    for n in tariffs))
                          for t in row['turns']):
        reasons.append('tariff_correction')             # «не X, а Y» / «не X, у меня Y»
    known = {' '.join(sorted(_words(s))) for s in symptoms} | {' '.join(sorted(_words(NORMAL_STATE)))}
    if any(f['attribute'] == 'service_state' and f.get('valid_to') is None and f.get('polarity', 1) == 1
           and ' '.join(sorted(_words(f['value']))) not in known for f in row['facts']):
        reasons.append('free_form_current_service_state')  # half of the current service states in gold
    return reasons


def _evidence_kept(original, noisy, value):
    """The words of a quoted fact value are still recognisable in the noisy source turn."""
    words = [w for w in _words(value) if w not in STOP_WORDS]
    spoken = _words(original)
    if not words or not set(words) <= spoken:
        return True   # the gold value is a paraphrase, noise cannot remove its quote
    heard = _words(noisy)
    glued = ''.join(sorted(heard, key=len, reverse=True))
    kept = sum(any(difflib.SequenceMatcher(None, w, h).ratio() >= 0.7 for h in heard) or w in glued for w in words)
    return kept * 2 >= len(words)


def evidence_safe_noisy_copies(rows, seed, tag, retries=4):
    """Noisy copies in which no gold fact loses its textual basis.

    Plain noise sometimes drops or cuts off the very words a gold fact quotes (about 15 % of
    quoted facts in train); the label then teaches the model to state a fact with no support.
    Such source turns are re-noised with another seed and, failing that, left clean.
    """
    copies, stats = noisy_copies(rows, seed, tag), Counter()
    for original, noisy in zip(rows, copies):
        level = int(noisy['call_id'].rsplit(tag, 1)[1]) / 100
        clean_turns = {t['idx']: t['text'] for t in original['turns']}
        by_turn = {}
        for fact in original['facts']:
            by_turn.setdefault(fact['source']['turn_idx'], []).append(fact['value'])
        for turn in noisy['turns']:
            values = by_turn.get(turn['idx'])
            if not values:
                continue
            text, attempt = turn['text'], 0
            while not all(_evidence_kept(clean_turns[turn['idx']], text, v) for v in values):
                attempt += 1
                if attempt > retries:
                    text = clean_turns[turn['idx']]
                    stats['source_turns_left_clean'] += 1
                    break
                rng = random.Random(f'{seed}:{tag}:{original["call_id"]}:{turn["idx"]}:retry{attempt}')
                text = corrupt_text(clean_turns[turn['idx']], level, rng)
            if attempt:
                stats['source_turns_renoised'] += 1
            turn['text'] = text
    return copies, stats


def build_focus(rows, tariffs, symptoms=(), seed=42, target=FOCUS_TARGET_CALLS):
    """Problem calls as evidence-safe noisy copies + a replay share + clean copies up to target."""
    reasons = {row['call_id']: focus_reasons(row, tariffs, symptoms) for row in rows}
    problem = [row for row in rows if reasons[row['call_id']]]
    ordinary = [row for row in rows if not reasons[row['call_id']]]
    rng = random.Random(f'{seed}:focus')
    replay = rng.sample(ordinary, min(len(ordinary), round(len(ordinary) * FOCUS_REPLAY_SHARE)))
    noisy_problem, stats = evidence_safe_noisy_copies(problem, seed, 'f')
    noisy_replay, more = evidence_safe_noisy_copies(replay[::2], seed, 'f')
    stats.update(more)
    selected = noisy_problem + noisy_replay + [_copy(row, 'c') for row in replay[1::2]]
    ranked = sorted(problem, key=lambda row: (-len(reasons[row['call_id']]), row['call_id']))
    selected += [_copy(row, 'c') for row in ranked[:max(0, target - len(selected))]]
    selected = selected[:target] if len(selected) > target else selected
    rng.shuffle(selected)
    info = {'target_calls': target, 'calls': len(selected), 'problem_calls': len(problem),
            'replay_calls': len(replay), 'reasons': dict(Counter(r for found in reasons.values() for r in found)),
            'noise': dict(stats)}
    return selected, info


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
    catalog = json.loads(Path(args.catalog).read_text(encoding='utf-8'))
    tariffs = catalog.get('tariffs', [])
    variants, reasons = build_variants(rows, tariffs, args.seed)
    variants['train_focus.jsonl'], focus_info = build_focus(rows, tariffs, catalog.get('symptoms', []), args.seed)
    val_accounts = {r['call_id'].split(':')[0] for r in read_jsonl(args.val)} if args.val else set()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'source': args.train, 'source_sha256': hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
                'seed': args.seed, 'noise_levels': NOISE_LEVELS, 'max_extra_copies': MAX_EXTRA_COPIES,
                'source_stats': describe(rows),
                'resampling_reasons': dict(Counter(r for found in reasons.values() for r in found)),
                'calls_resampled': sum(bool(found) for found in reasons.values()),
                'focus': focus_info, 'files': {}}
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
