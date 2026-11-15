"""Train-only self-training pilot: prepare, generate locally, assemble audited targets."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from case_io import read_jsonl, validate_gold, write_jsonl
from run_context import file_info
from schema import as_call, contract_signature, parse_answer, source_problems, to_answer
from training_data import ensure_training_splits


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def account(row):
    return row['call_id'].split(':')[0]


def input_only(row):
    # Do not forward arbitrary nested series metadata or labels to inference.
    return as_call({key: row[key] for key in ('call_id', 'started_at', 'turns')}).model_dump(
        mode='json', exclude_none=True)


def signature(row):
    turns = [(t['speaker'], ' '.join(t['text'].casefold().split())) for t in row['turns']]
    return hashlib.sha256(json.dumps(turns, ensure_ascii=False).encode()).hexdigest()


def disjoint(left, right):
    if {account(r) for r in left} & {account(r) for r in right}:
        raise ValueError('Account overlap between partitions')
    if {signature(r) for r in left} & {signature(r) for r in right}:
        raise ValueError('Transcript overlap between partitions (including renamed IDs)')


def rank(value, seed):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def split_train(train, val, fraction, seed):
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('pool fraction must be between 0 and 1')
    for row in train + val:
        validate_gold(row)
    ensure_training_splits(train, val)
    disjoint(train, val)
    accounts = sorted({account(r) for r in train}, key=lambda a: rank(a, seed))
    if len(accounts) < 2:
        raise ValueError('At least two train accounts are required')
    count = min(len(accounts) - 1, max(1, round(len(accounts) * fraction)))
    pool_accounts = set(accounts[:count])
    gold = [r for r in train if account(r) not in pool_accounts]
    pool = [r for r in train if account(r) in pool_accounts]
    disjoint(gold, pool)
    return gold, pool


def prepare(args):
    train_info, val_info = file_info(args.train), file_info(args.val)
    if (train_info['recognized_version'] != 'organizer-2.1-train'
            or val_info['recognized_version'] != 'organizer-2.1-val'):
        raise ValueError('Pilot requires the supplied organizer v2.1 train and val bytes')
    gold, pool = split_train(read_jsonl(args.train), read_jsonl(args.val), args.pool_fraction, args.seed)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_jsonl(root / 'gold_train.jsonl', gold)
    write_jsonl(root / 'pool_inputs.jsonl', [input_only(r) for r in pool])
    write_jsonl(root / 'audit_gold.jsonl', pool)
    manifest = {
        'kind': 'train-only-pilot', 'seed': args.seed, 'pool_fraction': args.pool_fraction,
        'source_train': train_info, 'source_val': val_info, 'contract': contract_signature(),
        'counts': {'gold_calls': len(gold), 'pool_calls': len(pool),
                   'gold_accounts': len({account(r) for r in gold}),
                   'pool_accounts': len({account(r) for r in pool})},
        'files': {name: file_info(root / name) for name in
                  ('gold_train.jsonl', 'pool_inputs.jsonl', 'audit_gold.jsonl')},
        'warning': 'A model previously trained on all train makes this an engineering pilot only.',
    }
    dump(root / 'manifest.json', manifest)
    print(json.dumps(manifest['counts']))


def checked_pilot(directory):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['kind'] != 'train-only-pilot' or manifest['contract'] != contract_signature():
        raise ValueError('Pilot kind or contract mismatch')
    # audit_gold deliberately never opened by generation or assembly.
    for name in ('gold_train.jsonl', 'pool_inputs.jsonl'):
        if file_info(root / name)['sha256'] != manifest['files'][name]['sha256']:
            raise ValueError(f'Pilot file changed: {name}')
    return root, manifest


def variant(row):
    changed = input_only(row)
    for turn in changed['turns']:
        # Whitespace-only sensitivity probe, not a simulation of ASR robustness.
        turn['text'] = '  ' + re.sub(r'\s+', '  ', turn['text']) + '  '
    return changed


def generate(args):
    root, pilot = checked_pilot(args.pilot)
    rows = read_jsonl(root / 'pool_inputs.jsonl')
    if any(set(r) - {'call_id', 'started_at', 'turns'} for r in rows):
        raise ValueError('Unexpected metadata in model inputs')
    if args.batch_size < 1:
        raise ValueError('batch size must be positive')
    model = Path(args.model)
    if (not model.is_dir() or not list(model.glob('*.safetensors'))
            or not (model / 'config.json').is_file() or (model / 'adapter_config.json').exists()):
        raise ValueError('Use a local merged model with safetensors weights')
    if args.teacher_scope == 'pilot-gold-only':
        experiment = json.loads((model / 'experiment.json').read_text(encoding='utf-8'))
        if (experiment.get('train_sha256') != pilot['files']['gold_train.jsonl']['sha256']
                or experiment.get('initial_adapter') or experiment.get('train_limit') is not None):
            raise ValueError('pilot-gold-only requires training from base on the complete gold partition')
    model_files = {str(p.relative_to(model)): file_info(p) for p in sorted(model.rglob('*'))
                   if p.is_file() and p.suffix in ('.json', '.safetensors', '.model')}
    import config
    if config.TEMPORAL_SERIES_HISTORY:
        raise ValueError('Pilot requires TEMPORAL_SERIES_HISTORY=0 for independent views')
    # Local Extractor only; save raw outputs to reject partial/salvaged answers.
    import extractor as backend
    backend.SAVE_RAW_RESPONSES = True
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'started.json', {'model_files': model_files, 'teacher_scope': args.teacher_scope,
                               'pilot_sha256': file_info(root / 'manifest.json')['sha256']})
    engine = backend.Extractor(str(model), device=args.device)
    with (out / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for begin in range(0, len(rows), args.batch_size):
            batch = rows[begin:begin + args.batch_size]
            views = []
            for inputs in (batch, [variant(r) for r in batch]):
                details = []
                results = engine.extract([as_call(r) for r in inputs], args.batch_size, measurements=details)
                if len(results) != len(batch) or len(details) != len(batch):
                    raise RuntimeError('Incomplete backend batch')
                views.append([{'result': r, 'measurement': d} for r, d in zip(results, details)])
            for index, row in enumerate(batch):
                stream.write(json.dumps({'call_id': row['call_id'],
                                         'views': [view[index] for view in views]}, ensure_ascii=False) + '\n')
            stream.flush()
            print(f'Generated {min(begin + args.batch_size, len(rows))}/{len(rows)}')
    dump(out / 'complete.json', {
        'pilot_sha256': file_info(root / 'manifest.json')['sha256'],
        'predictions_sha256': file_info(out / 'predictions.jsonl')['sha256'],
        'teacher_scope': args.teacher_scope, 'model_files': model_files,
        'calls': len(rows), 'contract': pilot['contract'],
        'settings': {k: getattr(config, k) for k in (
            'MAX_INPUT_TOKENS', 'MAX_NEW_TOKENS', 'JSON_CONSTRAINED', 'JSON_GRAMMAR_BACKEND',
            'TEMPORAL_NORMALIZATION', 'FACT_RECOVERY', 'RETRY_ON_TOKEN_LIMIT')},
        'batch_size': args.batch_size, 'device': engine.device, 'dtype': str(engine.model.dtype),
        'target_policy': 'strict raw answers; no postprocessing or source salvage',
        'code': {name: file_info(Path(__file__).with_name(name)) for name in
                 ('self_training.py', 'extractor.py', 'schema.py', 'prediction.py')},
    })


def canonical(facts):
    return sorted(json.dumps(f, sort_keys=True, ensure_ascii=False)
                  for f in json.loads(to_answer(facts.model_dump()))['facts'])


def words(text):
    return re.findall(r'\w+', text.casefold().replace('ё', 'е'))


def select_target(row, record):
    parsed = []
    if len(record.get('views', [])) != 2:
        return None, 'missing_views'
    for inp, view in zip((row, variant(row)), record['views']):
        result, detail = view['result'], view['measurement']
        if 'error' in result:
            return None, 'generation_error:' + str(result['error'])
        if detail.get('generation_limit_reached') or detail.get('retry'):
            return None, 'token_limit_or_retry'
        facts = parse_answer(detail.get('raw_response') or '')
        if facts is None or source_problems(facts, as_call(inp)):
            return None, 'invalid_raw_answer_or_source'
        if not facts.facts:
            return None, 'empty_answer'
        keys = canonical(facts)
        if len(set(keys)) != len(keys):
            return None, 'duplicate_facts'
        turns = {t['idx']: t['text'] for t in inp['turns']}
        current = {}
        for fact in facts.facts:
            if fact.confirmed is False:
                return None, 'unconfirmed_fact'
            # Deliberately conservative literal support, not a truth/confidence estimate.
            value = ' '.join(words(fact.value))
            source = ' '.join(words(turns[fact.source.turn_idx]))
            if not value or f' {value} ' not in f' {source} ':
                return None, 'nonliteral_value'
            if fact.attribute in ('tariff', 'service_state', 'ticket_status') and fact.valid_to is None:
                key = (fact.object.type, fact.object.name.casefold(), fact.attribute)
                state = (fact.value, fact.polarity)
                if key in current and current[key] != state:
                    return None, 'conflicting_open_states'
                current[key] = state
        parsed.append(facts)
    if canonical(parsed[0]) != canonical(parsed[1]):
        return None, 'view_disagreement'
    return json.loads(to_answer(parsed[0].model_dump()))['facts'], 'accepted'


def assemble(args):
    if not math.isfinite(args.max_pseudo_fraction) or not 0 < args.max_pseudo_fraction <= 0.5:
        raise ValueError('max pseudo fraction must be in (0, 0.5]')
    root, pilot = checked_pilot(args.pilot)
    generated = Path(args.generation)
    complete = json.loads((generated / 'complete.json').read_text(encoding='utf-8'))
    if (complete['pilot_sha256'] != file_info(root / 'manifest.json')['sha256']
            or complete['predictions_sha256'] != file_info(generated / 'predictions.jsonl')['sha256']):
        raise ValueError('Generation provenance mismatch')
    if file_info(args.val)['sha256'] != pilot['source_val']['sha256']:
        raise ValueError('Validation data changed')
    gold, pool, val = (read_jsonl(p) for p in
                       (root / 'gold_train.jsonl', root / 'pool_inputs.jsonl', args.val))
    disjoint(gold, pool)
    disjoint(gold + pool, val)
    records = {r['call_id']: r for r in read_jsonl(generated / 'predictions.jsonl')}
    if set(records) != {r['call_id'] for r in pool}:
        raise ValueError('Missing or extra prediction IDs')
    candidates, decisions = [], []
    for row in pool:
        facts, reason = select_target(row, records[row['call_id']])
        decisions.append({'call_id': row['call_id'], 'reason': reason})
        if facts is not None:
            target = {**row, 'facts': facts}
            validate_gold(target)
            candidates.append(target)
    cap = math.floor(len(gold) * args.max_pseudo_fraction / (1 - args.max_pseudo_fraction))
    selected = {r['call_id'] for r in sorted(candidates, key=lambda r: rank(r['call_id'], pilot['seed']))[:cap]}
    pseudo = [r for r in candidates if r['call_id'] in selected]
    for d in decisions:
        if d['reason'] == 'accepted' and d['call_id'] not in selected:
            d['reason'] = 'mixture_cap'
    mixed = gold + pseudo
    report = {
        'teacher_scope': complete['teacher_scope'], 'gold_calls': len(gold),
        'pseudo_calls': len(pseudo), 'pseudo_fraction': len(pseudo) / len(mixed),
        'reasons': dict(Counter(d['reason'] for d in decisions)),
        'accepted_attributes': dict(Counter(f['attribute'] for r in pseudo for f in r['facts'])),
        'pilot_sha256': complete['pilot_sha256'],
        'generation_manifest': file_info(generated / 'complete.json'),
        'max_pseudo_fraction': args.max_pseudo_fraction,
        'status': 'ready' if pseudo else 'no_accepted_pseudo_labels',
        'limitations': 'Literal support and whitespace agreement do not prove truth or completeness.',
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / 'decisions.jsonl', decisions)
    write_jsonl(out / 'pseudo_train.jsonl', pseudo)
    if pseudo:
        write_jsonl(out / 'mixed_train.jsonl', mixed)
        report['mixed_train'] = file_info(out / 'mixed_train.jsonl')
    dump(out / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not pseudo:
        raise ValueError('No accepted pseudo-labels; diagnostics saved, no mixed train created')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--train', required=True)
    p.add_argument('--val', required=True)
    p.add_argument('--pool-fraction', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', required=True)
    p.set_defaults(run=prepare)
    p = commands.add_parser('generate')
    p.add_argument('--pilot', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--teacher-scope', required=True, choices=('full-train-seen', 'pilot-gold-only'))
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--output-dir', required=True)
    p.set_defaults(run=generate)
    p = commands.add_parser('assemble')
    p.add_argument('--pilot', required=True)
    p.add_argument('--generation', required=True)
    p.add_argument('--val', required=True)
    p.add_argument('--max-pseudo-fraction', type=float, default=0.2)
    p.add_argument('--output-dir', required=True)
    p.set_defaults(run=assemble)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
