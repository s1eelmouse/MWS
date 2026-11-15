"""Reproduce the date-only A/B on frozen predictions; no GPU or network needed."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from case_io import read_jsonl, write_jsonl
from schema import as_call
from scoring import make_scorer
import temporal_case_v2 as candidate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def days(fact):
    return tuple(str(fact.get(field))[:10] for field in ('valid_from', 'valid_to'))


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', required=True, type=Path)
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--pred', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--base-revision', default='b2cfc1b')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new experiment directory')
    base_source = subprocess.check_output(
        ['git', 'show', args.base_revision + ':src/temporal_case_v2.py'], cwd=ROOT)
    args.output_dir.mkdir(parents=True)
    base_path = args.output_dir / 'base_temporal_case_v2.py'
    base_path.write_bytes(base_source)
    spec = importlib.util.spec_from_file_location('ab_base_case_v2', base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    gold_rows, predictions = read_jsonl(args.gold), read_jsonl(args.pred)
    gold = {r['call_id']: r for r in gold_rows}
    assert len(gold) == len(gold_rows), 'Duplicate gold IDs'
    inputs = {r['call_id']: r for r in predictions}
    assert len(inputs) == len(predictions) and set(inputs) == set(gold), 'Call ID mismatch'
    catalog = json.loads(args.catalog.read_text(encoding='utf-8'))
    outputs = []
    for name, module in [('before', base), ('after', candidate)]:
        layers, transformed = module.ValueLayers(catalog), {}
        for cid, row in gold.items():
            call = as_call(row)  # Drops gold before either implementation runs.
            facts, _audit = module.repair_case_v2_dates(inputs[cid]['facts'], call, layers)
            transformed[cid] = {**inputs[cid], 'facts': facts}
            if name == 'after':
                repeated, _ = module.repair_case_v2_dates(facts, call, layers)
                assert repeated == facts, 'Not idempotent: ' + cid
        write_jsonl(args.output_dir / (name + '.jsonl'), list(transformed.values()))
        outputs.append(transformed)
    before, after = outputs
    scorer = make_scorer('2.1', args.catalog)
    counts, attributes, changed_pairs = Counter(), {}, []
    for cid, row in gold.items():
        strip = lambda facts: [{k: v for k, v in f.items() if k not in ('valid_from', 'valid_to')}
                               for f in facts]
        assert strip(before[cid]['facts']) == strip(after[cid]['facts']), 'Non-date change'
        left = scorer.match_facts(row['facts'], before[cid]['facts'])
        right = scorer.match_facts(row['facts'], after[cid]['facts'])
        assert len(left) == len(right)
        for (gf, pf, _), (gg, pp, _) in zip(left, right):
            assert gf == gg
            if not scorer.temporal_eligible(gf, row):
                continue
            old_ok, new_ok = days(gf) == days(pf), days(gf) == days(pp)
            count = attributes.setdefault(gf['attribute'], Counter())
            count['eligible'] += 1
            count['before_correct'] += old_ok
            count['after_correct'] += new_ok
            if old_ok != new_ok:
                outcome = 'fixed' if new_ok else 'regressed'
                count[outcome] += 1
                counts[outcome] += 1
                changed_pairs.append({'call_id': cid, 'attribute': gf['attribute'],
                                      'outcome': outcome, 'gold': gf, 'before': pf, 'after': pp})
    result = {'base_revision': args.base_revision,
              'candidate_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'gold_sha256': digest(args.gold), 'catalog_sha256': digest(args.catalog),
              'predictions_sha256': digest(args.pred), 'base_layer_sha256': digest(base_path),
              'source_sha256': {p.name: digest(p) for p in sorted((ROOT / 'src').glob('*.py'))},
              'scorer': scorer.metadata(), 'before': scorer.evaluate(gold, before),
              'after': scorer.evaluate(gold, after), 'paired': dict(counts), 'attributes': attributes,
              'non_date_fields_unchanged': True, 'idempotent': True,
              'gold_passed_to_rules': False, 'fresh_model_inference': False,
              'qualification': 'Frozen predictions on old inputs; reused development val, not a blind test.'}
    (args.output_dir / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (args.output_dir / 'changed-pairs.json').write_text(json.dumps(changed_pairs, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('before', 'after', 'paired')}, indent=2))


if __name__ == '__main__':
    run()
