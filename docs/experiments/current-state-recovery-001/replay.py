"""Compare recovery v1/v2 on identical raw model answers, without GPU or training."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import fact_recovery
from prediction import parse_prediction
from schema import as_call, source_problems
from scoring import make_scorer


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]
    indexed = {r['call_id']: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate call IDs: ' + str(path))
    return indexed


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def states(scorer, gold, predictions):
    """Diagnostic outcomes using exactly the official current-state selector."""
    m = scorer.module
    result = {}
    for cid, row in gold.items():
        keys = {}
        for f in row['facts']:
            if f['attribute'] in m.SINGLE_VALUED and f.get('valid_to') is None:
                key = (f['object']['id'], f['attribute'])
                if key not in keys or f['valid_from'] > keys[key]['valid_from']:
                    keys[key] = f
        for key, gf in keys.items():
            candidates = [f for f in predictions[cid]['facts'] if f['attribute'] == gf['attribute']
                          and f.get('valid_to') is None and m._object_sim(gf, f) > 0]
            pf = max(candidates, key=lambda f: (f.get('valid_from') or '', m._value_sim(gf['value'], f['value']))) if candidates else None
            need = m.VALUE_SIM if pf and min(len(m._tokens(gf['value'])), len(m._tokens(pf['value']))) <= 2 else m.VALUE_SIM_LONG
            result[(cid, *key)] = bool(pf and m._value_sim(gf['value'], pf['value']) >= need)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', required=True, type=Path)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--reference-pred', type=Path, help='optional frozen server predictions; baseline must reproduce them')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output exists; choose a new directory')
    gold, raw = load(args.gold), load(args.raw)
    if set(raw) != set(gold):
        parser.error('Raw answers and gold must have the same call IDs')
    args.output_dir.mkdir(parents=True)
    base_path = Path(__file__).with_name('base_fact_recovery.py')
    spec = importlib.util.spec_from_file_location('frozen_recovery_v1', base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    scorer = make_scorer('2.1', args.catalog)
    current = fact_recovery.recover_facts
    reference = load(args.reference_pred) if args.reference_pred else None
    all_predictions, report = {}, {}
    try:
        for name, recovery in [('before', base.recover_facts), ('after', current)]:
            fact_recovery.recover_facts = recovery
            predictions, events = {}, []
            start = time.perf_counter()
            for cid, row in gold.items():
                call, audit = as_call(row), []  # Gold is stripped before rules run.
                facts, error, _summary = parse_prediction(raw[cid]['raw_response'], call,
                    temporal_mode='case_v2', temporal_decisions=audit,
                    catalog_path=str(args.catalog), date_catalog_path=str(args.catalog), fact_recovery=True)
                if error or source_problems(facts, call):
                    raise ValueError('Replay has invalid prediction/source: ' + cid + ': ' + str(error))
                payload = facts.model_dump(mode='json')['facts']
                predictions[cid] = {'call_id': cid, 'facts': payload}
                if name == 'before' and reference is not None and payload != reference[cid]['facts']:
                    raise ValueError('Baseline does not reproduce reference: ' + cid)
                if name == 'after':
                    repeated, _ = current(payload, call, args.catalog)
                    if repeated != payload:
                        raise ValueError('Recovery is not idempotent: ' + cid)
                events.extend({'call_id': cid, **e} for e in audit if e.get('component') == 'fact_recovery')
            elapsed = time.perf_counter() - start
            report[name] = {'metrics': scorer.evaluate(gold, predictions), 'processing_seconds': elapsed,
                            'recovery_audit_events': len(events)}
            all_predictions[name] = predictions
            (args.output_dir / (name + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
                for r in predictions.values()), encoding='utf-8')
            (args.output_dir / (name + '-audit.json')).write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding='utf-8')
            print(name, report[name], flush=True)
    finally:
        fact_recovery.recover_facts = current
    before = states(scorer, gold, all_predictions['before'])
    after = states(scorer, gold, all_predictions['after'])
    paired, changed = Counter(), []
    for key, old in before.items():
        if old != after[key]:
            outcome = 'fixed' if after[key] else 'regressed'
            paired[key[2] + ':' + outcome] += 1
            changed.append({'call_id': key[0], 'object_id': key[1], 'attribute': key[2], 'outcome': outcome})
    report.update(state_changes=dict(paired), calls=len(gold), scorer=scorer.metadata(),
                  gold_sha256=digest(args.gold), raw_sha256=digest(args.raw), catalog_sha256=digest(args.catalog),
                  reference_reproduced=reference is not None, reference_sha256=digest(args.reference_pred) if reference else None,
                  base_recovery_sha256=digest(base_path),
                  source_sha256={p.name: digest(p) for p in sorted((ROOT / 'src').glob('*.py'))},
                  gold_passed_to_rules=False, new_generation=False, training=False, idempotent=True,
                  all_sources_valid=True, qualification='Reused development val; not a blind hidden-test result.')
    (args.output_dir / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (args.output_dir / 'state-changes.json').write_text(json.dumps(changed, ensure_ascii=False, indent=2) + '\n')
    print('STATE CHANGES', dict(paired))


if __name__ == '__main__':
    main()
