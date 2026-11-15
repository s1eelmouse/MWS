"""Local evidence pack: recovery ablation, rule checks and escaped offline demo.

Never calls a model/API. Gold reaches the evaluator only. Detailed output contains
private transcripts: keep it locally, outside version control.
"""
import argparse
import hashlib
import html
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import unittest

import fact_recovery
from case_evaluation import evaluate_predictions
from prediction import parse_prediction
from schema import as_call, source_problems

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('fact_precision', 'fact_recall', 'temporal_recall', 'state_accuracy', 'grounding_turn')
LABELS = ('Точность фактов', 'Полнота фактов', 'Временная метрика', 'Текущее состояние', 'Реплика-источник')
CHECK_MODULES = ('test_current_state_recovery', 'test_temporal_case_v2',
                 'test_temporal_binding', 'test_state_intervals')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    if len({r['call_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate call IDs')
    if not rows:
        raise ValueError('Empty input')
    return {r['call_id']: r for r in rows}


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({'test': test.id(), 'status': 'passed'})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({'test': test.id(), 'status': 'failed'})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({'test': test.id(), 'status': 'error'})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.records.append({'test': test.id(), 'status': 'skipped', 'reason': reason})

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err:
            self.records.append({'test': str(subtest), 'status': 'failed_subtest'})


def rule_checks():
    sys.path.insert(0, str(ROOT / 'tests'))
    suite = unittest.TestLoader().loadTestsFromNames(CHECK_MODULES)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, resultclass=RecordedResult).run(suite)
    return {'kind': 'synthetic_rule_regression_not_model_adversarial_evaluation',
            'tests': result.testsRun, 'passed': result.wasSuccessful(),
            'skipped': len(result.skipped), 'records': result.records}, stream.getvalue()


def score_profile(gold, raw, catalog, recovery, enabled):
    previous = fact_recovery.recover_facts
    predictions, examples = [], []
    started = time.perf_counter()
    try:
        fact_recovery.recover_facts = recovery
        for cid, row in gold.items():
            call, audit = as_call(row), []
            result, error, summary = parse_prediction(raw[cid]['raw_response'], call,
                temporal_mode='case_v2', temporal_decisions=audit, fact_recovery=enabled,
                catalog_path=str(catalog), date_catalog_path=str(catalog))
            if error or source_problems(result, call):
                raise ValueError('Invalid output/source: ' + cid + ': ' + str(error))
            facts = result.model_dump(mode='json')['facts']
            predictions.append({'call_id': cid, 'facts': facts})
            examples.append({'call_id': cid, 'turns': [t.model_dump(mode='json') for t in call.turns],
                             'facts': facts, 'decisions': audit, 'summary': summary})
    finally:
        fact_recovery.recover_facts = previous
    seconds = time.perf_counter() - started
    quality, _ = evaluate_predictions(list(gold.values()), predictions,
                                     scorer_version='2.1', catalog_path=catalog)
    return predictions, examples, {'quality': quality, 'cpu_postprocess_seconds': seconds,
                                 'gpu_seconds': None, 'new_generation': False}


def evidence_rows(before, after):
    """Show only actual changed/additional facts, with exact available source text."""
    old = {r['call_id']: r['facts'] for r in before}
    result = []
    for row in after:
        turns = {t['idx']: t['text'] for t in row['turns']}
        for fact in row['facts']:
            if fact in old[row['call_id']]:
                continue
            idx = fact['source']['turn_idx']
            result.append({'call_id': row['call_id'], 'fact': fact,
                           'source_text': turns.get(idx, ''),
                           'before': [f for f in old[row['call_id']]
                                      if f['object'] == fact['object'] and f['attribute'] == fact['attribute']],
                           'decisions': row['decisions']})
    return result


def render_html(report, examples):
    esc = lambda value: html.escape(str(value), quote=True)
    heads = ''.join('<th>' + esc(v) + '</th>' for v in LABELS)
    rows = ''
    for name, profile in report['profiles'].items():
        metrics = profile['quality']['official']
        cells = ''.join('<td>' + ('—' if metrics[k] is None else f'{100*metrics[k]:.2f}%') + '</td>' for k in METRICS)
        rows += '<tr><th>' + esc(name) + '</th>' + cells + '</tr>'
    cards = []
    for example in examples:
        f = example['fact']
        cards.append('<details><summary>' + esc(f['attribute'] + ': ' + f['object']['name'] + ' → ' + f['value'])
            + '</summary><p>Звонок: ' + esc(example['call_id']) + '</p><blockquote>'
            + esc(example['source_text']) + '</blockquote><p>Цитата: '
            + esc(f['source'].get('quote') or 'не задана') + '</p><p>Начало: '
            + esc(f.get('valid_from')) + '; конец: ' + esc(f.get('valid_to'))
            + '</p><p>До:</p><pre>' + esc(json.dumps(example['before'], ensure_ascii=False, indent=2))
            + '</pre><p>Факт после:</p><pre>' + esc(json.dumps(f, ensure_ascii=False, indent=2))
            + '</pre><p>Решения правил, а не объяснение внутренних рассуждений модели:</p><pre>'
            + esc(json.dumps(example['decisions'], ensure_ascii=False, indent=2)) + '</pre></details>')
    checks = report['rule_checks']
    return '''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>CompactFactExtraction — доказательства</title><style>
body{font:16px system-ui;max-width:1120px;margin:36px auto;padding:0 20px;color:#172238;background:#f5f7fb}
table{border-collapse:collapse;width:100%;background:white}th,td{padding:12px;border-bottom:1px solid #ddd;text-align:left}
details{background:white;padding:16px;margin:12px 0;border-radius:10px}summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}blockquote{border-left:4px solid #2563eb;padding-left:16px}
</style><h1>CompactFactExtraction: качество и источники</h1>
<p>Локальный отчёт с данными звонков. Не публиковать и не показывать персональные данные без обезличивания.</p>
<p>Одинаковые сырые ответы модели, официальный оценщик 2.1. Это сравнение обработки на разработческом val;
не скрытый тест, не новое обучение и не кривая ресурсов GPU.</p>
<table><tr><th>Режим</th>''' + heads + '</tr>' + rows + '</table><h2>Проверки правил</h2><p>' + esc(checks['tests']) \
        + ' тестов; успешность: ' + esc(checks['passed']) + '. Это синтетические проверки компонентов, не проверка нейросети на adversarial-звонках.</p><h2>Изменённые факты: источник и даты</h2>' \
        + ''.join(cards) + '</html>'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('gold', 'raw', 'catalog', 'output-dir'):
        p.add_argument('--' + flag, required=True, type=Path)
    p.add_argument('--base-recovery', type=Path, default=ROOT / 'docs/experiments/current-state-recovery-001/base_fact_recovery.py')
    p.add_argument('--reference-pred', type=Path)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error('Output exists; choose a new directory')
    gold, raw = read_rows(args.gold), read_rows(args.raw)
    if set(gold) != set(raw):
        p.error('Gold/raw call IDs differ')
    spec = importlib.util.spec_from_file_location('frozen_defense_recovery', args.base_recovery)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    checks, check_log = rule_checks()
    if not checks['passed']:
        raise RuntimeError('Rule checks failed:\n' + check_log)
    args.output_dir.mkdir(parents=True)
    profiles, examples, rows = {}, {}, {}
    for name, recovery, enabled in [('off', base.recover_facts, False),
                                    ('v1', base.recover_facts, True),
                                    ('v2', fact_recovery.recover_facts, True)]:
        rows[name], examples[name], profiles[name] = score_profile(gold, raw, args.catalog, recovery, enabled)
        (args.output_dir / (name + '-predictions.jsonl')).write_text(''.join(
            json.dumps(r, ensure_ascii=False) + '\n' for r in rows[name]), encoding='utf-8')
        print(name, profiles[name]['quality']['official'], flush=True)
    if args.reference_pred:
        reference = read_rows(args.reference_pred)
        if {r['call_id']: r['facts'] for r in rows['v1']} != {k: r['facts'] for k, r in reference.items()}:
            raise ValueError('v1 does not reproduce supplied reference')
    report = {'profiles': profiles, 'rule_checks': checks, 'calls': len(gold),
              'reference_reproduced': args.reference_pred is not None,
              'hashes': {k: digest(getattr(args, k)) for k in ('gold', 'raw', 'catalog', 'base_recovery')},
              'source_hashes': {f.name: digest(f) for f in sorted((ROOT / 'src').glob('*.py'))},
              'training': False, 'gold_passed_to_rules': False, 'gpu_resource_curve_proven': False}
    evidence = evidence_rows(rows['v1'], examples['v2'])
    (args.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.output_dir / 'rule-checks.log').write_text(check_log, encoding='utf-8')
    (args.output_dir / 'evidence-private.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output_dir / 'index.html').write_text(render_html(report, evidence), encoding='utf-8')
    print('Evidence pack complete:', args.output_dir)


if __name__ == '__main__':
    main()
