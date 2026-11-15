"""Safety boundaries for local defense artifacts and resource plans."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import defense_evidence as evidence
from resource_matrix import plan


class DefenseEvidenceTest(unittest.TestCase):
    def test_transcript_and_fact_are_escaped_in_demo(self):
        attack = '<script>alert(1)</script>'
        report = {'profiles': {}, 'rule_checks': {'tests': 1, 'passed': True}}
        fact = {'attribute': 'tariff', 'object': {'name': attack}, 'value': attack,
                'source': {'turn_idx': 0, 'quote': attack}}
        rendered = evidence.render_html(report, [{'call_id': attack, 'fact': fact,
            'source_text': attack, 'before': [], 'decisions': []}])
        self.assertNotIn('<script>', rendered)
        self.assertIn('&lt;script&gt;', rendered)
        self.assertIn("default-src 'none'", rendered)

    def test_duplicate_ids_and_empty_input_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'rows.jsonl'
            for text in ('', '{"call_id":"x"}\n{"call_id":"x"}\n'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    evidence.read_rows(path)

    def test_profile_failure_restores_runtime_and_strips_gold(self):
        gold = {'x': {'call_id': 'x', 'started_at': '2026-11-15T12:00:00+03:00',
                      'turns': [{'idx': 0, 'speaker': 'client', 'text': 'текст'}],
                      'facts': [{'secret_gold': True}]}}
        original = evidence.fact_recovery.recover_facts
        def rejected(raw, call, **kwargs):
            self.assertNotIn('facts', call.model_dump())
            raise RuntimeError('synthetic failure')
        with patch.object(evidence, 'parse_prediction', rejected):
            with self.assertRaises(RuntimeError):
                evidence.score_profile(gold, {'x': {'raw_response': '{}'}}, 'catalog.json', lambda: None, True)
        self.assertIs(evidence.fact_recovery.recover_facts, original)

    def test_demo_excludes_unchanged_facts(self):
        fact = {'attribute': 'tariff', 'object': {'name': 'Смарт'}, 'value': 'Смарт',
                'source': {'turn_idx': 0}}
        before = [{'call_id': 'x', 'facts': [fact]}]
        after = [{'call_id': 'x', 'facts': [fact], 'turns': [{'idx': 0, 'text': 'текст'}], 'decisions': []}]
        self.assertEqual(evidence.evidence_rows(before, after), [])

    def test_resource_plan_preserves_same_protocol_and_three_batch_sizes(self):
        matrix = plan('/tmp/model with spaces', '/tmp/data/val.jsonl', '/tmp/new run', python='python')
        commands = matrix['commands']
        self.assertEqual([c[c.index('--batch-size')+1] for c in commands], ['1', '2', '4'])
        self.assertTrue(all(c[c.index('--model')+1] == str(Path('/tmp/model with spaces').resolve()) for c in commands))
        self.assertEqual(matrix['environment']['FACT_RECOVERY'], '1')
        self.assertEqual(matrix['environment']['TEMPORAL_SERIES_HISTORY'], '0')
        self.assertTrue(all('--limit' not in c for c in commands))
        self.assertEqual(matrix['environment']['HF_HUB_OFFLINE'], '1')


if __name__ == '__main__':
    unittest.main()
