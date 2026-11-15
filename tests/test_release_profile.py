"""Release identity and measurement consistency, without GPU or model loading."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import release_profile as profile
from release_validate import plan as validation_plan
from resource_matrix import plan as matrix_plan
from schema import contract_signature
from compare_runs import comparison_rows


class ReleaseProfileTest(unittest.TestCase):
    def test_clean_noisy_matrix_use_identical_processing(self):
        for run in [validation_plan('/tmp/model', '/tmp/data/val.jsonl', '/tmp/run'),
                    validation_plan('/tmp/model', '/tmp/data/val.jsonl', '/tmp/run', 'noisy'),
                    matrix_plan('/tmp/model', '/tmp/data/val.jsonl', '/tmp/run')]:
            self.assertEqual(run['environment'], profile.environment('/tmp/data/val.jsonl'))
            self.assertEqual(run['environment']['TEMPORAL_SERIES_HISTORY'], '1')
            self.assertEqual(run['environment']['ALLOW_LEGACY_CONTRACT'], '0')
        matrix = matrix_plan('/tmp/model', '/tmp/data/val.jsonl', '/tmp/run')
        self.assertEqual([c[c.index('--batch-size')+1] for c in matrix['commands']], ['1','2','4'])

    def test_noisy_input_is_created_before_generation_in_fresh_output(self):
        run = validation_plan('/tmp/model', '/tmp/data/val.jsonl', '/tmp/run', 'noisy')
        make, benchmark = run['commands']
        self.assertTrue(make[1].endswith('make_noisy_val.py'))
        self.assertEqual(make[make.index('--output')+1], benchmark[benchmark.index('--input')+1])
        self.assertEqual(make[make.index('--seed')+1], '42')
        self.assertFalse(any('train.py' in word for c in run['commands'] for word in c))

    def test_preflight_checks_exact_data_model_and_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); model=root/'model'; model.mkdir()
            (root/'val.jsonl').write_text('{}')
            (root/'catalog.json').write_text('{}')
            (model/'model.safetensors').write_bytes(b'fixture, not real weights')
            (model/'case_contract.json').write_text(json.dumps({'signature':contract_signature()}))
            hashes={n:profile.digest(root/n) for n in ('val.jsonl','catalog.json')}
            with patch.object(profile, 'DATA_SHA256', hashes), patch.object(profile, 'MODEL_SHA256', profile.digest(model/'model.safetensors')):
                self.assertEqual(profile.preflight(model,root/'val.jsonl')['contract']['policy'],'exact')
                (root/'val.jsonl').write_text('changed')
                with self.assertRaisesRegex(ValueError,'data/catalog'):
                    profile.preflight(model,root/'val.jsonl')

    def test_preflight_rejects_other_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for n in ('val.jsonl','catalog.json','model.safetensors'): (root/n).write_text('{}')
            with patch.object(profile,'DATA_SHA256',{n:profile.digest(root/n) for n in ('val.jsonl','catalog.json')}):
                with self.assertRaisesRegex(ValueError,'Weights differ'):
                    profile.preflight(root,root/'val.jsonl')

    def test_curve_rejects_different_rule_code_with_same_policy_name(self):
        reports=[{'hardware':{'source_sha256':{'src/fact_recovery.py': value}}} for value in ('a','a','b')]
        with self.assertRaisesRegex(ValueError,'source hashes'):
            comparison_rows(reports)


if __name__ == '__main__': unittest.main()
