"""Regression coverage for the saved 4090 contract and diagnostic-only reruns."""
import contextlib
import copy
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from schema import contract_signature, parse_answer, parse_answer_with_diagnostics
from extractor import Extractor
from benchmark import run_benchmark
from compare_runs import comparison_rows
from test_case_schema import CALL, FACT


class Tokens:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key):
        return Tokens([r[key[1]] for r in self.rows])

    def tolist(self):
        return self.rows


class Inputs(dict):
    def to(self, device):
        return self


class DiagnosticTest(unittest.TestCase):
    def test_existing_model_contract_does_not_change(self):
        self.assertEqual(contract_signature(), '40e9c449fffa0434ebd8cc5f7860cbe843c7d1ab9d5d28256392d20f24e72654')

    def test_json_syntax_and_schema_are_distinguished(self):
        _, error = parse_answer_with_diagnostics('{"facts": [broken]}')
        self.assertEqual(error['reason'], 'json_syntax')
        self.assertIn('position', error['json_error'])
        _, error = parse_answer_with_diagnostics('{"facts":[{"attribute":"made_up"}]}')
        self.assertEqual(error['error'], 'invalid_schema')
        self.assertTrue(error['validation_errors'])
        self.assertNotIn('input', error['validation_errors'][0])
        self.assertIsNotNone(parse_answer('<think>reason</think>```json\n{"facts":[]}\n```'))
        self.assertIsNone(parse_answer('no json'))

    def test_extractor_records_raw_parse_source_and_length_failures(self):
        bad_source = copy.deepcopy(FACT)
        bad_source['source']['turn_idx'] = 999
        raw = ['{"facts":[broken]}', '{"facts":[{}]}',
               json.dumps({'facts':[bad_source]}), '{"facts":[]}']
        class Tokenizer:
            pad_token_id = 0
            def __call__(self, prompts, **kwargs):
                return {'input_ids':[1,3]} if isinstance(prompts,str) else Inputs(input_ids=Tokens([[1,3]]*len(prompts)))
            def batch_decode(self, tokens, **kwargs):
                return raw
        backend = Extractor.__new__(Extractor)
        backend.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        backend.device, backend.with_schema = 'cpu', False
        backend.tok, backend.lock = Tokenizer(), threading.Lock()
        backend.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100),
            generation_config=SimpleNamespace(eos_token_id=2),
            generate=lambda **kwargs: Tokens([[1,3,6,2,0]]*3+[[1,3,6,7,8]]))
        calls=[{**CALL,'call_id':f'synthetic:{i}'} for i in range(4)]
        details=[]
        with patch('extractor.prompt_for',return_value='prompt'), \
                patch('extractor.MAX_NEW_TOKENS',3), \
                patch('extractor.RETRY_ON_TOKEN_LIMIT',False):
            answers=backend.extract(calls,4,details)
        self.assertEqual([x['error'] for x in answers],['invalid_json','invalid_schema','invalid_source','output_token_limit'])
        self.assertEqual([x['diagnostic']['raw_response'] for x in details],raw)
        self.assertTrue(details[2]['diagnostic']['source_problems'])
        self.assertEqual(details[3]['output_tokens'],3)

    def test_raw_failures_are_saved_separately_and_no_gold_reaches_backend(self):
        class Backend:
            device='cpu'
            def extract(self,calls,batch_size,measurements):
                self_no_gold = all(not hasattr(c,'facts') for c in calls)
                assert self_no_gold
                measurements.extend({'call_id':c.call_id,'diagnostic':{'error':'invalid_schema',
                    'raw_response':'{"facts":[{}]}','validation_errors':[{'type':'missing'}]}} for c in calls)
                return [{'error':'invalid_schema'} for c in calls]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            report=run_benchmark([CALL],Backend(),root,1,1)
            failure=json.loads((root/'failures.jsonl').read_text())
            self.assertEqual(failure['raw_response'],'{"facts":[{}]}')
            self.assertNotIn('diagnostic',json.loads((root/'calls.jsonl').read_text()))
            self.assertEqual(report['failed_calls'],1)
            self.assertEqual(report['quality']['official']['fact_recall'],0)

    def test_diagnostic_subset_is_not_a_comparison_point(self):
        with self.assertRaisesRegex(ValueError,'Diagnostic'):
            comparison_rows([{'diagnostic_subset':True}]*3)

    def test_pipeline_requires_explicit_choice_before_any_output_is_created(self):
        from run_experiment import main
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'must-not-exist'
            with patch.object(sys,'argv',['run_experiment.py','--output-dir',str(output)]):
                with self.assertRaises(SystemExit):main()
            self.assertFalse(output.exists())

    def test_pipeline_rejects_invalid_learning_rate_before_writing(self):
        from run_experiment import main
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'must-not-exist'
            argv = ['run_experiment.py', '--train-new', '--learning-rate', '0',
                    '--output-dir', str(output)]
            with patch.object(sys, 'argv', argv):
                with self.assertRaises(SystemExit):
                    main()
            self.assertFalse(output.exists())

    def test_subset_cli_filters_before_limit_and_records_provenance(self):
        from benchmark import main
        from case_io import write_jsonl
        class Backend:
            device='cpu'
            torch=SimpleNamespace(get_num_threads=lambda:8)
            model=SimpleNamespace(num_parameters=lambda:123,dtype='float32',config=SimpleNamespace())
            def extract(self,calls,batch_size,measurements):
                assert all(not hasattr(c,'facts') for c in calls)
                return [{'facts':[]} for c in calls]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            gold=root/'gold.jsonl'
            write_jsonl(gold,[{**CALL,'call_id':f'c{i}'} for i in range(3)])
            failure=root/'old-errors.jsonl'
            write_jsonl(failure,[{'call_id':'c0','matched':1},{'call_id':'c2','error':'invalid_json'}])
            output=root/'new'
            argv=['benchmark.py','--input',str(gold),'--failures',str(failure),
                  '--model','fake','--device','cpu','--limit','1','--output-dir',str(output)]
            with patch.object(sys,'argv',argv), patch('extractor.Extractor',return_value=Backend()), \
                    patch('benchmark.environment',return_value={'cpu':'fake'}):
                main()
            report=json.loads((output/'report.json').read_text())
            self.assertTrue(report['diagnostic_subset'])
            self.assertTrue(report['failures_sha256'])
            self.assertEqual(json.loads((output/'predictions.jsonl').read_text())['call_id'],'c2')


if __name__ == '__main__':
    unittest.main()
