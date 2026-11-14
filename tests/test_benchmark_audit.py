"""Progress and raw responses remain auditable with rules inside timed extraction."""
import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from benchmark import run_benchmark
from test_case_schema import CALL,FACT


class AuditTest(unittest.TestCase):
    def test_progress_raw_and_pre_history_snapshot_are_saved_without_untimed_rules(self):
        class Backend:
            device='cpu'
            temporal_normalization='task'
            def extract(self,calls,batch_size,measurements):
                assert all(not hasattr(c,'facts') for c in calls)
                results = []
                for c in calls:
                    before = {'facts': [copy.deepcopy(FACT)]} if c.call_id.endswith('0') else {'error': 'invalid_json'}
                    measurements.append({'call_id': c.call_id, 'raw_response': 'original generated text',
                                         'prediction_before_series': before})
                    after = copy.deepcopy(before)
                    if 'facts' in after:
                        after['facts'][0]['valid_from'] = None
                    results.append(after)
                return results
        rows=[{**copy.deepcopy(CALL),'call_id':f'synthetic:call{i}'} for i in range(2)]
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()) as log:
            dest=Path(temp)
            with patch('benchmark.apply_temporal_postprocess', side_effect=AssertionError('untimed rules')):
                report=run_benchmark(rows,Backend(),dest,1,1)
            self.assertIn('Completed 1/2; failed: 0',log.getvalue())
            self.assertIn('Completed 2/2; failed: 1',log.getvalue())
            self.assertEqual(report['failed_calls'],1)
            before=json.loads((dest/'predictions-before-series.jsonl').read_text().splitlines()[0])
            after=json.loads((dest/'predictions.jsonl').read_text().splitlines()[0])
            self.assertEqual(before['facts'][0],FACT)
            self.assertIsNone(after['facts'][0]['valid_from'])
            self.assertEqual(len((dest/'raw-answers.jsonl').read_text().splitlines()),2)
            self.assertNotIn('raw_response',json.loads((dest/'calls.jsonl').read_text().splitlines()[0]))
            self.assertNotIn('prediction_before_series',json.loads((dest/'calls.jsonl').read_text().splitlines()[0]))
            self.assertTrue(report['temporal_postprocess']['included_in_measurement'])
