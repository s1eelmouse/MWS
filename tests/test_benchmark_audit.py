"""Progress, raw responses and final series processing remain independently auditable."""
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
    def test_progress_and_raw_are_saved_without_gold_and_series_is_separate(self):
        class Backend:
            device='cpu'
            temporal_normalization='task'
            def extract(self,calls,batch_size,measurements):
                assert all(not hasattr(c,'facts') for c in calls)
                measurements.extend({'call_id':c.call_id,'raw_response':'original generated text'} for c in calls)
                return [{'facts':[copy.deepcopy(FACT)]} if c.call_id.endswith('0') else {'error':'invalid_json'} for c in calls]
        def process(calls,predictions,mode):
            after=copy.deepcopy(predictions)
            after[0]['facts'][0]['valid_from']=None
            return after,{'dates_changed':1},[{'call_id':calls[0].call_id,'field':'valid_from'}]
        rows=[{**copy.deepcopy(CALL),'call_id':f'synthetic:call{i}'} for i in range(2)]
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()) as log:
            dest=Path(temp)
            with patch('benchmark.apply_temporal_postprocess',side_effect=process):
                report=run_benchmark(rows,Backend(),dest,1,1)
            self.assertIn('Completed 1/2; failed: 0',log.getvalue())
            self.assertIn('Completed 2/2; failed: 1',log.getvalue())
            self.assertEqual(report['failed_calls'],1)
            before=json.loads((dest/'predictions-before-series.jsonl').read_text(encoding='utf-8').splitlines()[0])
            after=json.loads((dest/'predictions.jsonl').read_text(encoding='utf-8').splitlines()[0])
            self.assertEqual(before['facts'][0],FACT)
            self.assertIsNone(after['facts'][0]['valid_from'])
            self.assertEqual(len((dest/'raw-answers.jsonl').read_text(encoding='utf-8').splitlines()),2)
            self.assertNotIn('raw_response',json.loads((dest/'calls.jsonl').read_text(encoding='utf-8').splitlines()[0]))
            self.assertAlmostEqual(report['processing_seconds_including_postprocess'],
                                   report['measurement_seconds']+report['temporal_postprocess_seconds'])
            self.assertTrue((dest/'postprocess-memory.csv').exists())
