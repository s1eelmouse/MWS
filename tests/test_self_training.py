"""Data isolation, whole-answer rejection and provenance in the self-training pilot."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from case_io import read_jsonl, write_jsonl
from run_context import file_info
from schema import contract_signature
from self_training import (assemble, checked_pilot, disjoint, dump, generate, input_only,
                           select_target, split_train, variant)
from test_case_schema import CALL, FACT


def example(cid='pool:1'):
    row = copy.deepcopy(CALL)
    row['call_id'] = cid
    row['turns'][0]['text'] = f'Перезагрузка роутера — не помогло. Звонок {cid}'
    return row


def prediction(facts=None):
    facts = copy.deepcopy(facts if facts is not None else [FACT])
    # Quotes are optional evidence; the normal training target uses turn_idx only.
    for fact in facts:
        fact['source'] = {'turn_idx': 7}
    raw = json.dumps({'facts': facts}, ensure_ascii=False)
    return {'call_id': 'pool:1', 'views': [
        {'result': {'facts': facts}, 'measurement': {'raw_response': raw}}
        for _ in range(2)]}


class SelfTrainingTest(unittest.TestCase):
    def test_inputs_strip_gold_and_nested_metadata(self):
        row = example()
        row['series'] = {'facts': ['secret']}
        clean = input_only(row)
        self.assertEqual(set(clean), {'call_id', 'started_at', 'turns'})
        self.assertNotIn('secret', json.dumps(clean))
        self.assertNotEqual(variant(row)['turns'], clean['turns'])
        self.assertEqual(row['turns'], clean['turns'])

    def test_split_is_repeatable_and_keeps_accounts_together(self):
        rows = [example(f'{i}:{j}') for i in range(10) for j in range(2)]
        gold, pool = split_train(rows, [example('val:1')], .2, 42)
        self.assertEqual((gold, pool), split_train(rows, [example('val:1')], .2, 42))
        self.assertEqual(len(pool), 4)
        disjoint(gold, pool)

    def test_rejects_val_accounts_and_renamed_transcripts(self):
        with self.assertRaises(ValueError):
            split_train([example('a:1'), example('b:1')], [example('a:2')], .2, 42)
        renamed = example('b:1')
        renamed['turns'] = example('a:1')['turns']
        with self.assertRaisesRegex(ValueError, 'Transcript overlap'):
            disjoint([example('a:1')], [renamed])

    def test_accepts_consistent_supported_answer(self):
        target, reason = select_target(input_only(example()), prediction())
        self.assertEqual(reason, 'accepted')
        self.assertEqual(len(target), 1)
        self.assertEqual(target[0]['source'], {'turn_idx': 7})

    def test_rejects_whole_answer_for_one_bad_fact(self):
        wrong = copy.deepcopy(FACT)
        wrong['value'] = 'Несуществующий факт'
        target, reason = select_target(input_only(example()), prediction([FACT, wrong]))
        self.assertIsNone(target)
        self.assertEqual(reason, 'nonliteral_value')

    def test_rejects_bad_raw_source_even_if_backend_salvaged(self):
        record = prediction()
        raw = json.loads(record['views'][0]['measurement']['raw_response'])
        raw['facts'][0]['source']['turn_idx'] = 999
        record['views'][0]['measurement']['raw_response'] = json.dumps(raw)
        self.assertEqual(select_target(example(), record)[1], 'invalid_raw_answer_or_source')

    def test_rejects_errors_empty_answers_duplicates_and_limits(self):
        for record, reason in [(prediction([]), 'empty_answer'),
                               (prediction([FACT, FACT]), 'duplicate_facts')]:
            self.assertEqual(select_target(example(), record)[1], reason)
        for key in ('retry', 'generation_limit_reached'):
            record = prediction()
            record['views'][0]['measurement'][key] = True
            self.assertEqual(select_target(example(), record)[1], 'token_limit_or_retry')
        record = prediction()
        record['views'][0]['result'] = {'error': 'input_token_limit'}
        self.assertEqual(select_target(example(), record)[1], 'generation_error:input_token_limit')

    def test_rejects_semantic_disagreement(self):
        record = prediction()
        raw = json.loads(record['views'][1]['measurement']['raw_response'])
        raw['facts'][0]['valid_from'] = None
        record['views'][1]['measurement']['raw_response'] = json.dumps(raw)
        self.assertEqual(select_target(example(), record)[1], 'view_disagreement')

    def test_rejects_conflicting_current_states(self):
        row = example()
        row['turns'][0]['text'] = 'тариф А тариф Б'
        a = copy.deepcopy(FACT)
        a.update(attribute='tariff', value='тариф А')
        b = copy.deepcopy(a)
        b['value'] = 'тариф Б'
        self.assertEqual(select_target(row, prediction([a, b]))[1], 'conflicting_open_states')

    def make_pilot(self, directory):
        root = Path(directory) / 'pilot'
        root.mkdir()
        gold = [example(f'gold{i}:1') for i in range(5)]
        write_jsonl(root / 'gold_train.jsonl', gold)
        write_jsonl(root / 'pool_inputs.jsonl', [input_only(example())])
        val = Path(directory) / 'val.jsonl'
        write_jsonl(val, [example('val:1')])
        dump(root / 'manifest.json', {
            'kind': 'train-only-pilot', 'contract': contract_signature(), 'seed': 42,
            'source_val': file_info(val),
            'files': {name: file_info(root / name) for name in ('gold_train.jsonl', 'pool_inputs.jsonl')},
        })
        return root, val

    def test_full_local_generation_and_assembly_without_gold_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, val = self.make_pilot(tmp)
            model = Path(tmp) / 'model'
            model.mkdir()
            (model / 'model.safetensors').write_bytes(b'test-only')
            (model / 'config.json').write_text('{}')
            gen = Path(tmp) / 'gen'
            class FakeEngine:
                device = 'cpu'
                model = SimpleNamespace(dtype='float32')

                def extract(self, calls, batch_size, measurements):
                    for call in calls:
                        if 'facts' in call.model_dump():
                            raise AssertionError('Gold entered inference')
                        measurements.append(prediction()['views'][0]['measurement'])
                    return [prediction()['views'][0]['result'] for _ in calls]

            with patch('extractor.Extractor', return_value=FakeEngine()), \
                    patch('config.TEMPORAL_SERIES_HISTORY', False), \
                    patch('extractor.SAVE_RAW_RESPONSES', False):
                generate(SimpleNamespace(pilot=root, model=model, batch_size=1, device='cpu',
                                         teacher_scope='full-train-seen', output_dir=gen))
            out = Path(tmp) / 'assembled'
            args = SimpleNamespace(pilot=root, generation=gen, val=val,
                                   max_pseudo_fraction=.2, output_dir=out)
            assemble(args)
            self.assertEqual(len(read_jsonl(out / 'mixed_train.jsonl')), 6)
            self.assertEqual(len(read_jsonl(out / 'pseudo_train.jsonl')), 1)
            with self.assertRaises(FileExistsError):
                assemble(args)
            with (gen / 'predictions.jsonl').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                assemble(args)

    def test_changed_pilot_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.make_pilot(tmp)
            with (root / 'pool_inputs.jsonl').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                checked_pilot(root)


if __name__ == '__main__':
    unittest.main()
