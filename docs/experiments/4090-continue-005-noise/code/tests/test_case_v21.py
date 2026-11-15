"""New-case compatibility, official equivalence and explicitly versioned fixes."""
import copy
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from case_evaluation import evaluate_predictions
from case_io import validate_gold
from scoring import make_scorer
from schema import (Facts, LEGACY_CONTRACT_SIGNATURE, check_model_contract, contract_signature,
                    legacy_contract_signature, to_answer)
from training_data import adapter_origin
from test_case_schema import CALL, FACT


class CaseV21Test(unittest.TestCase):
    def state(self):
        row = copy.deepcopy(CALL)
        row['facts'][0].update(attribute='service_state', value='работает нормально')
        return row

    def test_empty_object_roundtrips_validation_target_and_json_schema(self):
        row = self.state()
        row['facts'][0]['object']['name'] = ''
        validate_gold(row)
        target = to_answer(row['facts'])
        self.assertEqual(json.loads(target)['facts'][0]['object']['name'], '')
        self.assertEqual(Facts.model_json_schema()['$defs']['ObjectRef']['properties']['name']['minLength'], 0)
        report, _ = evaluate_predictions([row], [row])
        self.assertEqual(report['official']['fact_recall'], 1)

    def test_only_known_legacy_contract_can_be_migrated_explicitly_without_editing_it(self):
        self.assertEqual(legacy_contract_signature(), LEGACY_CONTRACT_SIGNATURE)
        self.assertNotEqual(contract_signature(), LEGACY_CONTRACT_SIGNATURE)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'adapter'
            source.mkdir()
            manifest = source / 'case_contract.json'
            manifest.write_text(json.dumps({'signature': LEGACY_CONTRACT_SIGNATURE}))
            before = manifest.read_bytes()
            with self.assertRaises(ValueError):
                check_model_contract(source)
            accepted = check_model_contract(source, allow_legacy=True)
            self.assertFalse(accepted['weights_modified'])
            (source / 'adapter_config.json').write_text(json.dumps({'base_model_name_or_path': 'test/base', 'peft_type': 'LORA'}))
            (source / 'adapter_model.safetensors').write_bytes(b'fixture')
            origin = adapter_origin(source, root / 'new-cycle', 'test/base', allow_legacy=True)
            self.assertEqual(origin['contract_compatibility']['source_signature'], LEGACY_CONTRACT_SIGNATURE)
            self.assertEqual(manifest.read_bytes(), before)
            manifest.write_text('{"signature":"unknown"}')
            with self.assertRaises(ValueError):
                check_model_contract(source, allow_legacy=True)

    def test_official_source_bytes_pinned_and_datefix_is_optional_and_labelled(self):
        official, fixed = make_scorer('2.1'), make_scorer('2.1-datefix')
        self.assertEqual(official.sha256, '863f87737dd1108e772250967f43697fa2411b3394c190d192b7ff66666259c3')
        for text in ('через две недели', 'позапрошлый четверг'):
            with self.assertRaises(TypeError):
                official.module.recover_dates(text, date(2026, 11, 14))
            self.assertTrue(fixed.module.recover_dates(text, date(2026, 11, 14)))
        for text in ('вчера', 'через неделю', 'в прошлом месяце', 'на следующей неделе'):
            self.assertEqual(official.module.recover_dates(text, date(2026, 11, 14)),
                             fixed.module.recover_dates(text, date(2026, 11, 14)))
        self.assertIsNone(official.metadata()['local_patch'])
        self.assertEqual(fixed.metadata()['local_patch'], 'date-set-update-fix-1')
        self.assertEqual(official.path.read_bytes(), fixed.path.read_bytes())

    def test_catalogues_do_not_leak_between_independent_evaluations(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'a.json', Path(temp) / 'b.json'
            a.write_text('{"tariffs":["Тест Альфа"]}', encoding='utf-8')
            b.write_text('{"tariffs":["Тест Бета"]}', encoding='utf-8')
            first, second = make_scorer('2.1', a), make_scorer('2.1', b)
            self.assertEqual(first.module._catalog_for('tariff'), ['Тест Альфа'])
            self.assertEqual(second.module._catalog_for('tariff'), ['Тест Бета'])
            self.assertEqual(first.module._catalog_for('tariff'), ['Тест Альфа'])

    def test_state_objects_negation_and_per_attribute_metadata_use_v21(self):
        row = self.state()
        second = copy.deepcopy(row['facts'][0])
        second.update(object={'id': 'synthetic:dev1', 'type': 'device', 'name': 'модем'}, value='красный индикатор')
        row['facts'].append(second)
        report, _ = evaluate_predictions([row], [row])
        self.assertEqual(report['official']['state_accuracy'], 1)
        self.assertEqual(report['per_attribute']['service_state']['state_accuracy'], 1)
        self.assertEqual(report['official']['metrics_version'], '2.1')
        pred = copy.deepcopy(row)
        pred['facts'][0]['value'] = 'не работает нормально'
        self.assertEqual(make_scorer().match_facts([row['facts'][0]], [pred['facts'][0]]), [])

    def test_temporal_diagnostics_separate_action_and_recoverable_states(self):
        row = self.state()
        predicted = copy.deepcopy(row)
        predicted['facts'][0]['valid_from'] = '2026-11-10T00:00:00'
        report, _ = evaluate_predictions([row], [predicted])
        attr = report['per_attribute']['service_state']
        self.assertEqual(attr['n_temporal'], 1)
        self.assertEqual(attr['temporal_eligible_matched_facts'], 1)
        self.assertEqual(attr['temporal_date_errors'], 1)
        action, _ = evaluate_predictions([CALL], [CALL])
        self.assertEqual(action['official']['n_temporal'], 0)
        self.assertIsNone(action['official']['temporal_recall'])
        row['facts'][0]['attribute'] = 'agreement'
        self.assertEqual(evaluate_predictions([row], [row])[0]['per_attribute']['agreement']['attribute_kind'], 'temporal')

    def test_training_event_keeps_context_without_calling_it_official_quality(self):
        from training_observer import make_observer
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {'transformers': SimpleNamespace(TrainerCallback=object)}):
            root = Path(temp)
            observer = make_observer(root, SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
                                     context={'context_sha256': 'fixture', 'scoring': {'scorer_version': '2.1'}})
            observer.on_log(None, SimpleNamespace(global_step=4, epoch=0.1), None, logs={'loss': 0.5})
            event = json.loads((root / 'events.jsonl').read_text())
            self.assertEqual(event['context_sha256'], 'fixture')
            self.assertEqual(event['metrics_kind'], 'teacher_forced_loss_not_official_generation_quality')
            self.assertEqual(json.loads((root / 'context.json').read_text())['scoring']['scorer_version'], '2.1')
