"""Evidence, identity and polarity boundaries for deterministic current states."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fact_recovery import recover_facts
from schema import as_call, Facts, source_problems
from prediction import parse_prediction


def call(*texts):
    return as_call({'call_id': 'synthetic:current', 'started_at': '2026-11-15T12:00:00+03:00',
                    'turns': [{'idx': i, 'speaker': 'client' if i % 2 == 0 else 'operator', 'text': text}
                              for i, text in enumerate(texts)]})


def client(value='перезвонок', attribute='agreement', **extra):
    return {'object': {'name': 'Мария Ивановна', 'type': 'client', 'id': None},
            'attribute': attribute, 'value': value, 'polarity': 1,
            'valid_from': '2026-11-01T00:00:00+03:00', 'valid_to': None,
            'source': {'turn_idx': 0, 'span': None, 'quote': None}, **extra}


class CurrentStateRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.catalog = Path(self.temp.name) / 'catalog.json'
        self.catalog.write_text(json.dumps({'tariffs': ['Пульс', 'Смарт', 'Смарт+'],
                                           'services': ['домашний интернет', 'домашний телефон'],
                                           'devices': ['роутер'],
                                           'symptoms': ['нет гудков', 'работает нормально', 'нет сети']}))

    def recover(self, facts, *texts):
        source = call(*texts)
        before = copy.deepcopy(facts)
        result, audit = recover_facts(facts, source, self.catalog)
        self.assertEqual(facts, before)
        parsed = Facts.model_validate({'facts': result})
        self.assertFalse(source_problems(parsed, source))
        return result, audit

    def test_tariff_wrapper_does_not_change_the_named_plan(self):
        result, _ = self.recover([client('тариф Смарт подключен', 'tariff')], 'Тариф Смарт подключен.')
        self.assertEqual(result[0]['value'], 'Смарт')

    def test_grammatical_tariff_case_is_not_a_new_plan_name(self):
        catalog = json.loads(self.catalog.read_text())
        catalog['tariffs'].append('Улучшенный')
        self.catalog.write_text(json.dumps(catalog))
        result, _ = self.recover([client()], 'Я на Улучшенном сижу, плачу исправно.')
        self.assertEqual(result[-1]['value'], 'Улучшенный')

    def test_question_about_effect_does_not_retract_a_tariff_assertion(self):
        result, _ = self.recover([client()], 'Тариф у меня Пульс — это не влияет?')
        self.assertEqual(result[-1]['value'], 'Пульс')

    def test_tariff_question_requires_confirmation(self):
        result, _ = self.recover([client()], 'У вас тариф Смарт+?', 'Да, он самый Смарт+.')
        self.assertEqual(result[-1]['value'], 'Смарт+')
        result, _ = self.recover([client()], 'У вас тариф Смарт+?')
        self.assertEqual(len(result), 1)

    def test_negative_response_does_not_confirm_a_tariff(self):
        result, _ = self.recover([client()], 'У вас тариф Смарт?', 'Нет, не Смарт.')
        self.assertEqual(len(result), 1)
        result, _ = self.recover([client()], 'У вас тариф Смарт?', 'Да, не Смарт.')
        self.assertEqual(len(result), 1)

    def test_acknowledged_tariff_correction_closes_the_explicit_old_plan(self):
        old = client('Пульс', 'tariff')
        result, audit = self.recover([old], 'Какой у вас тариф?',
                                     'Сейчас тариф Смарт. Пульс раньше был.')
        self.assertIsNotNone(result[0]['valid_to'])
        self.assertEqual(result[-1]['value'], 'Смарт')
        self.assertTrue(any(e['reason'] == 'explicit_historical_tariff_closed' for e in audit))

    def test_unacknowledged_tariff_conflict_is_not_resolved_by_last_mention(self):
        old = client('Пульс', 'tariff')
        result, _ = self.recover([old], 'У меня сейчас тариф Смарт, Пульс тоже подключен.')
        self.assertEqual(result, [old])

    def test_existing_current_tariff_is_preserved_without_historical_evidence(self):
        initial = client('Пульс', 'tariff')
        result, _ = self.recover([initial], 'У меня тариф Смарт.')
        self.assertEqual(result, [initial])

    def test_future_and_proposal_do_not_create_a_current_plan(self):
        for text in ('Хочу тариф Смарт.', 'Завтра у меня будет тариф Смарт.', 'Предлагаю тариф Смарт.'):
            result, _ = self.recover([client()], text)
            self.assertEqual(len(result), 1, text)

    def test_named_service_is_recovered_with_exact_source(self):
        result, audit = self.recover([], 'Домашний телефон — нет гудков.')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['object']['name'], 'домашний телефон')
        self.assertEqual(result[0]['value'], 'нет гудков')
        self.assertEqual(result[0]['polarity'], 1)
        self.assertEqual(result[0]['source']['quote'], 'Домашний телефон — нет гудков')
        self.assertEqual(audit[0]['field'], 'service_state')

    def test_literal_activation_of_a_named_service_is_a_state(self):
        result, _ = self.recover([], 'Домашний интернет подключен.')
        self.assertEqual(result[0]['value'], 'Домашний интернет подключен')

    def test_state_question_requires_affirmative_response(self):
        result, _ = self.recover([], 'Домашний интернет работает?')
        self.assertFalse(result)
        result, _ = self.recover([], 'Домашний интернет работает?', 'Да, все верно.')
        self.assertEqual(len(result), 1)
        result, _ = self.recover([], 'Домашний интернет работает?', 'Да, не работает.')
        self.assertFalse(result)

    def test_connection_noun_is_not_an_activation_predicate(self):
        result, _ = self.recover([], 'Роутер пишет: нет типа подключения.')
        self.assertFalse(result)

    def test_television_is_not_a_television_set_object(self):
        catalog = json.loads(self.catalog.read_text())
        catalog['devices'].append('телевизор')
        self.catalog.write_text(json.dumps(catalog))
        result, _ = self.recover([], 'По телевидению все работает нормально.')
        self.assertFalse(result)

    def test_a_hypothetical_recovery_does_not_open_a_closed_fact(self):
        initial = {**client('нет сети', 'service_state'),
                   'object': {'name': 'домашний интернет', 'type': 'service', 'id': None},
                   'valid_to': '2026-11-10T00:00:00+03:00'}
        result, _ = self.recover([initial], 'Если домашний интернет работает, закроем заявку.')
        self.assertEqual(result, [initial])

    def test_historical_hypothetical_and_future_states_are_not_current(self):
        for text in ('Раньше домашний интернет работал нормально.',
                     'Если домашний интернет работает нормально, ждите.',
                     'Завтра домашний интернет будет подключен.',
                     'Кажется, домашний телефон — нет гудков.'):
            result, _ = self.recover([], text)
            self.assertFalse(result, text)

    def test_two_objects_in_one_assertion_are_not_guessed(self):
        result, _ = self.recover([], 'Домашний интернет и домашний телефон не работают.')
        self.assertFalse(result)

    def test_incidental_new_equipment_later_in_call_is_not_a_new_primary_object(self):
        result, _ = self.recover([], 'У меня вопрос по оплате.', 'Роутер работает нормально.')
        self.assertFalse(result)

    def test_device_already_identified_by_model_can_be_recovered_later(self):
        known = {**client('перезагрузка', 'action'),
                 'object': {'name': 'роутер', 'type': 'device', 'id': None}}
        result, _ = self.recover([known], 'У меня вопрос по оплате.', 'Роутер работает нормально.')
        self.assertEqual(result[-1]['attribute'], 'service_state')

    def test_existing_current_state_is_not_overridden(self):
        existing = {**client('нет сети', 'service_state'),
                    'object': {'name': 'домашний интернет', 'type': 'service', 'id': None}}
        result, _ = self.recover([existing], 'Домашний интернет работает нормально.')
        self.assertEqual(result, [existing])

    def test_one_object_does_not_block_another(self):
        existing = {**client('нет сети', 'service_state'),
                    'object': {'name': 'домашний интернет', 'type': 'service', 'id': None}}
        result, _ = self.recover([existing], 'Домашний телефон — нет гудков.')
        self.assertEqual(len(result), 2)

    def test_explicit_symptom_start_keeps_timezone(self):
        result, _ = self.recover([], 'Со вторника домашний телефон — нет гудков.')
        self.assertEqual(result[0]['valid_from'], '2026-11-10T00:00:00+03:00')

    def test_recovery_is_idempotent(self):
        result, _ = self.recover([client()], 'У меня тариф Смарт. Домашний интернет подключен.')
        repeated, _ = self.recover(result, 'У меня тариф Смарт. Домашний интернет подключен.')
        self.assertEqual(result, repeated)

    def test_normalizer_reports_updates_separately_from_added_facts(self):
        payload = json.dumps({'facts': [client('тариф Смарт подключен', 'tariff')]})
        result, error, summary = parse_prediction(payload, call('Тариф Смарт подключен.'),
            temporal_mode='off', catalog_path=str(self.catalog), fact_recovery=True)
        self.assertIsNone(error)
        self.assertEqual(result.facts[0].value, 'Смарт')
        self.assertEqual(summary['recovered_facts'], 0)
        self.assertEqual(summary['recovery_updates'], 1)


if __name__ == '__main__':
    unittest.main()
