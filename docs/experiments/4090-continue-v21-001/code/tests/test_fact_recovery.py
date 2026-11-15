"""Deterministic recovery of missing current tariff/status facts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fact_recovery import recover_facts
from schema import as_call


def call(text, started='2026-03-11T10:00:00'):
    return as_call({'call_id': 'synthetic:recovery', 'started_at': started,
                    'turns': [{'idx': 0, 'speaker': 'client', 'text': text},
                              {'idx': 1, 'speaker': 'operator', 'text': 'Фиксирую.'}]})


def client_fact():
    return {'object': {'name': 'Марина Сергеевна', 'type': 'client', 'id': None},
            'attribute': 'agreement', 'value': 'перезвонок', 'polarity': 1,
            'valid_from': None, 'valid_to': None,
            'source': {'turn_idx': 0}, 'speaker': 'client'}


class FactRecoveryTest(unittest.TestCase):
    def test_longest_catalog_name_and_plus_are_not_confused_with_shorter_tariffs(self):
        with tempfile.TemporaryDirectory() as temp:
            catalog = Path(temp) / 'catalog.json'
            catalog.write_text(json.dumps({'tariffs': ['Люкс', 'Орбита-Люкс', 'Экстра', 'Экстра+']}), encoding='utf-8')
            for value in ('Орбита-Люкс', 'Экстра+'):
                result, _ = recover_facts([client_fact()], call('У меня тариф ' + value + '.'), str(catalog))
                self.assertEqual(result[-1]['value'], value)

    def test_catalog_word_in_device_name_is_not_recovered_as_a_tariff(self):
        with tempfile.TemporaryDirectory() as temp:
            catalog = Path(temp) / 'catalog.json'
            catalog.write_text(json.dumps({'tariffs': ['Мини']}), encoding='utf-8')
            result, audit = recover_facts([client_fact()], call('У меня сейчас мини-приставка не работает.'), str(catalog))
            self.assertEqual(len(result), 1)
            self.assertFalse(audit)
    def test_negation_question_future_and_hypothesis_never_create_a_status(self):
        for text in ('Сейчас заявка 42 не выполнена.', 'Сейчас заявка 42 в работе?',
                     'Если заявка 42 сейчас в работе, то ждите.',
                     'Завтра заявка 42 будет в работе.', 'Вчера заявка 42 была в работе.'):
            with self.subTest(text=text):
                result, audit = recover_facts([], call(text))
                self.assertEqual(result, [])
                self.assertEqual(audit, [])

    def test_status_identity_and_existing_current_are_checked_per_ticket(self):
        existing = {'object': {'name': 'заявка 11', 'type': 'ticket'},
                    'attribute': 'ticket_status', 'value': 'принята', 'polarity': 1,
                    'valid_from': None, 'valid_to': None, 'source': {'turn_idx': 0}}
        result, audit = recover_facts([existing], call('Сейчас статус заявки 42 — в работе.'))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[-1]['object']['name'], 'заявка 42')
        self.assertEqual(audit[-1]['source_turn_idx'], 0)
        # No number and two known tickets is ambiguous; never borrow the first object.
        result, audit = recover_facts([existing, {**existing, 'object': {'name': 'заявка 42', 'type': 'ticket'},
                                             'valid_to': '2026-03-01T00:00:00'}],
                                     call('Сейчас статус — выполнена.'))
        self.assertEqual(len(result), 2)
        self.assertFalse(audit)

    def test_tariff_proposal_or_denial_is_not_a_current_tariff(self):
        with tempfile.TemporaryDirectory() as temp:
            catalog = Path(temp) / 'catalog.json'
            catalog.write_text(json.dumps({'tariffs': ['Пульс']}), encoding='utf-8')
            for text in ('У меня не тариф Пульс.', 'Сейчас хочу тариф Пульс.',
                         'У вас сейчас тариф Пульс?', 'Раньше у меня был тариф Пульс.'):
                result, audit = recover_facts([client_fact()], call(text), str(catalog))
                self.assertEqual(len(result), 1, text)
                self.assertFalse(audit)

    def test_two_different_clients_are_not_guessed(self):
        with tempfile.TemporaryDirectory() as temp:
            catalog = Path(temp) / 'catalog.json'
            catalog.write_text(json.dumps({'tariffs': ['Пульс']}), encoding='utf-8')
            for source in ([client_fact(), {**client_fact(), 'object': {'name': 'Другой клиент', 'type': 'client'}}],):
                result, audit = recover_facts(source, call('У меня сейчас тариф Пульс.'), str(catalog))
                self.assertEqual(len(result), len(source))
                self.assertFalse(audit)

    def test_one_explicit_unnamed_client_is_preserved_without_inventing_a_name(self):
        with tempfile.TemporaryDirectory() as temp:
            catalog = Path(temp) / 'catalog.json'
            catalog.write_text(json.dumps({'tariffs': ['Пульс']}), encoding='utf-8')
            source = [{**client_fact(), 'object': {'name': '', 'type': 'client'}}]
            result, audit = recover_facts(source, call('У меня сейчас тариф Пульс.'), str(catalog))
            self.assertEqual(len(result), 2)
            self.assertEqual(result[-1]['object']['name'], '')
            self.assertEqual(len(audit), 1)

    def test_missing_tariff_is_recovered_from_catalog_and_text(self):
        with tempfile.TemporaryDirectory() as temp:
            catalog = Path(temp) / 'catalog.json'
            catalog.write_text(json.dumps({'tariffs': ['Орбита-Люкс', 'Пульс']}), encoding='utf-8')
            facts, decisions = recover_facts(
                [client_fact()], call('У меня тариф Орбита-Люкс, сейчас подключён.'), str(catalog))
        tariff = [fact for fact in facts if fact['attribute'] == 'tariff']
        self.assertEqual(len(tariff), 1)
        self.assertEqual(tariff[0]['value'], 'Орбита-Люкс')
        self.assertEqual(tariff[0]['object']['name'], 'Марина Сергеевна')
        self.assertEqual(decisions[0]['reason'], 'missing_current_tariff')

    def test_existing_current_tariff_is_not_overridden(self):
        fact = {**client_fact(), 'attribute': 'tariff', 'value': 'Пульс'}
        facts, decisions = recover_facts(
            [fact], call('У меня тариф Орбита-Люкс, сейчас подключён.'), None)
        self.assertEqual([item for item in facts if item['attribute'] == 'tariff'], [fact])
        self.assertFalse(decisions)

    def test_historical_tariff_does_not_block_current_recovery(self):
        old = {**client_fact(), 'attribute': 'tariff', 'value': 'Пульс',
               'valid_to': '2026-03-10T00:00:00'}
        with tempfile.TemporaryDirectory() as temp:
            catalog = Path(temp) / 'catalog.json'
            catalog.write_text(json.dumps({'tariffs': ['Орбита-Люкс', 'Пульс']}), encoding='utf-8')
            facts, decisions = recover_facts(
                [old], call('У меня тариф Орбита-Люкс, сейчас подключён.'), str(catalog))
        current = [item for item in facts if item['attribute'] == 'tariff'
                   and not item.get('valid_to')]
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]['value'], 'Орбита-Люкс')
        self.assertTrue(decisions)

    def test_missing_current_status_is_recovered(self):
        old = {'object': {'name': 'заявка 42', 'type': 'ticket', 'id': None},
               'attribute': 'ticket_status', 'value': 'принята', 'polarity': 1,
               'valid_from': '2026-03-10T00:00:00', 'valid_to': '2026-03-11T00:00:00',
               'source': {'turn_idx': 0}, 'speaker': 'operator'}
        facts, decisions = recover_facts(
            [old], call('Сейчас статус по заявке 42 — в работе.'), None)
        statuses = [fact for fact in facts if fact['attribute'] == 'ticket_status']
        self.assertEqual(len(statuses), 2)
        self.assertEqual(statuses[-1]['value'], 'в работе')
        self.assertEqual(statuses[-1]['object']['name'], 'заявка 42')
        self.assertEqual(decisions[0]['reason'], 'missing_current_status')

    def test_current_status_is_not_duplicated(self):
        current = {'object': {'name': 'заявка 42', 'type': 'ticket', 'id': None},
                   'attribute': 'ticket_status', 'value': 'в работе', 'polarity': 1,
                   'valid_from': '2026-03-11T00:00:00', 'valid_to': None,
                   'source': {'turn_idx': 0}, 'speaker': 'operator'}
        facts, decisions = recover_facts(
            [current], call('Сейчас статус по заявке 42 — в работе.'), None)
        self.assertEqual(facts, [current])
        self.assertFalse(decisions)

    def test_status_without_current_context_is_not_invented(self):
        old = {'object': {'name': 'заявка 42', 'type': 'ticket', 'id': None},
               'attribute': 'ticket_status', 'value': 'принята', 'polarity': 1,
               'valid_from': '2026-03-10T00:00:00', 'valid_to': '2026-03-11T00:00:00',
               'source': {'turn_idx': 0}, 'speaker': 'operator'}
        facts, decisions = recover_facts([old], call('Заявка 42 была в работе.'), None)
        self.assertEqual(facts, [old])
        self.assertFalse(decisions)


if __name__ == '__main__':
    unittest.main()
