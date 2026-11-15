"""Deterministic review queue for disputed facts; no gold crosses the boundary."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from review_queue import build_queue
from schema import as_call


def call(*turns, cid='synthetic:review', started='2026-03-11T10:00:00'):
    return as_call({
        'call_id': cid, 'started_at': started,
        'turns': [{'idx': index, 'speaker': speaker, 'text': text}
                  for index, (speaker, text) in enumerate(turns)],
    })


def fact(value, *, attribute='tariff', turn_idx=0, speaker='client',
         valid_from=None, valid_to=None, name='Марина Сергеевна', kind='client'):
    return {
        'object': {'name': name, 'type': kind, 'id': None},
        'attribute': attribute, 'value': value, 'polarity': 1,
        'valid_from': valid_from, 'valid_to': valid_to,
        'source': {'turn_idx': turn_idx}, 'speaker': speaker,
    }


def pred(call_obj, *facts, error=None):
    row = {'call_id': call_obj.call_id, 'facts': list(facts)}
    if error:
        row['error'] = error
    return row


class ReviewQueueTest(unittest.TestCase):
    def test_supported_current_fact_is_not_queued(self):
        c = call(('client', 'У меня тариф Пятёрочка, всё работает.'))
        queue, summary = build_queue([c], [pred(c, fact('Пятёрочка'))])
        self.assertEqual(queue, [])
        self.assertEqual(summary['facts'], 1)
        self.assertEqual(summary['queued_facts'], 0)

    def test_low_support_is_queued(self):
        c = call(('client', 'У меня тариф Пятёрочка.'))
        queue, summary = build_queue([c], [pred(c, fact('Орбита-Люкс'))])
        self.assertEqual(queue[0]['reasons'], ['low_value_support'])
        self.assertEqual(queue[0]['priority'], 2)
        self.assertEqual(summary['by_reason']['low_value_support'], 1)

    def test_missing_source_is_high_priority(self):
        c = call(('client', 'У меня тариф Пятёрочка.'))
        queue, _ = build_queue([c], [pred(c, fact('Пятёрочка', turn_idx=99))])
        self.assertEqual(queue[0]['reasons'], ['source_turn_missing'])
        self.assertEqual(queue[0]['priority'], 3)
        self.assertIsNone(queue[0]['source'])

    def test_speaker_mismatch_is_high_priority(self):
        c = call(('client', 'У меня тариф Пятёрочка.'))
        queue, _ = build_queue([c], [pred(c, fact('Пятёрочка', speaker='operator'))])
        self.assertIn('source_speaker_mismatch', queue[0]['reasons'])
        self.assertEqual(queue[0]['priority'], 3)

    def test_competing_current_values_are_queued_together(self):
        c = call(('client', 'У меня тариф Пятёрочка. Раньше был Орбита-Люкс.'))
        first = fact('Пятёрочка')
        second = fact('Орбита-Люкс')
        queue, summary = build_queue([c], [pred(c, first, second)])
        self.assertEqual([item['fact_index'] for item in queue], [0, 1])
        self.assertTrue(all('competing_current_values' in item['reasons'] for item in queue))
        self.assertEqual(summary['by_reason']['competing_current_values'], 2)

    def test_long_agreement_is_queued(self):
        c = call(('client', 'Оператор перезвонит завтра и проверит линию.'),
                 ('operator', 'Хорошо.'))
        value = ('оператор перезвонит завтра проверит линию оборудование '
                 'кабель роутер телевизор приставка сигнал')
        queue, _ = build_queue([c], [pred(c, fact(value, attribute='agreement'))])
        self.assertIn('long_agreement', queue[0]['reasons'])

    def test_date_without_source_cue_is_queued(self):
        c = call(('client', 'У меня тариф Пятёрочка подключён.'))
        queue, _ = build_queue([c], [pred(c, fact(
            'Пятёрочка', valid_from='2026-01-01T00:00:00'))])
        self.assertIn('date_not_supported_by_source', queue[0]['reasons'])

    def test_offer_context_is_queued(self):
        c = call(('operator', 'Можете предложить тариф Пульс со скидкой?'),
                 ('client', 'Да, интересно.'))
        queue, _ = build_queue([c], [pred(c, fact('Пульс', attribute='tariff', speaker='operator'))])
        self.assertIn('possible_offer_or_hypothesis', queue[0]['reasons'])

    def test_gold_facts_are_not_used(self):
        c = call(('client', 'У меня тариф Пятёрочка.'))
        gold = {**fact('Секретный-Голд'), 'source': {'turn_idx': 0}}
        row = {**c.model_dump(mode='json'), 'facts': [gold]}
        c_with_gold = as_call(row)
        queue, summary = build_queue([c_with_gold], [pred(c_with_gold, fact('Пятёрочка'))])
        self.assertEqual(queue, [])
        self.assertEqual(summary['facts'], 1)

    def test_failed_prediction_is_counted_not_queued(self):
        c = call(('client', 'У меня тариф Пятёрочка.'))
        queue, summary = build_queue([c], [pred(c, error='invalid_json')])
        self.assertEqual(queue, [])
        self.assertEqual(summary['failed_calls'], 1)

    def test_max_items_keeps_summary_and_selects_top(self):
        c = call(('client', 'У меня тариф Пятёрочка.'))
        queue, summary = build_queue(
            [c], [pred(c, fact('Орбита-Люкс'), fact('Пульс'))], max_items=1)
        self.assertEqual(len(queue), 1)
        self.assertEqual(summary['queued_facts'], 2)
        self.assertEqual(summary['selected_facts'], 1)
        self.assertEqual(summary['max_items'], 1)
        self.assertEqual(queue[0]['priority'], 3)
        self.assertIn('competing_current_values', queue[0]['reasons'])


if __name__ == '__main__':
    unittest.main()
