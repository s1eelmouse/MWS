"""Task-mode date binding: explicit cues, action convention, interval safety, series carry."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess_dates import process_predictions
from prediction import parse_prediction
from schema import as_call
from temporal import normalize_fact_dates, repair_task_dates
from test_temporal import fixture


class TaskModeTest(unittest.TestCase):
    def test_status_uses_explicit_past_cue(self):
        call, fact = fixture('Вчера заявку приняли.')
        result, decision = normalize_fact_dates([fact], call, mode='task')
        self.assertEqual(result[0]['valid_from'], '2026-02-28T00:00:00')
        self.assertEqual(decision[0]['reason'], 'status_explicit_time')

    def test_status_endpoint_marks_valid_to_not_start(self):
        call, fact = fixture('До вчерашнего дня заявка была принята.',
                             valid_from='2026-01-02T00:00:00')
        result, _ = normalize_fact_dates([fact], call, mode='task')
        self.assertEqual(result[0]['valid_from'], '2026-01-02T00:00:00')
        self.assertEqual(result[0]['valid_to'], '2026-02-28T00:00:00')

    def test_action_uses_cue_then_call_time(self):
        call, fact = fixture('Четыре дня назад перезагрузила роутер.',
                             attribute='action', value='перезагрузка роутера',
                             valid_from='2026-01-01T00:00:00', start='2026-03-01T12:00:00')
        result, _ = normalize_fact_dates([fact], call, mode='task')
        self.assertEqual(result[0]['valid_from'], '2026-02-25T00:00:00')

        call, fact = fixture('Перезагрузка роутера не помогла.',
                             attribute='action', value='перезагрузка роутера',
                             valid_from='2026-01-01T00:00:00', start='2026-03-01T12:00:00')
        result, decision = normalize_fact_dates([fact], call, mode='task')
        self.assertEqual(result[0]['valid_from'], call.started_at)
        self.assertEqual(decision[0]['reason'], 'action_assertion_time')

    def test_agreement_uses_future_cue(self):
        call, fact = fixture('Завтра мастер приедет.', attribute='agreement',
                             value='выезд специалиста', valid_from='2026-02-27T00:00:00')
        result, _ = normalize_fact_dates([fact], call, mode='task')
        self.assertEqual(result[0]['valid_from'], '2026-03-02T00:00:00')

    def test_negative_fact_is_not_redated(self):
        call, fact = fixture('Вчера заявку приняли.')
        fact['polarity'] = -1
        self.assertEqual(normalize_fact_dates([fact], call, mode='task')[0], [fact])

    def test_interval_guard_keeps_closed_action(self):
        call, fact = fixture('Заявка обработана.', attribute='action',
                             value='обработка заявки', valid_from='2026-02-01T00:00:00',
                             valid_to='2026-02-20T00:00:00', start='2026-03-01T12:00:00')
        result, decision = repair_task_dates(fact, call)
        self.assertEqual(result, fact)
        self.assertEqual(decision['reason'], 'interval_conflict')

    def test_invalid_model_date_is_left_to_the_sanitizer(self):
        call, fact = fixture('Вчера заявку приняли.', valid_from='2026-02-29T00:00:00')
        result, decision = repair_task_dates(fact, call)
        self.assertEqual(result['valid_from'], '2026-02-29T00:00:00')
        self.assertEqual(decision['reason'], 'invalid_model_date_left_for_sanitizer')
        parsed, error, summary = parse_prediction(json.dumps({'facts': [fact]}), call,
                                                 temporal_mode='task')
        self.assertIsNone(error)
        self.assertIsNone(parsed.facts[0].valid_from)
        self.assertEqual(summary.get('invalid_dates_dropped'), 1)

    def test_series_carries_accepted_status_only_within_account(self):
        first = as_call({'call_id': 'series:0', 'started_at': '2026-03-01T12:00:00',
                         'series': {'account_id': 'account-a', 'call_index': 0},
                         'turns': [{'idx': 42, 'speaker': 'operator', 'text': 'Заявка принята.'}]})
        second = as_call({'call_id': 'series:1', 'started_at': '2026-03-05T12:00:00',
                          'series': {'account_id': 'account-a', 'call_index': 1},
                          'turns': [{'idx': 42, 'speaker': 'operator', 'text': 'Заявка принята.'}]})
        other = as_call({'call_id': 'series:2', 'started_at': '2026-03-06T12:00:00',
                         'series': {'account_id': 'account-b', 'call_index': 0},
                         'turns': [{'idx': 42, 'speaker': 'operator', 'text': 'Заявка принята.'}]})
        base = {'object': {'name': 'заявка', 'type': 'ticket', 'id': 'ticket:1'},
                'attribute': 'ticket_status', 'value': 'принята', 'polarity': 1,
                'valid_from': '2026-01-05T00:00:00', 'valid_to': None,
                'source': {'turn_idx': 42}, 'speaker': 'operator'}
        second_fact = {**copy.deepcopy(base), 'valid_from': '2026-02-01T00:00:00'}
        other_fact = {**copy.deepcopy(base), 'valid_from': '2026-03-01T00:00:00'}
        output, audit, counts = process_predictions(
            [first, second, other],
            [{'call_id': 'series:0', 'facts': [base]},
             {'call_id': 'series:1', 'facts': [second_fact]},
             {'call_id': 'series:2', 'facts': [other_fact]}],
            mode='task', allow_series_history=True)
        by_id = {row['call_id']: row for row in output}
        self.assertEqual(by_id['series:1']['facts'][0]['valid_from'], '2026-01-05T00:00:00')
        self.assertEqual(by_id['series:2']['facts'][0]['valid_from'], '2026-03-01T00:00:00')
        self.assertEqual(counts.get('series_accepted_carried'), 1)
        self.assertTrue(any(item.get('component') == 'series' for item in audit))

    def test_mode_off_preserves_model_dates(self):
        call, fact = fixture('Вчера заявку приняли.')
        output, _, _ = process_predictions(
            [call], [{'call_id': call.call_id, 'facts': [fact]}], mode='off')
        self.assertEqual(output[0]['facts'][0], fact)


if __name__ == '__main__':
    unittest.main()
