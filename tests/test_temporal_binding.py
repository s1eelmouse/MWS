"""Event/boundary binding and short replies, independent of val examples."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import as_call
from temporal import normalize_fact_dates
from temporal_binding import repair_bound_dates
from test_temporal import fixture


class TemporalBindingTest(unittest.TestCase):
    def test_both_boundaries_and_successor_from_are_bound_to_own_events(self):
        call, accepted = fixture('Позавчера заявку приняли, а вчера перевели в работу.',
                                  valid_from='2026-02-20T00:00:00')
        working = {**copy.deepcopy(accepted), 'value': 'в работе'}
        result, decisions = normalize_fact_dates([accepted, working], call, mode='binding')
        self.assertEqual(result[0]['valid_from'], '2026-02-27T00:00:00')
        self.assertEqual(result[0]['valid_to'], '2026-02-28T00:00:00')
        self.assertEqual(result[1]['valid_from'], '2026-02-28T00:00:00')
        self.assertTrue(any(e.get('binding') == 'explicit_status_transition' for e in decisions[0]['evidence']))

    def test_two_dates_of_same_status_have_different_roles(self):
        call, fact = fixture('Со вторника до пятницы заявка была принята.',
                              valid_from='2026-02-01T00:00:00')
        item, _ = repair_bound_dates(fact, call)
        self.assertEqual(item['valid_from'], '2026-02-24T00:00:00')
        self.assertEqual(item['valid_to'], '2026-02-27T00:00:00')

    def test_explicit_end_wins_over_inferred_transition(self):
        call, fact = fixture('Заявка была принята до пятницы, а вчера перевели в работу.',
                              valid_from='2026-02-01T00:00:00')
        item, _ = repair_bound_dates(fact, call)
        self.assertEqual(item['valid_to'], '2026-02-27T00:00:00')

    def test_comma_separated_time_adjunct_is_not_lost(self):
        for text in ['Позавчера, её приняли.', 'Её приняли, позавчера.']:
            call, fact = fixture(text, valid_from='2026-02-20T00:00:00')
            self.assertEqual(repair_bound_dates(fact, call)[0]['valid_from'], '2026-02-27T00:00:00')

    def test_unrelated_action_time_does_not_date_ticket(self):
        call, fact = fixture('Заявка принята, а вчера перезагрузил роутер.')
        self.assertEqual(repair_bound_dates(fact, call)[0], fact)

    def test_partial_period_is_not_replaced_by_other_status_date(self):
        call, fact = fixture('На прошлой неделе принята и вчера в работе.')
        self.assertEqual(repair_bound_dates(fact, call)[0], fact)

    def test_conflicting_repeated_status_is_abstention(self):
        call, fact = fixture('Вчера заявку приняли. Позавчера заявку приняли.')
        item, event = repair_bound_dates(fact, call)
        self.assertEqual(item, fact)
        self.assertEqual(event['reason'], 'conflicting_event_times')

    def test_short_confirmation_uses_only_previous_turns(self):
        call, fact = fixture('Да, всё верно.')
        row = call.model_dump()
        row['turns'] = [
            {'idx': 11, 'speaker': 'operator', 'text': 'Вчера заявку приняли?'},
            {'idx': 42, 'speaker': 'client', 'text': 'Да, всё верно.'},
            {'idx': 90, 'speaker': 'operator', 'text': 'Позавчера заявку приняли.'},
        ]
        fact['speaker'] = 'client'
        item, event = repair_bound_dates(fact, as_call(row))
        self.assertEqual(item['valid_from'], '2026-02-28T00:00:00')
        self.assertEqual(item['source'], fact['source'])
        self.assertEqual(event['evidence'][0]['turn_idx'], 11)

    def test_temporal_answer_inherits_question_event(self):
        call, fact = fixture('Позавчера.')
        row = call.model_dump()
        row['turns'].insert(0, {'idx': 12, 'speaker': 'operator', 'text': 'Когда заявку приняли?'})
        item, _ = repair_bound_dates(fact, as_call(row))
        self.assertEqual(item['valid_from'], '2026-02-27T00:00:00')

    def test_rejection_does_not_confirm_previous_date(self):
        call, fact = fixture('Нет, неверно.')
        row = call.model_dump()
        row['turns'].insert(0, {'idx': 12, 'speaker': 'operator', 'text': 'Вчера заявку приняли?'})
        self.assertEqual(repair_bound_dates(fact, as_call(row))[0], fact)

    def test_ticket_reference_mismatch_and_multiple_tickets_are_ambiguous(self):
        for text in ['Вчера заявку 9999 приняли.', 'Вчера заявку 1111 приняли, а заявку 2222 перевели в работу.']:
            call, fact = fixture(text)
            fact['object']['name'] = 'заявка 1234'
            self.assertEqual(repair_bound_dates(fact, call)[0], fact)

    def test_questions_hypotheses_and_negated_cues_do_not_supply_dates(self):
        for text in ['Вчера заявку приняли?', 'Если вчера заявку приняли.', 'Заявку приняли не вчера.']:
            call, fact = fixture(text)
            self.assertEqual(repair_bound_dates(fact, call)[0], fact)

    def test_historical_start_is_not_guessed_from_end(self):
        call, fact = fixture('До вчерашнего дня заявка была принята.', valid_from='2025-12-01T00:00:00')
        item, _ = repair_bound_dates(fact, call)
        self.assertEqual(item['valid_from'], fact['valid_from'])
        self.assertEqual(item['valid_to'], '2026-02-28T00:00:00')

    def test_timezone_is_preserved_and_reapplication_is_idempotent(self):
        call, fact = fixture('Вчера заявку приняли.', start='2026-03-01T12:00:00+03:00',
                              valid_from='2026-02-20T00:00:00+03:00')
        item, _ = repair_bound_dates(fact, call)
        self.assertEqual(item['valid_from'], '2026-02-28T00:00:00+03:00')
        self.assertEqual(repair_bound_dates(item, call)[0], item)

    def test_inputs_and_non_date_fields_are_preserved(self):
        call, fact = fixture('Вчера заявку приняли.')
        original = copy.deepcopy(fact)
        item, _ = repair_bound_dates(fact, call)
        for key in set(fact) - {'valid_from', 'valid_to'}:
            self.assertEqual(item[key], fact[key])
        self.assertEqual(fact, original)

    def test_closing_ticket_does_not_end_completed_status(self):
        call, fact = fixture('Сегодня статус выполнена мастер отчитался сегодня её закрыли.')
        fact['value'] = 'выполнена'
        item, _ = repair_bound_dates(fact, call)
        self.assertEqual(item['valid_to'], fact['valid_to'])

    def test_observation_is_not_status_onset(self):
        call, fact = fixture('Ещё позавчера она была принята.', valid_from='2026-02-15T00:00:00')
        self.assertEqual(repair_bound_dates(fact, call)[0], fact)

    def test_repeated_endpoint_does_not_become_start(self):
        call, fact = fixture('Статус принята держался до вчерашнего дня вчера он сменился.',
                              valid_from='2026-02-15T00:00:00')
        item, _ = repair_bound_dates(fact, call)
        self.assertEqual(item['valid_from'], fact['valid_from'])
        self.assertEqual(item['valid_to'], '2026-02-28T00:00:00')

    def test_future_notification_does_not_date_status(self):
        call, fact = fixture('Назначен выезд специалиста подтверждение придёт в субботу.')
        fact['value'] = 'назначен выезд специалиста'
        self.assertEqual(repair_bound_dates(fact, call)[0], fact)


if __name__ == '__main__':
    unittest.main()
