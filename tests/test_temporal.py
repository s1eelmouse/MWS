"""Calendar boundaries, evidence roles and refusal of ambiguous date changes."""
import copy
import json
from pathlib import Path
import sys
import unittest
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from compare_temporal import compare_date_predictions
from postprocess_dates import process_predictions
from prediction import parse_prediction
from schema import as_call, contract_signature
from temporal import extract_time_expressions, normalize_fact_dates, repair_fact_dates, shift_calendar_months


def fixture(text, *, value='принята', attribute='ticket_status', start='2026-03-01T12:00:00',
            valid_from='2026-02-27T00:00:00', valid_to=None):
    call = as_call({'call_id': 'synthetic:time', 'started_at': start,
                    'turns': [{'idx': 42, 'speaker': 'operator', 'text': text}]})
    fact = {'object': {'name': 'заявка', 'type': 'ticket', 'id': 'synthetic:ticket'}, 'attribute': attribute,
            'value': value, 'polarity': 1, 'valid_from': valid_from, 'valid_to': valid_to,
            'source': {'turn_idx': 42}, 'speaker': 'operator'}
    return call, fact


class CalendarTest(unittest.TestCase):
    def first(self, text, anchor='2026-03-01T12:00:00', **kw):
        values = extract_time_expressions(text, anchor, **kw)
        self.assertEqual(len(values), 1, values)
        return values[0]

    def test_leap_and_non_leap_year(self):
        self.assertEqual(self.first('вчера').start, '2026-02-28T00:00:00')
        self.assertEqual(self.first('вчера', '2024-03-01T12:00:00').start, '2024-02-29T00:00:00')

    def test_year_boundary_and_timezone(self):
        self.assertEqual(self.first('завтра', '2026-12-31T23:30:00+03:00').start, '2027-01-01T00:00:00+03:00')

    def test_russian_compound_numbers_and_weeks(self):
        self.assertEqual(self.first('двадцать один день назад').start, '2026-02-08T00:00:00')
        self.assertEqual(self.first('через две недели').start, '2026-03-15T00:00:00')
        self.assertEqual(self.first('неделю назад').start, '2026-02-22T00:00:00')

    def test_hour_arithmetic_preserves_anchor_clock(self):
        self.assertEqual(self.first('два часа назад', '2026-03-01T01:30:00+03:00').start, '2026-02-28T23:30:00+03:00')

    def test_explicit_invalid_date_is_not_clamped(self):
        value = self.first('2026-02-29')
        self.assertFalse(value.certain)
        self.assertEqual(value.reason, 'invalid_calendar_date')

    def test_absolute_named_and_numeric_dates(self):
        self.assertEqual(self.first('28.02.2026').start, '2026-02-28T00:00:00')
        self.assertEqual(self.first('28 февраля 2026 года').start, '2026-02-28T00:00:00')

    def test_missing_year_requires_direction(self):
        self.assertFalse(self.first('31 декабря').certain)
        self.assertEqual(self.first('31 декабря', '2026-01-02T10:00:00', direction='past').start, '2025-12-31T00:00:00')

    def test_weekday_same_day_is_ambiguous(self):
        self.assertFalse(self.first('в воскресенье', direction='past').certain)
        self.assertEqual(self.first('со вторника').start, '2026-02-24T00:00:00')

    def test_week_and_month_are_ranges(self):
        week = self.first('на прошлой неделе')
        self.assertFalse(week.certain)
        self.assertEqual((week.start, week.end), ('2026-02-16T00:00:00', '2026-02-22T00:00:00'))
        month = self.first('в прошлом месяце')
        self.assertEqual(month.end, '2026-02-28T00:00:00')
        self.assertEqual(self.first('в прошлом месяце', '2024-03-01T12:00:00').end, '2024-02-29T00:00:00')

    def test_clock_composes_only_with_explicit_day(self):
        value = self.first('завтра в 14:30')
        self.assertEqual(value.start, '2026-03-02T14:30:00')
        self.assertFalse(self.first('в 14:30').certain)
        self.assertFalse(self.first('после двух').certain)

    def test_spoken_clock_requires_explicit_meridiem_or_policy(self):
        self.assertEqual(self.first('завтра в два часа дня').start, '2026-03-02T14:00:00')
        value = self.first('завтра после двух', daytime_hours=True)
        self.assertEqual(value.start, '2026-03-02T14:00:00')
        self.assertEqual(value.relation, 'after')

    def test_calendar_month_shift_does_not_invent_february_31(self):
        with self.assertRaises(ValueError):
            shift_calendar_months(datetime(2026, 1, 31), 1)
        self.assertEqual(shift_calendar_months(datetime(2026, 1, 15), -1), datetime(2025, 12, 15))

    def test_offsets_refer_to_original_text(self):
        text = 'Эээ, вчера её приняли.'
        expression = self.first(text)
        self.assertEqual(text[slice(*expression.span)], 'вчера')


class FactTimeTest(unittest.TestCase):
    def test_source_idx_is_not_list_position_and_inputs_are_not_mutated(self):
        call, fact = fixture('Вчера заявка была принята.')
        original = copy.deepcopy(fact)
        result, event = repair_fact_dates(fact, call)
        self.assertEqual(result['valid_from'], '2026-02-28T00:00:00')
        self.assertEqual(event['status'], 'changed')
        self.assertEqual(fact, original)
        again, event2 = repair_fact_dates(result, call)
        self.assertEqual(again, result)
        self.assertEqual(event2['reason'], 'already_consistent')

    def test_two_statuses_bind_their_own_clauses(self):
        call, accepted = fixture('Позавчера заявку приняли, а вчера она перешла в работу.')
        working = {**copy.deepcopy(accepted), 'value': 'в работе'}
        result, _ = normalize_fact_dates([accepted, working], call)
        self.assertEqual(result[0]['valid_from'], '2026-02-27T00:00:00')
        self.assertEqual(result[1]['valid_from'], '2026-02-28T00:00:00')

    def test_end_time_does_not_overwrite_unknown_start(self):
        call, fact = fixture('До вчерашнего дня заявка была принята.', valid_from='2026-01-02T00:00:00')
        result, _ = repair_fact_dates(fact, call)
        self.assertEqual(result['valid_from'], fact['valid_from'])
        self.assertEqual(result, fact)
        call, fact = fixture('Вчера статус «принята» закрылся.', valid_from='2026-01-02T00:00:00')
        result, _ = repair_fact_dates(fact, call)
        self.assertEqual(result['valid_from'], fact['valid_from'])
        self.assertEqual(result['valid_to'], '2026-02-28T00:00:00')

    def test_two_dates_in_one_clause_are_not_resolved_by_proximity(self):
        call, fact = fixture('Вчера и позавчера заявка была принята.')
        result, event = repair_fact_dates(fact, call)
        self.assertEqual(result, fact)
        self.assertEqual(event['reason'], 'multiple_or_partial_times')

    def test_service_observation_does_not_establish_start(self):
        call, fact = fixture('Вчера работало нормально.', attribute='service_state', value='работает нормально')
        result, event = repair_fact_dates(fact, call)
        self.assertEqual(result, fact)
        self.assertEqual(event['reason'], 'unknown_historical_start')

    def test_service_onset_is_explicit(self):
        call, fact = fixture('Вчера вдруг нет доступа в интернет.', attribute='service_state', value='нет доступа в интернет')
        result, _ = repair_fact_dates(fact, call)
        self.assertEqual(result['valid_from'], '2026-02-28T00:00:00')

    def test_numeric_date_is_not_split_into_sentences(self):
        call, fact = fixture('28.02.2026 заявка была принята.')
        result, _ = repair_fact_dates(fact, call)
        self.assertEqual(result['valid_from'], '2026-02-28T00:00:00')

    def test_agreement_and_action_policies_remain_unchanged(self):
        for attribute in ['agreement', 'action']:
            call, fact = fixture('Вчера её приняли, завтра приедет мастер.', attribute=attribute)
            result, _ = repair_fact_dates(fact, call)
            self.assertEqual(result, fact)

    def test_counterfactual_question_and_negative_fact_are_not_repaired(self):
        for text in ['Если вчера заявка была принята.', 'Вчера заявка была принята?']:
            call, fact = fixture(text)
            self.assertEqual(repair_fact_dates(fact, call)[0], fact)
        call, fact = fixture('Вчера заявка была принята.')
        fact['polarity'] = -1
        self.assertEqual(repair_fact_dates(fact, call)[0], fact)

    def test_two_objects_with_same_evidence_are_not_merged(self):
        call, first = fixture('Вчера заявку приняли.')
        second = copy.deepcopy(first)
        second['object']['name'] = 'другая заявка'
        results, events = normalize_fact_dates([first, second], call)
        self.assertEqual(results, [first, second])
        self.assertTrue(all(e['reason'] == 'multiple_objects_same_evidence' for e in events))

    def test_interval_conflict_is_not_fixed_by_swapping(self):
        call, fact = fixture('Вчера заявку приняли.', valid_to='2026-02-20T00:00:00')
        result, event = repair_fact_dates(fact, call)
        self.assertEqual(result, fact)
        self.assertEqual(event['reason'], 'interval_conflict')

    def test_invalid_model_date_is_recovered_before_strict_validation(self):
        call, fact = fixture('Вчера заявку приняли.', valid_from='2026-02-29T00:00:00')
        decisions = []
        parsed, error, summary = parse_prediction(json.dumps({'facts': [fact]}), call,
                                                 temporal_mode='conservative', temporal_decisions=decisions)
        self.assertIsNone(error)
        self.assertEqual(parsed.facts[0].valid_from, '2026-02-28T00:00:00')
        self.assertEqual(summary['temporal_dates_repaired'], 1)
        self.assertTrue(decisions)

    def test_offline_boundary_strips_gold_and_preserves_other_fields(self):
        call, fact = fixture('Вчера заявку приняли.')
        raw = {**call.model_dump(), 'facts': [{'value': 'gold poison', 'valid_from': '1900-01-01'}]}
        row = {'call_id': call.call_id, 'facts': [fact]}
        output, _, _ = process_predictions([as_call(raw)], [row])
        other = copy.deepcopy(output[0]['facts'][0])
        other['valid_from'] = fact['valid_from']
        self.assertEqual(other, fact)
        self.assertEqual(row['facts'][0], fact)

    def test_compare_counts_regressions_and_rejects_removed_facts(self):
        call, fact = fixture('Вчера заявку приняли.')
        gold = {'call_id': call.call_id, **call.model_dump(), 'facts': [{**fact, 'valid_from': '2026-02-28T00:00:00'}]}
        before = [{'call_id': call.call_id, 'facts': [fact]}]
        after, _, _ = process_predictions([call], before)
        report, _ = compare_date_predictions([gold], before, after)
        self.assertEqual(report['net_correct_pairs'], 1)
        reverse, _ = compare_date_predictions([gold], after, before)
        self.assertEqual(reverse['transitions']['regressed'], 1)
        with self.assertRaisesRegex(ValueError, 'non-date fields'):
            compare_date_predictions([gold], before, [{'call_id': call.call_id, 'facts': []}])

    def test_existing_model_contract_is_unchanged(self):
        self.assertEqual(contract_signature(), '3ee403f3b8645663828a1208bd93e81a087b234afeb566c7c36882da63c4a274')


if __name__ == '__main__':
    unittest.main()
