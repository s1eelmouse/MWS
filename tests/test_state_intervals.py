"""State boundaries must follow the named event, rather than a nearby date."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from schema import CallInput, Facts
from temporal_case_v2 import ValueLayers, repair_case_v2_dates

CATALOG = {'tariffs': ['Смарт', 'Пульс'], 'symptoms': ['нет гудков', 'слабый сигнал вай-фай'],
           'statuses': ['принята', 'в работе', 'выполнена'], 'agreements': []}
START = '2026-11-13T12:00:00+03:00'


def fact(attr, value, **extra):
    return {'object': {'name': 'объект', 'type': 'service', 'id': None},
            'attribute': attr, 'value': value, 'polarity': 1,
            'valid_from': '2026-10-01T12:00:00+03:00', 'valid_to': None,
            'source': {'turn_idx': 0}, **extra}


def repair(item, text, *following):
    call = CallInput(call_id='synthetic:interval', started_at=START,
                     turns=[{'idx': i, 'speaker': 'client', 'text': t}
                            for i, t in enumerate((text, *following))])
    result, audit = repair_case_v2_dates([item], call, ValueLayers(CATALOG))
    Facts.model_validate({'facts': result})
    return result[0], audit[0]


class StateIntervalsTest(unittest.TestCase):
    def test_tariff_activation_and_since(self):
        for text in ('Вчера перешла на тариф Смарт.', 'Со вторника у меня тариф Смарт.'):
            item, audit = repair(fact('tariff', 'Смарт'), text)
            self.assertEqual(item['valid_from'][:10], '2026-11-12' if 'Вчера' in text else '2026-11-10')
            self.assertTrue(audit['evidence'])

    def test_tariff_until_changes_end_only(self):
        before = fact('tariff', 'Смарт')
        item, _ = repair(before, 'Тариф Смарт действовал до вчерашнего дня.')
        self.assertEqual(item['valid_from'], before['valid_from'])
        self.assertEqual(item['valid_to'][:10], '2026-11-12')

    def test_explicit_transfer_dates_both_tariffs(self):
        old, _ = repair(fact('tariff', 'Пульс'), 'Вчера перешла с Пульса на Смарт.')
        new, _ = repair(fact('tariff', 'Смарт'), 'Вчера перешла с Пульса на Смарт.')
        self.assertEqual(old['valid_to'][:10], '2026-11-12')
        self.assertEqual(new['valid_from'][:10], '2026-11-12')

    def test_plus_tariff_is_not_the_same_value(self):
        before = fact('tariff', 'Смарт')
        self.assertEqual(repair(before, 'Вчера подключили Смарт+.')[0], before)

    def test_tariff_observation_proposal_and_question_are_not_activation(self):
        before = fact('tariff', 'Смарт')
        for text in ('Вчера смотрела тариф Смарт.', 'Хочу завтра подключить тариф Смарт.',
                     'Вчера подключили тариф Смарт?', 'Возможно, вчера подключили Смарт.',
                     'Вчера тариф Смарт не был подключен.'):
            self.assertEqual(repair(copy.deepcopy(before), text)[0], before, text)

    def test_unrelated_date_does_not_date_a_tariff(self):
        before = fact('tariff', 'Смарт')
        self.assertEqual(repair(before, 'У меня тариф Смарт. Вчера перестал работать интернет.')[0], before)
        self.assertEqual(repair(before, 'У меня тариф Смарт.', 'Вчера перезагрузила роутер.')[0], before)

    def test_month_period_does_not_become_an_arbitrary_day(self):
        before = fact('tariff', 'Смарт')
        self.assertEqual(repair(before, 'Месяц назад перешла на тариф Смарт.')[0], before)

    def test_closed_symptom_is_not_reopened_by_an_unrelated_date(self):
        before = fact('service_state', 'нет гудков', valid_to='2026-11-01T00:00:00+03:00')
        self.assertEqual(repair(before, 'Нет гудков было раньше.', 'Вчера заявку приняли.')[0], before)

    def test_explicit_symptom_end_keeps_start(self):
        before = fact('service_state', 'нет гудков')
        item, _ = repair(before, 'Нет гудков было до вчерашнего дня.')
        self.assertEqual(item['valid_from'], before['valid_from'])
        self.assertEqual(item['valid_to'][:10], '2026-11-12')

    def test_ongoing_symptom_reopens_from_since_date(self):
        before = fact('service_state', 'нет гудков', valid_to='2026-11-01T00:00:00+03:00')
        item, _ = repair(before, 'Со вторника нет гудков.')
        self.assertEqual(item['valid_from'][:10], '2026-11-10')
        self.assertIsNone(item['valid_to'])


class TicketEvidenceTest(unittest.TestCase):
    def run_case(self, texts, items):
        call = CallInput(call_id='synthetic:tickets', started_at=START,
                         turns=[{'idx': i, 'speaker': 'operator', 'text': t}
                                for i, t in enumerate(texts)])
        result, audit = repair_case_v2_dates(items, call, ValueLayers(CATALOG))
        Facts.model_validate({'facts': result})
        return result, audit

    def ticket(self, value, number='42', **extra):
        return fact('ticket_status', value,
                    object={'name': 'заявка ' + number, 'type': 'ticket', 'id': None}, **extra)

    def test_explicit_transition_gives_endpoint_and_next_start(self):
        items = [self.ticket('принята'), self.ticket('в работе')]
        output, audit = self.run_case([
            'Заявка 42 создана.', 'До вчера она еще была принята, а сейчас в работе.'], items)
        self.assertEqual(output[0]['valid_to'][:10], '2026-11-12')
        self.assertEqual(output[1]['valid_from'][:10], '2026-11-12')
        self.assertTrue(any(e['source_turn_idx'] == 1 for e in audit[1]['evidence']))

    def test_explicit_other_turn_repairs_date_of_the_same_ticket(self):
        output, _ = self.run_case(['Заявка 42 в работе.', 'Заявка 42 со вторника в работе.'],
                                   [self.ticket('в работе')])
        self.assertEqual(output[0]['valid_from'][:10], '2026-11-10')

    def test_unnumbered_evidence_cannot_cross_multiple_tickets(self):
        items = [self.ticket('в работе')]
        output, _ = self.run_case(['Заявка 42 зарегистрирована, заявка 99 закрыта.',
                                   'Со вторника в работе.'], items)
        self.assertEqual(output, items)

    def test_other_number_cannot_date_mine(self):
        items = [self.ticket('в работе')]
        output, _ = self.run_case(['Заявка 42 зарегистрирована.',
                                   'Заявка 99 со вторника в работе.'], items)
        self.assertEqual(output, items)

    def test_conflicting_explicit_dates_are_not_resolved_by_first_or_last(self):
        items = [self.ticket('в работе')]
        output, _ = self.run_case(['Заявка 42 зарегистрирована.',
                                   'Заявка 42 со вторника в работе.',
                                   'Заявка 42 со среды в работе.'], items)
        self.assertEqual(output, items)

    def test_model_dates_do_not_complete_a_chain(self):
        items = [self.ticket('принята'), self.ticket('в работе', valid_from='2026-11-10T00:00:00+03:00')]
        output, _ = self.run_case(['Заявка 42.'], items)
        self.assertEqual(output, items)

    def test_questions_and_negation_do_not_supply_evidence(self):
        items = [self.ticket('в работе')]
        for text in ('Заявка 42 вчера была в работе?', 'Заявка 42 вчера не была в работе.',
                     'Возможно, заявка 42 со вторника в работе.'):
            output, _ = self.run_case(['Заявка 42.', text], items)
            self.assertEqual(output, items)

    def test_visit_date_and_phone_call_date_do_not_date_a_status(self):
        items = [self.ticket('в работе')]
        for text in ('Выезд специалиста в воскресенье, заявка 42 остается в работе.',
                     'В четверг был контрольный звонок, заявка 42 в работе.'):
            output, _ = self.run_case(['Заявка 42 в работе.', text], items)
            self.assertEqual(output, items)

    def test_still_normal_does_not_prove_end_of_normal_state(self):
        before = fact('service_state', 'работает нормально')
        self.assertEqual(repair(before, 'Еще вчера все работало нормально.')[0], before)

    def test_beginning_and_end_are_applied_together(self):
        before = fact('service_state', 'нет гудков')
        item, _ = repair(before, 'Со вторника до вчерашнего дня нет гудков.')
        self.assertEqual((item['valid_from'][:10], item['valid_to'][:10]), ('2026-11-10', '2026-11-12'))

    def test_timezone_immutability_and_idempotence(self):
        before = fact('tariff', 'Смарт')
        original = copy.deepcopy(before)
        item, _ = repair(before, 'Вчера перешла на тариф Смарт.')
        self.assertEqual(before, original)
        self.assertTrue(item['valid_from'].endswith('+03:00'))
        self.assertEqual(repair(item, 'Вчера перешла на тариф Смарт.')[0], item)

    def test_negative_fact_is_preserved(self):
        before = fact('tariff', 'Смарт', polarity=-1)
        self.assertEqual(repair(before, 'Вчера подключили Смарт.')[0], before)

    def test_question_elsewhere_does_not_erase_an_asserted_onset(self):
        before = fact('service_state', 'нет гудков')
        item, _ = repair(before, 'Что происходит? Со вторника все пропало! Нет гудков!')
        self.assertEqual(item['valid_from'][:10], '2026-11-10')

    def test_source_question_does_not_establish_a_new_symptom(self):
        before = fact('service_state', 'нет гудков')
        self.assertEqual(repair(before, 'Вчера появились нет гудков?')[0], before)

    def test_previous_named_onset_can_explain_a_confirmation(self):
        before = fact('service_state', 'нет гудков', source={'turn_idx': 1})
        item, _ = repair(before, 'Все работало нормально, а два дня назад пропало: нет гудков.',
                          'Да, нет гудков.')
        self.assertEqual(item['valid_from'][:10], '2026-11-11')

    def test_until_for_normal_is_not_until_for_the_following_symptom(self):
        before = fact('service_state', 'нет гудков')
        item, _ = repair(before, 'До вторника все работало нормально, со вторника нет гудков.')
        self.assertEqual(item['valid_from'][:10], '2026-11-10')
        self.assertIsNone(item['valid_to'])


if __name__ == '__main__':
    unittest.main()
