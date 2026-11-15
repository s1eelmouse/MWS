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
