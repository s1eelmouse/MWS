"""Independent status assertions: no organizer val examples or IDs."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fact_recovery import recover_facts
from schema import as_call, Facts, source_problems


def call(*texts, speaker='operator', started='2026-07-08T12:00:00'):
    return as_call({'call_id': 'synthetic:status-scope', 'started_at': started,
                    'turns': [{'idx': i*3, 'speaker': speaker, 'text': text} for i,text in enumerate(texts)]})


def status(value='принята', *, number=735, start='2026-07-07T00:00:00', end=None, idx=0):
    return {'object': {'name': 'заявка '+str(number), 'type': 'ticket', 'id': None},
            'attribute': 'ticket_status', 'value': value, 'polarity': 1,
            'valid_from': start, 'valid_to': end, 'source': {'turn_idx':idx}, 'speaker':'operator'}


class TicketAssertionsTest(unittest.TestCase):
    def run_policy(self, facts, c):
        original = copy.deepcopy(facts)
        result, audit = recover_facts(facts, c)
        self.assertEqual(facts, original, 'Caller-owned facts must not be mutated')
        parsed = Facts.model_validate({'facts': result})
        self.assertEqual(source_problems(parsed, c), [])
        self.assertEqual(recover_facts(result,c)[0], result, 'Recovery must be idempotent')
        return result,audit

    def test_today_acceptance_is_not_rejected_by_unrelated_future_refund(self):
        c=call('Заявка 735 принята сегодня, возврат будет через девять дней.')
        result,audit=self.run_policy([],c)
        self.assertEqual([f['value'] for f in result],['принята'])
        self.assertEqual(result[0]['valid_from'],'2026-07-08T00:00:00')
        self.assertEqual(audit[-1]['action'],'add')

    def test_unrelated_negation_does_not_negate_acceptance(self):
        result,_=self.run_policy([],call('Заявка 735 принята сегодня, интернет не заработал.'))
        self.assertEqual(result[0]['value'],'принята')

    def test_future_visit_does_not_date_the_status(self):
        result,_=self.run_policy([],call('Заявка 735 принята сегодня мастер приедет через два дня.'))
        self.assertEqual(result[0]['valid_from'],'2026-07-08T00:00:00')

    def test_closed_work_is_a_current_completed_status(self):
        result,_=self.run_policy([],call('Сейчас заявка 735 выполнена, работы закрыты.'))
        self.assertEqual(result[0]['value'],'выполнена')
        self.assertIsNone(result[0]['valid_to'])

    def test_explicit_same_day_transition_closes_old_state(self):
        c=call('Заявка 735 принята.', 'Сегодня статус заявки 735 сменился: сейчас в работе.')
        result,audit=self.run_policy([status()],c)
        self.assertEqual(result[0]['valid_to'],'2026-07-08T00:00:00')
        self.assertEqual(result[-1]['value'],'в работе')
        self.assertEqual(result[-1]['source']['turn_idx'],3)
        self.assertTrue(any(a['action']=='close' for a in audit))

    def test_supported_history_without_boundary_is_retained(self):
        c=call('Заявка 735 была принята.', 'Сейчас заявка 735 в работе.')
        result,audit=self.run_policy([status()],c)
        self.assertEqual(result,[status()])
        self.assertEqual(audit[-1]['action'],'abstained')

    def test_explicit_assertion_replaces_unsupported_model_status(self):
        c=call('Сейчас заявка 735 в работе.')
        result,audit=self.run_policy([status('выполнена')],c)
        self.assertEqual([f['value'] for f in result],['в работе'])
        self.assertTrue(any(a['action']=='remove' for a in audit))

    def test_number_and_later_source_cannot_be_overridden(self):
        c=call('Сейчас заявка 735 в работе.', 'Заявка 735 принята.')
        result,_=self.run_policy([status(idx=3)],c)
        self.assertEqual(result,[status(idx=3)])
        c=call('Сейчас заявка 736 в работе.')
        result,_=self.run_policy([status()],c)
        self.assertEqual(result[0],status())
        self.assertEqual(result[1]['object']['name'],'заявка 736')

    def test_two_numbers_without_local_subject_do_not_borrow_identity(self):
        c=call('Обсуждаем заявки 735 и 736.', 'Сейчас статус в работе.')
        result,_=self.run_policy([status(),status(number=736)],c)
        self.assertEqual(len(result),2)

    def test_two_numbered_clauses_keep_their_subjects(self):
        c=call('Заявка 735 сейчас в работе, заявка 736 сегодня принята.')
        result,_=self.run_policy([],c)
        self.assertEqual({(f['object']['name'],f['value']) for f in result},
                         {('заявка 735','в работе'),('заявка 736','принята')})

    def test_negation_question_conditional_future_and_history_abstain(self):
        for text in ('Сейчас заявка 735 не выполнена.', 'Заявка 735 сегодня принята?',
                     'Если заявка 735 принята сегодня, возврат будет позже.',
                     'Завтра заявка 735 будет в работе.', 'Заявка 735 в работе будет завтра.',
                     'Вчера заявка 735 была в работе.', 'У заявки 735 раньше статус выполнена.',
                     'Сейчас проверю статус заявки 735, возможно выполнена.'):
            with self.subTest(text=text):
                result,_=self.run_policy([],call(text))
                self.assertEqual(result,[])

    def test_unresolved_negative_prediction_and_unknown_status_are_not_erased(self):
        negative={**status('выполнена'),'polarity':-1}
        result,audit=self.run_policy([negative],call('Сейчас заявка 735 выполнена.'))
        self.assertEqual(result,[negative])
        self.assertEqual(audit[-1]['action'],'abstained')
        free=status('идёт проверка счёта')
        result,_=self.run_policy([free],call('Сейчас заявка 735 в работе.'))
        self.assertEqual(result,[free])

    def test_timezone_of_explicit_day_is_preserved(self):
        result,_=self.run_policy([],call('Сегодня заявка 735 принята.',started='2026-07-08T12:00:00+03:00'))
        self.assertEqual(result[0]['valid_from'],'2026-07-08T00:00:00+03:00')

    def test_accepted_status_does_not_turn_into_an_unspecified_work_stage(self):
        result,_=self.run_policy([],call('Заявка 735 принята в работу сегодня.'))
        self.assertEqual([f['value'] for f in result],['принята'])

    def test_result_after_lookup_is_a_separate_assertion(self):
        c=call('Сейчас статус заявки 735 проверю… Да вот: заявка 735 сегодня принята.')
        result,_=self.run_policy([],c)
        self.assertEqual([f['value'] for f in result],['принята'])

    def test_status_nouns_and_historical_suffixes_are_not_present_assertions(self):
        for text in ('После выполнения заявки 735 настройки сохранены.',
                     'Статус заявки 735 «принята» у неё значился.',
                     'По заявке 735 статус «принята» действовал до вторника.',
                     'Сейчас должно быть указано: заявка 735 выполнена.',
                     'Какой сейчас статус заявки 735 выполнена',
                     'Скажите заявка 735 сейчас в работе',
                     'Сейчас заявка 735 выполнена не была.',
                     'Сейчас заявка 735 в работе ли'):
            with self.subTest(text=text):
                result,_=self.run_policy([],call(text))
                self.assertEqual(result,[])

    def test_unknown_transition_cannot_mix_timezone_conventions(self):
        old=status(start='2026-07-07T00:00:00+03:00')
        result,audit=self.run_policy([old],call('Заявка 735 принята.',
            'Сегодня статус заявки 735 сменился: сейчас в работе.'))
        self.assertEqual(result,[old])
        self.assertEqual(audit[-1]['action'],'abstained')

    def test_clipped_number_never_creates_a_second_ticket(self):
        old=status(number=7358)
        c=call('Заявка 7358 принята.', 'Сейчас статус заявки 735 — в работе.')
        result,_=self.run_policy([old],c)
        self.assertEqual(result,[old])


if __name__ == '__main__':
    unittest.main()
