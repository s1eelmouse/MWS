"""Merged history policy: resource bounds, replay safety and session isolation."""
import copy
from concurrent.futures import ThreadPoolExecutor
import math
import unittest

from test_series_history import sample
from series_history import SeriesDateMemory
from postprocess_dates import process_predictions


class HistoryResourcesTest(unittest.TestCase):
    def first(self, account='a'):
        return sample('first', '2026-03-01T12:00:00', '2026-02-25T00:00:00', account=account)

    def second(self, account='a'):
        return sample('second', '2026-03-05T12:00:00', '2026-03-02T00:00:00', account=account)

    def test_ttl_account_and_byte_caps(self):
        clock = [0.0]
        memory = SeriesDateMemory(max_accounts=2, max_bytes=2500, account_bytes=1200,
                                  ttl_seconds=10, clock=lambda: clock[0])
        for i in range(20):
            call, row = self.first(str(i))
            memory.apply([call], [row])
            stats = memory.stats()
            self.assertLessEqual(stats['accounts'], 2)
            self.assertLessEqual(stats['serialized_payload_bytes'], 2500)
            self.assertLessEqual(stats['entries'], 2)
        clock[0] = 11
        self.assertEqual(memory.stats()['entries'], 0)
        self.assertEqual(memory.stats()['serialized_payload_bytes'], 0)
        self.assertGreater(memory.stats()['counts']['history_expired_entries'], 0)

    def test_per_account_cap_and_invalid_limits(self):
        memory = SeriesDateMemory(max_account_entries=2)
        for i in range(20):
            call, row = self.first()
            row['facts'][0]['object']['id'] = str(i)
            memory.apply([call], [row])
        self.assertEqual(memory.stats()['entries'], 2)
        for options in ({'max_entries': True}, {'max_bytes': 0}, {'max_accounts': -1},
                        {'ttl_seconds': math.inf}, {'ttl_seconds': math.nan}):
            with self.assertRaises(ValueError):
                SeriesDateMemory(**options)

    def test_sessions_and_account_types_do_not_mix(self):
        for account in ('other', 1, '1'):
            memory = SeriesDateMemory()
            call, row = self.first(1 if account == '1' else 'a')
            memory.apply([call], [row])
            call, row = self.second(account)
            self.assertEqual(memory.apply([call], [row])[0][0], row)
        memory = SeriesDateMemory()
        call, row = self.first()
        memory.apply([call], [row])
        call, row = self.second()
        call.series['session_id'] = 'new'
        self.assertEqual(memory.apply([call], [row])[0][0], row)

    def test_conflicting_retry_is_not_allowed_to_overwrite_history(self):
        memory = SeriesDateMemory()
        call, row = self.first()
        memory.apply([call], [row])
        different = call.model_copy(deep=True)
        different.turns[0].text = 'Заявка снова принята вчера.'
        changed = copy.deepcopy(row)
        changed['facts'][0]['valid_from'] = '2026-02-28T00:00:00'
        _, audit, counts = memory.apply([different], [changed])
        self.assertEqual(counts['history_conflicting_replay'], 1)
        self.assertEqual(audit[0]['reason'], 'conflicting_replay')
        call, row = self.second()
        result, _, _ = memory.apply([call], [row])
        self.assertEqual(result[0]['facts'][0]['valid_from'], '2026-02-25T00:00:00')

    def test_unknown_new_episode_invalidates_old_date(self):
        memory = SeriesDateMemory()
        memory.apply(*[[x] for x in self.first()])
        call, row = self.second()
        call.turns[0].text = 'Заявка снова принята.'
        row['facts'][0]['valid_from'] = None
        memory.apply([call], [row])
        self.assertEqual(memory.stats()['counts']['history_episode_invalidations'], 1)
        # The later unknown episode must remain a barrier to an older retry.
        older_call, older_row = self.first()
        memory.apply([older_call], [older_row])
        call, row = sample('third', '2026-03-06T12:00:00', '2026-03-04T00:00:00')
        self.assertEqual(memory.apply([call], [row])[0][0], row)

    def test_confirmed_explicit_start_has_priority_but_end_does_not_block_start(self):
        memory = SeriesDateMemory()
        memory.apply(*[[x] for x in self.first()])
        call, row = self.second()
        from schema import Turn
        call.turns = [Turn(idx=10, speaker='operator', text='Заявка принята вчера?'),
                      Turn(idx=12, speaker='client', text='Да, верно.')]
        row['facts'][0]['valid_from'] = '2026-03-04T00:00:00'
        self.assertEqual(memory.apply([call], [row])[0][0], row)
        memory = SeriesDateMemory()
        memory.apply(*[[x] for x in self.first()])
        call, row = self.second()
        call.turns[0].text = 'Да, верно.'
        call.turns.insert(0, Turn(idx=10, speaker='operator',
                                  text='Заявка была принята до вчерашнего дня?'))
        result, _, _ = memory.apply([call], [row])
        self.assertEqual(result[0]['facts'][0]['valid_from'], '2026-02-25T00:00:00')

    def test_resource_curve_refuses_different_history_or_decoder_settings(self):
        from compare_runs import comparison_rows
        base = {'status': 'complete', 'protocol': 'v1', 'dataset_sha256': 'same',
                'selected_calls_sha256': 'same', 'contract_signature': 'same', 'limits': {},
                'quality': {'scorer_sha256': 'same'}, 'warmup_batches': 1}
        for name, value in [('history_settings', {'max_payload_bytes': 200}),
                            ('series_history_enabled', True), ('grammar_backend', 'cpu')]:
            with self.assertRaisesRegex(ValueError, 'history settings'):
                comparison_rows([base, base, {**base, name: value}])

    def test_future_seed_mixed_timezones_and_conflicting_intervals_abstain(self):
        memory = SeriesDateMemory()
        call, row = self.first()
        row['facts'][0]['valid_from'] = '2027-01-01T00:00:00'
        memory.apply([call], [row])
        self.assertEqual(memory.stats()['entries'], 0)
        call, row = self.first()
        memory.apply([call], [row])
        call, row = self.second()
        call.started_at += '+03:00'
        self.assertEqual(memory.apply([call], [row])[0][0], row)
        call, row = self.second()
        row['facts'][0]['valid_from'] = None
        row['facts'][0]['valid_to'] = '2026-02-20T00:00:00'
        result, audit, counts = memory.apply([call], [row])
        self.assertEqual(result[0], row)
        self.assertEqual(counts['history_interval_conflicts'], 1)
        self.assertEqual(audit[0]['reason'], 'history_interval_conflict')

    def test_stable_id_survives_renaming_and_misalignment_is_rejected(self):
        memory = SeriesDateMemory()
        call, row = self.first()
        row['facts'][0]['object']['id'] = 'ticket-1234'
        memory.apply([call], [row])
        call, row = self.second()
        row['facts'][0]['object'].update(id='ticket-1234', name='обращение 1234')
        self.assertEqual(memory.apply([call], [row])[0][0]['facts'][0]['valid_from'],
                         '2026-02-25T00:00:00')
        row['call_id'] = 'wrong'
        with self.assertRaises(ValueError):
            memory.apply([call], [row])

    def test_same_bounded_store_used_by_offline_stream(self):
        memory = SeriesDateMemory()
        for call, row in [self.first(), self.second()]:
            result, _, _ = process_predictions([call], [row], mode='task',
                                               allow_series_history=True, memory=memory)
        self.assertEqual(result[0]['facts'][0]['valid_from'], '2026-02-25T00:00:00')
        memory.clear()
        self.assertEqual(memory.stats()['peak_serialized_payload_bytes'], 0)
        self.assertEqual(memory.stats()['counts'], {})

    def test_concurrent_accounts_are_isolated_and_bounded(self):
        memory = SeriesDateMemory(max_accounts=8)
        def work(index):
            call, row = self.first(str(index))
            return memory.apply([call], [row])[0][0]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(work, range(32)))
        self.assertEqual(len(results), 32)
        self.assertLessEqual(memory.stats()['accounts'], 8)


if __name__ == '__main__':
    unittest.main()
