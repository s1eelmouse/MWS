"""Causal history, account/ticket isolation and offline/online consistency."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from postprocess_dates import process_predictions
from schema import as_call
from series_history import SeriesDateMemory


def sample(call_id, started, value, *, account='a', ticket='1234', text='Заявка принята.', index=0):
    call = as_call({'call_id': call_id, 'started_at': started,
                    'series': {'account_id': account, 'call_index': index},
                    'turns': [{'idx': 12, 'speaker': 'operator', 'text': text}]})
    fact = {'object': {'name': 'заявка ' + ticket, 'type': 'ticket'},
            'attribute': 'ticket_status', 'value': 'принята', 'polarity': 1,
            'valid_from': value, 'valid_to': None, 'source': {'turn_idx': 12},
            'speaker': 'operator'}
    return call, {'call_id': call_id, 'facts': [fact]}


class SeriesHistoryTest(unittest.TestCase):
    def setUp(self):
        self.first = sample('a:0', '2026-03-01T12:00:00', '2026-02-25T00:00:00')
        self.second = sample('a:1', '2026-03-05T12:00:00', '2026-03-02T00:00:00', index=1)

    def test_default_offline_mode_is_stateless(self):
        calls, rows = zip(self.first, self.second)
        result, _, counts = process_predictions(calls, rows, mode='task')
        self.assertEqual(result[1]['facts'][0]['valid_from'], rows[1]['facts'][0]['valid_from'])
        self.assertNotIn('series_accepted_carried', counts)

    def test_history_matches_streaming_requests_and_keeps_input_order(self):
        calls, rows = zip(self.first, self.second)
        bulk, _, _ = SeriesDateMemory().apply(list(reversed(calls)), list(reversed(rows)))
        memory = SeriesDateMemory()
        memory.apply([calls[0]], [rows[0]])
        online, audit, _ = memory.apply([calls[1]], [rows[1]])
        self.assertEqual(bulk[0], online[0])
        self.assertEqual(online[0]['facts'][0]['valid_from'], '2026-02-25T00:00:00')
        self.assertEqual(audit[0]['source_call_id'], 'a:0')
        self.assertEqual(rows[1]['facts'][0]['valid_from'], '2026-03-02T00:00:00')

    def test_never_reads_future_even_when_call_index_disagrees(self):
        future_call, future_row = sample('future', '2026-04-01T12:00:00', '2026-03-20T00:00:00', index=0)
        past_call, past_row = sample('past', '2026-03-01T12:00:00', '2026-02-25T00:00:00', index=9)
        result, _, _ = SeriesDateMemory().apply([future_call, past_call], [future_row, past_row])
        self.assertEqual(result[1], past_row)
        memory = SeriesDateMemory()
        memory.apply([future_call], [future_row])
        result, audit, _ = memory.apply([past_call], [past_row])
        self.assertEqual(result[0], past_row)
        self.assertEqual(audit[0]['reason'], 'history_not_strictly_earlier')

    def test_accounts_and_ticket_numbers_are_independent(self):
        first_call, first_row = self.first
        for account, ticket in [('b', '1234'), ('a', '9999')]:
            call, row = sample('other', '2026-03-05T12:00:00', '2026-03-02T00:00:00', account=account, ticket=ticket)
            result, _, _ = SeriesDateMemory().apply([first_call, call], [first_row, row])
            self.assertEqual(result[1], row)

    def test_generic_ticket_name_is_insufficient_identity(self):
        calls, rows = zip(self.first, self.second)
        rows = copy.deepcopy(rows)
        for row in rows:
            row['facts'][0]['object']['name'] = 'заявка'
        result, _, _ = SeriesDateMemory().apply(calls, rows)
        self.assertEqual(result, list(rows))

    def test_explicit_date_and_new_episode_override_memory(self):
        for text in ['Вчера заявка была принята.', 'Заявка снова принята.']:
            call, row = sample('new', '2026-03-05T12:00:00', '2026-03-04T00:00:00', text=text)
            result, _, _ = SeriesDateMemory().apply([self.first[0], call], [self.first[1], row])
            self.assertEqual(result[1], row)

    def test_cache_is_bounded_and_can_be_reset_after_warmup(self):
        memory = SeriesDateMemory(max_entries=1)
        memory.apply([self.first[0]], [self.first[1]])
        other_call, other_row = sample('other', '2026-03-02T12:00:00', '2026-02-28T00:00:00', account='b')
        _, _, counts = memory.apply([other_call], [other_row])
        self.assertEqual(counts['series_entries_evicted'], 1)
        self.assertEqual(len(memory._entries), 1)
        memory.clear()
        self.assertEqual(len(memory._entries), 0)

    def test_negative_facts_are_not_carried(self):
        call, row = copy.deepcopy(self.second)
        row['facts'][0]['polarity'] = -1
        result, _, _ = SeriesDateMemory().apply([self.first[0], call], [self.first[1], row])
        self.assertEqual(result[1], row)

    def test_provenance_keeps_origin_across_three_calls(self):
        third = sample('a:2', '2026-03-06T12:00:00', '2026-03-03T00:00:00', index=2)
        memory = SeriesDateMemory()
        for call, row in [self.first, self.second]:
            memory.apply([call], [row])
        _, events, _ = memory.apply([third[0]], [third[1]])
        self.assertEqual(events[0]['source_call_id'], self.first[0].call_id)
        self.assertEqual(events[0]['source_started_at'], self.first[0].started_at)

    def test_equal_date_does_not_erase_original_provenance(self):
        call, row = copy.deepcopy(self.second)
        row['facts'][0]['valid_from'] = self.first[1]['facts'][0]['valid_from']
        memory = SeriesDateMemory()
        memory.apply([self.first[0], call], [self.first[1], row])
        third_call, third_row = sample('a:2', '2026-03-06T12:00:00', '2026-03-03T00:00:00')
        _, events, _ = memory.apply([third_call], [third_row])
        self.assertEqual(events[0]['source_call_id'], self.first[0].call_id)

    def test_extractor_applies_memory_and_records_decisions(self):
        from extractor import Extractor
        backend = Extractor.__new__(Extractor)
        backend.series_memory = SeriesDateMemory()
        calls, rows = zip(self.first, self.second)
        by_id = {row['call_id']: row for row in rows}

        def generate(batch, measurements):
            for call in batch:
                measurements.append({'call_id': call.call_id})
            return [copy.deepcopy(by_id[call.call_id]) for call in batch]

        backend._extract_batch = generate
        records = []
        result = backend.extract(calls, batch_size=1, measurements=records)
        self.assertEqual(result[1]['facts'][0]['valid_from'], rows[0]['facts'][0]['valid_from'])
        self.assertEqual(records[1]['normalization']['series_accepted_carried'], 1)
        self.assertEqual(records[1]['series_decisions'][0]['source_call_id'], calls[0].call_id)
        backend.reset_series_history()
        self.assertEqual(len(backend.series_memory._entries), 0)

    def test_http_requests_share_only_this_backend_history(self):
        from extractor import Extractor
        from fastapi.testclient import TestClient
        from serve import create_app
        backend = Extractor.__new__(Extractor)
        backend.series_memory = SeriesDateMemory(max_entries=20)
        backend.series_history_enabled = True
        backend.temporal_normalization = 'task'
        calls, rows = zip(self.first, self.second)
        by_id = {row['call_id']: {'facts': row['facts']} for row in rows}
        backend._extract_safely = lambda batch: [copy.deepcopy(by_id[c.call_id]) for c in batch]
        with TestClient(create_app(backend)) as client:
            for call in calls:
                response = client.post('/extract', json=call.model_dump(mode='json'))
                self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['facts'][0]['valid_from'], rows[0]['facts'][0]['valid_from'])
            health = client.get('/health').json()
            self.assertTrue(health['series_history_enabled'])
            self.assertEqual(health['series_history_max_entries'], 20)

    def test_vllm_backend_uses_same_history_without_a_gpu(self):
        from serve_vllm import VLLMExtractor
        backend = VLLMExtractor()
        backend.series_memory = SeriesDateMemory()
        with patch('serve_vllm.chat', side_effect=[json.dumps({'facts': row['facts']})
                                                  for _, row in [self.first, self.second]]):
            for call, _ in [self.first, self.second]:
                result = backend.extract([call])
        self.assertEqual(result[0]['facts'][0]['valid_from'], self.first[1]['facts'][0]['valid_from'])


if __name__ == '__main__':
    unittest.main()
