"""Bounded, causal date memory shared by offline processing and HTTP backends."""
from collections import Counter, OrderedDict, defaultdict
import copy
from datetime import datetime, timezone
import re
import threading


def _date(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def _time_key(value):
    parsed = _date(value)
    aware = parsed.tzinfo is not None
    normalized = parsed.astimezone(timezone.utc) if aware else parsed
    return aware, normalized.isoformat()


def _account(call):
    value = (call.series or {}).get('account_id')
    return value if isinstance(value, str) and value.strip() else None


def _ticket(fact):
    obj = fact['object']
    if obj.get('type') not in (None, 'ticket'):
        return None
    if isinstance(obj.get('id'), str) and obj['id'].strip():
        return ('id', obj['id'])
    match = re.search(r'\b(?:заявк[а-я]*|обращени[а-я]*)\s*(?:№\s*|номер\s*)?(\d+)\b', obj['name'].lower())
    return ('number', match[1]) if match else None


def _explicit_or_new(fact, call):
    # A local explicit date/new episode overrides remembered model state.
    from temporal import CLAUSE_BREAK, STATUS_PATTERNS, extract_time_expressions
    turn = next((t for t in call.turns if t.idx == fact['source']['turn_idx']), None)
    if turn is None:
        return True
    for clause in CLAUSE_BREAK.split(turn.text):
        lower = clause.lower().replace('ё', 'е')
        if not re.search(STATUS_PATTERNS['принята'], lower):
            continue
        if re.search(r'\b(?:повторно|снова|заново|новая\s+заявка)\b', lower):
            return True
        if any(x.certain for x in extract_time_expressions(clause, call.started_at, direction='past')):
            return True
    return False


class SeriesDateMemory:
    """Use earlier observations of the same identified ticket/account only.

    Out-of-order requests abstain; they never read a future observation or replace
    newer memory. The object belongs to one backend, not a module-global cache.
    """
    def __init__(self, max_entries=10000):
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            raise ValueError('max_entries must be a positive integer')
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def apply(self, calls, results):
        if len(calls) != len(results) or len({c.call_id for c in calls}) != len(calls):
            raise ValueError('History requires one unique result per call')
        output = copy.deepcopy(list(results))
        by_id = {c.call_id: row for c, row in zip(calls, output)}
        events, counts = [], Counter()
        # Chronology, not call_index or file order, controls access to history.
        ordered = sorted(calls, key=lambda c: (_account(c) or c.call_id, *_time_key(c.started_at), c.call_id))
        with self._lock:
            for call in ordered:
                account = _account(call)
                row = by_id[call.call_id]
                if not account or row.get('error'):
                    continue
                groups = defaultdict(list)
                for index, fact in enumerate(row.get('facts', [])):
                    if fact.get('attribute') == 'ticket_status' and fact.get('polarity', 1) == 1 \
                            and fact.get('value', '').strip().lower() == 'принята':
                        ticket = _ticket(fact)
                        if ticket is not None:
                            groups[(account, ticket)].append((index, fact))
                for key, group in groups.items():
                    if len({f.get('valid_from') for _, f in group}) != 1:
                        counts['series_ambiguous_dates'] += 1
                        continue
                    previous = self._entries.get(key)
                    current_time = _time_key(call.started_at)
                    if previous and (previous['time'][0] != current_time[0]
                                     or previous['time'][1] >= current_time[1]):
                        events.append({'call_id': call.call_id, 'component': 'series',
                                       'status': 'abstained', 'reason': 'history_not_strictly_earlier'})
                        counts['series_out_of_order_abstained'] += 1
                        continue
                    origin = call.call_id
                    origin_started_at = call.started_at
                    explicit = any(_explicit_or_new(fact, call) for _, fact in group)
                    for index, fact in group:
                        if previous and not explicit and previous['date']:
                            value = previous['date']
                            try:
                                proposed = _date(value)
                                end = _date(fact['valid_to']) if fact.get('valid_to') else None
                                allowed = proposed <= _date(call.started_at) and (end is None or proposed <= end)
                            except (ValueError, TypeError):
                                allowed = False
                            if allowed:
                                # Keep the original provenance even when no date changes.
                                origin = previous['origin']
                                origin_started_at = previous['origin_started_at']
                            if allowed and fact.get('valid_from') != value:
                                events.append({'call_id': call.call_id, 'component': 'series',
                                               'fact_index': index, 'field': 'valid_from',
                                               'before': fact.get('valid_from'), 'after': value,
                                               'source_call_id': previous['origin'],
                                               'source_started_at': previous['origin_started_at'],
                                               'last_observed_at': previous['observed_at'],
                                               'status': 'changed', 'reason': 'accepted_status_carried_within_series'})
                                fact['valid_from'] = value
                                counts['series_accepted_carried'] += 1
                    self._entries[key] = {'date': group[0][1].get('valid_from'), 'time': current_time,
                                          'origin': origin, 'origin_started_at': origin_started_at,
                                          'observed_at': call.started_at}
                    self._entries.move_to_end(key)
                    while len(self._entries) > self.max_entries:
                        self._entries.popitem(last=False)
                        counts['series_entries_evicted'] += 1
        return output, events, dict(counts)

    def clear(self):
        with self._lock:
            self._entries.clear()
