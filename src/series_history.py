"""Bounded, causal date memory shared by offline processing and HTTP backends."""
from collections import Counter, OrderedDict, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import threading
import time


def _date(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def _time_key(value):
    parsed = _date(value)
    aware = parsed.tzinfo is not None
    normalized = parsed.astimezone(timezone.utc) if aware else parsed
    return aware, normalized.isoformat()


def _account(call):
    series = call.series or {}
    value, session = series.get('account_id'), series.get('session_id', '')
    if (type(value) not in (str, int) or not str(value).strip() or len(str(value)) > 256
            or type(session) not in (str, int) or len(str(session)) > 256):
        return None
    return _digest([type(value).__name__, value, type(session).__name__, session])


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def _ticket(fact):
    obj = fact['object']
    if obj.get('type') not in (None, 'ticket'):
        return None
    if isinstance(obj.get('id'), str) and obj['id'].strip():
        return ('id', obj['id']) if len(obj['id']) <= 256 else None
    match = re.search(r'\b(?:заявк[а-я]*|обращени[а-я]*)\s*(?:№\s*|номер\s*)?(\d+)\b', obj['name'].lower())
    return ('number', match[1]) if match else None


def _explicit_or_new(fact, call):
    # A local explicit date/new episode overrides remembered model state.
    from temporal import (CLAUSE_BREAK, STATUS_PATTERNS, extract_time_expressions,
                          _cue_dates, _datetime)
    turn = next((t for t in call.turns if t.idx == fact['source']['turn_idx']), None)
    if turn is None:
        return True
    # A short confirmation may obtain its explicit time from the preceding question.
    from temporal_binding import repair_bound_dates
    _, bound = repair_bound_dates(fact, call)
    if any(evidence.get('field') == 'valid_from' for evidence in bound.get('evidence', [])):
        return True
    if re.search(r'\b(?:повторно|снова|заново|новая\s+заявка)\b', turn.text.lower()):
        return True
    for clause in CLAUSE_BREAK.split(turn.text):
        lower = clause.lower().replace('ё', 'е')
        if not re.search(STATUS_PATTERNS['принята'], lower):
            continue
        if re.search(r'\b(?:повторно|снова|заново|новая\s+заявка)\b', lower):
            return True
        if any(x.certain for x in extract_time_expressions(clause, call.started_at, direction='past')):
            return True
        if _cue_dates(clause, _datetime(call.started_at).date()):
            return True  # Respect the same ASR cues recognized by task mode.
    return False


class SeriesDateMemory:
    """Use earlier observations of the same identified ticket/account only.

    Out-of-order requests abstain; they never read a future observation or replace
    newer memory. The object belongs to one backend, not a module-global cache.
    """
    def __init__(self, max_entries=10000, *, max_accounts=512, max_bytes=8 * 1024 * 1024,
                 account_bytes=32768, max_account_entries=64, ttl_seconds=3600,
                 clock=time.monotonic):
        for value in (max_entries, max_accounts, max_bytes, account_bytes, max_account_entries):
            if type(value) is not int or value < 1:
                raise ValueError('History capacity limits must be positive integers')
        if isinstance(ttl_seconds, bool) or not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError('History TTL must be finite and positive')
        self.max_entries = max_entries
        self.max_accounts, self.max_bytes = max_accounts, max_bytes
        self.account_bytes, self.max_account_entries = account_bytes, max_account_entries
        self.ttl_seconds, self._clock = ttl_seconds, clock
        self._entries = OrderedDict()
        self._accounts = OrderedDict()
        self._account_bytes = Counter()
        self._payload_bytes = self._peak_bytes = 0
        self._counts, self._seconds = Counter(), 0.0
        self._lock = threading.RLock()

    def _remove(self, key, reason, counts):
        account, _, payload = self._entries.pop(key)
        size = len(payload) + len(key)
        self._payload_bytes -= size
        self._account_bytes[account] -= size
        self._accounts[account].pop(key)
        if not self._accounts[account]:
            del self._accounts[account]
            del self._account_bytes[account]
        counts[reason] += 1

    def _expire(self, now, counts):
        while self._entries and now - next(iter(self._entries.values()))[1] >= self.ttl_seconds:
            self._remove(next(iter(self._entries)), 'history_expired_entries', counts)

    def _store(self, account, key, value, now, counts):
        payload = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        size = len(payload) + len(key)
        if size > min(self.account_bytes, self.max_bytes):
            counts['history_oversize_abstained'] += 1
            return
        if key in self._entries:
            self._remove(key, 'history_replaced_entries', counts)
        while account in self._accounts and (
                len(self._accounts[account]) >= self.max_account_entries
                or self._account_bytes[account] + size > self.account_bytes):
            self._remove(next(iter(self._accounts[account])), 'history_account_evictions', counts)
        while account not in self._accounts and len(self._accounts) >= self.max_accounts:
            oldest = next(iter(self._accounts))
            for entry in list(self._accounts[oldest]):
                self._remove(entry, 'history_account_evictions', counts)
        while self._entries and (len(self._entries) >= self.max_entries
                                 or self._payload_bytes + size > self.max_bytes):
            self._remove(next(iter(self._entries)), 'series_entries_evicted', counts)
        self._entries[key] = (account, now, payload)
        self._accounts.setdefault(account, OrderedDict())[key] = None
        self._accounts.move_to_end(account)
        self._account_bytes[account] += size
        self._payload_bytes += size
        self._peak_bytes = max(self._peak_bytes, self._payload_bytes)

    def apply(self, calls, results):
        start = time.perf_counter()
        if len(calls) != len(results) or len({c.call_id for c in calls}) != len(calls):
            raise ValueError('History requires one unique result per call')
        if any(row.get('call_id', call.call_id) != call.call_id for call, row in zip(calls, results)):
            raise ValueError('History call IDs differ or results are not aligned')
        output = copy.deepcopy(list(results))
        by_id = {c.call_id: row for c, row in zip(calls, output)}
        events, counts = [], Counter()
        # Chronology, not call_index or file order, controls access to history.
        ordered = sorted(calls, key=lambda c: (_account(c) or c.call_id, *_time_key(c.started_at), c.call_id))
        with self._lock:
            now = self._clock()
            self._expire(now, counts)
            for call in ordered:
                account = _account(call)
                row = by_id[call.call_id]
                if not account or len(call.call_id) > 256 or row.get('error'):
                    continue
                fingerprint = _digest(call.model_dump(mode='json', exclude_none=True))
                groups = defaultdict(list)
                for index, fact in enumerate(row.get('facts', [])):
                    if fact.get('attribute') == 'ticket_status' and fact.get('polarity', 1) == 1 \
                            and fact.get('value', '').strip().lower() == 'принята':
                        ticket = _ticket(fact)
                        if ticket is not None:
                            groups[_digest([account, ticket])].append((index, fact))
                for key, group in groups.items():
                    if len({f.get('valid_from') for _, f in group}) != 1:
                        counts['series_ambiguous_dates'] += 1
                        continue
                    cached = self._entries.get(key)
                    previous = json.loads(cached[2]) if cached else None
                    if (previous and previous['call_id'] == call.call_id
                            and previous['fingerprint'] != fingerprint):
                        counts['history_conflicting_replay'] += 1
                        events.append({'call_id': call.call_id, 'component': 'series',
                                       'status': 'abstained', 'reason': 'conflicting_replay'})
                        continue
                    current_time = _time_key(call.started_at)
                    replay = bool(previous and tuple(previous['time']) == current_time
                                  and previous['call_id'] == call.call_id
                                  and _time_key(previous['origin_started_at']) < current_time)
                    if previous and (previous['time'][0] != current_time[0]
                                     or (previous['time'][1] >= current_time[1] and not replay)):
                        events.append({'call_id': call.call_id, 'component': 'series',
                                       'status': 'abstained', 'reason': 'history_not_strictly_earlier'})
                        counts['series_out_of_order_abstained'] += 1
                        continue
                    origin = call.call_id
                    origin_started_at = call.started_at
                    explicit = any(_explicit_or_new(fact, call) for _, fact in group)
                    if previous and not explicit and previous['date']:
                        # A group is one observation: never partly carry a date
                        # when one of its intervals would become invalid.
                        try:
                            from schema import Facts
                            candidates = [dict(fact, valid_from=previous['date']) for _, fact in group]
                            Facts.model_validate({'facts': candidates})
                            if _date(previous['date']) > _date(call.started_at):
                                raise ValueError('future date')
                        except (ValueError, TypeError):
                            counts['history_interval_conflicts'] += 1
                            events.append({'call_id': call.call_id, 'component': 'series',
                                           'status': 'abstained', 'reason': 'history_interval_conflict'})
                            continue
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
                    date = group[0][1].get('valid_from')
                    observation = {'date': date, 'time': current_time, 'call_id': call.call_id,
                                   'fingerprint': fingerprint, 'origin': origin,
                                   'origin_started_at': origin_started_at, 'observed_at': call.started_at}
                    try:
                        valid = bool(date and _date(date) <= _date(call.started_at))
                        # Also validates intervals and mixed timezone awareness.
                        from schema import Facts
                        Facts.model_validate({'facts': [fact for _, fact in group]})
                    except (TypeError, ValueError):
                        valid = False
                    if not valid:
                        counts['history_invalid_date_abstained'] += 1
                        # A new explicit episode invalidates the previous episode, even
                        # if its new onset is unknown or not usable yet.
                        if explicit:
                            # Keep a chronological barrier: an older request arriving
                            # later must not resurrect the previous episode.
                            observation.update(date=None, origin=call.call_id,
                                               origin_started_at=call.started_at)
                            self._store(account, key, observation, now, counts)
                            counts['history_episode_invalidations'] += 1
                        continue
                    self._store(account, key, observation, now, counts)
            self._counts.update(counts)
            self._seconds += time.perf_counter() - start
        return output, events, dict(counts)

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._accounts.clear()
            self._account_bytes.clear()
            self._payload_bytes = self._peak_bytes = 0
            self._counts.clear()
            self._seconds = 0.0

    def stats(self):
        with self._lock:
            self._expire(self._clock(), self._counts)
            return {**self.settings(), 'entries': len(self._entries), 'accounts': len(self._accounts),
                    'serialized_payload_bytes': self._payload_bytes,
                    'peak_serialized_payload_bytes': self._peak_bytes,
                    'processing_seconds': self._seconds, 'counts': dict(self._counts)}

    def settings(self):
        return {'policy': 'identified_ticket_explicit_priority_v3',
                'max_entries': self.max_entries, 'max_accounts': self.max_accounts,
                'max_payload_bytes': self.max_bytes, 'max_account_payload_bytes': self.account_bytes,
                'max_account_entries': self.max_account_entries, 'ttl_seconds': self.ttl_seconds}


def new_series_memory():
    """One factory keeps offline, Transformers and vLLM resource limits identical."""
    import config
    return SeriesDateMemory(config.TEMPORAL_SERIES_MAX_ENTRIES,
                            max_accounts=config.HISTORY_MAX_ACCOUNTS,
                            max_bytes=config.HISTORY_MAX_BYTES,
                            ttl_seconds=config.HISTORY_TTL_SECONDS)
