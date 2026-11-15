"""Collect explicit boundaries of one identified ticket within the current call.

No status-order completion, no predicted dates used as evidence, and no history.
An unnumbered turn is usable only when the transcript identifies a single ticket.
"""
import copy
import re
from collections import defaultdict


def bind_ticket_evidence(items, call, decisions, stamp, read_status):
    from temporal_case_v2 import (TICKET_NUMBER, STATUS_PATTERNS, _status_key,
                                  canonical_text, day_cues, NOT_STATUS, HEDGE,
                                  _change, _ordered, _clauses_with_separator)
    from datetime import datetime
    from types import SimpleNamespace
    call_day = datetime.fromisoformat(call.started_at.replace('Z', '+00:00')).date()
    all_numbers = {number for turn in call.turns
                   for number in TICKET_NUMBER.findall(turn.text.lower())}
    for item, decision in zip(items, decisions):
        number = TICKET_NUMBER.search(item['object'].get('name', '').lower())
        key = _status_key(item['value'])
        if item['attribute'] != 'ticket_status' or item.get('polarity', 1) != 1 \
                or not number or key not in STATUS_PATTERNS or decision['reason'] in (
                    'invalid_model_date_left_for_sanitizer', 'multiple_objects_same_evidence', 'source_missing'):
            continue
        candidates = defaultdict(set)
        proof = defaultdict(list)
        for turn in call.turns:
            named = set(TICKET_NUMBER.findall(turn.text.lower()))
            if named != {number[1]} and not (not named and all_numbers == {number[1]}):
                continue
            probe = copy.deepcopy(item)
            probe['valid_from'] = probe['valid_to'] = None
            changes = []
            # A visit/phone-call deadline in a summary does not date the status.
            chunks = []
            for chunk, separator in _clauses_with_separator(turn.text):
                lower = canonical_text(chunk)
                unrelated = re.search(r'\b(?:перезвон[а-я]*|звони[а-я]*|звонок|контрольн[а-я]*|приед[а-я]*|визит[а-я]*)\b', lower)
                visit = re.search(r'\bвыезд[а-я]*\b', lower) and not re.search(
                    STATUS_PATTERNS['назначен выезд специалиста'], lower)
                if not (unrelated or visit):
                    chunks.append(chunk + separator)
            read_status(probe, call_day, SimpleNamespace(text=' '.join(chunks), idx=turn.idx), changes, stamp)
            for change in changes:
                candidates[change['field']].add(change['after'])
                proof[change['field']].append({'source_turn_idx': turn.idx,
                                               'field': change['field'], 'date': change['after'],
                                               'reason': 'explicit_same_ticket_boundary'})

            # «до вчера была принята, а сейчас в работе»: one explicit transition
            # gives the old endpoint and the new start, even across clause breaks.
            text = canonical_text(turn.text)
            if '?' in text or HEDGE.search(text):
                continue
            mentions = sorted((m.start(), m.end(), status)
                              for status, pattern in STATUS_PATTERNS.items()
                              for m in re.finditer(pattern, text)
                              if not NOT_STATUS.search(text[max(0, m.start() - 15):m.start()]))
            cues = day_cues(text, call_day, canonical=True)
            for day, role, position in cues:
                if role != 'until' or len(mentions) < 2:
                    continue
                old = min(mentions, key=lambda m: min(abs(position - m[0]), abs(position - m[1])))
                following = [m for m in mentions if m[0] > max(old[0], position) and m[2] != old[2]]
                if not following:
                    continue
                new = following[0]
                between = text[old[1]:new[0]]
                if not re.search(r'\b(?:сейчас|теперь|сменил[а-я]*|изменил[а-я]*|перевел[а-я]*)\b', between):
                    continue
                # A separate date for the new status defeats this inference.
                if any(d != day and p > old[1] for d, _r, p in cues):
                    continue
                field = 'valid_to' if key == old[2] else 'valid_from' if key == new[2] else None
                if field:
                    candidates[field].add(stamp(day))
                    proof[field].append({'source_turn_idx': turn.idx, 'field': field,
                                         'date': stamp(day), 'reason': 'explicit_ticket_transition'})
        proposed = {field: next(iter(values)) for field, values in candidates.items()
                    if len(values) == 1}
        start = proposed.get('valid_from', item.get('valid_from'))
        end = proposed.get('valid_to', item.get('valid_to'))
        if not _ordered(start, end):
            continue
        for field, value in proposed.items():
            _change(item, field, value, decision['changes'])
            decision['evidence'].extend(proof[field])
