"""Text-grounded boundaries for tariffs and services; no labels or cross-call memory.

Only named values and explicit event dates qualify. An observation, a proposal,
or an unrelated date in the next turn does not establish the start of a state.
"""
import re

TRANSITION = re.compile(r'\b(?:переш[а-я]*|перех[а-я]*|смен[а-я]*|помен[а-я]*|подключ[а-я]*|активир[а-я]*)')
UNCERTAIN = re.compile(r'\b(?:если|возможно|кажется|наверное|предлага[а-я]*|можем|могу|хочу|хотел[а-я]*|планир[а-я]*)\b')
OBSERVATION = re.compile(r'\b(?:еще|смотр[а-я]*|провер[а-я]*|звони[а-я]*|квитанц[а-я]*|обещ[а-я]*)\b')
NORMAL_PATTERN = re.compile(r'\bработа[а-я]*\s+(?:нормальн[а-я]*|стабильн[а-я]*)\b')
CHUNKS = re.compile(r'(?<!\d)[.!?;…]+|[.!?;…]+(?!\d)|\b(?:а|но|затем|потом|теперь)\b', re.I)


def _mentions(text, value):
    # Keep offsets in the canonical text, including ASR cue expansion.
    from temporal_case_v2 import _words, _close, NORMAL
    if value.lower() == NORMAL:
        return [(m.start(), m.end()) for m in NORMAL_PATTERN.finditer(text)]
    wanted = _words(value)
    tokens = list(re.finditer(r'[а-яa-z0-9+]+', text))
    result = []
    for i in range(len(tokens) - len(wanted) + 1):
        group = tokens[i:i + len(wanted)]
        if wanted and all(w == t.group() or len(w) >= 5 and ('+' in w) == ('+' in t.group())
                          and _close(w, t.group())
                          for w, t in zip(wanted, group)):
            result.append((group[0].start(), group[-1].end()))
    return result


def _distance(position, span):
    return min(abs(position - span[0]), abs(position - span[1]))


def explicit_boundaries(item, turn, call_day, entries):
    """Return unique (start, end) days and evidence belonging to this value.

    Calendar parsing is shared with case_v2. Value mentions compete for each
    cue, so a date next to a symptom cannot date the tariff in the same turn.
    """
    from temporal_case_v2 import canonical_text, day_cues, NORMAL
    starts, ends, evidence = set(), set(), []
    text = canonical_text(turn.text)
    start = 0
    chunks = []
    for separator in CHUNKS.finditer(text):
        chunks.append((text[start:separator.start()], separator.group()))
        start = separator.end()
    chunks.append((text[start:], ''))
    for chunk, separator in chunks:
        if '?' in separator or UNCERTAIN.search(chunk):
            continue
        if item['attribute'] == 'tariff' and re.search(
                r'\bне\s+(?:был[аои]?\s+)?(?:подключ[а-я]*|смен[а-я]*|переш[а-я]*|активир[а-я]*)', chunk):
            continue
        own = _mentions(chunk, item['value'])
        if not own:
            continue
        competitors = []
        for value in entries:
            if value.lower() != item['value'].lower():
                competitors.extend(_mentions(chunk, value))
        cues = day_cues(chunk, call_day, canonical=True)
        for day, role, position in cues:
            nearest = min(own, key=lambda span: _distance(position, span))
            if role == 'until' and any(r == 'since' and position < p < nearest[0]
                                       for _d, r, p in cues):
                continue  # «до вторника работало, со вторника нет гудков»
            # Ambiguity is an abstention, including equally close values.
            prefix = chunk[max(0, nearest[0] - 20):nearest[0]]
            transfer = (item['attribute'] == 'tariff' and TRANSITION.search(chunk)
                        and len({d for d, _r, _p in cues}) == 1
                        and re.search(r'\bс(?:о)?\s+.+?\bна\s+', chunk)
                        and re.search(r'\b(?:со?|на)\s+[«"\s]*$', prefix))
            if not transfer and any(_distance(position, other) <= _distance(position, nearest)
                   for other in competitors if not (other[0] >= nearest[0] and other[1] <= nearest[1])):
                continue
            if re.search(r'\bне\s+(?:был[аои]?\s+)?$', prefix):
                continue
            if role == 'until':
                ends.add(day)
                field = 'valid_to'
            elif item['attribute'] == 'tariff':
                if OBSERVATION.search(chunk):
                    continue
                transition = TRANSITION.search(chunk)
                if role != 'since' and transition is None:
                    continue
                # «перешла с Пульса на Смарт»: the old tariff ends, the new begins.
                old = transition is not None and re.search(r'\bс(?:о)?\s+[«"\s]*$', prefix)
                if old and re.search(r'\bна\s+', chunk[nearest[1]:]):
                    ends.add(day)
                    field = 'valid_to'
                else:
                    starts.add(day)
                    field = 'valid_from'
            elif item['value'].lower() == NORMAL:
                # «вчера работало» is an observation, not a known start/end.
                if role != 'since':
                    continue
                starts.add(day)
                field = 'valid_from'
            else:
                if OBSERVATION.search(chunk):
                    continue
                starts.add(day)
                field = 'valid_from'
            evidence.append({'source_turn_idx': turn.idx, 'field': field,
                             'day': day.isoformat(), 'text': chunk.strip(),
                             'ongoing': field == 'valid_from'
                             and (role == 'since' or item['attribute'] == 'service_state'
                                  and item['value'].lower() != NORMAL)
                             and not re.search(r'\b(?:был[аои]?|действовал[а-я]*|работал[а-я]*)\b', chunk)})
    return (next(iter(starts)) if len(starts) == 1 else None,
            next(iter(ends)) if len(ends) == 1 else None, evidence)


def service_onset_context(item, call, turn, call_day, entries):
    """A named symptom may follow its onset phrase, or be confirmed next turn.

    Require an onset cue and the same symptom; never borrow a date from an
    arbitrary adjacent turn (e.g. a router reboot or ticket registration).
    """
    from temporal_case_v2 import canonical_text, day_cues, NORMAL
    if item['value'].lower() == NORMAL:
        return None
    candidates = []
    ordered = sorted(call.turns, key=lambda t: t.idx)
    pos = next(i for i, t in enumerate(ordered) if t.idx == turn.idx)
    for i in (pos, pos - 1, pos + 1):
        if not 0 <= i < len(ordered):
            continue
        source = ordered[i]
        text = canonical_text(source.text)
        if UNCERTAIN.search(text) or not _mentions(text, item['value']):
            continue
        cursor = 0
        questioned = False
        for separator in CHUNKS.finditer(text):
            if '?' in separator.group() and _mentions(text[cursor:separator.start()], item['value']):
                questioned = True
            cursor = separator.end()
        if questioned:
            continue
        # A different catalog symptom in this context makes event ownership uncertain.
        own = _mentions(text, item['value'])
        if any(not any(a <= left and right <= b for a, b in own)
               for value in entries if value.lower() not in (item['value'].lower(), NORMAL)
               for left, right in _mentions(text, value)):
            continue
        cues = day_cues(text, call_day, canonical=True)
        days = {d for d, role, _ in cues if role == 'since' or
                re.search(r'\b(?:пропал[а-я]*|появил[а-я]*|начал[а-я]*|перестал[а-я]*|отрезало|вдруг|бац|до\s+этого)\b', text)}
        if len(days) == 1 and not any(role == 'until' and d not in days for d, role, _ in cues):
            candidates.append((next(iter(days)), source.idx))
    if len({d for d, _ in candidates}) == 1:
        return candidates[0]
    return None


def bind_explicit_interval(item, turn, call_day, entries, changes, stamp):
    """Apply boundaries atomically; retain an unmentioned boundary."""
    from temporal_case_v2 import _change, _ordered
    start, end, evidence = explicit_boundaries(item, turn, call_day, entries)
    proposed_start = stamp(start) if start else item.get('valid_from')
    ongoing = start and not end and any(e.get('ongoing') for e in evidence)
    proposed_end = stamp(end) if end else None if ongoing else item.get('valid_to')
    if not _ordered(proposed_start, proposed_end):
        return []
    if start:
        _change(item, 'valid_from', proposed_start, changes)
    if end:
        _change(item, 'valid_to', proposed_end, changes)
    elif ongoing and item.get('valid_to') is not None:
        changes.append({'field': 'valid_to', 'before': item['valid_to'], 'after': None})
        item['valid_to'] = None
    return evidence
