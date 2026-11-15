"""Within-call temporal evidence binding. No gold or cross-call history is read."""
from collections import defaultdict
import copy
from dataclasses import dataclass
import re

from temporal import (STATUS_PATTERNS, _datetime, _norm, extract_time_expressions,
                      repair_task_dates)


BREAK = re.compile(r'(?<!\d)[.!?;…]+|[.!?;…]+(?!\d)|,|\b(?:а|но|затем|потом|теперь)\b', re.I)
TRANSITION = re.compile(r'\b(?:перев[её][дл][а-я]*|переш[её]л[а-я]*|взял[а-я]*|стала?|измени[а-я]*|затем|потом|дальше|после\s+этого)\b', re.I)
RANK = {'принята': 0, 'в работе': 1, 'назначен выезд специалиста': 2, 'выполнена': 3}
FILLERS = {'да', 'ну', 'это', 'как', 'раз', 'еще', 'буквально', 'тогда', 'именно', 'около', 'в', 'с', 'со', 'до'}
CONFIRM = {'да', 'ага', 'угу', 'все', 'верно', 'правильно', 'именно', 'так', 'подходит', 'конечно', 'спасибо'}


@dataclass
class Segment:
    left: int
    right: int
    after: str
    text: str


def _segments(text):
    result, start = [], 0
    for match in BREAK.finditer(text):
        if text[start:match.start()].strip():
            result.append(Segment(start, match.start(), match.group(), text[start:match.start()]))
        start = match.end()
    if text[start:].strip():
        result.append(Segment(start, len(text), '', text[start:]))
    return result


def _statuses(text):
    return [name for name, pattern in STATUS_PATTERNS.items()
            if re.search(pattern, text.lower().replace('ё', 'е'))]


def _times(text, call):
    expressions = extract_time_expressions(text, call.started_at, direction='past')
    result = []
    for expr in expressions:
        if not expr.certain or expr.precision not in ('day', 'minute'):
            continue
        before = text[max(0, expr.span[0] - 5):expr.span[0]].lower()
        if re.search(r'\bне\s*$', before):
            continue  # Retraction/idiom, not a positive event timestamp.
        if _datetime(expr.start).date() <= _datetime(call.started_at).date():
            result.append(expr)
    return result


def _time_only(text, call):
    expressions = _times(text, call)
    if not expressions:
        return False
    masked = list(text)
    for expr in expressions:
        masked[expr.span[0]:expr.span[1]] = ' ' * (expr.span[1] - expr.span[0])
    return set(_norm(''.join(masked)).split()) <= FILLERS


def _field(text, expr):
    prefix = text[max(0, expr.span[0] - 20):expr.span[0]].lower().replace('ё', 'е')
    if re.search(r'\bдо\s*$', prefix) or expr.relation == 'before' or re.match(r'^до\s', expr.text.lower()):
        return 'valid_to'
    lower = text.lower().replace('ё', 'е')
    # Closing the ticket starts "completed"; only explicitly ending the status
    # itself is evidence for valid_to.
    closing = re.search(r'\b(?:статус|этап)\b[^,.!?;]{0,45}\b(?:закончил[а-я]*|завершил[а-я]*)\b', lower)
    if closing and abs(closing.end() - expr.span[0]) <= 20:
        return 'valid_to'
    if re.search(r'\b(?:еще|все\s+еще)\s*$', prefix):
        return None  # Observation at that time does not establish an onset.
    # An unpunctuated future event (e.g. an SMS arriving) cannot date a status.
    if re.search(r'\b(?:прид[её]т|приед[её]т|позвонит|будет)\b', prefix):
        return None
    return 'valid_from'


def _collect(text, call, turn_idx, *, allow_question=False):
    candidates = defaultdict(list)
    parts = _segments(text)
    events = []
    for i, part in enumerate(parts):
        values = _statuses(part.text)
        if not values:
            continue
        lower = part.text.lower().replace('ё', 'е')
        if ('?' in part.after and not allow_question) or re.search(r'\b(?:если|возможно|может|кажется|наверное)\b', lower):
            continue
        if re.search(r'\bне\s+(?:была?\s+)?(?:приня[а-я]*|в\s+работе|выполн[а-я]*)', lower):
            continue
        left, right = part.left, part.right
        # Comma-separated temporal adjuncts attach only when they contain no other event.
        if not _times(part.text, call) and i and parts[i - 1].after == ',' and _time_only(parts[i - 1].text, call):
            left = parts[i - 1].left
        if not _times(text[left:right], call) and i + 1 < len(parts) and part.after == ',' and _time_only(parts[i + 1].text, call):
            right = parts[i + 1].right
        window = text[left:right]
        expressions = _times(window, call)
        if len(values) > 1:
            unresolved = any(not e.certain for e in extract_time_expressions(window, call.started_at, direction='past'))
            if len({e.start for e in expressions}) != 1 or unresolved or not re.search(r'\bи\b', window.lower()):
                continue
            if re.search(r'\b(?:раньше|перед|до\s+этого)\b', window.lower()):
                continue
        fields = defaultdict(list)
        for expression in expressions:
            field = _field(window, expression)
            if field is None:
                continue
            evidence = {**expression.as_dict(), 'span': [left + expression.span[0], left + expression.span[1]],
                        'turn_idx': turn_idx, 'field': field, 'binding': 'status_clause'}
            fields[field].append((expression.start, evidence))
            for value in values:
                candidates[(value, field)].append((expression.start, evidence))
        # Repeating an endpoint without punctuation does not also create a start:
        # "до вчерашнего дня вчера он сменился".
        end_dates = {date for date, _ in fields.get('valid_to', [])}
        if end_dates:
            for value in values:
                key = (value, 'valid_from')
                candidates[key] = [(date, ev) for date, ev in candidates[key]
                                   if date not in end_dates or re.match(r'^(?:с|со)\s', ev['text'].lower())]
            fields['valid_from'] = [(date, ev) for date, ev in fields.get('valid_from', [])
                                    if date not in end_dates or re.match(r'^(?:с|со)\s', ev['text'].lower())]
        for value in values:
            events.append((value, left, right, fields))
    # Close only an explicit transition of this ticket in this same utterance.
    for old, new in zip(events, events[1:]):
        old_name, old_left, old_right, _ = old
        new_name, new_left, new_right, fields = new
        from_values = fields.get('valid_from', [])
        if RANK[new_name] <= RANK[old_name] or len({v for v, _ in from_values}) != 1:
            continue
        if candidates.get((old_name, 'valid_to')):
            continue  # Explicit endpoint wins over an inferred boundary.
        if not TRANSITION.search(text[old_right:new_right]):
            continue
        value, evidence = from_values[0]
        candidates[(old_name, 'valid_to')].append((value, {**evidence, 'field': 'valid_to',
                                                           'binding': 'explicit_status_transition'}))
    return candidates


def _confirmation(text):
    words = _norm(text).split()
    return bool(words) and len(words) <= 10 and set(words) <= CONFIRM


def _ticket_reference_safe(fact, text):
    numbers = set(re.findall(r'\b(?:заявк[а-я]*|обращени[а-я]*)\s*(?:№\s*|номер\s*)?(\d+)\b', text.lower()))
    wanted = re.search(r'\b(?:заявк[а-я]*|обращени[а-я]*)\s*(?:№\s*|номер\s*)?(\d+)\b', fact['object']['name'].lower())
    return len(numbers) <= 1 and (not numbers or not wanted or wanted[1] in numbers)


def repair_bound_dates(fact, call):
    """Preserve task conventions for other attributes; improve ticket evidence binding."""
    if fact['attribute'] != 'ticket_status':
        return repair_task_dates(fact, call)
    item = copy.deepcopy(fact)
    decision = {'attribute': fact['attribute'], 'source_turn_idx': fact['source']['turn_idx'],
                'status': 'unchanged', 'reason': 'no_bound_event_time', 'changes': [], 'evidence': []}
    if fact.get('polarity', 1) != 1:
        decision['reason'] = 'negative_fact'
        return item, decision
    value = _norm(fact['value'])
    if value not in STATUS_PATTERNS:
        return item, decision
    position = next((i for i, turn in enumerate(call.turns) if turn.idx == fact['source']['turn_idx']), None)
    if position is None:
        decision['reason'] = 'source_missing'
        return item, decision
    source = call.turns[position]
    if not _ticket_reference_safe(fact, source.text):
        decision['reason'] = 'ambiguous_ticket_reference'
        return item, decision
    candidates = _collect(source.text, call, source.idx)
    # Read earlier turns only for an actual short confirmation or temporal answer.
    if not candidates and (_confirmation(source.text) or _time_only(source.text, call)):
        for previous in call.turns[max(0, position - 2):position]:
            if set(_statuses(previous.text)) != {value} or not _ticket_reference_safe(fact, previous.text):
                continue
            if _confirmation(source.text):
                antecedent = _collect(previous.text, call, previous.idx, allow_question=True)
                for key, values in antecedent.items():
                    candidates[key].extend((date, {**evidence, 'binding': 'confirmed_previous_turn'}) for date, evidence in values)
            elif '?' in previous.text:
                for expression in _times(source.text, call):
                    field = 'valid_to' if re.search(r'\b(?:до\s+какого|закрыл|закончил)', previous.text.lower()) else 'valid_from'
                    candidates[(value, field)].append((expression.start, {**expression.as_dict(),
                                                         'turn_idx': source.idx, 'field': field,
                                                         'binding': 'answer_to_previous_question'}))
    for field in ('valid_from', 'valid_to'):
        values = candidates.get((value, field), [])
        if len({date for date, _ in values}) > 1:
            decision.update(status='abstained', reason='conflicting_event_times')
            return copy.deepcopy(fact), decision
        if values:
            proposed, evidence = values[0]
            decision['evidence'].append(evidence)
            if item.get(field) != proposed:
                decision['changes'].append({'field': field, 'before': item.get(field), 'after': proposed})
                item[field] = proposed
    if item.get('valid_from') and item.get('valid_to'):
        try:
            if _datetime(item['valid_from']) > _datetime(item['valid_to']):
                raise ValueError('reversed')
        except (ValueError, TypeError):
            decision.update(status='abstained', reason='interval_conflict', changes=[])
            return copy.deepcopy(fact), decision
    if decision['changes']:
        decision.update(status='changed', reason='bound_event_time')
    elif decision['evidence']:
        decision['reason'] = 'already_consistent'
    return item, decision
