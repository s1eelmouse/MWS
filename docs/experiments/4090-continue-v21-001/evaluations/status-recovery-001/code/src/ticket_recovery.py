"""Local, numbered-ticket status assertions; no model, gold or cross-call state.

Only current assertions qualify. A new observation cannot supply an unknown
historical boundary: conflicting supported history is left intact with an audit.
"""
import copy
from datetime import datetime
import re

POLICY = 'explicit-current-per-object-v2-ticket-assertions'
NUMBER = re.compile(r'\b(?:заявк[а-я]*|обращени[а-я]*)\s*(?:№\s*|номер\s*)?(\d+)\b')
STATUS = {
    'назначен выезд специалиста': re.compile(r'\b(?:назначен(?:а|о|ы)?\s+(?:\w+\s+){0,2}выезд|выезд\s+назначен(?:а|о|ы)?)\b'),
    'выполнена': re.compile(r'\bвыполнен(?:а|о|ы)?\b'),
    'в работе': re.compile(r'\b(?:в\s+работе|(?:взят[а-я]*|передан[а-я]*)\s+в\s+работу)\b'),
    'принята': re.compile(r'\bпринят(?:а|о|ы)?\b'),
}
SENTENCES = re.compile(r'[^.!?;…\n]+[.!?;…]?')
BREAK = re.compile(r'[,;]|\b(?:а|но|затем|потом|теперь)\b')
CURRENT = re.compile(r'\b(?:сейчас|сегодня|текущ[а-я]*|на\s+данный\s+момент|уже|статус)\b')
UNCERTAIN = re.compile(r'\b(?:если|возможно|наверное|кажется|может\s+быть|допустим|должн[а-я]*\s+быть)\b')
PLANNED = re.compile(r'\b(?:будет|будут|завтра|послезавтра|хочу|хотел[а-я]*|'
                     r'предлага[а-я]*|планир[а-я]*|обещал[а-я]*|обновлю|переведу|проверю)\b')
PAST = re.compile(r'\b(?:был[аои]?|пыла|значил[а-я]*|действовал[а-я]*|раньше|ранее|прежде|вчера|позавчера|назад|до|прошл[а-я]*)\b')
CHANGE = re.compile(r'\b(?:сменил[а-я]*|обновил[а-я]*|перевел[а-я]*|изменил[а-я]*)\b')
NEGATED = re.compile(r'\b(?:не|нет)\s+(?:(?:был[аои]?|является|значится|статус[а-я]*)\s+)*$')
OTHER_EVENT = re.compile(r'\b(?:мастер[а-я]*|инженер[а-я]*|специалист[а-я]*\s+(?:приед|позвон)|'
                         r'возврат[а-я]*|перезвон[а-я]*|приед[а-я]*|звонок|визит[а-я]*|срок[а-я]*)\b')


def status_key(value):
    hits = [key for key, pattern in STATUS.items() if pattern.search(value.lower().replace('ё', 'е'))]
    return hits[0] if len(hits) == 1 else None


def ticket_number(obj):
    hit = NUMBER.search((obj.get('name') or '').lower())
    return hit[1] if hit else None


def _object(facts, number):
    matches = [f['object'] for f in facts if f.get('attribute') == 'ticket_status'
               and ticket_number(f['object']) == number]
    ids = {o.get('id') for o in matches if o.get('id')}
    if len(ids) > 1:
        return None
    if matches:
        return copy.deepcopy(next((o for o in matches if o.get('id')), matches[0]))
    return {'name': 'заявка ' + number, 'type': 'ticket', 'id': None}


def _chunks(text):
    start = 0
    for hit in BREAK.finditer(text):
        yield start, text[start:hit.start()]
        start = hit.end()
    yield start, text[start:]


def _ordered(start, end):
    if not start:
        return True
    try:
        return datetime.fromisoformat(start.replace('Z', '+00:00')) <= datetime.fromisoformat(end.replace('Z', '+00:00'))
    except (TypeError, ValueError):
        return False


def assertions(facts, call):
    """Collect positive present assertions, scoped to a clause and one ticket.

    Whole-sentence questions/conditions are conservative abstentions. A clause
    following a comma may have its own unrelated negation/future event. Subject
    identity can cross clauses only when the sentence/turn/call has one number.
    """
    from temporal_case_v2 import canonical_text
    all_numbers = {n for t in call.turns for n in NUMBER.findall(t.text.lower())}
    all_numbers.update(n for f in facts if f.get('attribute') == 'ticket_status'
                       for n in [ticket_number(f['object'])] if n)
    result = []
    for turn in sorted(call.turns, key=lambda t: t.idx):
        text = canonical_text(turn.text)
        turn_numbers = set(NUMBER.findall(text))
        for sentence in SENTENCES.finditer(text):
            body = sentence.group()
            if '?' in body or UNCERTAIN.search(body):
                continue
            sentence_numbers = set(NUMBER.findall(body))
            for chunk_start, chunk in _chunks(body):
                # Contrasts isolate the scopes of "не принята, а в работе" and
                # "была принята, но сейчас ..." without borrowing a future visit.
                if chunk:
                    mentions = sorted((m.start(), m.end(), key)
                                      for key, pattern in STATUS.items() for m in pattern.finditer(chunk))
                    for i, (start, end, key) in enumerate(mentions):
                        left = mentions[i-1][1] if i else 0
                        right = mentions[i+1][0] if i+1 < len(mentions) else len(chunk)
                        prefix, suffix = chunk[left:start], chunk[end:right]
                        # Dates of a visit/refund after a status are not status dates.
                        other = OTHER_EVENT.search(suffix)
                        local_suffix = suffix[:other.start()] if other else suffix
                        scope = prefix + chunk[start:end] + local_suffix
                        if re.search(r'\b(?:какой|какая|каком|уточните|назовите|подскажите|скажите)\b', prefix) \
                                or re.match(r'\s+ли\b', suffix):
                            continue
                        if re.match(r'\s+не\s+(?:был[аои]?|является|значится)\b', suffix):
                            continue
                        if (PLANNED.search(body[:chunk_start]) or PAST.search(body[:chunk_start])) \
                                and not re.search(r'\b(?:сейчас|сегодня|теперь)\b', scope):
                            continue
                        if NEGATED.search(prefix) or PLANNED.search(prefix) or re.match(
                                r'\s*(?:будет|будут)\b', suffix):
                            continue
                        if PAST.search(scope) and not re.search(r'\b(?:сейчас|теперь)\b', scope):
                            continue
                        if re.search(r'\bдо\s+(?:вчера|сегодня|завтра|пятницы|понедельника)\b', local_suffix):
                            continue
                        explicit = bool(CURRENT.search(scope))
                        named = set(NUMBER.findall(chunk))
                        # Bare present declarations by the operator qualify, but
                        # an isolated participle from the client does not.
                        if not explicit and not (turn.speaker == 'operator' and named):
                            continue
                        pool = named or sentence_numbers or turn_numbers or all_numbers
                        if len(pool) != 1:
                            continue
                        number = next(iter(pool))
                        known = {ticket_number(f['object']) for f in facts
                                 if f.get('attribute') == 'ticket_status'}
                        if number not in known and any(n != number and n.startswith(number)
                                                       for n in all_numbers):
                            # An ASR-clipped number must not create a second ticket
                            # or silently inherit the full ticket's identity.
                            continue
                        obj = _object(facts, number)
                        if obj is None:
                            continue
                        today = bool(re.search(r'\bсегодня\b', scope))
                        now_change = bool(CHANGE.search(scope) and re.search(r'\b(?:сейчас|сегодня)\b', scope))
                        anchor = datetime.fromisoformat(call.started_at.replace('Z', '+00:00'))
                        # Existing recovery uses observation time when onset is
                        # not stated. Never present that fallback as a transition.
                        date = anchor.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() if today else call.started_at
                        result.append({'value': key, 'object': obj, 'number': number,
                                       'turn_idx': turn.idx, 'speaker': turn.speaker,
                                       'valid_from': date, 'transition_boundary': date if now_change else None,
                                       'explicit_current': explicit, 'text': scope.strip(),
                                       'date_basis': 'explicit_today' if today else 'current_observation',
                                       'order': (turn.idx, sentence.start()+chunk_start+start)})
    return result


def reconcile_ticket_statuses(facts, call):
    output = copy.deepcopy(facts)
    evidence = assertions(output, call)
    by_number = {}
    for item in evidence:
        by_number.setdefault(item['number'], []).append(item)
    audit = []
    turns = {t.idx: t for t in call.turns}

    def log(item, action, reason, **extra):
        audit.append({'component': 'fact_recovery', 'field': 'ticket_status', 'action': action,
                      'reason': reason, 'policy': POLICY, 'object': copy.deepcopy(item['object']),
                      'value': item['value'], 'source_turn_idx': item['turn_idx'],
                      'date_basis': item['date_basis'], 'evidence': item['text'], **extra})

    for number, choices in by_number.items():
        latest_order = max(c['order'] for c in choices)
        latest = [c for c in choices if c['order'] == latest_order]
        if len({c['value'] for c in latest}) != 1:
            continue
        item = latest[-1]
        related = [f for f in output if f.get('attribute') == 'ticket_status'
                   and ticket_number(f['object']) == number]
        opened = [f for f in related if f.get('valid_to') is None and f.get('polarity', 1) == 1]
        if any(f.get('polarity') == -1 and status_key(f['value']) == item['value']
               and f['source']['turn_idx'] >= item['turn_idx'] for f in related):
            log(item, 'abstained', 'explicit_negative_prediction_conflict')
            continue
        conflicting = [f for f in opened if status_key(f['value']) != item['value']]
        if conflicting and (item['speaker'] != 'operator' or not item['explicit_current']):
            log(item, 'abstained', 'replacement_requires_explicit_operator_assertion')
            continue
        edits = []
        for fact in conflicting:
            idx = fact['source']['turn_idx']
            source = turns.get(idx)
            key = status_key(fact['value'])
            # Do not remove a later claim, an unknown/free status, or evidence
            # that names another ticket. All are ambiguity, not proof of error.
            if source is None or idx > item['turn_idx'] or key is None:
                edits = None
                break
            named = set(NUMBER.findall(source.text.lower()))
            if named and named != {number}:
                edits = None
                break
            supported = bool(STATUS[key].search(source.text.lower().replace('ё', 'е')))
            boundary = item['transition_boundary']
            if not supported:
                edits.append(('remove', fact, None))
            elif boundary and _ordered(fact.get('valid_from'), boundary):
                edits.append(('close', fact, boundary))
            else:
                edits = None
                break
        if edits is None:
            log(item, 'abstained', 'supported_history_without_exact_transition_boundary')
            continue
        for action, fact, boundary in edits:
            before = copy.deepcopy(fact)
            if action == 'remove':
                output.remove(fact)
                log(item, action, 'unsupported_current_status', before=before)
            else:
                fact['valid_to'] = boundary
                log(item, action, 'explicit_current_status_transition', before=before, after=copy.deepcopy(fact))
        if any(status_key(f['value']) == item['value'] for f in opened if f not in conflicting):
            continue
        fresh = {'object': copy.deepcopy(item['object']), 'attribute': 'ticket_status',
                 'value': item['value'], 'polarity': 1, 'valid_from': item['valid_from'],
                 'valid_to': None, 'source': {'turn_idx': item['turn_idx'], 'span': None, 'quote': None},
                 'speaker': item['speaker'], 'confirmed': None}
        output.append(fresh)
        log(item, 'add', 'missing_current_status', fact_index=len(output)-1, after=copy.deepcopy(fresh))
    return output, audit
