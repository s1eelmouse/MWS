"""Recover missing current tariff/status facts from explicit text and the catalog.

Package 2.1 scores state by object+attribute and only looks at facts with
valid_to=None.  If the model omitted the current state, the call fails no matter
how good its dates are.  These rules add a candidate only when that candidate is
missing; they never override a predicted current state.
"""
import collections
import json
import re
from pathlib import Path

STATUS_PATTERNS = (
    ('назначен выезд специалиста', r'назначен[а-я]*\s+(?:\w+\s+){0,2}выезд[а-я]*'),
    ('выполнена', r'выполнен[а-я]*'),
    ('в работе', r'в\s+работ[еу]'),
    ('принята', r'принят[а-я]*'),
)
TARIFF_CONTEXT = re.compile(
    r'тариф|пакет|подключ|сейчас|текущ|переш|смен|польз|у вас|у меня|ваш|мой', re.I)
STATUS_CONTEXT = re.compile(
    r'сейчас|на данный момент|статус|уже|текущ|на сегодня', re.I)
WORD = re.compile(r'[а-яa-z0-9]+')


def _norm(text):
    return ' '.join(WORD.findall(text.lower().replace('ё', 'е')))


def _load_tariffs(catalog_path):
    if not catalog_path:
        return []
    path = Path(catalog_path)
    if path.is_dir():
        path = path / 'catalog.json'
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding='utf-8'))
    return [str(item) for item in payload.get('tariffs', [])]


def _client_object(facts):
    objects = [(fact.get('object') or {}) for fact in facts
               if (fact.get('object') or {}).get('type') == 'client']
    if not objects:
        return None
    name, object_id = collections.Counter(
        (obj.get('name') or '', obj.get('id')) for obj in objects).most_common(1)[0][0]
    return {'name': name, 'type': 'client', 'id': object_id}


def _ticket_object(facts, turns):
    for fact in facts:
        if fact.get('attribute') == 'ticket_status':
            obj = fact.get('object') or {}
            return {'name': obj.get('name') or '', 'type': obj.get('type') or 'ticket',
                    'id': obj.get('id')}
    text = ' '.join(turn.text.lower().replace('ё', 'е') for turn in turns)
    match = re.search(r'заявк[а-я]*\s*(\d+)', text)
    if match:
        return {'name': 'заявка ' + match.group(1), 'type': 'ticket', 'id': None}
    return None


def _tariff_candidate(turns, tariffs):
    catalog = [(_norm(name), name) for name in tariffs if len(_norm(name)) >= 4]
    found = []
    for turn in turns:
        normalized = _norm(turn.text)
        for token, name in catalog:
            for match in re.finditer(r'(?<![а-яa-z0-9])' + re.escape(token) +
                                     r'(?![а-яa-z0-9])', normalized):
                position = match.start()
                context = normalized[max(0, position - 70):position + len(token) + 70]
                if TARIFF_CONTEXT.search(context):
                    found.append((name, turn.idx, turn.speaker))
    if not found:
        return None
    counts = collections.Counter(name for name, _, _ in found)
    best = max(counts, key=lambda name: (counts[name],
                                         max(index for value, index, _ in found
                                             if value == name)))
    index = max(index for value, index, _ in found if value == best)
    speaker = next(speaker for value, idx, speaker in found
                   if value == best and idx == index)
    return best, index, speaker


def _status_candidate(turns):
    found = []
    for turn in turns:
        text = turn.text.lower().replace('ё', 'е')
        for value, pattern in STATUS_PATTERNS:
            for match in re.finditer(pattern, text):
                context = text[max(0, match.start() - 70):match.end() + 70]
                if STATUS_CONTEXT.search(context):
                    found.append((value, turn.idx, turn.speaker))
    return found[-1] if found else None


def recover_facts(facts, call, catalog_path=None):
    """Return (new_facts, decisions); only adds a missing current state."""
    output = [dict(fact) for fact in facts]
    decisions = []
    if not any(fact.get('attribute') == 'tariff' and not fact.get('valid_to')
               for fact in output):
        tariffs = _load_tariffs(catalog_path)
        candidate = _tariff_candidate(call.turns, tariffs) if tariffs else None
        obj = _client_object(output)
        if candidate and obj is not None:
            value, turn_idx, speaker = candidate
            output.append({
                'object': obj, 'attribute': 'tariff', 'value': value, 'polarity': 1,
                'valid_from': call.started_at, 'valid_to': None,
                'source': {'turn_idx': turn_idx, 'span': None, 'quote': None},
                'speaker': speaker, 'confirmed': None,
            })
            decisions.append({'component': 'fact_recovery', 'field': 'tariff',
                              'reason': 'missing_current_tariff', 'value': value,
                              'source_turn_idx': turn_idx})
    current_status = any(fact.get('attribute') == 'ticket_status'
                         and not fact.get('valid_to') for fact in output)
    if not current_status:
        candidate = _status_candidate(call.turns)
        obj = _ticket_object(output, call.turns)
        if candidate and obj is not None:
            value, turn_idx, speaker = candidate
            output.append({
                'object': obj, 'attribute': 'ticket_status', 'value': value, 'polarity': 1,
                'valid_from': call.started_at, 'valid_to': None,
                'source': {'turn_idx': turn_idx, 'span': None, 'quote': None},
                'speaker': speaker, 'confirmed': None,
            })
            decisions.append({'component': 'fact_recovery', 'field': 'ticket_status',
                              'reason': 'missing_current_status', 'value': value,
                              'source_turn_idx': turn_idx})
    return output, decisions
