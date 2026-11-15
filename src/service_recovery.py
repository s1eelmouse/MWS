"""Recover a missing service/device state only from a locally named assertion.

Objects come from the input catalog or existing predictions, never from labels.
Multiple named objects, questions without confirmation, hypothetical/future and
historical statements are abstentions. Existing current states are preserved.
"""
import copy
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
import re
from difflib import SequenceMatcher

CLAUSE = re.compile(r'(?<!\d)[.!?;\n]+|[.!?;\n]+(?!\d)|\b(?:а|но|зато|теперь)\b', re.I)
HEDGE = re.compile(r'\b(?:если|возможно|наверное|кажется|может|будет|будут|завтра|хочу|хотел[а-я]*|предлага[а-я]*|должен|должна|должно)\b')
HISTORICAL = re.compile(r'\b(?:раньше|ранее|прежде|работал[аои]?|был[аои]?|стоял[аои]?)\b')
PREDICATE = re.compile(r'\b(?:не\s+)?(?:работает|работают|подключ[её]н(?:а|о|ы|ным|ной)?|активен|активна|активно|активны|заблокирован(?:а|о|ы)?)\b')
CONFIRM = re.compile(r'^\s*(?:ну\s+)?(?:да\b|именно\b|верно\b|все\s+верно\b|так\s+и\s+есть\b)')


@lru_cache(maxsize=8)
def _catalog(path, mtime, size):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _tokens(text):
    from catalog_hints import words
    return [(word, match.start(), match.end())
            for match in re.finditer(r'[а-яёa-z0-9+]+', text.lower()) for word in words(match.group())]


@lru_cache(maxsize=4096)
def _value_tokens(value):
    from catalog_hints import words
    return tuple(words(value))


@lru_cache(maxsize=50_000)
def _object_word(w, token):
    if w == token:
        return True
    if min(len(w), len(token)) <= 3:
        return False
    if abs(len(w) - len(token)) <= 2 and (w.startswith(token) or token.startswith(w)):
        return True
    return SequenceMatcher(None, w, token).ratio() >= .86


def _matches(tokens, value, *, object_name=False):
    from catalog_hints import close
    wanted = _value_tokens(value)
    result = []
    for i in range(len(tokens) - len(wanted) + 1):
        found = tokens[i:i + len(wanted)]
        same = _object_word if object_name else close
        if wanted and all(('+' in w) == ('+' in token[0]) and same(w, token[0])
                          for w, token in zip(wanted, found)):
            result.append((found[0][1], found[-1][2]))
    return result


def _parts(text):
    start = 0
    for match in CLAUSE.finditer(text):
        yield text[start:match.start()], start, '?' in match.group()
        start = match.end()
    yield text[start:], start, False


def recover_services(facts, call, catalog_path):
    from fact_recovery import _norm, _same_object
    from temporal_case_v2 import day_cues, _iso
    if not catalog_path:
        return copy.deepcopy(facts), []
    path = Path(catalog_path)
    if path.is_dir():
        path /= 'catalog.json'
    info = path.stat()
    catalog = _catalog(str(path.resolve()), info.st_mtime_ns, info.st_size)
    turns = sorted(call.turns, key=lambda t: t.idx)
    first_client = next((t.text for t in turns if t.speaker == 'client'), '')
    introduction = _tokens(first_client)
    objects = []
    for f in facts:
        obj = f['object']
        if obj.get('type') in ('service', 'device') and obj.get('name'):
            if not any(obj == other for other in objects):
                objects.append(copy.deepcopy(obj))
    for section, kind in (('services', 'service'), ('devices', 'device')):
        for name in catalog.get(section, []):
            if kind == 'device' and not _matches(introduction, name, object_name=True):
                # Incidental equipment in a later troubleshooting instruction is
                # not a new state object unless the model already identified it.
                continue
            if not any(_norm(obj['name']) == _norm(name) for obj in objects):
                objects.append({'name': name, 'type': kind, 'id': None})
    output, audit, grouped = copy.deepcopy(facts), [], {}
    started = datetime.fromisoformat(call.started_at.replace('Z', '+00:00'))
    symptoms = [v for v in catalog.get('symptoms', []) if v != 'больше не используется']
    for pos, turn in enumerate(turns):
        for text, offset, question in _parts(turn.text):
            low = text.lower().replace('ё', 'е')
            if HEDGE.search(low) or HISTORICAL.search(low):
                continue
            if question:
                answer = turns[pos + 1].text.lower().replace('ё', 'е') if pos + 1 < len(turns) else ''
                if not CONFIRM.search(answer) or re.search(r'\b(?:не|нет|но|другой)\b', answer[:90]):
                    continue
            predicates = list(PREDICATE.finditer(low))
            if not predicates and not re.search(r'\b(?:нет|пропал[а-я]*|не\s+\w+|пищит|зависает|тормозит|слаб[а-я]*|обрыв[а-я]*)\b', low):
                continue
            tokens = _tokens(text)
            found = [(obj, span) for obj in objects for span in _matches(tokens, obj['name'], object_name=True)]
            found = [(obj, span) for obj, span in found if not any(
                other[0] <= span[0] and other[1] >= span[1] and other[1] - other[0] > span[1] - span[0]
                for _other_obj, other in found)]
            names = {_norm(obj['name']) for obj, _ in found}
            if len(names) != 1:
                continue
            identities = {(obj.get('id'), obj.get('type')) for obj, _ in found}
            if len(identities) != 1:
                continue
            obj, obj_span = found[0]
            related = [f for f in output if f['attribute'] == 'service_state' and _same_object(f['object'], obj)]
            if any(f.get('valid_to') is None for f in related):
                continue
            matched = [(value, span) for value in symptoms for span in _matches(tokens, value)]
            matched = [(value, span) for value, span in matched if not any(
                other[0] <= span[0] and other[1] >= span[1] and other[1] - other[0] > span[1] - span[0]
                for _v, other in matched)]
            value, end = None, None
            if len({v for v, _span in matched}) == 1:
                value, span = matched[0]
                near_predicates = [m for m in predicates if 0 <= m.start() - obj_span[1] <= 30]
                if near_predicates and not any(span[0] <= m.start() < span[1] for m in near_predicates):
                    continue  # A summary symptom elsewhere is not this object's local predicate.
                if re.search(r'\bне\s*$', low[max(0, span[0] - 5):span[0]]) and not value.startswith(('не ', 'нет ')):
                    continue
                end = max(obj_span[1], span[1])
            elif not matched:
                near = [m for m in predicates if 0 <= m.start() - obj_span[1] <= 55]
                if len(near) != 1:
                    continue
                end = near[0].end()
                value = text[obj_span[0]:end].strip(' ,:—-')
            if not value or end is None or end - obj_span[0] > 200:
                continue
            # No unscoped inference for another subject or a quoted person's denial.
            if re.search(r'\b(?:не\s+пользуюсь|не\s+подключал[а-я]*|не\s+заказывал[а-я]*)\b', low):
                continue
            date = call.started_at
            if value in symptoms:
                cues = day_cues(text, started.date())
                days = {d for d, role, _p in cues if role == 'since'}
                if len(days) == 1:
                    date = _iso(next(iter(days)), started.tzinfo)
            lo, hi = offset + obj_span[0], offset + end
            item = {'object': copy.deepcopy(obj), 'attribute': 'service_state', 'value': value,
                    'polarity': 1, 'valid_from': date, 'valid_to': None,
                    'source': {'turn_idx': turn.idx, 'span': [lo, hi], 'quote': turn.text[lo:hi]},
                    'speaker': turn.speaker, 'confirmed': None}
            grouped.setdefault(_norm(obj['name']), []).append(item)
    for name, candidates in grouped.items():
        latest = max(f['source']['turn_idx'] for f in candidates)
        current = [f for f in candidates if f['source']['turn_idx'] == latest]
        if len({f['value'] for f in current}) != 1:
            continue
        item = current[0]
        if any(f['attribute'] == 'service_state' and _same_object(f['object'], item['object'])
               and f.get('valid_to') is None for f in output):
            continue
        output.append(item)
        audit.append({'component': 'fact_recovery', 'fact_index': len(output) - 1,
                      'field': 'service_state', 'reason': 'explicit_named_service_state',
                      'value': item['value'], 'object': copy.deepcopy(item['object']),
                      'source_turn_idx': item['source']['turn_idx']})
    return output, audit
