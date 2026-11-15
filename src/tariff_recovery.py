"""Recover or correct omitted current states from explicit evidence, never gold.

Package 2.1 scores current state by object+attribute and only looks at facts with
valid_to=None.  If the model omitted the current state, the call fails no matter
how good its dates are.  This module adds a candidate only from explicit text and
the supplied catalog; it never reads gold labels.

Policy v2 also corrects an existing current fact when a later explicit statement
contradicts it (an explicit switch, confirmation or system/current assertion).
The old current fact is closed at the call timestamp, not silently overwritten.
Negations, offers, plans, questions without a first-person/system assertion,
historical mentions and device names such as «мини-приставка» are rejected.
"""
import copy
from functools import lru_cache
import json
from pathlib import Path
import re

WORD = re.compile(r'[а-яa-z0-9]+')
TICKET = re.compile(r'\b(?:заявк[а-я]*|обращени[а-я]*)\s*(?:№\s*|номер\s*)?(\d+)\b')
CLAUSES = re.compile(r'[^.!?;\n]+[.!?;]?')
UNASSERTED = re.compile(r'\b(?:не|нет|если|возможно|наверное|кажется|хотел[а-я]*|'
                       r'хочу|предлага[а-я]*|планир[а-я]*|будет|будут|завтра|раньше|'
                       r'ранее|прежде|бы[лв][а-я]*|вчера)\b')
CURRENT = re.compile(r'\b(?:сейчас|текущ[а-я]*|на\s+данный\s+момент|на\s+сегодня|уже|статус)\b')
TARIFF_MENTION = re.compile(r'\b(?:т[ао][рф]и[фв]|фариф|тари|пакет|плачу\s+по)\b')
TARIFF_FIRST = re.compile(r'\b(?:у\s+меня|мой|моя|мое|мои|плачу|пользуюсь)\b')
TARIFF_SECOND = re.compile(r'\b(?:у\s+вас|ваш|ваша|ваше|ваши)\b')
TARIFF_ASSERT = re.compile(r'\b(?:вижу|смотрю|по\s+тарифу|в\s+базе|в\s+системе|'
                           r'у\s+нас|активен|подключен[а-я]*|идёт|идет|верно|правильно)')
TARIFF_PRESENT = re.compile(r'\b(?:сейчас|теперь|оста[её]мся|оста[её]тся|подключ[её]н|'
                            r'активен|действует|никто\s+не\s+менял)')
TARIFF_OFFER = re.compile(r'\b(?:хотите|хотит|подключить|подключим|предлага|предлож|'
                          r'давайте|рассмотр|интересу|нужен|нужна|можно|если)')
TARIFF_DEVICE = re.compile(r'(?:приставк|роутер|телевизор|устройств|телефон|камер|'
                           r'модем|колонк|планшет|ноутбук)')
TARIFF_SWITCH = re.compile(r'\b(?:перешл[а-я]*|сменил[а-я]*|перевел[а-я]*|'
                           r'перевели|сменили)\b')
CONFIRM = re.compile(r'\b(?:да|ага|угу|верно|точно|правильно|именно|'
                     r'он\s+самый|оно\s+самое)\b')
STATUS_PATTERNS = (
    ('назначен выезд специалиста', re.compile(r'\bназначен[а-я]*\s+(?:\w+\s+){0,2}выезд[а-я]*\b')),
    ('выполнена', re.compile(r'\bвыполнен[а-я]*\b')),
    ('в работе', re.compile(r'\bв\s+работ[еу]\b')),
    ('принята', re.compile(r'\bпринят[а-я]*\b')),
)


def _norm(text):
    return ' '.join(WORD.findall(text.lower().replace('ё', 'е')))


def _tariff_norm(text):
    # '+' distinguishes catalog tariffs; hyphens separate words, as in the package.
    return ' '.join(re.findall(r'[а-яa-z0-9+]+', text.lower().replace('ё', 'е')))


def _value_key(value):
    return _tariff_norm(value)


@lru_cache(maxsize=8)
def _tariffs(path, modified_ns, size):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    values = payload.get('tariffs', [])
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError('Catalog tariffs must be a list of strings')
    return tuple((value, re.compile(r'(?<![а-яa-z0-9+])' + re.escape(_tariff_norm(value)) + r'(?![а-яa-z0-9+])'))
                 for value in values if len(_tariff_norm(value)) >= 4)


def _load_tariffs(path):
    if not path:
        return ()
    path = Path(path)
    if path.is_dir():
        path /= 'catalog.json'
    info = path.stat()  # A configured missing catalog is an explicit error.
    return _tariffs(str(path.resolve()), info.st_mtime_ns, info.st_size)


def _identity(obj):
    if obj.get('id'):
        return ('id', obj['id'])
    name = _norm(obj.get('name', ''))
    return ('name', name, obj.get('type')) if name else None


def _same_object(a, b):
    if a.get('id') and b.get('id'):
        return a['id'] == b['id']
    if not a.get('id') and not b.get('id') and not a.get('name') and not b.get('name'):
        return a.get('type') == b.get('type') == 'client'
    return bool(_norm(a.get('name', ''))) and _norm(a['name']) == _norm(b.get('name', ''))


def _unique_objects(facts, kind):
    objects = {}
    for fact in facts:
        obj = fact.get('object') or {}
        if obj.get('type') == kind or (kind == 'ticket' and fact.get('attribute') == 'ticket_status'):
            key = _identity(obj)
            if key is None and kind == 'client':
                key = ('anonymous', kind)
            if key is not None:
                objects.setdefault(key, copy.deepcopy(obj))
    return list(objects.values())


def _tariff_hits(text, tariffs):
    """Catalog names in a clause, without the shorter name inside a longer one."""
    normalized = _tariff_norm(text)
    hits = [(value, hit) for value, pattern in tariffs for hit in pattern.finditer(normalized)]
    hits = [(value, hit) for value, hit in hits if not any(
        other.start() <= hit.start() and other.end() >= hit.end()
        and other.end() - other.start() > hit.end() - hit.start() for _, other in hits)]
    return normalized, hits


def _tariff_candidate(text, tariffs, *, question, speaker, previous_text):
    """Catalog evidence with the role of each mention kept separate."""
    normalized, hits = _tariff_hits(text, tariffs)
    # Do not treat "мой" in tariff "Мой онлайн" as a personal assertion.
    masked = list(normalized)
    for _name,hit in hits:
        masked[hit.start():hit.end()] = ' '*(hit.end()-hit.start())
    outside = ''.join(masked)
    intent = re.compile(r'\b(?:если|возможно|наверное|кажется|хочу|хотел[а-я]*|'
                        r'планир[а-я]*|предлага[а-я]*|завтра|будет|будут)\b')
    inquiry = re.compile(r'\b(?:какой|какая|случайно|подскаж[а-я]*|уточн[а-я]*|'
                         r'не\s+переходил[а-я]*)\b')
    # A completed switch has a grammatical destination, not "last name seen".
    switch = TARIFF_SWITCH.search(normalized)
    if switch:
        prefix = normalized[:switch.start()]
        if question or intent.search(prefix) or re.search(r'\bне\s*$',prefix):
            return [],False
        targets = [(name,hit) for name,hit in hits if hit.start()>switch.end()
                   and re.search(r'\bна\s+(?:тариф\s+)?$',normalized[switch.end():hit.start()])]
        if len(targets)!=1:
            return [],False
        name,hit=targets[0]
        if re.search(r'\b(?:предлагал[а-я]*|хотел[а-я]*|если)\b',normalized[switch.end():hit.start()]):
            return [],False
        return [name],True
    if not hits:
        return [],False
    # An assertion followed by a question about consequences is not an inquiry
    # about the tariff's identity. Preserve the assertion's own evidence only.
    tag = re.search(r'(?:—|-|,)\s*это\b',text)
    consequence_question = bool(question and speaker=='client' and tag
                                and TARIFF_FIRST.search(outside))
    system_identity_question = bool(question and speaker=='operator'
        and TARIFF_ASSERT.search(outside) and re.search(r'это\s+вы\s*\?',text))
    if question and not consequence_question and not system_identity_question:
        return [],False
    if intent.search(outside) and not consequence_question:
        return [],False
    if inquiry.search(outside):
        return [],False
    # Confirmation must refer to a previously named actual tariff, not an offer.
    if speaker=='client' and len(normalized.split())<=16 and CONFIRM.search(outside):
        previous_names={n for n,_h in _tariff_hits(previous_text,tariffs)[1]}
        values={n for n,_h in hits}
        if len(values)==1 and values<=previous_names and not TARIFF_OFFER.search(previous_text) \
                and not re.search(r'\b(?:не|нет|если|возможно)\b',previous_text):
            return list(values),True
    names=[]
    for name,hit in hits:
        prefix=normalized[:hit.start()];suffix=normalized[hit.end():]
        external_prefix=outside[:hit.start()];external_suffix=outside[hit.end():]
        if re.search(r'\b(?:не|нет)\s+(?:тариф\s+)?$',prefix):continue
        if re.search(r'\b(?:раньше|ранее|прежде|был[а-я]*)\b',external_prefix) \
                and not re.search(r'\b(?:сейчас|теперь)\b',external_prefix):continue
        if re.match(r'\s*(?:не\s+подключен[а-я]*|отключен[а-я]*|больше\s+не)',suffix):continue
        if TARIFF_OFFER.search(external_prefix) and not consequence_question:continue
        if re.search(r'\b(?:предлагал[а-я]*|рассматривал[а-я]*)\b',external_suffix):continue
        personal=TARIFF_FIRST.search(external_prefix) or TARIFF_SECOND.search(external_prefix)
        present=TARIFF_PRESENT.search(external_prefix+' '+external_suffix)
        system=speaker=='operator' and TARIFF_ASSERT.search(external_prefix)
        if not (personal or present or system):continue
        if TARIFF_DEVICE.search(outside) and not TARIFF_MENTION.search(outside):continue
        names.append(name)
    names=list(dict.fromkeys(names))
    return (names,True) if len(names)==1 else ([],False)


def _candidates(facts, call, tariffs):
    clients = _unique_objects(facts, 'client')
    if not clients:
        # The package allows an unnamed client; use it instead of inventing a name.
        clients = [{'name': '', 'type': 'client', 'id': None}]
    if len(clients) != 1:
        clients = []

    turns = sorted(call.turns, key=lambda turn: turn.idx)
    previous = {}
    last = ''
    for turn in turns:
        previous[turn.idx] = last
        last = turn.text.lower().replace('ё', 'е')

    candidates = []
    for turn in turns:
        for match in CLAUSES.finditer(turn.text.lower().replace('ё', 'е')):
            text = match.group()
            question = '?' in text
            if len(clients) != 1:
                continue
            values, force = _tariff_candidate(
                text, tariffs, question=question, speaker=turn.speaker,
                previous_text=previous.get(turn.idx, ''))
            if len(values) == 1:
                candidates.append(('tariff', values[0], clients[0], turn, force))
    return candidates


def recover_facts(facts, call, catalog_path=None):
    """Return copies plus an audit. Explicit current evidence wins for its object."""
    output = copy.deepcopy(facts)
    tariffs = _load_tariffs(catalog_path)
    grouped = {}
    for attribute, value, obj, turn, force in _candidates(output, call, tariffs):
        grouped.setdefault((attribute, _identity(obj)), []).append((value, obj, turn, force))
    decisions = []
    for (attribute, _key), candidates in grouped.items():
        forced = [candidate for candidate in candidates if candidate[3]]
        pool = forced if forced else candidates
        latest = max(turn.idx for _, _, turn, _ in pool)
        current = [(value, obj, turn, force) for value, obj, turn, force in pool
                   if turn.idx == latest]
        if len({value for value, _, _, _ in current}) != 1:
            continue
        value, obj, turn, force = current[-1]
        related = [fact for fact in output if fact.get('attribute') == attribute
                   and _same_object(fact.get('object') or {}, obj)]
        current_related = [fact for fact in related
                           if not fact.get('valid_to') and fact.get('polarity', 1) == 1]
        if any(_value_key(fact.get('value', '')) == _value_key(value) for fact in current_related):
            continue
        if any(fact.get('polarity') == -1 and _value_key(fact.get('value', '')) == _value_key(value)
               for fact in related):
            continue
        if current_related and not force:
            continue
        if force:
            for fact in current_related:
                if _value_key(fact.get('value', '')) != _value_key(value):
                    if not fact.get('valid_from') or fact['valid_from'] <= call.started_at:
                        fact['valid_to'] = call.started_at
        output.append({'object': obj, 'attribute': attribute, 'value': value, 'polarity': 1,
                       'valid_from': call.started_at, 'valid_to': None,
                       'source': {'turn_idx': turn.idx, 'span': None, 'quote': None},
                       'speaker': turn.speaker, 'confirmed': None})
        if attribute == 'tariff':
            reason = 'explicit_current_tariff_correction' if current_related else 'missing_current_tariff'
        else:
            reason = 'explicit_current_status_correction' if current_related else 'missing_current_status'
        decisions.append({'component': 'fact_recovery', 'fact_index': len(output) - 1,
                          'field': attribute, 'reason': reason, 'value': value,
                          'object': copy.deepcopy(obj), 'source_turn_idx': turn.idx,
                          'policy': 'unified-guarded-tariffs-v1', 'action': 'add'})
    return output, decisions
