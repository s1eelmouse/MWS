"""Recover explicit current tariff assertions and acknowledged corrections.

Catalog matching is literal or a bounded adjective case form, longest-first. A proposed or merely queried
tariff is never current without an affirmative response. Existing model states
are superseded only when the transcript explicitly identifies them as historical.
"""
import copy
import re
from datetime import datetime
from functools import lru_cache

HEDGE = re.compile(r'\b(?:если|возможно|наверное|кажется|хочу|хотел[а-я]*|предлага[а-я]*|планир[а-я]*|завтра|будет|будут)\b')
PAST = re.compile(r'\b(?:раньше|ранее|прежде|давно|был[аои]?|стоял[а-я]*)\b')
CURRENT = re.compile(r'\b(?:сейчас|теперь|текущ[а-я]*|у\s+меня|у\s+вас|плачу\s+по|сижу|пользуюсь|подключен[а-я]*)\b')
AFFIRM = re.compile(r'^\s*(?:ну\s+)?(?:да\b|верно\b|точно\b|именно\b|он\s+самый\b|все\s+верно\b)')
WRAPPER = {'тариф', 'тарифный', 'план', 'пакет', 'подключен', 'подключена', 'активен',
           'активный', 'текущий', 'действующий', 'сейчас', 'у', 'клиента'}


@lru_cache(maxsize=512)
def _inflected_pattern(name):
    from fact_recovery import _tariff_norm
    tokens = _tariff_norm(name).split()
    parts, changed = [], False
    for token in tokens:
        if len(token) >= 5 and token.endswith(('ый', 'ий', 'ой')):
            parts.append(re.escape(token[:-2]) + r'(?:ый|ий|ой|ого|его|ому|ему|ым|им|ом|ем)')
            changed = True
        else:
            parts.append(re.escape(token))
    return re.compile(r'(?<![а-яa-z0-9+])' + r'\s+'.join(parts) + r'(?![а-яa-z0-9+])') if changed else None


def hits(text, tariffs):
    found = [(name, m) for name, pattern in tariffs for m in pattern.finditer(text)]
    for name, _pattern in tariffs:
        pattern = _inflected_pattern(name)
        if pattern is not None:
            found.extend((name, m) for m in pattern.finditer(text)
                         if not any(other == name and hit.span() == m.span() for other, hit in found))
    return [(name, m) for name, m in found if not any(
        n.start() <= m.start() and n.end() >= m.end() and n.end() - n.start() > m.end() - m.start()
        for _other, n in found)]


def refine_tariffs(facts, call, tariffs):
    from fact_recovery import _tariff_norm, _norm, _unique_objects, _same_object
    output, audit = copy.deepcopy(facts), []
    if not tariffs:
        return output, audit
    # A known tariff with a generic wrapper still denotes the same tariff.
    for i, fact in enumerate(output):
        if fact['attribute'] != 'tariff':
            continue
        text = _tariff_norm(fact['value'])
        found = hits(text, tariffs)
        if len(found) == 1:
            name, match = found[0]
            outside = set((text[:match.start()] + ' ' + text[match.end():]).split())
            if outside <= WRAPPER and fact['value'] != name:
                audit.append({'component': 'fact_recovery', 'fact_index': i, 'field': 'tariff',
                              'reason': 'canonical_tariff_wrapper', 'before': fact['value'], 'value': name,
                              'source_turn_idx': fact['source']['turn_idx']})
                fact['value'] = name
    clients = _unique_objects(output, 'client')
    if len(clients) != 1:
        return output, audit
    obj = clients[0]
    turns = sorted(call.turns, key=lambda t: t.idx)
    candidates = []
    for pos, turn in enumerate(turns):
        # Split contrast/correction clauses, retaining the local question scope.
        for clause in re.split(r'[.!;\n]+|\b(?:а|но|зато)\b', turn.text.lower().replace('ё', 'е')):
            text = _tariff_norm(clause)
            found = hits(text, tariffs)
            for name, match in found:
                prefix = text[max(0, match.start() - 70):match.start()]
                suffix = text[match.end():match.end() + 65]
                if HEDGE.search(prefix) or re.search(r'\b(?:не|нет)\s+(?:тариф\s+)?$', prefix):
                    continue
                if re.match(r'\s*(?:не\s+подключен[а-я]*|отключен[а-я]*|больше\s+не)', suffix):
                    continue
                if PAST.search(prefix + ' ' + suffix) and not re.search(r'\b(?:теперь|сейчас)\b', prefix + ' ' + suffix):
                    continue
                asserted = bool(CURRENT.search(text))
                context = bool(re.search(r'\b(?:тариф[а-я]*|пакет[а-я]*|плачу\s+по)\b', text)
                               or re.search(r'\bна\s+$', prefix) and re.search(r'\bсижу\b', suffix))
                # A repeated catalog name can answer a tariff question from the previous turn.
                previous = _tariff_norm(turns[pos - 1].text) if pos else ''
                tariff_context = any(re.search(r'тариф|пакет', turns[j].text.lower())
                                     for j in range(max(0, pos - 2), pos))
                if tariff_context and re.search(r'\b(?:теперь|сейчас)\b', text):
                    context = True
                confirmed = bool(AFFIRM.search(text) and name in [n for n, _m in hits(previous, tariffs)]
                                 and re.search(r'тариф|пакет', previous))
                if '?' in clause:
                    next_text = _tariff_norm(turns[pos + 1].text) if pos + 1 < len(turns) else ''
                    confirmed = bool(AFFIRM.search(next_text) and not re.search(r'\b(?:не|нет|но|другой)\b', next_text[:90]))
                    # «Тариф у меня X — это не влияет?» asserts X, queries its effect.
                    effect_question = bool(re.search(r'\bу\s+меня\b', prefix)
                                           and re.search(r'влия|меша|важно', suffix))
                    if not (confirmed or effect_question):
                        continue
                if not (context and asserted or confirmed):
                    continue
                candidates.append((name, turn, pos))
    if not candidates:
        return output, audit
    latest = max(turn.idx for _, turn, _ in candidates)
    selected = [(name, turn, pos) for name, turn, pos in candidates if turn.idx == latest]
    if len({name for name, _turn, _pos in selected}) != 1:
        return output, audit
    value, turn, pos = selected[0]
    related = [f for f in output if f['attribute'] == 'tariff' and _same_object(f['object'], obj)]
    if any(f['polarity'] == -1 and _tariff_norm(f['value']) == _tariff_norm(value) for f in related):
        return output, audit
    current = [f for f in related if f['polarity'] == 1 and f.get('valid_to') is None]
    if any(_tariff_norm(f['value']) == _tariff_norm(value) for f in current):
        return output, audit
    historical = []
    for old in current:
        # The old tariff must actually be called old, not just contradicted.
        old_name = _tariff_norm(old['value'])
        for source in turns[max(0, pos - 1):pos + 2]:
            for chunk in re.split(r'[.!?;\n]+|\b(?:а|но)\b', source.text.lower().replace('ё', 'е')):
                normalized = _tariff_norm(chunk)
                if old_name and old_name in normalized and PAST.search(normalized):
                    historical.append(old)
                    break
    if current and len({id(f) for f in historical}) != len(current):
        return output, audit
    for old in current:
        if old.get('valid_from'):
            try:
                if datetime.fromisoformat(old['valid_from'].replace('Z', '+00:00')) > datetime.fromisoformat(call.started_at.replace('Z', '+00:00')):
                    return output, audit
            except (TypeError, ValueError):
                return output, audit
    for old in current:
        old['valid_to'] = call.started_at
        audit.append({'component': 'fact_recovery', 'field': 'tariff',
                      'reason': 'explicit_historical_tariff_closed', 'value': old['value'],
                      'source_turn_idx': turn.idx})
    new = {'object': copy.deepcopy(obj), 'attribute': 'tariff', 'value': value, 'polarity': 1,
           'valid_from': call.started_at, 'valid_to': None,
           'source': {'turn_idx': turn.idx, 'span': None, 'quote': None},
           'speaker': turn.speaker, 'confirmed': None}
    output.append(new)
    audit.append({'component': 'fact_recovery', 'fact_index': len(output) - 1, 'field': 'tariff',
                  'reason': 'explicit_current_tariff', 'value': value, 'source_turn_idx': turn.idx})
    return output, audit
