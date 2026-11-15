"""Recover omitted current states from explicit evidence, never from gold labels.

Based on organizer-v21-state-accuracy's recovery idea; adds causal, per-object
guards for negation, planned/historical statements and multiple clients/tickets.
"""
import copy
from functools import lru_cache
import json
from pathlib import Path
import re

from ticket_recovery import POLICY as RECOVERY_POLICY, reconcile_ticket_statuses

WORD = re.compile(r'[а-яa-z0-9]+')
CLAUSES = re.compile(r'[^.!?;\n]+[.!?;]?')
TARIFF_CURRENT = re.compile(r'\b(?:сейчас|текущ[а-я]*|у\s+меня|у\s+вас|ваш|мой|подключен[а-я]*)\b')
TARIFF_MENTION = re.compile(r'\b(?:т[ао][рф]и[фв]|фариф|тари|пакет|плачу\s+по)\b')


def _norm(text):
    return ' '.join(WORD.findall(text.lower().replace('ё', 'е')))


def _tariff_norm(text):
    # '+' distinguishes catalog tariffs; hyphens separate words, as in the package.
    return ' '.join(re.findall(r'[а-яa-z0-9+]+', text.lower().replace('ё', 'е')))


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


def _candidates(facts, call, tariffs):
    clients = _unique_objects(facts, 'client')
    candidates = []
    for turn in sorted(call.turns, key=lambda t: t.idx):
        for match in CLAUSES.finditer(turn.text.lower().replace('ё', 'е')):
            text = match.group()
            if '?' in text:
                continue
            normalized = _tariff_norm(text)
            if len(clients) == 1 and (TARIFF_CURRENT.search(text) or TARIFF_MENTION.search(text)):
                names = []
                hits = [(value, hit) for value, pattern in tariffs for hit in pattern.finditer(normalized)]
                # A short tariff name inside a longer exact catalog name is not another tariff.
                hits = [(value, hit) for value, hit in hits if not any(
                    other.start() <= hit.start() and other.end() >= hit.end()
                    and other.end() - other.start() > hit.end() - hit.start() for _, other in hits)]
                for value, hit in hits:
                    prefix, suffix = normalized[max(0, hit.start() - 70):hit.start()], normalized[hit.end():hit.end() + 35]
                    context = normalized[max(0, hit.start() - 70):hit.end() + 35]
                    if not TARIFF_MENTION.search(context):
                        continue  # «мини-приставка» is a device mention, not tariff «Мини».
                    if re.search(r'\b(?:если|возможно|наверное|кажется)\b', prefix):
                        continue
                    if re.search(r'\b(?:не|нет)\s+(?:тариф\s+)?$', prefix):
                        continue
                    if re.search(r'\b(?:хочу|хотел[а-я]*|предлага[а-я]*|планир[а-я]*|завтра|будет|будут)\b', prefix):
                        continue
                    if re.search(r'\b(?:раньше|ранее|прежде|был[а-я]*)\b', prefix) and not re.search(r'\bсейчас\b', prefix):
                        continue
                    if re.match(r'\s*(?:не\s+подключен[а-я]*|отключен[а-я]*|больше\s+не)', suffix):
                        continue
                    names.append(value)
                names = list(dict.fromkeys(names))
                if len(names) == 1:
                    candidates.append(('tariff', names[0], clients[0], turn))
    return candidates


def recover_facts(facts, call, catalog_path=None):
    """Tariff recovery stays unchanged; ticket assertions have their own scope."""
    output = copy.deepcopy(facts)
    grouped = {}
    for attribute, value, obj, turn in _candidates(output, call, _load_tariffs(catalog_path)):
        grouped.setdefault((attribute, _identity(obj)), []).append((value, obj, turn))
    decisions = []
    for (attribute, _key), candidates in grouped.items():
        latest = max(turn.idx for _, _, turn in candidates)
        current = [(value, obj, turn) for value, obj, turn in candidates if turn.idx == latest]
        if len({value for value, _, _ in current}) != 1:
            continue
        value, obj, turn = current[-1]
        related = [fact for fact in output if fact.get('attribute') == attribute
                   and _same_object(fact.get('object') or {}, obj)]
        if any(not fact.get('valid_to') and fact.get('polarity', 1) == 1 for fact in related):
            continue
        if any(fact.get('polarity') == -1 and _norm(fact.get('value', '')) == _norm(value) for fact in related):
            continue
        output.append({'object': obj, 'attribute': attribute, 'value': value, 'polarity': 1,
                       'valid_from': call.started_at, 'valid_to': None,
                       'source': {'turn_idx': turn.idx, 'span': None, 'quote': None},
                       'speaker': turn.speaker, 'confirmed': None})
        decisions.append({'component': 'fact_recovery', 'fact_index': len(output) - 1,
                          'field': attribute, 'reason': 'missing_current_tariff' if attribute == 'tariff'
                          else 'missing_current_status', 'value': value, 'object': copy.deepcopy(obj),
                          'source_turn_idx': turn.idx, 'policy': RECOVERY_POLICY, 'action': 'add'})
    output, status_decisions = reconcile_ticket_statuses(output, call)
    return output, decisions + status_decisions
