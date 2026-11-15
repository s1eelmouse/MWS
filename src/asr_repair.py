"""Noise-tolerant view of a call for the recovery rules (NOISE_ROBUST_RECOVERY=1).

The tariff and ticket recovery rules match catalog names and status words exactly.
ASR noise of the kind seen in the organizer corpus — vowel reduction and voicing
(«Пятёрачка», «зейчас»), a dropped, doubled or swapped letter («рабте», «тарфи»),
glued words («уменя», «вработе») — makes them miss facts the model also missed.

This module gives the rules a repaired copy of the call:
  * a word is replaced by a target word only if both have the same phonetic key up
    to one edit; targets are catalog tariff words and a short list of cue words;
  * a word that differs from its target only in the ending (an inflection such as
    «местные» / «местный») is not touched;
  * tariff words are repaired only as a whole catalog name and only in a turn that
    mentions a tariff, or right after one;
  * glued function words («у», «в», «не», «меня», …) are split off when the rest is
    a cue word.
It also snaps a predicted tariff or ticket status value to the catalog entry when the
same rule identifies exactly one entry.

Turn indices, speakers and the number of turns never change; gold is never read.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

WORD = re.compile(r'[а-яёa-z0-9+]+', re.IGNORECASE)
PHONETIC = str.maketrans({
    'о': 'а', 'ё': 'и', 'е': 'и', 'я': 'и', 'э': 'и', 'ы': 'и',
    'д': 'т', 'з': 'с', 'ц': 'с', 'б': 'п', 'в': 'ф', 'г': 'к', 'ж': 'ш', 'щ': 'ш',
})
CUE_WORDS = (
    'тариф', 'тарифе', 'тарифу', 'тарифом', 'тарифа', 'заявка', 'заявки', 'заявку', 'заявке',
    'обращение', 'принята', 'принят', 'выполнена', 'выполнен', 'назначен', 'назначена',
    'выезд', 'работе', 'работу', 'сейчас', 'статус', 'текущий', 'подключен', 'теперь',
)
GLUED = ('меня', 'мой', 'моя', 'ваш', 'ваша', 'не', 'на', 'по', 'у', 'в')
FUNCTION_WORDS = {'меня', 'нас', 'вас', 'него', 'нее'}
TARIFF_CONTEXT = re.compile(r'\b(?:тариф[а-я]*|пакет[а-я]*)\b')
STATUSES = ('принята', 'в работе', 'назначен выезд специалиста', 'выполнена')


def key(word):
    return word.lower().replace('ё', 'е').translate(PHONETIC)


def within_one_edit(a, b):
    """One dropped, doubled or inserted letter, or two adjacent letters swapped.

    Substitutions count only through the phonetic key (voicing, vowel reduction);
    a free substitution («придет» / «принят») is a different word, not noise.
    """
    if a == b:
        return True
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return (len(diff) == 2 and diff[1] == diff[0] + 1
                and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    if abs(len(a) - len(b)) != 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def word_matches(word, target):
    """ASR-damaged form of target, not an inflection of it and not a different short word."""
    low, goal = word.lower().replace('ё', 'е'), target.lower().replace('ё', 'е')
    if low == goal:
        return True
    if len(goal) <= 2 or not low or key(low[0]) != key(goal[0]):
        return False
    stem = max(3, len(goal) - 2)
    if low[:stem] == goal[:stem] and low[stem:] != goal[stem:]:
        return False  # same stem, other ending: inflection («работы», «местные»)
    if len(goal) <= 5:
        return key(low) == key(goal)
    return within_one_edit(key(low), key(goal))


def tariff_word_matches(word, target):
    """Tariff names are ordinary words («Простой», «Орбита», «Мини»): only a pure
    phonetic confusion of a word of five or more letters counts as the same word."""
    low, goal = word.lower().replace('ё', 'е'), target.lower().replace('ё', 'е')
    if low == goal:
        return True
    if len(goal) <= 4 or not goal.isalpha() or len(low) != len(goal):
        return False
    stem = len(goal) - 2
    if low[:stem] == goal[:stem]:
        return False  # only the ending differs: an inflection
    return key(low) == key(goal)


@lru_cache(maxsize=8)
def _tariffs(path, modified_ns, size):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    names = []
    for name in data.get('tariffs', ()):
        tokens = WORD.findall(name.lower().replace('ё', 'е'))
        if tokens:
            names.append((name, tokens))
    return tuple(names)


def load_tariffs(path):
    if not path or not Path(path).is_file():
        return ()
    stat = Path(path).stat()
    return _tariffs(str(Path(path).resolve()), stat.st_mtime_ns, stat.st_size)


def _split_glued(token):
    low = token.lower()
    for prefix in GLUED:
        rest = low[len(prefix):]
        if low.startswith(prefix) and len(rest) >= 4 and (
                rest in FUNCTION_WORDS or any(word_matches(rest, cue) for cue in CUE_WORDS)):
            return [prefix, rest]
    return None


def _repair_cues(tokens):
    out = []
    for token in tokens:
        low = token.lower()
        if any(low == cue for cue in CUE_WORDS) or not low.isalpha():
            out.append(token)
            continue
        split = _split_glued(token)
        if split:
            out.extend(split)
            continue
        hits = [cue for cue in CUE_WORDS if len(cue) >= 5 and word_matches(low, cue)]
        if len(set(hits)) == 1 and not low.startswith('работа'):
            out.append(hits[0])
        else:
            out.append(token)
    return out


def _repair_tariffs(tokens, tariffs):
    lows = [t.lower().replace('ё', 'е') for t in tokens]
    out = list(tokens)
    position = 0
    while position < len(lows):
        best = None
        for name, parts in tariffs:
            end = position + len(parts)
            if end > len(lows) or all(lows[position + i] == p for i, p in enumerate(parts)):
                continue
            if all(tariff_word_matches(lows[position + i], p) for i, p in enumerate(parts)):
                if best is not None and best[0] != name:
                    best = ('ambiguous', parts)
                    break
                best = (name, parts)
        if best and best[0] != 'ambiguous':
            out[position:position + len(best[1])] = list(best[1])
            lows[position:position + len(best[1])] = list(best[1])
            position += len(best[1])
        else:
            position += 1
    return out


def repair_text(text, tariffs, *, tariff_context):
    """Repaired text as space-joined tokens with sentence punctuation kept."""
    pieces = re.findall(r'[а-яёa-z0-9+]+|[.!?;,…—-]', text, re.IGNORECASE)
    words = [p for p in pieces if WORD.fullmatch(p)]
    if not words:
        return text
    repaired = []
    for piece in pieces:
        repaired.extend(_repair_cues([piece]) if WORD.fullmatch(piece) else [piece])
    if tariff_context and tariffs:
        repaired = _repair_tariffs(repaired, tariffs)
    return re.sub(r'\s+([.!?;,…])', r'\1', ' '.join(repaired))


def repair_call(call, catalog_path=None):
    """Copy of the call with repaired turn texts; indices and speakers unchanged."""
    tariffs = load_tariffs(catalog_path)
    turns, previous_mentions = [], False
    for turn in sorted(call.turns, key=lambda t: t.idx):
        cues = repair_text(turn.text, (), tariff_context=False)
        mentions = bool(TARIFF_CONTEXT.search(cues.lower()))
        text = repair_text(turn.text, tariffs, tariff_context=mentions or previous_mentions)
        turns.append(turn.model_copy(update={'text': text}))
        previous_mentions = mentions
    order = {turn.idx: turn for turn in turns}
    return call.model_copy(update={'turns': [order[t.idx] for t in call.turns]})


def snap_values(facts, catalog_path=None):
    """Tariff and ticket_status values damaged by ASR are replaced by their catalog entry."""
    tariffs = load_tariffs(catalog_path)
    statuses = [(s, s.split()) for s in STATUSES]
    output, changes = [], []
    for index, fact in enumerate(facts):
        item = dict(fact)
        table = tariffs if fact.get('attribute') == 'tariff' else statuses \
            if fact.get('attribute') == 'ticket_status' else ()
        value = item.get('value')
        if table and isinstance(value, str):
            tokens = WORD.findall(value.lower().replace('ё', 'е'))
            exact = [name for name, parts in table if parts == tokens]
            match = tariff_word_matches if fact.get('attribute') == 'tariff' else word_matches
            close = {name for name, parts in table
                     if len(parts) == len(tokens) and all(match(t, p) for t, p in zip(tokens, parts))}
            if not exact and len(close) == 1:
                item['value'] = close.pop()
                changes.append({'fact_index': index, 'component': 'asr_repair', 'action': 'snap_value',
                                'field': 'value', 'before': value, 'after': item['value']})
        output.append(item)
    return output, changes
