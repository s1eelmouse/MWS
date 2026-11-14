"""Date binding for the organizer scorer v2.x (`TEMPORAL_NORMALIZATION=case_v2`).

Scorer v2 checks `valid_from` AND `valid_to` by day for service_state, tariff,
ticket_status and agreement; `action` is no longer scored temporally. This layer
therefore adds what the `task` mode does not read:

* single-day expressions from the scorer's own bank (позавчера, «пару дней назад»,
  «дней пять назад», «неделю назад», «в/на <день недели>», «через неделю»);
* service_state: onset of a symptom (`valid_from`) and the end of the preceding
  «работает нормально» (`valid_to`);
* the two labelling layers of the supplied gold, which do not depend on ASR noise:
  a value that is a catalog entry carries a timeline date (symptom onset, scheduled
  agreement, ticket status), while a free formulation is dated by the call itself
  and stays open. For the values this module classifies as free that holds for 4 133
  of 4 133 gold facts in train and 393 of 393 in val (package v2.1); schema.md states
  the same rule ("дата не названа — дата звонка"). Values that merely resemble a
  catalog entry are left to the model, because the scorer matches them to catalog gold;
* structure: agreements and symptoms are open intervals. Closing a ticket status at
  the start of the next one is available (close_chains) but off: it depends on the
  other predicted statuses being right and lowered the baseline predictions' score.

Week/month periods («на прошлой неделе», «месяц назад», «в прошлом месяце») are
deliberately not converted: in the supplied gold they map to a random day of the
period, so a rule cannot beat the model's own guess. Only the input call and the
predicted facts are used; gold labels never reach this module.
"""
import copy
import json
import re
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from temporal import CLAUSE_BREAK, WEEKDAYS, _canonicalize_cues, _value_clauses

NUMBER_WORDS = {
    'один': 1, 'одну': 1, 'два': 2, 'две': 2, 'пару': 2, 'три': 3, 'четыре': 4, 'пять': 5,
    'шесть': 6, 'семь': 7, 'восемь': 8, 'девять': 9, 'десять': 10,
}
_NUM = r'(\d{1,2}|' + '|'.join(NUMBER_WORDS) + r')'
_DAYS = r'(?:день|дня|дней|дню|суток|сутки)'
_WD_GEN = r'(понедельника|вторника|среды|четверга|пятницы|субботы|воскресенья)'
_WD_ACC = r'(понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье)'
_THIS = r'(?:эт[оуи]т?\s+|ближайш[а-я]+\s+)?'

STATUS_ORDER = ('принята', 'в работе', 'назначен выезд специалиста', 'выполнена')
STATUS_PATTERNS = {
    'принята': r'приня(?:т[а-я]*|л[а-я]*)',
    'в работе': r'в\s+работ[еу]',
    'назначен выезд специалиста': r'назнач[а-я]*\s+(?:\w+\s+){0,2}выезд[а-я]*',
    'выполнена': r'выполн[а-я]*',
}
NORMAL = 'работает нормально'
CATALOG_SECTIONS = {'service_state': ('symptoms',), 'tariff': ('tariffs',),
                    'ticket_status': ('statuses',), 'agreement': ('agreements',)}
HEDGE = re.compile(r'\b(?:если|возможно|может\s+быть|наверное|кажется)\b')
IDIOM = re.compile(r'\bне\s+вчера\s+родил[а-я]*')
NEGATED = re.compile(r'\bне\s*$')
TICKET_NUMBER = re.compile(r'\b(?:заявк[а-я]*|обращени[а-я]*)\s*(?:№\s*|номер\s*)?(\d+)\b')
NOT_STATUS = re.compile(r'\bне\s+(?:был[аои]?\s+)?$')
TRANSLIT = {'wifi': 'вай фай', 'wi': 'вай', 'fi': 'фай', 'вайфай': 'вай фай', 'tv': 'тв', 'sms': 'смс'}


def _words(text):
    words = []
    for word in re.findall(r'[а-яa-z0-9+]+', text.lower().replace('ё', 'е')):
        words.extend(TRANSLIT.get(word, word).split())
    return words


def _stem(word):
    return word[:-2] if len(word) >= 7 else word[:-1] if len(word) >= 5 else word


def _close(a, b):
    return a == b or _stem(a) == _stem(b) or SequenceMatcher(None, a, b).ratio() >= 0.75


class ValueLayers:
    """Tells catalog values from free formulations; anything in between is 'unsure'.

    'free' is claimed only when no catalog entry of the attribute resembles the value:
    «подтверждение по смс» or «заявка 5401 в работе» resemble catalog entries, the
    scorer matches them to catalog gold, so their dates are left to the model.
    """
    RESEMBLES = 0.6

    def __init__(self, catalog):
        if not isinstance(catalog, dict):
            raise ValueError('Catalog must be a JSON object')
        self.entries = {attr: [_words(v) for section in sections for v in catalog.get(section, ())]
                        for attr, sections in CATALOG_SECTIONS.items()}
        self.entries['service_state'].append(_words(NORMAL))
        self.exact = {attr: {' '.join(e) for e in entries} for attr, entries in self.entries.items()}

    def layer(self, attribute, value):
        if attribute not in self.entries:
            return 'unsure'
        words = _words(value)
        if ' '.join(words) in self.exact[attribute]:
            return 'catalog'
        for entry in self.entries[attribute]:
            covered = sum(any(_close(w, v) for v in words) for w in entry)
            if entry and covered / len(entry) >= self.RESEMBLES:
                return 'unsure'
        return 'free'


@lru_cache(maxsize=4)
def load_layers(path):
    """Value layers from the organizer catalog.json; a missing file is an explicit error."""
    return ValueLayers(json.loads(Path(path).read_text(encoding='utf-8')))


def _number(text):
    return int(text) if text.isdigit() else NUMBER_WORDS[text]


def _iso(day, tzinfo=None):
    """Midnight of the day; keeps the call's UTC offset so intervals never mix naive and aware."""
    return datetime.combine(day, datetime.min.time(), tzinfo=tzinfo).isoformat()


def _day(value):
    return value[:10] if isinstance(value, str) and value else None


def canonical_text(text):
    """Lower-case text with typical ASR variants of cue words restored."""
    return _canonicalize_cues(text).lower().replace('ё', 'е')


def day_cues(text, anchor, *, future=False, canonical=False):
    """Single-day cues of one text chunk as (date, role, position).

    role: 'since' (с/со …), 'until' (до …) or 'at'. With future=False only past and
    present days are returned, with future=True only present and future days: the
    caller chooses the direction from the attribute, never from the gold date.
    """
    lower = text if canonical else canonical_text(text)
    # «не вчера родилась» is an idiom, not a date; other cues of the same run-on stay.
    lower = IDIOM.sub(lambda m: ' ' * len(m.group()), lower)
    found = []

    def add(delta, match, role='at'):
        if NEGATED.search(lower[max(0, match.start() - 5):match.start()]):
            return  # «не завтра, а в среду»: the negated day is not the date
        if (delta > 0) == future or delta == 0:
            try:
                found.append((anchor + timedelta(days=delta), role, match.start()))
            except OverflowError:
                pass

    def role_before(position):
        before = lower[max(0, position - 6):position]
        if re.search(r'\bдо\s+$', before):
            return 'until'
        return 'since' if re.search(r'\bсо?\s+$', before) else 'at'

    for match in re.finditer(r'\b(позавчера|послезавтра|вчера|сегодня|завтра)\b', lower):
        delta = {'позавчера': -2, 'вчера': -1, 'сегодня': 0, 'завтра': 1, 'послезавтра': 2}[match.group(1)]
        add(delta, match, role_before(match.start()))
    for match in re.finditer(r'\b(вчерашн|сегодняшн|завтрашн)[а-я]+', lower):
        delta = {'вчерашн': -1, 'сегодняшн': 0, 'завтрашн': 1}[match.group(1)]
        add(delta, match, role_before(match.start()))
    for pattern in (rf'\b{_NUM}\s+{_DAYS}\s+(?:тому\s+)?назад\b', rf'\b{_DAYS}\s+{_NUM}\s+назад\b'):
        for match in re.finditer(pattern, lower):
            add(-_number(match.group(1)), match)
    for match in re.finditer(r'\b(?:(две|2)\s+недели|неделю)\s+(?:тому\s+)?назад\b', lower):
        add(-14 if match.group(1) else -7, match)
    for match in re.finditer(rf'\bчерез\s+{_NUM}\s+{_DAYS}\b', lower):
        add(_number(match.group(1)), match)
    for match in re.finditer(r'\bчерез\s+(?:(две|2)\s+недели|неделю)\b', lower):
        add(14 if match.group(1) else 7, match)
    for match in re.finditer(rf'\b(со?|до)\s+{_THIS}{_WD_GEN}\b', lower):
        back = (anchor.weekday() - WEEKDAYS[match.group(2)]) % 7 or 7
        add(-back, match, 'until' if match.group(1) == 'до' else 'since')
    for match in re.finditer(rf'\b(во?|на|к)\s+{_THIS}{_WD_ACC}\b', lower):
        target = WEEKDAYS[match.group(2)]
        if future:
            add((target - anchor.weekday()) % 7 or 7, match)
        elif match.group(1) in ('в', 'во'):
            add(-((anchor.weekday() - target) % 7 or 7), match)
    return found


def _unique(cues, roles=None):
    days = {day for day, role, _ in cues if roles is None or role in roles}
    return next(iter(days)) if len(days) == 1 else None


def _chunks(turn_text, value):
    """Clauses that name the value; the whole turn when the value is paraphrased."""
    clauses = [c for c in _value_clauses(turn_text, value) if not HEDGE.search(c.lower())]
    return clauses, (clauses or [turn_text])


def _nearby_day(call, turn, value, call_day, *, future, reach=1):
    """Unique cue day of the value clause, else of the source turn, else of adjacent turns."""
    named, chunks = _chunks(turn.text, value)
    for texts in ([chunks] if named else []) + [[turn.text]]:
        cues = [cue for text in texts for cue in day_cues(text, call_day, future=future)]
        if cues:
            return _unique(cues), named
    order = sorted(call.turns, key=lambda t: t.idx)
    position = next(i for i, t in enumerate(order) if t.idx == turn.idx)
    for distance in range(1, reach + 1):
        cues = []
        for index in (position - distance, position + distance):
            if 0 <= index < len(order):
                cues += day_cues(order[index].text, call_day, future=future)
        if cues:
            return _unique(cues), named
    return None, named


def _change(item, field, value, changes):
    if item.get(field) != value and _day(item.get(field)) != _day(value):
        changes.append({'field': field, 'before': item.get(field), 'after': value})
        item[field] = value


def _ordered(start, end):
    return not (start and end) or _day(start) <= _day(end)


def _status_key(value):
    return ' '.join(value.lower().replace('ё', 'е').split())


TRANSITION = re.compile(r'\b(?:сменил[а-я]*|обновил[а-я]*|перевел[а-я]*|изменил[а-я]*|перешл[а-я]*|ушл[а-я]*)')
STILL = re.compile(r'\bеще\b')


def _clauses_with_separator(text):
    """Clauses of a turn together with the separator that ends each of them."""
    start = 0
    for separator in CLAUSE_BREAK.finditer(text):
        yield text[start:separator.start()], separator.group()
        start = separator.end()
    yield text[start:], ''


def _clauses(text):
    return [clause for clause, _separator in _clauses_with_separator(text)]


def _same_ticket(item, text):
    """A turn that names another ticket number, or several, does not date this ticket."""
    numbers = set(TICKET_NUMBER.findall(text.lower()))
    wanted = TICKET_NUMBER.search(item['object'].get('name', '').lower())
    return len(numbers) <= 1 and (not numbers or not wanted or wanted[1] in numbers)


def _bind_status(item, call_day, turn, changes, stamp):
    """Bind a cue to the nearest status mention of its clause.

    «четыре дня назад была принята» starts the status; «находилась в работе до пятницы»
    and «была принята, вчера статус сменился» end it; «ещё вчера была принята» only says
    that the status still held, so it sets neither boundary.
    """
    key = _status_key(item['value'])
    if key not in STATUS_PATTERNS or not _same_ticket(item, turn.text):
        return
    starts, ends = set(), set()
    for clause, separator in _clauses_with_separator(turn.text):
        lower = canonical_text(clause)
        mentions = [(m.start(), m.end(), status) for status, pattern in STATUS_PATTERNS.items()
                    for m in re.finditer(pattern, lower)]
        own = [m for m in mentions if m[2] == key]
        if not own:
            continue
        # A question, a guess or «не была принята» states no status time.
        if '?' in separator or HEDGE.search(lower):
            continue
        if all(NOT_STATUS.search(lower[max(0, m[0] - 15):m[0]]) for m in own):
            continue
        change = TRANSITION.search(lower)
        for day, role, position in day_cues(lower, call_day, canonical=True):
            nearest = min(mentions, key=lambda m: min(abs(position - m[0]), abs(position - m[1])))
            if nearest[2] != key:
                continue
            if STILL.search(lower[max(0, position - 6):position + 22]):
                continue
            if role == 'until' or (change and nearest[1] <= change.start() and position > nearest[1]):
                ends.add(day)
            elif not change or nearest[0] >= change.start() or position < nearest[0]:
                starts.add(day)
    start = next(iter(starts)) if len(starts) == 1 else None
    end = next(iter(ends)) if len(ends) == 1 else None
    if start and end and start == end:
        end = None
    if start and _ordered(stamp(start), stamp(end) if end else item.get('valid_to')):
        _change(item, 'valid_from', stamp(start), changes)
    if end and _ordered(item.get('valid_from'), stamp(end)):
        _change(item, 'valid_to', stamp(end), changes)


def _open(item, changes):
    if item.get('valid_to') is not None:
        changes.append({'field': 'valid_to', 'before': item['valid_to'], 'after': None})
        item['valid_to'] = None


def _bind_free(item, call, changes):
    """Free formulation: dated by the call, open interval."""
    _open(item, changes)
    _change(item, 'valid_from', call.started_at, changes)


def _bind_state(item, call, call_day, turn, changes, stamp):
    if _status_key(item['value']) == NORMAL:
        # «до пятницы работало», «ещё вчера работало»: the normal state ends at the onset.
        day, _named = _nearby_day(call, turn, item['value'], call_day, future=False, reach=0)
        if day and day < call_day and _ordered(item.get('valid_from'), stamp(day)):
            _change(item, 'valid_to', stamp(day), changes)
        return
    _open(item, changes)
    day, _named = _nearby_day(call, turn, item['value'], call_day, future=False)
    if day and day < call_day:
        _change(item, 'valid_from', stamp(day), changes)


def _bind_agreement(item, call, call_day, turn, changes, stamp):
    _open(item, changes)
    day, _named = _nearby_day(call, turn, item['value'], call_day, future=True)
    if day and day > call_day:
        _change(item, 'valid_from', stamp(day), changes)


def _close_ticket_chains(items, decisions, call_day, stamp):
    """A status of a ticket ends on the day the next status of that ticket starts."""
    tickets = {}
    for index, item in enumerate(items):
        if item['attribute'] == 'ticket_status' and item.get('polarity', 1) == 1 \
                and _status_key(item['value']) in STATUS_ORDER:
            name = ' '.join(re.findall(r'[а-яa-z0-9]+', item['object'].get('name', '').lower()))
            tickets.setdefault(name, []).append(index)
    for indices in tickets.values():
        ranks = [STATUS_ORDER.index(_status_key(items[i]['value'])) for i in indices]
        if len(set(ranks)) != len(ranks):
            continue  # the same status twice: two tickets under one name, do not guess
        chain = [i for _, i in sorted(zip(ranks, indices))]
        rank = dict(zip(indices, ranks))
        for current, following in zip(chain, chain[1:]):
            start = items[following].get('valid_from')
            adjacent = rank[following] - rank[current] == 1
            if not adjacent and items[current].get('valid_to'):
                continue  # a status in between may be missing from the extraction
            if start and _day(start) <= call_day.isoformat() \
                    and _ordered(items[current].get('valid_from'), start):
                _change(items[current], 'valid_to', stamp(datetime.fromisoformat(_day(start)).date()),
                        decisions[current]['changes'])


def _parse(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def _keep_interval_valid(item, fact, decision):
    """The contract rejects valid_to < valid_from; rules work by day, timestamps may differ."""
    start, end = item.get('valid_from'), item.get('valid_to')
    if not (start and end):
        return
    try:
        reversed_interval = _parse(end) < _parse(start)
    except (TypeError, ValueError):
        reversed_interval = True
    if not reversed_interval:
        return
    if _day(start) == _day(end):
        item['valid_to'] = start  # same day: a zero-length interval, equal by day for the scorer
        return
    item['valid_from'], item['valid_to'] = fact.get('valid_from'), fact.get('valid_to')
    decision.update(status='abstained', reason='interval_conflict', changes=[])


def repair_case_v2_dates(facts, call, layers, *, close_chains=False):
    """Return copies of the facts and one audit decision per fact; nothing is dropped."""
    if not isinstance(layers, ValueLayers):
        layers = ValueLayers(layers)
    started = datetime.fromisoformat(call.started_at.replace('Z', '+00:00'))
    call_day = started.date()

    def stamp(day):
        return _iso(day, started.tzinfo)

    turns = {turn.idx: turn for turn in call.turns}
    items, decisions = [], []
    for fact in facts:
        item = copy.deepcopy(fact)
        decision = {'attribute': fact['attribute'], 'source_turn_idx': fact['source']['turn_idx'],
                    'status': 'unchanged', 'reason': 'case_v2_left_unchanged', 'changes': [], 'evidence': []}
        turn = turns.get(fact['source']['turn_idx'])
        valid = True
        for field in ('valid_from', 'valid_to'):
            if fact.get(field):
                try:
                    datetime.fromisoformat(fact[field].replace('Z', '+00:00'))
                except (TypeError, ValueError):
                    valid = False
        layer = layers.layer(fact['attribute'], fact['value'])
        decision['layer'] = layer
        if not valid:
            decision['reason'] = 'invalid_model_date_left_for_sanitizer'
        elif turn is None:
            decision['reason'] = 'source_missing'
        elif fact.get('polarity', 1) != 1:
            decision['reason'] = 'negative_fact'
        elif layer == 'free':
            _bind_free(item, call, decision['changes'])
        elif layer == 'unsure':
            if fact['attribute'] in CATALOG_SECTIONS:
                decision['reason'] = 'resembles_catalog_entry_left_to_model'
        elif fact['attribute'] == 'ticket_status':
            _bind_status(item, call_day, turn, decision['changes'], stamp)
        elif fact['attribute'] == 'service_state':
            _bind_state(item, call, call_day, turn, decision['changes'], stamp)
        elif fact['attribute'] == 'agreement':
            _bind_agreement(item, call, call_day, turn, decision['changes'], stamp)
        items.append(item)
        decisions.append(decision)
    if close_chains:
        _close_ticket_chains(items, decisions, call_day, stamp)
    for fact, item, decision in zip(facts, items, decisions):
        _keep_interval_valid(item, fact, decision)
        if decision['changes']:
            decision.update(status='changed', reason='case_v2_date_binding')
    return items, decisions
