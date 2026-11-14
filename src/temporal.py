"""Russian calendar arithmetic and conservative fact/date binding, without gold.

Calendar periods remain periods. Ambiguous dates are not guessed. All relative
expressions use the supplied call timestamp; the machine clock is never read.
"""
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
import calendar
import copy
import re


@dataclass(frozen=True)
class TimeExpression:
    text: str
    span: tuple[int, int]
    start: str | None
    end: str | None
    precision: str
    relation: str
    certain: bool
    rule: str
    reason: str | None = None

    def as_dict(self):
        return asdict(self)


NUMBERS = {
    'один': 1, 'одна': 1, 'одну': 1, 'одного': 1, 'два': 2, 'две': 2,
    'двух': 2, 'три': 3, 'трех': 3, 'четыре': 4, 'четырех': 4,
    'пять': 5, 'пяти': 5, 'шесть': 6, 'шести': 6, 'семь': 7, 'семи': 7,
    'восемь': 8, 'восьми': 8, 'девять': 9, 'девяти': 9, 'десять': 10,
    'десяти': 10, 'одиннадцать': 11, 'двенадцать': 12, 'тринадцать': 13,
    'четырнадцать': 14, 'пятнадцать': 15, 'шестнадцать': 16,
    'семнадцать': 17, 'восемнадцать': 18, 'девятнадцать': 19,
    'двадцать': 20, 'тридцать': 30,
}
NUMBER = r'(?:\d{1,4}|(?:двадцать|тридцать)(?:\s+(?:один|одну|два|две|три|четыре|пять|шесть|семь|восемь|девять))?|' + '|'.join(sorted(NUMBERS, key=len, reverse=True)) + r')'
UNIT = r'(?:день|дня|дней|сутки|суток|неделю|недели|недель|час|часа|часов|минуту|минуты|минут)'
MONTHS = {
    'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5,
    'июня': 6, 'июля': 7, 'августа': 8, 'сентября': 9, 'октября': 10,
    'ноября': 11, 'декабря': 12,
}
WEEKDAYS = {
    'понедельник': 0, 'понедельника': 0, 'вторник': 1, 'вторника': 1,
    'среда': 2, 'среду': 2, 'среды': 2, 'четверг': 3, 'четверга': 3,
    'пятница': 4, 'пятницу': 4, 'пятницы': 4, 'суббота': 5, 'субботу': 5, 'субботы': 5,
    'воскресенье': 6, 'воскресенья': 6,
}
RELATIVE = {
    'позавчера': -2, 'вчера': -1, 'сегодня': 0, 'завтра': 1, 'послезавтра': 2,
}


def _number(text):
    return int(text) if text.isdigit() else sum(NUMBERS[word] for word in text.split())


def _datetime(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def _midnight(value):
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def shift_calendar_months(value, months):
    """Same day in a shifted calendar month; invalid month days raise, never clamp."""
    ordinal = value.year * 12 + value.month - 1 + months
    year, month0 = divmod(ordinal, 12)
    return value.replace(year=year, month=month0 + 1)


def extract_time_expressions(text, started_at, *, direction=None, daytime_hours=False):
    """Return calendar candidates with original character offsets.

    direction is required to infer years or unqualified weekdays. A weekday equal
    to the call weekday stays ambiguous. Period ends are inclusive calendar days.
    Bare 12-hour times require a time-of-day policy. daytime_hours explicitly opts
    into interpreting hours 1..7 as 13..19; fact repair never enables this policy.
    """
    if direction not in (None, 'past', 'future'):
        raise ValueError('direction must be past, future or None')
    anchor = _datetime(started_at)
    day = _midnight(anchor)
    lower = text.lower().replace('ё', 'е')
    results = []

    def add(match, start=None, end=None, precision='day', relation='at',
            certain=True, rule='', reason=None):
        span = match.span()
        if any(span[0] < x.span[1] and x.span[0] < span[1] for x in results):
            return
        results.append(TimeExpression(text[span[0]:span[1]], span,
                                     start.isoformat() if start is not None else None,
                                     end.isoformat() if end is not None else None,
                                     precision, relation, certain, rule, reason))

    # Absolute years are explicit; invalid dates remain explicit unresolved evidence.
    absolute = r'\b(?:\d{4}-\d{2}-\d{2}(?:[tT]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)?|\d{1,2}\.\d{1,2}\.\d{4})\b'
    for match in re.finditer(absolute, text):
        value = match.group()
        try:
            date = (datetime.strptime(value, '%d.%m.%Y').replace(tzinfo=anchor.tzinfo)
                    if '.' in value else _datetime(value))
            if date.tzinfo is None and anchor.tzinfo is not None:
                date = date.replace(tzinfo=anchor.tzinfo)
            add(match, date, precision='minute' if 't' in value.lower() else 'day', rule='absolute_date')
        except ValueError:
            add(match, certain=False, rule='absolute_date', reason='invalid_calendar_date')

    months = '|'.join(MONTHS)
    for match in re.finditer(r'\b(?P<day>\d{1,2})\s+(?P<month>' + months + r')(?:\s+(?P<year>\d{4})(?:\s+года)?)?\b', lower):
        dd, mm = int(match['day']), MONTHS[match['month']]
        year = int(match['year']) if match['year'] else anchor.year
        if not match['year'] and direction is None:
            add(match, certain=False, rule='named_date', reason='year_not_specified')
            continue
        try:
            date = day.replace(year=year, month=mm, day=dd)
            if not match['year']:
                if direction == 'past' and date > day:
                    date = date.replace(year=year - 1)
                elif direction == 'future' and date < day:
                    date = date.replace(year=year + 1)
            add(match, date, rule='named_date')
        except ValueError:
            add(match, certain=False, rule='named_date', reason='invalid_calendar_date')

    for match in re.finditer(r'\b(позавчера|послезавтра|вчера|сегодня|завтра|вчерашнего\s+дня|сегодняшнего\s+дня)\b', lower):
        word = match.group()
        offset = -1 if word.startswith('вчерашнего') else 0 if word.startswith('сегодняшнего') else RELATIVE[word]
        try:
            add(match, day + timedelta(days=offset), rule='relative_day')
        except OverflowError:
            add(match, certain=False, rule='relative_day', reason='calendar_overflow')

    patterns = [
        (r'\b(?:через|спустя)\s+(?:(?P<n>' + NUMBER + r')\s+)?(?P<unit>' + UNIT + r')\b', 1),
        (r'\b(?:(?P<n>' + NUMBER + r')\s+)?(?P<unit>' + UNIT + r')\s+назад\b', -1),
    ]
    for pattern, sign in patterns:
        for match in re.finditer(pattern, lower):
            n = _number(match['n']) if match['n'] else 1
            unit = match['unit']
            seconds = 60 if unit.startswith('мин') else 3600 if unit.startswith('час') else None
            try:
                date = (anchor + timedelta(seconds=sign * n * seconds) if seconds else
                        day + timedelta(days=sign * n * (7 if unit.startswith('недел') else 1)))
                add(match, date, precision='minute' if seconds else 'day', rule='relative_duration')
            except (OverflowError, ValueError):
                add(match, certain=False, rule='relative_duration', reason='calendar_overflow')

    # A week/month is not reduced to an arbitrary exact day.
    for match in re.finditer(r'\b(?:на\s+|в\s+|до\s+конца\s+)?(?P<which>прошлой|этой|следующей)\s+неделе?и?\b', lower):
        shift = {'прошлой': -7, 'этой': 0, 'следующей': 7}[match['which']]
        try:
            monday = day - timedelta(days=day.weekday()) + timedelta(days=shift)
            add(match, monday, monday + timedelta(days=6), precision='week', relation='period',
                certain=False, rule='calendar_week', reason='period_not_single_day')
        except OverflowError:
            add(match, certain=False, rule='calendar_week', reason='calendar_overflow')
    for match in re.finditer(r'\b(?:в\s+|с\s+|до\s+конца\s+)?(?P<which>прошлом|этом|следующем|прошлого|этого|следующего)\s+месяц(?:е|а)\b', lower):
        shift = -1 if match['which'].startswith('прошл') else 1 if match['which'].startswith('следующ') else 0
        ordinal = anchor.year * 12 + anchor.month - 1 + shift
        year, month0 = divmod(ordinal, 12)
        try:
            first = day.replace(year=year, month=month0 + 1, day=1)
            last = first.replace(day=calendar.monthrange(year, month0 + 1)[1])
            add(match, first, last, precision='month', relation='period', certain=False,
                rule='calendar_month', reason='period_not_single_day')
        except ValueError:
            add(match, certain=False, rule='calendar_month', reason='calendar_overflow')

    for match in re.finditer(r'\b(?P<n>' + NUMBER + r')\s+(?P<unit>месяц|месяца|месяцев|год|года|лет)\s+назад\b', lower):
        n = _number(match['n']) * (12 if match['unit'] in ('год', 'года', 'лет') else 1)
        try:
            shifted = shift_calendar_months(day, -n)
            add(match, shifted, certain=False, rule='calendar_month_shift', reason='calendar_month_convention_required')
        except ValueError:
            add(match, certain=False, rule='calendar_month_shift', reason='invalid_calendar_date')

    weekdays = '|'.join(sorted(WEEKDAYS, key=len, reverse=True))
    for match in re.finditer(r'\b(?P<pre>со?|во?|до)\s+(?P<weekday>' + weekdays + r')\b', lower):
        wanted = WEEKDAYS[match['weekday']]
        inferred = direction if direction is not None else 'past' if match['pre'] in ('с', 'со') else None
        if inferred is None or wanted == day.weekday():
            add(match, certain=False, rule='weekday', reason='ambiguous_weekday')
        else:
            delta = (wanted - day.weekday()) % 7 if inferred == 'future' else -((day.weekday() - wanted) % 7)
            try:
                add(match, day + timedelta(days=delta), rule='weekday', relation='before' if match['pre'] == 'до' else 'at')
            except OverflowError:
                add(match, certain=False, rule='weekday', reason='calendar_overflow')

    # Only explicit 24-hour clock notation is unambiguous without a business-hours policy.
    for match in re.finditer(r'\b(?P<pre>в|к|после|до)\s+(?P<h>\d{1,2}):(?P<m>\d{2})\b', lower):
        h, m = int(match['h']), int(match['m'])
        relation = {'после': 'after', 'до': 'before', 'к': 'by', 'в': 'at'}[match['pre']]
        if h > 23 or m > 59:
            add(match, certain=False, precision='time', relation=relation, rule='clock', reason='invalid_clock_time')
            continue
        dates = [x for x in results if x.certain and x.precision == 'day'
                 and 0 <= match.start() - x.span[1] <= 12
                 and not re.search(r'[.!?;]', text[x.span[1]:match.start()])]
        if len(dates) == 1:
            previous = dates[0]
            date = _datetime(previous.start).replace(hour=h, minute=m)
            combined = TimeExpression(text[previous.span[0]:match.end()], (previous.span[0], match.end()),
                                      date.isoformat(), None, 'minute', relation, True, previous.rule + '+clock')
            results.remove(previous)
            results.append(combined)
        else:
            add(match, precision='time', relation=relation, certain=False, rule='clock', reason='day_not_specified')
    spoken = r'\b(?P<pre>после|до|к|в)\s+(?P<h>' + NUMBER + r')(?:\s+час(?:а|ов)?)?(?:\s+(?P<part>утра|ночи|дня|вечера))?\b'
    for match in re.finditer(spoken, lower):
        if any(match.start() < x.span[1] and x.span[0] < match.end() for x in results):
            continue
        h = _number(match['h'])
        part = match['part']
        if 1 <= h <= 12 and part:
            h = h % 12 + (12 if part in ('дня', 'вечера') else 0)
        elif 1 <= h <= 7 and daytime_hours:
            h += 12
        elif h < 13:
            add(match, precision='time', certain=False, rule='spoken_clock', reason='day_or_meridiem_not_specified')
            continue
        dates = [x for x in results if x.certain and x.precision == 'day'
                 and 0 <= match.start() - x.span[1] <= 12
                 and not re.search(r'[.!?;]', text[x.span[1]:match.start()])]
        if h > 23 or len(dates) != 1:
            add(match, precision='time', certain=False, rule='spoken_clock', reason='invalid_clock_time' if h > 23 else 'day_not_specified')
            continue
        previous = dates[0]
        date = _datetime(previous.start).replace(hour=h)
        results.remove(previous)
        results.append(TimeExpression(text[previous.span[0]:match.end()], (previous.span[0], match.end()),
                                      date.isoformat(), None, 'minute',
                                      {'после': 'after', 'до': 'before', 'к': 'by', 'в': 'at'}[match['pre']],
                                      True, previous.rule + '+spoken_clock'))
    return sorted(results, key=lambda x: x.span)


STATUS_PATTERNS = {
    'принята': r'\bприня(?:т[а-я]*|л[а-я]*)\b',
    'в работе': r'\bв\s+работ[еу]\b',
    'назначен выезд специалиста': r'\bназнач[а-я]*\s+(?:\w+\s+){0,2}выезд[а-я]*\b',
    'выполнена': r'\bвыполн[а-я]*\b',
}
CLAUSE_BREAK = re.compile(r'(?<!\d)[.!?;…]+|[.!?;…]+(?!\d)|\b(?:а|но|затем|потом|теперь|сейчас)\b', re.I)
STOP = {'в', 'во', 'на', 'и', 'с', 'со', 'не', 'нет', 'для', 'у', 'по', 'уже', 'это', 'все'}


def _norm(text):
    return ' '.join(re.findall(r'[а-яa-z0-9]+', text.lower().replace('ё', 'е')))


def _anchor_present(fact, clause):
    value = _norm(fact['value'])
    pattern = STATUS_PATTERNS.get(value) if fact['attribute'] == 'ticket_status' else None
    if pattern:
        return bool(re.search(pattern, clause))
    tokens = [w for w in value.split() if w not in STOP and len(w) >= 4]
    words = _norm(clause).split()
    if not tokens:
        return False
    # Require every content word; partial word overlap does not select an event.
    return all(any(w.startswith(t[:5]) for w in words) for t in tokens)


def repair_fact_dates(fact, call):
    """Return a copy plus an audit decision, using only input and predicted fact.

    Action dates and agreement policy are deliberately preserved: the supplied
    corpus mixes event time and assertion time. No hidden history is consulted.
    """
    item = copy.deepcopy(fact)
    event = {'attribute': fact['attribute'], 'source_turn_idx': fact['source']['turn_idx'],
             'status': 'unchanged', 'reason': None, 'changes': [], 'evidence': []}
    if fact['attribute'] in ('agreement', 'action'):
        event['reason'] = 'agreement_policy_unresolved' if fact['attribute'] == 'agreement' else 'action_time_policy_preserved'
        return item, event
    if fact.get('polarity', 1) != 1:
        event['reason'] = 'negative_fact'
        return item, event
    if re.search(r'\b(?:вчера|сегодня|завтра|назад)\b', fact['value'].lower()):
        event['reason'] = 'value_contains_time_expression'
        return item, event
    turn = next((t for t in call.turns if t.idx == fact['source']['turn_idx']), None)
    if turn is None:
        event['reason'] = 'source_missing'
        return item, event
    candidates = {'valid_from': [], 'valid_to': []}
    start = 0
    clauses = []
    for separator in CLAUSE_BREAK.finditer(turn.text):
        clauses.append((start, separator.start(), separator.group()))
        start = separator.end()
    clauses.append((start, len(turn.text), ''))
    for left, right, punctuation in clauses:
        clause = turn.text[left:right].lower().replace('ё', 'е')
        if not _anchor_present(fact, clause):
            continue
        if '?' in punctuation or re.search(r'\b(?:если|возможно|может|кажется|наверное)\b', clause):
            continue
        expressions = extract_time_expressions(turn.text[left:right], call.started_at, direction='past')
        usable = [x for x in expressions if x.certain and x.precision in ('day', 'minute')]
        # More than one time expression needs a semantic relation; never pick the first.
        if len(usable) != 1 or any(not x.certain for x in expressions):
            if expressions:
                event['reason'] = 'multiple_or_partial_times'
            continue
        expression = usable[0]
        when = _datetime(expression.start)
        if when.date() > _datetime(call.started_at).date():
            continue
        before = clause[max(0, expression.span[0] - 18):expression.span[0]]
        until = bool(re.search(r'\bдо\s*$', before))
        ending = bool(re.search(r'\b(?:закрыл[а-я]*|закончил[а-я]*|завершил[а-я]*)\b', clause)
                      or until and re.search(r'\bвключительно\b', clause))
        if until and not ending:
            event['reason'] = 'endpoint_inclusivity_unspecified'
            continue
        onset = bool(re.search(r'\b(?:с|со)\s*$', before) or
                     re.search(r'\b(?:начал[а-я]*|появил[а-я]*|пропал[а-я]*|перестал[а-я]*|вдруг|переш[ео]л[а-я]*|подключил[а-я]*)\b', clause))
        if ending:
            field = 'valid_to'
        elif _norm(fact['value']) == 'работает нормально' and not re.search(r'\b(?:с|со)\s*$', before):
            event['reason'] = 'unknown_historical_start'
            continue
        elif fact['attribute'] == 'ticket_status' or onset:
            field = 'valid_from'
        else:
            event['reason'] = 'observation_does_not_prove_start'
            continue
        candidates[field].append(expression.start)
        event['evidence'].append({**expression.as_dict(), 'span': [left + expression.span[0], left + expression.span[1]], 'field': field})
    if any(len(set(values)) > 1 for values in candidates.values()):
        event['reason'] = 'conflicting_evidence'
        return item, event
    for field, values in candidates.items():
        if values and item.get(field) != values[0]:
            event['changes'].append({'field': field, 'before': item.get(field), 'after': values[0]})
            item[field] = values[0]
    if item.get('valid_from') and item.get('valid_to'):
        try:
            if _datetime(item['valid_to']) < _datetime(item['valid_from']):
                event.update(status='abstained', reason='interval_conflict', changes=[])
                return copy.deepcopy(fact), event
        except (ValueError, TypeError):
            event.update(status='abstained', reason='invalid_remaining_boundary', changes=[])
            return copy.deepcopy(fact), event
    if event['changes']:
        event.update(status='changed', reason='explicit_event_time')
    elif event['evidence']:
        event['reason'] = 'already_consistent'
    elif event['reason'] is None:
        event['reason'] = 'no_unambiguous_event_time'
    return item, event


TASK_STOP = {'в', 'во', 'на', 'и', 'с', 'со', 'не', 'нет', 'для', 'у', 'по', 'уже', 'это',
             'все', 'тариф', 'заявка', 'клиент', 'оператор', 'услуга'}


def _clauses(text):
    start = 0
    for separator in CLAUSE_BREAK.finditer(text):
        yield text[start:separator.start()]
        start = separator.end()
    yield text[start:]


ASR_CONFUSIONS = {
    'о': 'а', 'а': 'о', 'е': 'и', 'и': 'е', 'я': 'е', 'ё': 'е', 'э': 'е',
    'т': 'д', 'д': 'т', 'с': 'з', 'з': 'с', 'б': 'п', 'п': 'б', 'в': 'ф', 'ф': 'в',
    'г': 'к', 'к': 'г', 'ж': 'ш', 'ш': 'ж', 'щ': 'ш', 'ц': 'с', 'ы': 'и',
}
CUE_WORDS = (
    'вчера', 'сегодня', 'позавчера', 'послезавтра', 'завтра', 'назад', 'через',
    'понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье',
)
CUE_STOP = {
    'вера', 'вечер', 'вечера', 'вечеру', 'вечерам', 'путаница', 'путаницу',
    'завтрак', 'завтрака', 'над', 'нада', 'еда', 'еду',
}


def _one_edit_variants(word):
    variants = set()
    for index, char in enumerate(word):
        replacement = ASR_CONFUSIONS.get(char)
        if replacement:
            variants.add(word[:index] + replacement + word[index + 1:])
        variants.add(word[:index] + word[index + 1:])
        if index + 1 < len(word):
            variants.add(word[:index] + word[index + 1] + char + word[index + 2:])
        variants.add(word[:index] + char + char + word[index + 1:])
    return {item for item in variants if item and item != word}


@lru_cache(maxsize=1)
def _cue_variant_map():
    """Typical ASR corruptions of cue words, matched exactly, never by loose distance."""
    mapping = {}
    for cue in CUE_WORDS:
        first = _one_edit_variants(cue)
        second = set()
        for item in first:
            second.update(_one_edit_variants(item))
        for item in {cue} | first | second:
            if item in CUE_STOP:
                continue
            mapping.setdefault(item, cue)
    return mapping


def _canonicalize_cues(text):
    mapping = _cue_variant_map()
    parts = []
    for token in re.findall(r'[а-яёa-z0-9]+|[^а-яёa-z0-9]+', text, flags=re.I):
        lower = token.lower().replace('ё', 'е')
        cue = mapping.get(lower)
        if cue and cue != lower:
            parts.append(cue)
        else:
            parts.append(lower if token[:1].isalpha() else token)
    return ''.join(parts)


def _cue_dates(text, anchor, *, future=False, fuzzy=True):
    """Return (date, label, position) for explicit calendar cues in one clause."""
    if fuzzy:
        text = _canonicalize_cues(text)
    lower = text.lower().replace('ё', 'е')
    found = []
    if not future and re.search(r'не\s+вчера\s+родил', lower):
        return found
    match = re.search(r'\bвчера\b|вчерашн', lower)
    if match and not future:
        found.append((anchor - timedelta(days=1), 'вчера', match.start()))
    match = re.search(r'\bсегодня\b|сейчас', lower)
    if match and not future:
        found.append((anchor, 'сегодня', match.start()))
    for match in re.finditer(r'(\d+|один|одну|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять)\s+'
                             r'(день|дня|дней)\s+назад', lower):
        amount = _number(match.group(1))
        if amount:
            found.append((anchor - timedelta(days=amount), f'{amount} дней назад', match.start()))
    for match in re.finditer(r'\b(?:с|со)\s+([а-я]+)', lower):
        target = WEEKDAYS.get(match.group(1))
        if target is not None:
            back = (anchor.weekday() - target) % 7 or 7
            found.append((anchor - timedelta(days=back), 'с ' + match.group(1), match.start()))
    if future:
        if re.search(r'послезавтра', lower):
            found.append((anchor + timedelta(days=2), 'послезавтра', lower.find('послезавтра')))
        elif re.search(r'\bзавтра\b', lower):
            found.append((anchor + timedelta(days=1), 'завтра', lower.find('завтра')))
        for match in re.finditer(r'через\s+(\d+|один|одну|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять)\s+'
                                 r'(день|дня|дней|неделю|недели|недель)', lower):
            amount = _number(match.group(1))
            if amount:
                days = 7 * amount if match.group(2).startswith('недел') else amount
                found.append((anchor + timedelta(days=days), f'через {amount}', match.start()))
        for match in re.finditer(r'\bв\s+(понедельник|вторник|сред[ау]|четверг|пятниц[ау]|суббот[ау]|воскресенье)\b', lower):
            target = WEEKDAYS[match.group(1)]
            found.append((anchor + timedelta(days=(target - anchor.weekday()) % 7),
                          'в ' + match.group(1), match.start()))
    return found


def _value_clauses(turn_text, value):
    words = [word for word in re.findall(r'[а-яa-z0-9]+', value.lower().replace('ё', 'е'))
             if len(word) >= 4 and word not in TASK_STOP]
    if not words:
        return []
    clauses = []
    for clause in _clauses(turn_text):
        normalized = re.findall(r'[а-яa-z0-9]+', clause.lower().replace('ё', 'е'))
        if all(any(word.startswith(token[:5]) for word in normalized) for token in words):
            clauses.append(clause)
    return clauses


def _iso_day(value):
    return value.isoformat() + 'T00:00:00'


def _date_change_allowed(item, field, value):
    try:
        proposed = _datetime(value)
        other = item.get('valid_to' if field == 'valid_from' else 'valid_from')
        if not other:
            return True
        current = _datetime(other)
        return proposed <= current if field == 'valid_from' else current <= proposed
    except (TypeError, ValueError):
        return False


def _task_status(fact, call, turn):
    pattern = STATUS_PATTERNS.get(fact['value'].strip().lower())
    if not pattern:
        return copy.deepcopy(fact), None
    anchor = _datetime(call.started_at).date()
    for clause in _clauses(turn.text):
        if not re.search(pattern, clause.lower().replace('ё', 'е')):
            continue
        found = _cue_dates(clause, anchor)
        dates = sorted({item[0] for item in found})
        if len(dates) != 1:
            continue
        position = min(item[2] for item in found)
        before = clause.lower().replace('ё', 'е')[max(0, position - 12):position]
        field = 'valid_to' if re.search(r'\bдо\s*$', before) else 'valid_from'
        item = copy.deepcopy(fact)
        value = _iso_day(dates[0])
        if not _date_change_allowed(item, field, value):
            return item, {'status': 'abstained', 'reason': 'interval_conflict',
                          'changes': [], 'evidence': []}
        if item.get(field) != value:
            item[field] = value
            return item, {'status': 'changed', 'reason': 'status_explicit_time',
                          'changes': [{'field': field, 'before': fact.get(field), 'after': value}],
                          'evidence': [{'text': found[0][1], 'field': field}]}
        return item, {'status': 'unchanged', 'reason': 'already_consistent', 'changes': [],
                      'evidence': [{'text': found[0][1], 'field': field}]}
    return copy.deepcopy(fact), None


def _task_action(fact, call, turn):
    anchor = _datetime(call.started_at).date()
    chunks = _value_clauses(turn.text, fact['value']) or [turn.text]
    found = []
    for chunk in chunks:
        found.extend(_cue_dates(chunk, anchor, fuzzy=False))
    dates = sorted({item[0] for item in found})
    value = _iso_day(dates[0]) if len(dates) == 1 else call.started_at
    if fact.get('valid_from') == value:
        return copy.deepcopy(fact), {'status': 'unchanged', 'reason': 'already_consistent',
                                     'changes': []}
    if not _date_change_allowed(fact, 'valid_from', value):
        return copy.deepcopy(fact), {'status': 'abstained', 'reason': 'interval_conflict',
                                     'changes': []}
    item = copy.deepcopy(fact)
    item['valid_from'] = value
    return item, {'status': 'changed',
                  'reason': 'action_explicit_time' if len(dates) == 1 else 'action_assertion_time',
                  'changes': [{'field': 'valid_from', 'before': fact.get('valid_from'), 'after': value}],
                  'evidence': [{'text': found[0][1]}] if found else []}


def _task_agreement(fact, call, turn):
    anchor = _datetime(call.started_at).date()
    chunks = _value_clauses(turn.text, fact['value']) or [turn.text]
    found = []
    for chunk in chunks:
        found.extend(_cue_dates(chunk, anchor, future=True))
    dates = sorted({item[0] for item in found})
    if len(dates) != 1:
        return copy.deepcopy(fact), None
    value = _iso_day(dates[0])
    if fact.get('valid_from') == value:
        return copy.deepcopy(fact), {'status': 'unchanged', 'reason': 'already_consistent',
                                     'changes': []}
    if not _date_change_allowed(fact, 'valid_from', value):
        return copy.deepcopy(fact), {'status': 'abstained', 'reason': 'interval_conflict',
                                     'changes': []}
    item = copy.deepcopy(fact)
    item['valid_from'] = value
    return item, {'status': 'changed', 'reason': 'agreement_event_time',
                  'changes': [{'field': 'valid_from', 'before': fact.get('valid_from'), 'after': value}],
                  'evidence': [{'text': found[0][1]}]}


def repair_task_dates(fact, call):
    """Deterministic task-mode binding: explicit cue wins, undated action is call time."""
    item = copy.deepcopy(fact)
    event = {'attribute': fact['attribute'], 'source_turn_idx': fact['source']['turn_idx'],
             'status': 'unchanged', 'reason': None, 'changes': [], 'evidence': []}
    if fact.get('polarity', 1) != 1:
        event['reason'] = 'negative_fact'
        return item, event
    for field in ('valid_from', 'valid_to'):
        if fact.get(field):
            try:
                _datetime(fact[field])
            except (TypeError, ValueError):
                event['reason'] = 'invalid_model_date_left_for_sanitizer'
                return item, event
    turn = next((t for t in call.turns if t.idx == fact['source']['turn_idx']), None)
    if turn is None:
        event['reason'] = 'source_missing'
        return item, event
    if fact['attribute'] == 'ticket_status':
        item, decision = _task_status(fact, call, turn)
    elif fact['attribute'] == 'action':
        item, decision = _task_action(fact, call, turn)
    elif fact['attribute'] == 'agreement':
        item, decision = _task_agreement(fact, call, turn)
    else:
        decision = None
    if decision:
        event.update(decision)
    else:
        event['reason'] = 'task_mode_left_unchanged'
    return item, event


def normalize_fact_dates(facts, call, mode='conservative'):
    """No mutation, dropping, reordering or access to gold labels."""
    if mode not in ('conservative', 'task'):
        raise ValueError('temporal mode must be conservative or task')
    owners = {}
    for fact in facts:
        key = (fact['source']['turn_idx'], fact['attribute'], _norm(fact['value']))
        obj = fact['object']
        owners.setdefault(key, set()).add((obj.get('id'), _norm(obj['name']), obj.get('type')))
    output, decisions = [], []
    for index, fact in enumerate(facts):
        key = (fact['source']['turn_idx'], fact['attribute'], _norm(fact['value']))
        if len(owners[key]) > 1:
            item = copy.deepcopy(fact)
            decision = {'attribute': fact['attribute'], 'source_turn_idx': fact['source']['turn_idx'],
                        'status': 'abstained', 'reason': 'multiple_objects_same_evidence',
                        'changes': [], 'evidence': []}
        elif mode == 'conservative':
            item, decision = repair_fact_dates(fact, call)
        else:
            item, decision = repair_task_dates(fact, call)
        output.append(item)
        decisions.append({'fact_index': index, **decision})
    return output, decisions
