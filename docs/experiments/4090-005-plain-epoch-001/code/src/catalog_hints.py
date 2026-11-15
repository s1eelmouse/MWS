"""Catalog hints per attribute: which catalog values a call probably talks about.

The organizer catalog has 422 entries, too many for every prompt. This module builds a
per-attribute hint table from catalog.json (plus how often each value occurs in train)
and finds, for one call, the few entries whose words occur in the transcript. Matching
is by word prefix and similarity, so typical ASR damage does not hide an entry. Only the
transcript and the catalog are read at lookup time; gold facts are never used there.
"""
import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

SECTIONS = {
    'service_state': ('symptoms',), 'tariff': ('tariffs',), 'ticket_status': ('statuses',),
    'action': ('actions',), 'agreement': ('agreements',), 'object': ('services', 'devices'),
}
STOP = {'и', 'в', 'во', 'на', 'по', 'от', 'до', 'для', 'не', 'с', 'со', 'к', 'за', 'из', 'у', 'о', 'же'}
TRANSLIT = {'wifi': 'вай фай', 'wi': 'вай', 'fi': 'фай', 'вайфай': 'вай фай', 'tv': 'тв', 'sms': 'смс'}
NORMAL_STATE = 'работает нормально'


def words(text):
    result = []
    for word in re.findall(r'[а-яa-z0-9+]+', text.lower().replace('ё', 'е')):
        result.extend(TRANSLIT.get(word, word).split())
    return result


@lru_cache(maxsize=1_000_000)
def close(entry_word, text_word):
    """Same word up to inflection or one or two ASR edits."""
    if entry_word == text_word:
        return True
    if len(entry_word) <= 3 or len(text_word) <= 3 or abs(len(entry_word) - len(text_word)) > 4:
        return False
    prefix = min(5, len(entry_word) - 1, len(text_word) - 1)
    if prefix >= 4 and entry_word[:prefix] == text_word[:prefix]:
        return True
    matcher = SequenceMatcher(None, entry_word, text_word)
    return matcher.real_quick_ratio() >= 0.8 and matcher.quick_ratio() >= 0.8 and matcher.ratio() >= 0.8


def build_hints(catalog, train_rows=()):
    """Hint table: attribute -> entries with their matching keys and train frequency."""
    seen = Counter()
    for row in train_rows:
        for fact in row.get('facts', ()):
            seen[(fact['attribute'], ' '.join(words(fact['value'])))] += 1
            seen[('object', ' '.join(words(fact['object'].get('name') or '')))] += 1
    table = {}
    for attribute, sections in SECTIONS.items():
        entries = []
        values = [(section, value) for section in sections for value in catalog.get(section, ())]
        if attribute == 'service_state':
            values.append(('symptoms', NORMAL_STATE))
        for section, value in values:
            keys = [w for w in words(value) if w not in STOP]
            entries.append({'value': value, 'section': section, 'keys': keys or words(value),
                            'train_count': seen[(attribute, ' '.join(words(value)))]})
        table[attribute] = entries
    return table


def find_hints(call, table, limit=8):
    """Catalog candidates for one call: attribute -> values, best supported first."""
    turns = call['turns'] if isinstance(call, dict) else [t.model_dump() for t in call.turns]
    turn_words = [set(words(turn['text'])) for turn in turns]
    vocabulary = set().union(*turn_words) if turn_words else set()
    matched = {}  # catalog key -> transcript words it resembles, computed once per call

    def resembling(key):
        if key not in matched:
            matched[key] = {word for word in vocabulary if close(key, word)}
        return matched[key]

    result = {}
    for attribute, entries in table.items():
        found = []
        for entry in entries:
            keys = entry['keys']
            if sum(bool(resembling(key)) for key in keys) < (len(keys) if len(keys) < 4 else len(keys) - 1):
                continue
            best = max((sum(bool(resembling(key) & text) for key in keys) for text in turn_words), default=0)
            # every key for short entries, all but one for entries of four or more keys
            if keys and best >= (len(keys) if len(keys) < 4 else len(keys) - 1):
                found.append((best / len(keys), entry['train_count'], entry['value']))
        found.sort(key=lambda item: (-item[0], -item[1], item[2]))
        result[attribute] = [value for _, _, value in found[:limit]]
    return result


def coverage(rows, table, limit=8):
    """How often a gold catalog value is among the hints of its call (diagnostic only)."""
    exact = {a: {' '.join(words(e['value'])) for e in entries} for a, entries in table.items()}
    total, hit, sizes = Counter(), Counter(), Counter()
    for row in rows:
        hints = find_hints(row, table, limit)
        for attribute, values in hints.items():
            sizes[attribute] += len(values)
        offered = {a: {' '.join(words(v)) for v in values} for a, values in hints.items()}
        for fact in row['facts']:
            for attribute, value in ((fact['attribute'], fact['value']),
                                     ('object', fact['object'].get('name') or '')):
                key = ' '.join(words(value))
                if key in exact[attribute]:
                    total[attribute] += 1
                    hit[attribute] += key in offered[attribute]
    return {a: {'gold_catalog_values': total[a], 'in_hints': hit[a],
                'recall': round(hit[a] / total[a], 4) if total[a] else None,
                'hints_per_call': round(sizes[a] / max(1, len(rows)), 2)} for a in table}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--train', help='adds how often each value occurs in train')
    parser.add_argument('--check', action='append', default=[],
                        help='labelled file to report hint coverage on; may repeat')
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--output', required=True, help='attribute_hints.json')
    args = parser.parse_args()
    catalog = json.loads(Path(args.catalog).read_text(encoding='utf-8'))
    table = build_hints(catalog, read_jsonl(args.train) if args.train else ())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(table, stream, ensure_ascii=False, indent=1)
        stream.write('\n')
    report = {'output': str(output), 'entries': {a: len(e) for a, e in table.items()},
              'coverage': {path: coverage(read_jsonl(path), table, args.limit) for path in args.check}}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
