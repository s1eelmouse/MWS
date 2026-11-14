"""Exact, ambiguity-aware catalog matching; never fuzzy-merges entities."""
import copy
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata


def normalize_label(value):
    value = unicodedata.normalize('NFKC', value).casefold().replace('ё', 'е')
    value = re.sub(r'[‐‑–—]', '-', value)
    value = value.strip(' \t\r\n\"\'«»„“”')
    return ' '.join(value.split())


@dataclass(frozen=True)
class CatalogMatch:
    canonical: str | None
    candidates: tuple[str, ...]
    reason: str


class CatalogResolver:
    def __init__(self, catalog, aliases=None):
        self.index = {}
        for kind, labels in catalog.items():
            if not isinstance(labels, list) or any(not isinstance(x, str) or not x.strip() for x in labels):
                raise ValueError('Catalog values must be lists of nonempty strings')
            index = self.index.setdefault(kind, {})
            for label in labels:
                index.setdefault(normalize_label(label), set()).add(label)
        # Aliases must be explicitly supplied and point to actual catalog entries.
        for kind, mapping in (aliases or {}).items():
            if kind not in self.index:
                raise ValueError('Alias refers to an unknown catalog kind')
            for alias, names in mapping.items():
                names = [names] if isinstance(names, str) else names
                if not names:
                    raise ValueError('Alias must have at least one target')
                for name in names:
                    if name not in catalog[kind]:
                        raise ValueError('Alias target is absent from catalog')
                    self.index[kind].setdefault(normalize_label(alias), set()).add(name)

    def resolve(self, value, kinds):
        kinds = [kinds] if isinstance(kinds, str) else kinds
        candidates = set()
        for kind in kinds:
            candidates.update(self.index.get(kind, {}).get(normalize_label(value), ()))
        values = tuple(sorted(candidates))
        return CatalogMatch(values[0] if len(values) == 1 else None, values,
                            'exact_normalized_match' if len(values) == 1 else
                            'ambiguous' if values else 'no_exact_match')


@lru_cache(maxsize=4)
def load_catalog(path):
    return CatalogResolver(json.loads(Path(path).read_text(encoding='utf-8')))


def normalize_catalog_facts(facts, resolver):
    """Canonicalize object spelling and whole tariff values; preserve all IDs/roles."""
    result, decisions = [], []
    for index, fact in enumerate(facts):
        item = copy.deepcopy(fact)
        kind = {'service': 'services', 'device': 'devices'}.get(item['object'].get('type'))
        kinds = [kind] if kind else ['services', 'devices'] if item['object'].get('type') is None else []
        match = resolver.resolve(item['object']['name'], kinds)
        if match.canonical is not None and match.canonical != item['object']['name']:
            decisions.append({'fact_index': index, 'field': 'object.name',
                              'before': item['object']['name'], 'after': match.canonical,
                              'rule': match.reason})
            item['object']['name'] = match.canonical
        if item['attribute'] == 'tariff':
            match = resolver.resolve(item['value'], 'tariffs')
            if match.canonical is not None and match.canonical != item['value']:
                decisions.append({'fact_index': index, 'field': 'value',
                                  'before': item['value'], 'after': match.canonical,
                                  'rule': match.reason})
                item['value'] = match.canonical
        result.append(item)
    return result, decisions
