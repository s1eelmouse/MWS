"""Подсчёт точности, полноты и F1 по полям.

Здесь нет сторонних библиотек, поэтому модуль легко проверять тестами.

Как считается. Значение каждого поля превращается в набор строк: у обычного поля
в наборе один элемент, у списка — по элементу на значение. Строки сравниваются
без учёта регистра и лишних пробелов.
    верно найдено (tp)  — элемент есть и в эталоне, и в ответе;
    лишнее (fp)         — элемент есть только в ответе;
    пропущено (fn)      — элемент есть только в эталоне.
Неверное значение обычного поля даёт сразу и лишнее, и пропущенное.

Это наша рабочая формула. Когда организаторы назовут свою, замените её здесь.
"""


def items(value) -> set:
    """Набор нормализованных строк для одного значения поля; пустое значение — пустой набор."""
    if value is None:
        return set()
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return {" ".join(str(item).lower().split()) for item in value if item is not None} - {""}


def _summary(counts: dict) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, **counts}


def score(pairs) -> dict:
    """pairs — список пар (эталон, ответ), оба — словари с фактами.

    Возвращает {"micro": {...}, "fields": {имя поля: {...}}}; в каждом словаре
    precision, recall, f1 и счётчики tp, fp, fn.
    """
    totals = {}
    for gold, pred in pairs:
        for key in set(gold) | set(pred):
            want, got = items(gold.get(key)), items(pred.get(key))
            counts = totals.setdefault(key, {"tp": 0, "fp": 0, "fn": 0})
            counts["tp"] += len(want & got)
            counts["fp"] += len(got - want)
            counts["fn"] += len(want - got)
    micro = {name: sum(counts[name] for counts in totals.values()) for name in ("tp", "fp", "fn")}
    return {
        "micro": _summary(micro),
        "fields": {key: _summary(counts) for key, counts in sorted(totals.items())},
    }
