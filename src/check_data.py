"""Проверка размеченного файла: соответствие схеме и дословность цитат.

    python src/check_data.py data/labeled.jsonl

Помогает при ручной проверке: показывает строки, где учитель пересказал текст
вместо цитаты или вышел за схему.
"""
import json
import sys

import config  # noqa: F401  (настраивает печать в консоль)
from schema import Facts

QUOTED = ("product", "problem", "request", "error_code")


def find_problems(row: dict) -> list:
    try:
        facts = Facts.model_validate(row["facts"])
        text = row["text"].lower()
    except (KeyError, TypeError, ValueError) as err:
        return [f"не по схеме: {' '.join(str(err).split())[:160]}"]
    found = []
    for name in QUOTED:
        value = getattr(facts, name)
        if value and value.lower() not in text:
            found.append(f"{name}: «{value}» нет в тексте дословно")
    found += [f"ids: «{item}» нет в тексте" for item in facts.ids if item.lower() not in text]
    return found


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Запуск: python src/check_data.py путь/к/файлу.jsonl")
    total = bad = 0
    with open(sys.argv[1], encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            total += 1
            try:
                problems = find_problems(json.loads(line))
            except ValueError:
                problems = ["строка не является JSON"]
            if problems:
                bad += 1
                print(f"строка {number}: " + "; ".join(problems))
    print(f"Проверено строк: {total}, с замечаниями: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
