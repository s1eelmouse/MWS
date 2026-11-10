"""Делит размеченный файл на train, val и test.

    python src/split.py --input data/labeled.jsonl

Каждое обращение попадает в свою часть по хешу текста. Поэтому после доразметки и
повторного запуска старые примеры остаются там же, где были: тест не утекает в обучение.
"""
import argparse
import hashlib
import json
from pathlib import Path


def part_of(text: str, val: int, test: int) -> str:
    bucket = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < test:
        return "test"
    if bucket < test + val:
        return "val"
    return "train"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/labeled.jsonl", help="размеченный файл")
    parser.add_argument("--out-dir", default="data", help="куда положить train.jsonl, val.jsonl, test.jsonl")
    parser.add_argument("--val", type=int, default=10, help="доля val в процентах")
    parser.add_argument("--test", type=int, default=10, help="доля test в процентах")
    args = parser.parse_args()

    rows = {}  # текст -> строка; повтор текста заменяет прежнюю разметку
    with open(args.input, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "text" not in row or "facts" not in row:
                raise SystemExit(f"{args.input}, строка {number}: нужны поля text и facts")
            rows[row["text"]] = row

    parts = {"train": [], "val": [], "test": []}
    for text, row in rows.items():
        parts[part_of(text, args.val, args.test)].append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in part:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)}")
    empty = [name for name, part in parts.items() if not part]
    if empty:
        print(f"Внимание: пустые части: {', '.join(empty)}. Данных мало, добавьте примеров.")


if __name__ == "__main__":
    main()
