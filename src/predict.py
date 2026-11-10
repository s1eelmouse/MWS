"""Прогон файла обращений через модель напрямую, без сервиса.

    python src/predict.py --input data/test.jsonl --output out/pred.jsonl
    python src/eval.py --gold data/test.jsonl --pred out/pred.jsonl

Обращения идут пачками, поэтому на большом наборе это быстрее, чем запросы по одному.
Модель без дообучения (строка «до» в таблице результатов):
    python src/predict.py --model Qwen/Qwen3-0.6B --with-schema --output out/pred_base.jsonl
"""
import argparse
import json
import os
import time
from pathlib import Path

from config import MERGED_DIR
from extractor import Extractor


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/test.jsonl", help='файл со строками {"text": "..."}')
    parser.add_argument("--output", default="out/pred.jsonl")
    parser.add_argument("--model", default=os.getenv("MODEL_DIR", MERGED_DIR),
                        help="папка слитой модели или имя модели на HuggingFace")
    parser.add_argument("--with-schema", action="store_true",
                        help="добавить в запрос схему и правила: для модели без дообучения")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f if line.strip()]
    if not texts:
        raise SystemExit(f"В файле {args.input} нет строк")

    extractor = Extractor(args.model, with_schema=args.with_schema)
    extractor.extract(texts[:1])  # прогрев: первый вызов в замер не входит

    start = time.perf_counter()
    results = []
    for begin in range(0, len(texts), args.batch_size):
        results += extractor.extract(texts[begin : begin + args.batch_size], args.batch_size)
        print(f"обработано {len(results)} из {len(texts)}")
    seconds = time.perf_counter() - start

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for text, result in zip(texts, results):
            row = {"text": text, **result} if "error" in result else {"text": text, "facts": result}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    failed = sum("error" in result for result in results)
    print(f"Модель: {args.model}; обращений: {len(texts)}, без валидного ответа: {failed}")
    print(f"Время: {seconds:.1f} с, {len(texts) / seconds:.1f} обращений в секунду "
          f"(пачки по {args.batch_size})")
    print(f"Ответы записаны в {args.output}")


if __name__ == "__main__":
    main()
