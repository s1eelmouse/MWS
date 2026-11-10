"""Разметка обращений моделью-учителем (любой API в формате OpenAI).

Вход:   data/raw.jsonl      — в каждой строке {"text": "..."}
Выход:  data/labeled.jsonl  — в каждой строке {"text": "...", "facts": {...}}

Перед запуском задайте переменные окружения. Пример для PowerShell:
    $env:TEACHER_URL = "https://адрес-сервиса/v1"
    $env:TEACHER_KEY = "ключ"
    $env:TEACHER_MODEL = "имя-модели"

    python src/label.py --limit 20     # сначала проба на 20 обращениях
    python src/label.py                # затем весь файл

Скрипт можно прервать и запустить снова: уже размеченные тексты он пропустит.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config  # noqa: F401  (настраивает печать в консоль)
from llm_api import chat
from schema import build_messages, parse_answer

ENV = ("TEACHER_URL", "TEACHER_KEY", "TEACHER_MODEL")


def label(text: str, json_mode: bool):
    """Возвращает факты или None, если учитель ответил не по схеме."""
    params = {"temperature": 0}
    if json_mode:
        params["response_format"] = {"type": "json_object"}
    raw = chat(os.environ["TEACHER_URL"], os.environ["TEACHER_KEY"], os.environ["TEACHER_MODEL"],
               build_messages(text, with_schema=True), **params)
    return parse_answer(raw)


def read_texts(path) -> list:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/raw.jsonl")
    parser.add_argument("--output", default="data/labeled.jsonl")
    parser.add_argument("--limit", type=int, help="разметить не больше стольких обращений")
    parser.add_argument("--workers", type=int, default=4, help="сколько запросов слать параллельно")
    parser.add_argument("--no-json-mode", action="store_true",
                        help="не просить у API режим JSON (если сервис его не поддерживает)")
    args = parser.parse_args()

    missing = [name for name in ENV if not os.getenv(name)]
    if missing:
        raise SystemExit(f"Задайте переменные окружения: {', '.join(missing)}")

    done = set(read_texts(args.output))
    todo = [text for text in dict.fromkeys(read_texts(args.input)) if text not in done]
    todo = todo[: args.limit]
    print(f"Уже размечено: {len(done)}, осталось в этом запуске: {len(todo)}")

    saved = skipped = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(args.workers)
    try:
        with open(args.output, "a", encoding="utf-8") as out:
            answers = pool.map(lambda text: label(text, not args.no_json_mode), todo)
            for text, facts in zip(todo, answers):
                if facts is None:
                    skipped += 1  # в файл не попадает, поэтому следующий запуск попробует ещё раз
                    continue
                out.write(json.dumps({"text": text, "facts": facts.model_dump()}, ensure_ascii=False) + "\n")
                out.flush()
                saved += 1
                if saved % 50 == 0:
                    print(f"размечено {saved}")
    finally:
        # При ошибке API или Ctrl+C не ждём оставшуюся очередь: размеченное уже в файле.
        pool.shutdown(wait=True, cancel_futures=True)
    print(f"Готово: размечено {saved}, пропущено из-за ответа не по схеме: {skipped}")


if __name__ == "__main__":
    main()
