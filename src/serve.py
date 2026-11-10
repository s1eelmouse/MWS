"""Микросервис извлечения фактов на FastAPI и Transformers.

Запуск из корня репозитория:
    uvicorn serve:app --app-dir src --port 8000

Проверка в браузере: http://127.0.0.1:8000/docs
Чтобы сервис был виден с других машин, добавьте --host 0.0.0.0.

Модель без дообучения (строка «до» в таблице результатов), PowerShell:
    $env:MODEL_DIR = "Qwen/Qwen3-0.6B"; $env:WITH_SCHEMA = "1"
    uvicorn serve:app --app-dir src --port 8000
"""
import os
import queue
import threading
from concurrent.futures import Future

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from config import MERGED_DIR
from extractor import Extractor

MODEL_DIR = os.getenv("MODEL_DIR", MERGED_DIR)
MAX_BATCH = 16  # сколько одновременных запросов склеивать в одну пачку
extractor = Extractor(MODEL_DIR, with_schema=os.getenv("WITH_SCHEMA") == "1")
app = FastAPI(title="CompactFactExtraction")


class Ticket(BaseModel):
    text: str


class Batch(BaseModel):
    texts: list[str]


def safe_extract(texts: list) -> list:
    """Сервис отвечает всегда: сбой модели превращается в запасной ответ, а не в ошибку 500."""
    try:
        return extractor.extract(texts, MAX_BATCH)
    except Exception as err:  # noqa: BLE001
        return [{"error": "internal_error", "raw": str(err)} for _ in texts]


# Запросы, пришедшие одновременно, обрабатываются одной пачкой: видеокарта тратит на пачку
# почти столько же времени, сколько на одно обращение. Запросы ждут в очереди, а фоновый
# поток забирает из неё всё, что накопилось, пока модель была занята.
waiting = queue.Queue()


def serve_queue():
    while True:
        items = [waiting.get()]  # ждём первый запрос
        while len(items) < MAX_BATCH:
            try:
                items.append(waiting.get_nowait())  # и забираем остальные, если они уже есть
            except queue.Empty:
                break
        results = safe_extract([text for text, _ in items])
        for (_, answer), result in zip(items, results):
            answer.set_result(result)


threading.Thread(target=serve_queue, daemon=True).start()


@app.post("/extract")
def extract(ticket: Ticket):
    answer = Future()
    waiting.put((ticket.text, answer))
    return answer.result()


@app.post("/extract_batch")
def extract_batch(batch: Batch):
    return safe_extract(batch.texts)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_DIR}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")  # корневой адрес открывает страницу с описанием сервиса
