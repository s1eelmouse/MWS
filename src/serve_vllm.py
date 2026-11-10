"""Тот же сервис, но модель работает в vLLM — быстрый вариант.

vLLM считает параллельные запросы одной пачкой и выдаёт ответ строго по JSON-схеме.
Он работает только на Linux; на Windows его запускают в WSL.

1. В Linux или WSL:
       vllm serve out/merged --served-model-name extractor --max-model-len 4096 --port 8001
2. В окружении проекта, из корня репозитория:
       uvicorn serve_vllm:app --app-dir src --port 8000

Адреса /extract, /extract_batch и /health те же, что у serve.py, поэтому eval.py
и mcp_server.py работают с обоими вариантами без изменений.
"""
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from pydantic import BaseModel

from config import MAX_NEW_TOKENS
from llm_api import chat
from schema import Facts, build_messages, parse_answer

# 127.0.0.1, а не localhost: в Windows запрос на localhost идёт на 2 секунды дольше.
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8001/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "extractor")
ANSWER_FORMAT = {"type": "json_schema",
                 "json_schema": {"name": "facts", "schema": Facts.model_json_schema()}}

app = FastAPI(title="CompactFactExtraction (vLLM)")
pool = ThreadPoolExecutor(32)


class Ticket(BaseModel):
    text: str


class Batch(BaseModel):
    texts: list[str]


def ask(text: str) -> dict:
    try:
        raw = chat(VLLM_URL, "-", VLLM_MODEL, build_messages(text), temperature=0,
                   max_tokens=MAX_NEW_TOKENS, response_format=ANSWER_FORMAT,
                   chat_template_kwargs={"enable_thinking": False})
    except Exception as err:  # noqa: BLE001  (сервис отвечает даже при сбое vLLM)
        return {"error": "backend_failed", "raw": str(err)}
    facts = parse_answer(raw)
    return facts.model_dump() if facts else {"error": "invalid_json", "raw": raw}


@app.post("/extract")
def extract(ticket: Ticket):
    return ask(ticket.text)


@app.post("/extract_batch")
def extract_batch(batch: Batch):
    return list(pool.map(ask, batch.texts))


@app.get("/health")
def health():
    return {"status": "ok", "backend": VLLM_URL}
