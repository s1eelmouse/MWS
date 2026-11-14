"""Optional vLLM backend sharing exactly the same HTTP and organizer call contract."""
import os
from concurrent.futures import ThreadPoolExecutor

from config import MAX_NEW_TOKENS
from llm_api import chat
from schema import Facts, build_messages, parse_answer, source_problems
from serve import create_app

ANSWER_FORMAT = {"type": "json_schema",
                 "json_schema": {"name": "facts", "schema": Facts.model_json_schema()}}


class VLLMExtractor:
    def extract(self, calls, batch_size=1):
        def ask(call):
            try:
                raw = chat(
                    os.getenv("VLLM_URL", "http://127.0.0.1:8001/v1"), "-",
                    os.getenv("VLLM_MODEL", "extractor"), build_messages(call),
                    temperature=0, max_tokens=MAX_NEW_TOKENS, response_format=ANSWER_FORMAT,
                    chat_template_kwargs={"enable_thinking": False})
            except Exception:
                return {"error": "backend_failed"}
            facts = parse_answer(raw)
            if facts is None:
                return {"error": "invalid_json"}
            if source_problems(facts, call):
                return {"error": "invalid_source"}
            return facts.model_dump(mode="json")
        with ThreadPoolExecutor(max_workers=min(8, len(calls) or 1)) as pool:
            return list(pool.map(ask, calls))


app = create_app(VLLMExtractor())
