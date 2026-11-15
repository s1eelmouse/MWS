"""Optional vLLM backend sharing exactly the same HTTP and organizer call contract."""
import os
from concurrent.futures import ThreadPoolExecutor

from config import (MAX_NEW_TOKENS, TEMPORAL_NORMALIZATION,
                    TEMPORAL_SERIES_HISTORY, FACT_RECOVERY, TEMPORAL_CATALOG_PATH)
from series_history import new_series_memory
from llm_api import chat
from prediction import parse_prediction
from schema import Facts, build_messages, source_problems, contract_signature
from serve import create_app

ANSWER_FORMAT = {"type": "json_schema",
                 "json_schema": {"name": "facts", "schema": Facts.model_json_schema()}}


class VLLMExtractor:
    constrained_json = True  # response_format asks vLLM for schema-guided decoding

    def __init__(self):
        self.model_contract = {'target_signature': contract_signature(), 'policy': 'external-backend-unverified'}
        self.temporal_normalization = TEMPORAL_NORMALIZATION
        self.fact_recovery = FACT_RECOVERY
        from fact_recovery import RECOVERY_POLICY
        self.fact_recovery_policy = RECOVERY_POLICY if FACT_RECOVERY else None
        self.temporal_catalog_sha256 = None
        if TEMPORAL_NORMALIZATION == 'case_v2':
            from pathlib import Path
            import hashlib
            from temporal_case_v2 import load_layers
            load_layers(TEMPORAL_CATALOG_PATH)
            self.temporal_catalog_sha256 = hashlib.sha256(Path(TEMPORAL_CATALOG_PATH).read_bytes()).hexdigest()
        self.series_history_enabled = TEMPORAL_SERIES_HISTORY
        self.series_memory = new_series_memory() if TEMPORAL_SERIES_HISTORY else None

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
            facts, problem, _ = parse_prediction(raw, call)
            if facts is None:
                return {"error": problem["error"]}
            if source_problems(facts, call):
                return {"error": "invalid_source"}
            return facts.model_dump(mode="json")
        with ThreadPoolExecutor(max_workers=min(8, len(calls) or 1)) as pool:
            results = list(pool.map(ask, calls))
        if self.series_memory is not None:
            results, _, _ = self.series_memory.apply(calls, results)
        return results


app = create_app(VLLMExtractor())
