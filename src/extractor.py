"""Case-aware Transformers inference. Heavy ML imports happen only on model loading."""
import threading
import time
import hashlib
from pathlib import Path

from config import (CATALOG_PATH, JSON_CONSTRAINED, JSON_GRAMMAR_BACKEND, JSON_GRAMMAR_THREADS,
                    MAX_INPUT_TOKENS, MAX_NEW_TOKENS, MERGED_DIR, PROMPT_LOOKUP_TOKENS,
                    RETRY_ON_TOKEN_LIMIT, RETRY_REPETITION_PENALTY, TEMPORAL_NORMALIZATION,
                    SAVE_RAW_RESPONSES)
from catalog_normalization import load_catalog
from json_constraints import GrammarLogitsProcessor, compile_json_grammar
from prediction import parse_prediction, salvage_sources
from schema import as_call, check_model_contract, source_problems
from training_data import prompt_for


class Extractor:
    def __init__(self, model_dir: str = MERGED_DIR, with_schema: bool = False, device: str = "auto"):
        check_model_contract(model_dir, with_schema)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.with_schema = with_schema
        self.temporal_normalization = TEMPORAL_NORMALIZATION
        self.catalog_sha256 = None
        if CATALOG_PATH:
            load_catalog(CATALOG_PATH)
            self.catalog_sha256 = hashlib.sha256(Path(CATALOG_PATH).read_bytes()).hexdigest()
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; refusing CPU fallback")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, device_map=self.device).eval()
        settings = self.model.generation_config
        settings.do_sample, settings.temperature, settings.top_p, settings.top_k = False, None, None, None
        self.constrained_json = False
        self.grammar_compile_seconds = None
        if JSON_CONSTRAINED:
            start = time.perf_counter()
            try:
                self.compiled_json_grammar = compile_json_grammar(
                    self.tok, self.model.config, JSON_GRAMMAR_THREADS)
            except Exception as exc:
                raise RuntimeError(
                    "Could not compile the Facts JSON grammar. Install a compatible xgrammar "
                    "or set JSON_CONSTRAINED=0 to run without constrained decoding."
                ) from exc
            self.grammar_compile_seconds = time.perf_counter() - start
            self.constrained_json = True
        self.lock = threading.Lock()

    def _json_logits_processor(self):
        if not getattr(self, "constrained_json", False):
            return None
        return GrammarLogitsProcessor(self.compiled_json_grammar, JSON_GRAMMAR_BACKEND)

    def extract(self, calls: list, batch_size: int = 1, measurements: list | None = None) -> list:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        calls = [as_call(call) for call in calls]
        results = []
        for start in range(0, len(calls), batch_size):
            batch = calls[start:start + batch_size]
            # Benchmarks must not silently change the requested batch on OOM.
            results.extend(self._extract_safely(batch) if measurements is None
                           else self._extract_batch(batch, measurements))
        return results

    def _extract_safely(self, calls):
        try:
            return self._extract_batch(calls)
        except RuntimeError as err:
            if "out of memory" not in str(err).lower() or len(calls) == 1:
                raise
            if self.device == "cuda":
                self.torch.cuda.empty_cache()
            half = len(calls) // 2
            return self._extract_safely(calls[:half]) + self._extract_safely(calls[half:])

    def _extract_batch(self, calls, measurements=None, repetition_penalty=None, allow_retry=True):
        results = [None] * len(calls)
        details = [{"call_id": c.call_id, "input_tokens": None, "output_tokens": 0,
                    "generation_limit_reached": False} for c in calls]
        if measurements is not None:
            measurements.extend(details)
        prompts, indices = [], []
        context = getattr(self.model.config, "max_position_embeddings", 32768)
        budget = min(MAX_INPUT_TOKENS, context - MAX_NEW_TOKENS)
        for index, call in enumerate(calls):
            try:
                prompt = prompt_for(self.tok, call, self.with_schema)
                length = len(self.tok(prompt, add_special_tokens=False)["input_ids"])
                details[index]["input_tokens"] = length
                if length > budget:
                    results[index] = {"error": "input_too_long", "input_tokens": length, "budget": budget}
                    continue
            except ValueError:
                results[index] = {"error": "input_too_long"}
                continue
            prompts.append(prompt)
            indices.append(index)
        if not prompts:
            return results
        for index in indices:
            # An interrupted generation has an unknown output length, not zero.
            details[index]["output_tokens"] = None
        # Prompt lookup proposes tokens without grammar masking; constrained decoding wins.
        speedup = ({"prompt_lookup_num_tokens": PROMPT_LOOKUP_TOKENS}
                   if len(prompts) == 1 and PROMPT_LOOKUP_TOKENS
                   and not getattr(self, "constrained_json", False) else {})
        processor = self._json_logits_processor()
        grammar = {"logits_processor": [processor]} if processor is not None else {}
        generation = ({"repetition_penalty": repetition_penalty}
                      if repetition_penalty is not None else {})
        with self.lock, self.torch.inference_mode():
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            out = self.model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=self.tok.pad_token_id, **speedup, **grammar, **generation)
            generated = out[:, enc["input_ids"].shape[1]:]
            answers = self.tok.batch_decode(generated, skip_special_tokens=True)
            eos = self.model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            for index, tokens in zip(indices, generated.tolist()):
                end = next((i + 1 for i, token in enumerate(tokens) if token in eos_ids), None)
                details[index]["output_tokens"] = end if end is not None else len(tokens)
                details[index]["generation_limit_reached"] = end is None and len(tokens) >= MAX_NEW_TOKENS
        for index, raw in zip(indices, answers):
            if SAVE_RAW_RESPONSES and measurements is not None:
                details[index]['raw_response'] = raw
            temporal_decisions = []
            facts, problem, normalization = parse_prediction(raw, calls[index],
                                                              temporal_decisions=temporal_decisions)
            if temporal_decisions:
                details[index]['temporal_decisions'] = temporal_decisions
            if normalization:
                details[index]["normalization"] = normalization
            retry_problem = None
            if details[index]["generation_limit_reached"]:
                problem = {"error": "output_token_limit", "reason": "max_new_tokens_without_eos",
                           "parse_diagnostic": problem}
                if RETRY_ON_TOKEN_LIMIT and allow_retry:
                    initial_output_tokens = details[index]["output_tokens"]
                    retry_details = []
                    retry_results = self._extract_batch(
                        [calls[index]], measurements=retry_details,
                        repetition_penalty=RETRY_REPETITION_PENALTY, allow_retry=False)
                    retry = retry_results[0]
                    if SAVE_RAW_RESPONSES and measurements is not None:
                        details[index]['initial_raw_response'] = raw
                        details[index]['raw_response'] = retry_details[0].get('raw_response')
                    details[index]["retry"] = {
                        "reason": "output_token_limit",
                        "repetition_penalty": RETRY_REPETITION_PENALTY,
                        "initial_output_tokens": initial_output_tokens,
                        "output_tokens": retry_details[0].get("output_tokens"),
                        "generation_limit_reached": retry_details[0].get("generation_limit_reached"),
                        "error": retry.get("error"),
                    }
                    if "error" not in retry:
                        results[index] = retry
                        details[index]["output_tokens"] = retry_details[0].get("output_tokens")
                        details[index]["generation_limit_reached"] = retry_details[0].get(
                            "generation_limit_reached")
                        if retry_details[0].get("normalization"):
                            details[index]["normalization"] = retry_details[0]["normalization"]
                        if retry_details[0].get("temporal_decisions"):
                            details[index]["temporal_decisions"] = retry_details[0][
                                "temporal_decisions"]
                        continue
                    if retry_details[0].get("diagnostic"):
                        retry_problem = {
                            **retry_details[0]["diagnostic"],
                            "retry_from": "output_token_limit",
                            "retry_repetition_penalty": RETRY_REPETITION_PENALTY,
                        }
                        problem = retry_problem
            elif facts is not None:
                problems = source_problems(facts, calls[index])
                if problems:
                    salvaged, dropped = salvage_sources(facts, calls[index])
                    if salvaged is None:
                        problem = {"error": "invalid_source", "reason": "source_validation",
                                   "source_problems": problems}
                    else:
                        facts = salvaged
                        normalization = dict(normalization or {})
                        normalization["invalid_source_facts_dropped"] = (
                            normalization.get("invalid_source_facts_dropped", 0) + dropped)
                        details[index]["normalization"] = normalization
            if problem:
                results[index] = {"error": problem["error"]}
                if measurements is not None:
                    details[index]["diagnostic"] = (retry_problem if retry_problem is not None
                                                    else {**problem, "raw_response": raw})
            else:
                results[index] = facts.model_dump(mode="json")
        return results
