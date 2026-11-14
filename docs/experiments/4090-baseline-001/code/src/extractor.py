"""Case-aware Transformers inference. Heavy ML imports happen only on model loading."""
import threading

from config import MAX_INPUT_TOKENS, MAX_NEW_TOKENS, MERGED_DIR, PROMPT_LOOKUP_TOKENS
from schema import as_call, parse_answer, source_problems, check_model_contract
from training_data import prompt_for


class Extractor:
    def __init__(self, model_dir: str = MERGED_DIR, with_schema: bool = False):
        check_model_contract(model_dir, with_schema)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.with_schema = with_schema
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, device_map=self.device).eval()
        settings = self.model.generation_config
        settings.do_sample, settings.temperature, settings.top_p, settings.top_k = False, None, None, None
        self.lock = threading.Lock()

    def extract(self, calls: list, batch_size: int = 1) -> list:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        calls = [as_call(call) for call in calls]
        results = []
        for start in range(0, len(calls), batch_size):
            results.extend(self._extract_safely(calls[start:start + batch_size]))
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

    def _extract_batch(self, calls):
        results = [None] * len(calls)
        prompts, indices = [], []
        context = getattr(self.model.config, "max_position_embeddings", 32768)
        budget = min(MAX_INPUT_TOKENS, context - MAX_NEW_TOKENS)
        for index, call in enumerate(calls):
            try:
                prompt = prompt_for(self.tok, call, self.with_schema)
                length = len(self.tok(prompt, add_special_tokens=False)["input_ids"])
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
        speedup = {"prompt_lookup_num_tokens": PROMPT_LOOKUP_TOKENS} if len(prompts) == 1 and PROMPT_LOOKUP_TOKENS else {}
        with self.lock, self.torch.inference_mode():
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            out = self.model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=self.tok.pad_token_id, **speedup)
            answers = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for index, raw in zip(indices, answers):
            facts = parse_answer(raw)
            if facts is None:
                results[index] = {"error": "invalid_json"}
            elif source_problems(facts, calls[index]):
                results[index] = {"error": "invalid_source"}
            else:
                results[index] = facts.model_dump(mode="json")
        return results
