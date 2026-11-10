"""Загрузка модели и извлечение фактов. Этим модулем пользуются serve.py и predict.py."""
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MAX_NEW_TOKENS, MERGED_DIR, PROMPT_LOOKUP_TOKENS
from schema import build_messages, parse_answer


class Extractor:
    def __init__(self, model_dir: str = MERGED_DIR, with_schema: bool = False):
        """model_dir — папка слитой модели или имя модели на HuggingFace.

        with_schema=True добавляет в запрос схему и правила: так честно замерять
        модель без дообучения. Дообученной модели хватает короткого запроса.
        """
        self.with_schema = with_schema
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"  # в пачке короткие запросы дополняются слева
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        # device_map кладёт веса сразу на видеокарту, не занимая оперативную память целой копией модели
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, device_map=self.device).eval()
        settings = self.model.generation_config  # жадное декодирование: ответ не зависит от случая
        settings.do_sample, settings.temperature, settings.top_p, settings.top_k = False, None, None, None
        self.lock = threading.Lock()  # к модели одновременно обращается один поток

    def extract(self, texts: list, batch_size: int = 16) -> list:
        """Для каждого текста возвращает словарь фактов либо {"error": ..., "raw": ...}."""
        results = []
        for start in range(0, len(texts), batch_size):
            results += self._extract_safely(texts[start : start + batch_size])
        return results

    def _extract_safely(self, texts: list) -> list:
        """Если на пачку не хватило видеопамяти, делит её пополам и пробует снова."""
        try:
            return self._extract_batch(texts)
        except RuntimeError as err:
            if "out of memory" not in str(err).lower() or len(texts) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(texts) // 2
            return self._extract_safely(texts[:half]) + self._extract_safely(texts[half:])

    @torch.inference_mode()
    def _extract_batch(self, texts: list) -> list:
        # Тот же запрос, что при обучении (train.py). Расхождение здесь портит качество.
        prompts = [
            self.tok.apply_chat_template(build_messages(text, self.with_schema), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
            for text in texts
        ]
        # Копирование из запроса (см. config.py) в Transformers работает только для одного текста.
        speedup = {"prompt_lookup_num_tokens": PROMPT_LOOKUP_TOKENS} if len(texts) == 1 and PROMPT_LOOKUP_TOKENS else {}
        with self.lock:
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            out = self.model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id, **speedup)
            answers = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        results = []
        for raw in answers:
            facts = parse_answer(raw)
            results.append(facts.model_dump() if facts else {"error": "invalid_json", "raw": raw})
        return results
