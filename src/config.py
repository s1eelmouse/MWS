"""Общие настройки проекта.

Меняйте значения здесь или задавайте переменные окружения с теми же именами.
Пример для PowerShell:  $env:BASE_MODEL = "Qwen/Qwen3-0.6B"
"""
import math
import os
import sys

# Стартовая модель; альтернативы проверяются по полному числу параметров <= 1.5B.
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "out/adapter")  # сюда train.py сохраняет адаптер LoRA
MERGED_DIR = os.getenv("MERGED_DIR", "out/merged")     # сюда merge.py сохраняет готовую модель

MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "0"))  # 0 = no character cap; never truncate
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "12288"))
MAX_SEQUENCE_TOKENS = int(os.getenv("MAX_SEQUENCE_TOKENS", "16384"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))

# Constrained decoding against the Facts schema. It guarantees JSON syntax and the
# contract shape before validation. Set JSON_CONSTRAINED=0 only for an explicit A/B run.
JSON_CONSTRAINED = os.getenv("JSON_CONSTRAINED", "1").strip().lower() not in ("0", "false", "no", "off")
JSON_GRAMMAR_THREADS = int(os.getenv("JSON_GRAMMAR_THREADS", "8"))
JSON_GRAMMAR_BACKEND = os.getenv("JSON_GRAMMAR_BACKEND", "auto").strip().lower()
if JSON_GRAMMAR_THREADS < 1:
    raise ValueError("JSON_GRAMMAR_THREADS must be at least 1")
if JSON_GRAMMAR_BACKEND not in ("auto", "cpu", "cuda", "triton", "torch_compile", "torch_native"):
    raise ValueError("JSON_GRAMMAR_BACKEND must be auto, cpu, triton, torch_compile or torch_native")

# A call that hits max_new_tokens gets one retry with this penalty. Normal calls
# are unchanged; the retry is recorded in calls.jsonl.
RETRY_ON_TOKEN_LIMIT = os.getenv("RETRY_ON_TOKEN_LIMIT", "1").strip().lower() not in ("0", "false", "no", "off")
RETRY_REPETITION_PENALTY = float(os.getenv("RETRY_REPETITION_PENALTY", "1.2"))
if not math.isfinite(RETRY_REPETITION_PENALTY) or RETRY_REPETITION_PENALTY <= 0:
    raise ValueError("RETRY_REPETITION_PENALTY must be finite and positive")

# Ускорение одиночного запроса. Факты — цитаты из обращения, поэтому модель за один шаг
# проверяет сразу несколько токенов, скопированных из запроса. Ответ тот же, шагов меньше.
PROMPT_LOOKUP_TOKENS = int(os.getenv("PROMPT_LOOKUP_TOKENS", "0"))

# Opt-in deterministic postprocessing. Preserve the historical model contract.
TEMPORAL_NORMALIZATION = os.getenv("TEMPORAL_NORMALIZATION", "off").strip().lower()
if TEMPORAL_NORMALIZATION not in ("off", "conservative"):
    raise ValueError("TEMPORAL_NORMALIZATION must be off or conservative")
CATALOG_PATH = os.getenv("CATALOG_PATH", "").strip()

# Печать не должна ронять скрипт из-за символа, которого нет в кодировке консоли Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")
