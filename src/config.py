"""Общие настройки проекта.

Меняйте значения здесь или задавайте переменные окружения с теми же именами.
Пример для PowerShell:  $env:BASE_MODEL = "Qwen/Qwen3-0.6B"
"""
import math
import os
from pathlib import Path
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

# Deterministic date binding. "task" uses explicit text cues and the call timestamp;
# "conservative" is the earlier, narrower rule set; "off" preserves model dates.
# "case_v2" follows organizer scorer v2.x (src/temporal_case_v2.py) and needs catalog.json
# to tell catalog values from free formulations.
TEMPORAL_MODES = ("off", "conservative", "task", "case_v2")
TEMPORAL_NORMALIZATION = os.getenv("TEMPORAL_NORMALIZATION", "task").strip().lower()
if TEMPORAL_NORMALIZATION not in TEMPORAL_MODES:
    raise ValueError("TEMPORAL_NORMALIZATION must be off, conservative, task or case_v2")
_DEFAULT_CATALOG = Path("data/organizer_v2_1/catalog.json")
_FALLBACK_CATALOG = Path("case_data/catalog.json")
_catalog_default = str(_DEFAULT_CATALOG) if _DEFAULT_CATALOG.is_file() else (
    str(_FALLBACK_CATALOG) if _FALLBACK_CATALOG.is_file() else "")
CATALOG_PATH = os.getenv("CATALOG_PATH", _catalog_default).strip()
# Deterministic recovery of a missing current tariff/status for package 2.1.
FACT_RECOVERY = os.getenv("FACT_RECOVERY", "1").strip().lower() not in ("0", "false", "no", "off")
# Read by case_v2 only; unlike CATALOG_PATH it does not rewrite object names or tariff values.
TEMPORAL_CATALOG_PATH = os.getenv("TEMPORAL_CATALOG_PATH", CATALOG_PATH).strip()
if TEMPORAL_NORMALIZATION == "case_v2" and not TEMPORAL_CATALOG_PATH:
    raise ValueError("TEMPORAL_NORMALIZATION=case_v2 requires TEMPORAL_CATALOG_PATH (catalog.json)")
SAVE_RAW_RESPONSES = os.getenv("SAVE_RAW_RESPONSES", "0").strip().lower() in ("1", "true", "yes", "on")

# Печать не должна ронять скрипт из-за символа, которого нет в кодировке консоли Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")
