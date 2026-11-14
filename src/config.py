"""Общие настройки проекта.

Меняйте значения здесь или задавайте переменные окружения с теми же именами.
Пример для PowerShell:  $env:BASE_MODEL = "Qwen/Qwen3-0.6B"
"""
import math
import os
from pathlib import Path
import sys

_DATA_ROOT = Path(__file__).resolve().parents[1] / 'data/organizer-v2.1'
CASE_DATA_DIR = os.getenv("CASE_DATA_DIR", os.getenv('ORGANIZER_DIR', str(_DATA_ROOT)))
# Compatibility alias for teammate commands; one authoritative dataset path.
ORGANIZER_DIR = Path(CASE_DATA_DIR)
SCORER_VERSION = os.getenv("SCORER_VERSION", "2.1").strip()
if SCORER_VERSION not in ("1", "2.1", "2.1-datefix"):
    raise ValueError("SCORER_VERSION must be 1, 2.1 or 2.1-datefix")
ALLOW_LEGACY_CONTRACT = os.getenv("ALLOW_LEGACY_CONTRACT", "0").lower() in ("1", "true", "yes", "on")

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
# "conservative" is the earlier rule set; "binding" is experimental status-event
# binding; "off" preserves model dates. Cross-call memory is separately opt-in.
TEMPORAL_NORMALIZATION = os.getenv("TEMPORAL_NORMALIZATION", "task").strip().lower()
TEMPORAL_MODES = ('off', 'conservative', 'task', 'binding', 'case_v2')
if TEMPORAL_NORMALIZATION not in TEMPORAL_MODES:
    raise ValueError("TEMPORAL_NORMALIZATION must be off, conservative, task, binding or case_v2")
CATALOG_PATH = os.getenv("CATALOG_PATH", "").strip()
TEMPORAL_CATALOG_PATH = os.getenv('TEMPORAL_CATALOG_PATH', CATALOG_PATH or str(Path(CASE_DATA_DIR) / 'catalog.json')).strip()
FACT_RECOVERY = os.getenv('FACT_RECOVERY', '0').strip().lower() in ('1', 'true', 'yes', 'on')
SAVE_RAW_RESPONSES = os.getenv("SAVE_RAW_RESPONSES", "0").strip().lower() in ("1", "true", "yes", "on")
TEMPORAL_SERIES_HISTORY = os.getenv("TEMPORAL_SERIES_HISTORY", "0").strip().lower() not in ("0", "false", "no", "off")
TEMPORAL_SERIES_MAX_ENTRIES = int(os.getenv("TEMPORAL_SERIES_MAX_ENTRIES", "10000"))
HISTORY_MAX_ACCOUNTS = int(os.getenv("HISTORY_MAX_ACCOUNTS", "512"))
HISTORY_MAX_BYTES = int(os.getenv("HISTORY_MAX_BYTES", str(8 * 1024 * 1024)))
HISTORY_TTL_SECONDS = float(os.getenv("HISTORY_TTL_SECONDS", "3600"))
if min(HISTORY_MAX_ACCOUNTS, HISTORY_MAX_BYTES) < 1:
    raise ValueError("History account and byte limits must be positive")
if not math.isfinite(HISTORY_TTL_SECONDS) or HISTORY_TTL_SECONDS <= 0:
    raise ValueError("History TTL must be finite and positive")
if TEMPORAL_SERIES_MAX_ENTRIES < 1:
    raise ValueError("TEMPORAL_SERIES_MAX_ENTRIES must be positive")
if TEMPORAL_SERIES_HISTORY and TEMPORAL_NORMALIZATION not in ('task', 'binding', 'case_v2'):
    raise ValueError("TEMPORAL_SERIES_HISTORY requires task, binding or case_v2 mode")

# Печать не должна ронять скрипт из-за символа, которого нет в кодировке консоли Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")
