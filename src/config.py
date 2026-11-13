"""Общие настройки проекта.

Меняйте значения здесь или задавайте переменные окружения с теми же именами.
Пример для PowerShell:  $env:BASE_MODEL = "Qwen/Qwen3-0.6B"
"""
import os
import sys

# Стартовая модель; альтернативы проверяются по полному числу параметров <= 1.5B.
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "out/adapter")  # сюда train.py сохраняет адаптер LoRA
MERGED_DIR = os.getenv("MERGED_DIR", "out/merged")     # сюда merge.py сохраняет готовую модель

MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "0"))  # 0 = no character cap; never truncate
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "8192"))
MAX_SEQUENCE_TOKENS = int(os.getenv("MAX_SEQUENCE_TOKENS", "16384"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))

# Ускорение одиночного запроса. Факты — цитаты из обращения, поэтому модель за один шаг
# проверяет сразу несколько токенов, скопированных из запроса. Ответ тот же, шагов меньше.
PROMPT_LOOKUP_TOKENS = int(os.getenv("PROMPT_LOOKUP_TOKENS", "0"))

# Печать не должна ронять скрипт из-за символа, которого нет в кодировке консоли Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")
