"""Общие настройки проекта.

Меняйте значения здесь или задавайте переменные окружения с теми же именами.
Пример для PowerShell:  $env:BASE_MODEL = "Qwen/Qwen3-0.6B"
"""
import os
import sys

# Модель-кандидат с HuggingFace: Qwen/Qwen3-1.7B, Qwen/Qwen3-0.6B, MTSAIR/Cotype-Nano
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-1.7B")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "out/adapter")  # сюда train.py сохраняет адаптер LoRA
MERGED_DIR = os.getenv("MERGED_DIR", "out/merged")     # сюда merge.py сохраняет готовую модель

MAX_TEXT_CHARS = 3000  # обращение обрезается до этой длины и при обучении, и в сервисе
MAX_NEW_TOKENS = 256   # потолок длины ответа модели в токенах

# Ускорение одиночного запроса. Факты — цитаты из обращения, поэтому модель за один шаг
# проверяет сразу несколько токенов, скопированных из запроса. Ответ тот же, шагов меньше.
PROMPT_LOOKUP_TOKENS = 10  # 0 — выключить

# Печать не должна ронять скрипт из-за символа, которого нет в кодировке консоли Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")
