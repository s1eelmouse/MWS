"""Вливает адаптер LoRA в базовую модель и сохраняет готовую модель в float16.

    python src/merge.py

Слитая модель отвечает быстрее, чем модель с подключённым адаптером, и её можно
сжимать дальше. Результат — папка out/merged.
"""
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ADAPTER_DIR, MERGED_DIR


def main():
    # Базовая модель берётся из самого адаптера: так её нельзя перепутать.
    base = PeftConfig.from_pretrained(ADAPTER_DIR).base_model_name_or_path
    print(f"Базовая модель: {base}; адаптер: {ADAPTER_DIR}")

    # Слияние идёт на процессоре, модель целиком лежит в оперативной памяти. В float16 ей нужно
    # вдвое меньше места, чем в float32 (Qwen3-1.7B — около 3,5 ГБ), а результат получается тем же.
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
    except MemoryError:
        raise SystemExit("Не хватило оперативной памяти для загрузки модели. "
                         "Закройте тяжёлые программы (браузер, IDE) и запустите ещё раз.") from None
    model = PeftModel.from_pretrained(model, ADAPTER_DIR).merge_and_unload()
    model.save_pretrained(MERGED_DIR)
    AutoTokenizer.from_pretrained(base).save_pretrained(MERGED_DIR)

    size = sum(file.stat().st_size for file in Path(MERGED_DIR).rglob("*") if file.is_file())
    print(f"Готово: {MERGED_DIR}, {size / 1e9:.2f} ГБ")


if __name__ == "__main__":
    main()
