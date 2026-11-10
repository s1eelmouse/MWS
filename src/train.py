"""Дообучение малой модели методом QLoRA.

    python src/train.py                           # data/train.jsonl и data/val.jsonl, 3 эпохи
    python src/train.py --limit 200 --epochs 1    # быстрая проба: доходит ли скрипт до конца
    python src/train.py --batch-size 2            # если не хватает видеопамяти

Какую модель учить, задаёт BASE_MODEL в config.py или одноимённая переменная окружения.
Результат — адаптер LoRA в out/adapter. Следующий шаг: python src/merge.py
"""
import os

# Работаем на одной видеокарте: так проще и предсказуемее (в Kaggle их две).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,  # noqa: E402
                          Trainer, TrainingArguments)

from config import ADAPTER_DIR, BASE_MODEL, MAX_NEW_TOKENS  # noqa: E402
from schema import build_messages, to_answer  # noqa: E402

LEARNING_RATE = 2e-4
BATCH_SIZE = 4        # примеров за один проход по видеокарте; меняется ключом --batch-size
EFFECTIVE_BATCH = 16  # примеров на один шаг обучения: при меньшем BATCH_SIZE градиент копится дольше
LORA_RANK = 16        # ошибка на train не падает — поднимите до 32


class Rows(Dataset):
    """Готовые примеры в виде, который принимает Trainer."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def read_jsonl(path, limit=None) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()][:limit]


def make_batch(rows: list, pad_id: int, tail_only: bool = True) -> dict:
    """Собирает пачку из примеров разной длины.

    Короткие примеры дополняются слева, поэтому ответы всех примеров стоят в конце.
    Предсказания нужны только там, где ответ: в хвосте длиной keep (ответ и одна позиция
    перед ним). Считать их для всех позиций запроса — в разы больше видеопамяти при том
    же результате. tail_only=False оставляет все позиции: для моделей без logits_to_keep.
    """
    width = max(len(row["input_ids"]) for row in rows)
    keep = width
    if tail_only:
        keep = max(sum(label != -100 for label in row["labels"]) for row in rows) + 1
    input_ids, mask, labels = [], [], []
    for row in rows:
        pad = width - len(row["input_ids"])
        input_ids.append([pad_id] * pad + row["input_ids"])
        mask.append([0] * pad + [1] * len(row["input_ids"]))
        labels.append(([-100] * pad + row["labels"])[-keep:])
    batch = {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(mask),
             "labels": torch.tensor(labels)}
    if tail_only:
        batch["logits_to_keep"] = keep
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--val", default="data/val.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, help="взять только первые N примеров (для пробы)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="примеров за один проход; не хватает видеопамяти — поставьте 2 или 1")
    parser.add_argument("--no-4bit", action="store_true",
                        help="обычная LoRA без 4-битной загрузки: быстрее, но нужно больше видеопамяти")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch не видит видеокарту с CUDA. Поставьте PyTorch с поддержкой CUDA "
                         "или обучайте в Kaggle либо Colab.")

    # bfloat16 умеют видеокарты NVIDIA 30-й серии и новее; у T4 и более старых — float16.
    bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if bf16 else torch.float16

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.no_4bit:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype, device_map={"": 0})
    else:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant, device_map={"": 0})
        model = prepare_model_for_kbit_training(model)
    # "all-linear" — адаптеры на все линейные слои, кроме выходного: подходит любой модели.
    model = get_peft_model(model, LoraConfig(r=LORA_RANK, lora_alpha=2 * LORA_RANK, lora_dropout=0.05,
                                             task_type="CAUSAL_LM", target_modules="all-linear"))
    model.print_trainable_parameters()

    # Умеет ли модель считать предсказания только для конца последовательности (Qwen умеет).
    tail_only = "logits_to_keep" in inspect.signature(model.get_base_model().forward).parameters

    def prompt_for(text):
        # Тот же запрос, что в сервисе (extractor.py). Расхождение здесь портит качество.
        return tok.apply_chat_template(build_messages(text), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    def encode(row):
        question = tok(prompt_for(row["text"]), add_special_tokens=False)["input_ids"]
        # Для моделей не из семейства Qwen проверьте, что eos_token — это конец ответа в их шаблоне.
        answer = tok(to_answer(row["facts"]) + tok.eos_token, add_special_tokens=False)["input_ids"]
        # -100 означает «здесь ошибку не считать»: модель учится только на ответе.
        # Имена полей совпадают с аргументами модели: поля с другими именами Trainer отбрасывает.
        return {"input_ids": question + answer, "labels": [-100] * len(question) + answer}

    train_rows, val_rows = read_jsonl(args.train, args.limit), read_jsonl(args.val, args.limit)
    if not train_rows:
        raise SystemExit(f"В файле {args.train} нет примеров")
    print(f"Модель: {BASE_MODEL}; примеров train: {len(train_rows)}, val: {len(val_rows)}")

    accumulate = max(1, EFFECTIVE_BATCH // args.batch_size)
    steps = math.ceil(len(train_rows) / (args.batch_size * accumulate)) * args.epochs
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="out/run", num_train_epochs=args.epochs, learning_rate=LEARNING_RATE,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=accumulate, lr_scheduler_type="cosine",
            warmup_steps=max(1, steps // 20), bf16=bf16, fp16=not bf16,
            logging_steps=max(1, min(10, steps // 10)),
            eval_strategy="epoch" if val_rows else "no", save_strategy="no",
            label_names=["labels"], report_to="none"),
        train_dataset=Rows([encode(row) for row in train_rows]),
        eval_dataset=Rows([encode(row) for row in val_rows]) if val_rows else None,
        data_collator=lambda rows: make_batch(rows, tok.pad_token_id, tail_only),
    )
    try:
        trainer.train()
    except RuntimeError as err:
        if "out of memory" not in str(err).lower():
            raise
        raise SystemExit("Не хватило памяти. Запустите с меньшим пакетом, например --batch-size 2, и закройте "
                         "тяжёлые программы (браузер, IDE): в Windows видеопамять расходует и общую память.") from None
    print(f"Пик видеопамяти при обучении: {torch.cuda.max_memory_allocated() / 2**30:.1f} ГБ")
    model.save_pretrained(ADAPTER_DIR)
    print(f"Адаптер сохранён в {ADAPTER_DIR}. Дальше: python src/merge.py")

    # Проверка глазами: ответ модели на одно обращение, которого она не видела при обучении.
    sample = (val_rows or train_rows)[0]
    torch.cuda.empty_cache()
    model.eval()
    try:
        enc = tok(prompt_for(sample["text"]), return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
    except RuntimeError as err:  # адаптер уже сохранён, поэтому сбой здесь обучение не портит
        print("Пробный ответ получить не удалось:", str(err).splitlines()[0])
        return
    print("Обращение:   ", sample["text"])
    print("Ответ модели:", tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    print("Эталон:      ", to_answer(sample["facts"]))


if __name__ == "__main__":
    main()
