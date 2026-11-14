"""QLoRA on organizer calls, with resumable checkpoints and explicit token budgets."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import inspect
import json
import math
import time
from pathlib import Path

from case_io import read_jsonl, validate_gold
from config import ADAPTER_DIR, BASE_MODEL, MAX_SEQUENCE_TOKENS
from training_data import (adapter_origin, check_output_directory, encode_row,
                           ensure_training_splits, file_digest, save_manifest)
from schema import contract_signature


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_batch(rows: list, pad_id: int, tail_only: bool = True) -> dict:
    """Собирает пачку из примеров разной длины.

    Короткие примеры дополняются слева, поэтому ответы всех примеров стоят в конце.
    Предсказания нужны только там, где ответ: в хвосте длиной keep (ответ и одна позиция
    перед ним). Считать их для всех позиций запроса — в разы больше видеопамяти при том
    же результате. tail_only=False оставляет все позиции: для моделей без logits_to_keep.
    """
    import torch
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/organizer/train.jsonl")
    parser.add_argument("--val", default="data/organizer/val.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-sequence-tokens", type=int, default=MAX_SEQUENCE_TOKENS)
    parser.add_argument("--output-dir", default=ADAPTER_DIR)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--init-adapter-dir",
                        help="start a NEW cycle from adapter weights, with a fresh optimizer")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--check-data", action="store_true",
                        help="validate data and split isolation without GPU or model downloads")
    args = parser.parse_args()
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.init_adapter_dir and args.resume_from_checkpoint:
        parser.error("Choose starting adapter weights OR resume an interrupted cycle")
    origin = (adapter_origin(args.init_adapter_dir, args.output_dir, BASE_MODEL)
              if args.init_adapter_dir else None)
    if min(args.epochs, args.batch_size, args.save_steps, args.max_sequence_tokens) <= 0:
        parser.error("epochs, batch size, save steps and sequence budget must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be positive or -1")
    train_rows = read_jsonl(args.train, args.limit)
    val_rows = read_jsonl(args.val, args.limit)
    ensure_training_splits(train_rows, val_rows)
    for row in train_rows + val_rows:
        validate_gold(row)
    if args.check_data:
        print(json.dumps({"train_calls": len(train_rows), "val_calls": len(val_rows),
                          "model": BASE_MODEL, "gpu_run": False}, ensure_ascii=False, indent=2))
        return
    output = check_output_directory(args.output_dir, args.resume_from_checkpoint)
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                                  Trainer, TrainingArguments, set_seed)
    except ImportError as exc:
        raise SystemExit("Install GPU dependencies from requirements.txt and CUDA-enabled PyTorch first") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is unavailable. Data validation works here; training must run on the RTX PC.")
    set_seed(args.seed)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    encoded_train = [encode_row(tok, row, args.max_sequence_tokens) for row in train_rows]
    encoded_val = [encode_row(tok, row, args.max_sequence_tokens) for row in val_rows]
    load_options = {"dtype": dtype, "device_map": {"": 0}}
    if not args.no_4bit:
        load_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_options)
    context_limit = getattr(model.config, 'max_position_embeddings', args.max_sequence_tokens)
    if any(len(row['input_ids']) > context_limit for row in encoded_train + encoded_val):
        raise SystemExit('An example exceeds the model context window; no examples were truncated')
    parameter_count = model.num_parameters()
    if parameter_count > 1_500_000_000:
        raise SystemExit(f"Base model has {parameter_count} parameters, above the 1.5B case budget")
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if origin:
        model = PeftModel.from_pretrained(model, origin['directory'], is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=.05,
            task_type="CAUSAL_LM", target_modules="all-linear"))
    model.print_trainable_parameters()
    tail_only = "logits_to_keep" in inspect.signature(model.get_base_model().forward).parameters
    manifest = {
        "model": BASE_MODEL, "base_parameters": parameter_count, "seed": args.seed,
        "train_sha256": file_digest(args.train), "val_sha256": file_digest(args.val),
        "schema_sha256": file_digest(Path(__file__).with_name("schema.py")),
        "train_limit": args.limit, "max_sequence_tokens": args.max_sequence_tokens,
        "batch_size": args.batch_size, "four_bit": not args.no_4bit,
    }
    if origin or args.learning_rate != 2e-4:
        manifest.update(initial_adapter=origin, learning_rate=args.learning_rate,
                        optimizer_policy='fresh cycle; checkpoint resume restores optimizer')
    if args.resume_from_checkpoint:
        previous = json.loads((output / 'experiment.json').read_text(encoding='utf-8'))
        if previous.get('initial_adapter'):
            manifest.update(initial_adapter=previous['initial_adapter'],
                            learning_rate=args.learning_rate,
                            optimizer_policy=previous['optimizer_policy'])
    save_manifest(output / "experiment.json", manifest)
    save_manifest(output / 'case_contract.json', {'signature': contract_signature()})
    # Each resumed invocation gets its own journal; previous histories are retained.
    from measurements import environment, write_json
    from training_observer import make_observer
    attempt = output / f"attempt-{time.time_ns()}"
    attempt.mkdir()
    write_json(attempt / "environment.json", environment())
    from config import MAX_INPUT_TOKENS, MAX_NEW_TOKENS
    write_json(attempt / "settings.json", {
        **vars(args), "model": BASE_MODEL, "learning_rate": args.learning_rate,
        "initial_adapter": origin,
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": .05,
        "max_input_tokens": MAX_INPUT_TOKENS, "max_new_tokens": MAX_NEW_TOKENS,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_revision": getattr(model.config, "_commit_hash", None),
    })
    accumulate = max(1, 16 // args.batch_size)
    steps = args.max_steps if args.max_steps > 0 else math.ceil(
        len(train_rows) / (args.batch_size * accumulate)) * args.epochs
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output / "checkpoints"), num_train_epochs=args.epochs,
            max_steps=args.max_steps, learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
            gradient_accumulation_steps=accumulate, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            lr_scheduler_type="cosine", warmup_steps=max(1, steps // 20),
            bf16=bf16, fp16=not bf16, seed=args.seed, data_seed=args.seed,
            logging_steps=1 if args.max_steps > 0 else 10,
            eval_strategy="epoch" if encoded_val else "no",
            save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
            label_names=["labels"], prediction_loss_only=True, report_to="none"),
        train_dataset=Rows(encoded_train),
        eval_dataset=Rows(encoded_val) if encoded_val else None,
        data_collator=lambda rows: make_batch(rows, tok.pad_token_id, tail_only),
        callbacks=[make_observer(attempt, torch)],
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    status, failure = "failed", None
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        status = "complete"
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        if "out of memory" in str(exc).lower():
            raise SystemExit("CUDA memory exhausted. Checkpoints are retained; reduce batch size "
                             "or use a larger GPU. No transcript was silently truncated.") from exc
        raise
    finally:
        write_json(attempt / "status.json", {
            "status": status, "error": failure, "global_step": trainer.state.global_step,
            "elapsed_seconds": time.perf_counter() - start,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
        write_json(attempt / "loss-history.json", trainer.state.log_history)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    model.save_pretrained(output)
    tok.save_pretrained(output)
    stats = {
        "elapsed_seconds": seconds, "completed_global_steps": trainer.state.global_step,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(0), "base_parameters": parameter_count,
        "torch": torch.__version__,
    }
    stats_path = output / f"training_stats_step_{trainer.state.global_step}_{time.time_ns()}.json"
    with stats_path.open("x", encoding="utf-8") as stream:
        json.dump(stats, stream, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Adapter and checkpoints saved in {output}")


if __name__ == "__main__":
    main()
