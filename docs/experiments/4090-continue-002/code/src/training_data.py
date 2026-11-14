"""CPU-testable encoding and experiment safeguards; no ML imports."""
import hashlib
import json
from pathlib import Path

from case_io import validate_gold
from config import MAX_INPUT_TOKENS, MAX_NEW_TOKENS
from schema import build_messages, to_answer


def prompt_for(tokenizer, row, with_schema=False):
    return tokenizer.apply_chat_template(
        build_messages(row, with_schema), tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )


def encode_row(tokenizer, row, max_sequence_tokens):
    validate_gold(row)
    question = tokenizer(prompt_for(tokenizer, row), add_special_tokens=False)["input_ids"]
    answer = tokenizer(to_answer(row["facts"]) + tokenizer.eos_token,
                       add_special_tokens=False)["input_ids"]
    if (len(question) > MAX_INPUT_TOKENS or len(answer) > MAX_NEW_TOKENS
            or len(question) + len(answer) > max_sequence_tokens):
        raise ValueError(
            f"{row['call_id']}: input={len(question)}, target={len(answer)} tokens exceed limits. "
            "Transcript and targets were NOT truncated; choose a larger budget or implement windows."
        )
    return {"input_ids": question + answer, "labels": [-100] * len(question) + answer}


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapter_origin(directory, output_directory, base_model):
    """Validate a local starting adapter and record its exact bytes before GPU loading."""
    directory, output = Path(directory).resolve(), Path(output_directory).resolve()
    if directory == output or directory in output.parents or output in directory.parents:
        raise ValueError('Starting adapter and new output must be separate directories')
    config_path = directory / 'adapter_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('base_model_name_or_path') != base_model or config.get('peft_type') != 'LORA':
        raise ValueError('Starting adapter must be LoRA for the selected base model')
    from schema import check_model_contract
    check_model_contract(directory)
    weights = [p for p in directory.iterdir() if p.name in ('adapter_model.safetensors', 'adapter_model.bin')]
    if len(weights) != 1:
        raise ValueError('Starting adapter must have exactly one adapter weight file')
    return {'directory': str(directory), 'files': {p.name: file_digest(p) for p in
            [config_path, directory / 'case_contract.json', *weights]}}


def ensure_training_splits(train_rows, val_rows):
    if not train_rows:
        raise ValueError("Training file is empty")
    train_accounts = {r["call_id"].split(":")[0] for r in train_rows}
    val_accounts = {r["call_id"].split(":")[0] for r in val_rows}
    if train_accounts & val_accounts:
        raise ValueError("Training and validation contain the same account")


def check_output_directory(path, resume=None):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and resume is None:
        raise ValueError(f"{path} is not empty; choose a new output directory or explicitly resume")
    if resume is not None:
        if not (path / 'experiment.json').is_file():
            raise ValueError('Resume requires the original experiment.json manifest')
        checkpoint = Path(resume).resolve()
        if not checkpoint.is_dir() or path.resolve() not in checkpoint.parents:
            raise ValueError("Resume checkpoint must exist inside this experiment directory")
        if not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("Checkpoint has no trainer_state.json")
    return path


def save_manifest(path, manifest):
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != manifest:
            raise ValueError("Experiment manifest differs; refusing to mix data/model/settings")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
