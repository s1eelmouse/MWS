"""Merge a case LoRA adapter into a fresh model directory; existing weights are protected."""
import argparse
import json
import shutil
from pathlib import Path

from config import ADAPTER_DIR, MERGED_DIR
from schema import check_model_contract, contract_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", default=ADAPTER_DIR)
    parser.add_argument("--output", default=MERGED_DIR)
    parser.add_argument('--allow-legacy-contract', action='store_true')
    args = parser.parse_args()
    adapter, output = Path(args.adapter_dir), Path(args.output)
    if not (adapter / "adapter_config.json").is_file():
        parser.error("Adapter is not saved yet: adapter_config.json is missing")
    compatibility = check_model_contract(adapter, allow_legacy=args.allow_legacy_contract)
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; choose a fresh model directory")
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = PeftConfig.from_pretrained(adapter).base_model_name_or_path
    print(f"Base: {base}; adapter: {adapter}")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
    if model.num_parameters() > 1_500_000_000:
        raise SystemExit("Model exceeds the conservative total-parameter budget of 1.5B")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer_source = adapter if (adapter / "tokenizer_config.json").exists() else base
    AutoTokenizer.from_pretrained(tokenizer_source).save_pretrained(output)
    if (adapter / "experiment.json").is_file():
        shutil.copyfile(adapter / "experiment.json", output / "experiment.json")
    (output / 'case_contract.json').write_text(json.dumps(
        {**contract_manifest(), 'source_contract': compatibility}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"Merged model saved in {output}")


if __name__ == "__main__":
    main()
