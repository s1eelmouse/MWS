"""Read-only validation of the evaluation inputs, model and compiled grammar."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from case_io import read_jsonl
from schema import check_model_contract
from config import JSON_CONSTRAINED, TEMPORAL_NORMALIZATION
from json_constraints import compile_json_grammar


def main():
    manifest = ROOT / 'EVALUATION_TRANSFER_MANIFEST.json'
    if manifest.exists():
        for name, digest in json.loads(manifest.read_text())['files'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f'Code hash mismatch: {name}'
    dataset = Path(os.environ['VAL_FILE'])
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == 'a2cd8a6e3583e0c0db8cc700bef0387775ad44d0639723781cea2543baf7019e', 'Different val bytes; review the experiment settings'
    rows = read_jsonl(dataset)
    assert len(rows) == len({r['call_id'] for r in rows}) == 383
    model = Path(os.environ['MODEL_DIR'])
    check_model_contract(model)
    weights = json.loads(Path(__file__).with_name('expected-model-weights.json').read_text())
    for name, digest in weights.items():
        assert hashlib.sha256((model / name).read_bytes()).hexdigest() == digest, f'Wrong second-cycle model weights: {name}'
    assert JSON_CONSTRAINED and TEMPORAL_NORMALIZATION == 'task'
    for package in ['torch', 'transformers', 'xgrammar', 'psutil', 'pydantic']:
        print(package, importlib.metadata.version(package), flush=True)
    import torch
    assert torch.cuda.is_available(), 'CUDA unavailable'
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    compile_json_grammar(tokenizer, config, max_threads=8)
    print('PREFLIGHT OK: code hashes, second-cycle weights, 383 val calls, CUDA and Facts grammar.', flush=True)


if __name__ == '__main__':
    main()
