"""Pinned inference profile for the measured 005 release; no training or downloads."""
import hashlib
import json
from pathlib import Path

MODEL_SHA256 = 'cef9fcde9f1f2a812e42c6122a0f5094fbaa456f6077596cf03ebb86e4d552f6'
DATA_SHA256 = {
    'val.jsonl': 'e12610354aa69dff83e1d7c1e8c94736c97d5761d20af58790435f4be1941808',
    'catalog.json': 'b7579c704e03e76e174e54bf7e7099171bc62301a2cfd8f5b3c7838b552701cb'}
ENV = {'SCORER_VERSION': '2.1', 'TEMPORAL_NORMALIZATION': 'case_v2',
       'FACT_RECOVERY': '1', 'TEMPORAL_SERIES_HISTORY': '1',
       'TEMPORAL_SERIES_MAX_ENTRIES': '10000', 'HISTORY_MAX_ACCOUNTS': '512',
       'HISTORY_MAX_BYTES': '8388608', 'HISTORY_TTL_SECONDS': '3600',
       'JSON_CONSTRAINED': '1', 'JSON_GRAMMAR_BACKEND': 'torch_native',
       'JSON_GRAMMAR_THREADS': '8', 'SAVE_RAW_RESPONSES': '1',
       'MAX_INPUT_TOKENS': '12288', 'MAX_SEQUENCE_TOKENS': '16384',
       'MAX_NEW_TOKENS': '4096', 'MAX_TEXT_CHARS': '0', 'PROMPT_LOOKUP_TOKENS': '0',
       'RETRY_ON_TOKEN_LIMIT': '1', 'RETRY_REPETITION_PENALTY': '1.2',
       'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'PYTHONUNBUFFERED': '1',
       'HF_HUB_OFFLINE': '1', 'ALLOW_LEGACY_CONTRACT': '0', 'WITH_SCHEMA': '0'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment(gold):
    gold = Path(gold).resolve()
    return {**ENV, 'CASE_DATA_DIR': str(gold.parent),
            'CATALOG_PATH': str(gold.with_name('catalog.json')),
            'TEMPORAL_CATALOG_PATH': str(gold.with_name('catalog.json'))}


def preflight(model, gold):
    """Fail before GPU loading if weights/data are not the recorded release."""
    model, gold = Path(model).resolve(), Path(gold).resolve()
    paths = {'val.jsonl': gold, 'catalog.json': gold.with_name('catalog.json')}
    actual = {name: digest(path) for name, path in paths.items()}
    if actual != DATA_SHA256:
        raise ValueError('Release data/catalog checksum differs; do not mix organizer versions')
    actual_model = digest(model / 'model.safetensors')
    if actual_model != MODEL_SHA256:
        raise ValueError('Weights differ from measured 005; choose and document a separate release profile')
    from schema import check_model_contract
    contract = check_model_contract(model, with_schema=False, allow_legacy=False)
    if contract['policy'] != 'exact':
        raise ValueError('Release requires the exact v2.1 model contract')
    return {'model_sha256': actual_model, 'data_sha256': actual,
            'contract': contract, 'environment': environment(gold)}
