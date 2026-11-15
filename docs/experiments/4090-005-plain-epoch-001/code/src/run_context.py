"""Versioned dataset, grammar and scorer provenance shared by train/eval/benchmark."""
import hashlib
import json
from pathlib import Path

from config import CATALOG_PATH
from schema import Facts, contract_manifest
from scoring import make_scorer

KNOWN_DATA = {'0878bfd983e14c75b51f32f04d961c5b82b10fa243845a25fe1c811cb4882466': 'organizer-2.1-train',
              'e12610354aa69dff83e1d7c1e8c94736c97d5761d20af58790435f4be1941808': 'organizer-2.1-val',
              '041dacaa1f6974b2c5b915eaa0f58eeafa537b0c6aa77d5282df533677029d6a': 'organizer-v1-train',
              'a2cd8a6e3583e0c0db8cc700bef0387775ad44d0639723781cea2543baf7019e': 'organizer-v1-val'}


def file_info(path):
    if not path:
        return None
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size,
            'recognized_version': KNOWN_DATA.get(digest, 'custom-or-different-bytes')}


def catalog_for(dataset, explicit=None):
    if explicit:
        path = Path(explicit)
    elif CATALOG_PATH:
        path = Path(CATALOG_PATH)
    else:
        path = Path(dataset).parent / 'catalog.json'
        if not path.exists():
            return None
    if not path.is_file():
        raise ValueError(f'Catalog file not found: {path}')
    return str(path)


def context(*, train=None, val=None, catalog=None, scorer_version=None):
    scorer = make_scorer(scorer_version, catalog)
    schema = json.dumps(Facts.model_json_schema(), sort_keys=True, ensure_ascii=False).encode()
    result = {'contract': contract_manifest(), 'json_schema_sha256': hashlib.sha256(schema).hexdigest(),
              'scoring': scorer.metadata(), 'train': file_info(train), 'val': file_info(val),
              'catalog': file_info(catalog)}
    result['context_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return result
