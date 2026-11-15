import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('model_transfer', ROOT / 'docs/migrations/4090-to-v100/model-transfer.py')
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


class ModelTransferTest(unittest.TestCase):
    def fixture(self, root):
        model = root / 'old/out/merged-4090-continue-002'
        data = root / 'old/data/organizer'
        adapter = model.parent / '4090-continue-002'
        for directory in (model, data, adapter / 'checkpoints/checkpoint-246'):
            directory.mkdir(parents=True)
        weights = b'synthetic weights, not a downloaded model'
        (model / 'model.safetensors').write_bytes(weights)
        for name in ('config.json', 'tokenizer_config.json', 'case_contract.json'):
            (model / name).write_text('{}')
        for name in ('val.jsonl', 'train.jsonl', 'catalog.json'):
            (data / name).write_text('{}\n')
        (adapter / 'adapter_config.json').write_text('{}')
        (adapter / 'adapter_model.safetensors').write_bytes(b'synthetic adapter')
        (adapter / 'checkpoints/checkpoint-246/optimizer.pt').write_bytes(b'not needed for next fresh cycle')
        return model, data, hashlib.sha256(weights).hexdigest()

    def test_minimal_package_installs_identical_model_data_and_adapter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, data, expected = self.fixture(root)
            archive = root / 'transfer.tar.gz'
            with patch.object(transfer, 'EXPECTED_WEIGHT_SHA256', expected):
                transfer.pack(archive, model, data)
            target = root / 'team_shpana'
            target.mkdir()
            transfer.install(archive, target)
            self.assertEqual((target / 'models/continue-002/model.safetensors').read_bytes(),
                             (model / 'model.safetensors').read_bytes())
            self.assertTrue((target / 'adapters/continue-002/adapter_model.safetensors').is_file())
            self.assertFalse((target / 'adapters/continue-002/checkpoints').exists())
            self.assertIn('JSON_GRAMMAR_BACKEND=torch_native', (target / 'runtime.env').read_text())
            self.assertEqual(json.loads((target / 'out/v100-assets.json').read_text())['model'],
                             str(target / 'models/continue-002'))
            with self.assertRaises(FileExistsError):
                transfer.install(archive, target)

    def test_corrupted_transfer_stops_before_extracting_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, data, expected = self.fixture(root)
            archive = root / 'transfer.tar.gz'
            with patch.object(transfer, 'EXPECTED_WEIGHT_SHA256', expected):
                transfer.pack(archive, model, data)
            archive.write_bytes(archive.read_bytes() + b'broken')
            target = root / 'team_shpana'
            target.mkdir()
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                transfer.install(archive, target)
            self.assertFalse((target / 'models').exists())
