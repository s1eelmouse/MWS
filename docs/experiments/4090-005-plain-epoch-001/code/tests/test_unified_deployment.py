"""Installer rejects ambiguous/corrupt archives and never replaces existing files."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('install005', ROOT / 'evaluation/install_model_005.py')
install005 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install005)


def archive_at(path, entries):
    with tarfile.open(path, 'w:gz') as archive:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def model_entries(prefix='experiment/merged'):
    return [(prefix + '/' + name, value) for name, value in (
        ('config.json', b'{}'), ('tokenizer.json', b'{}'),
        ('tokenizer_config.json', b'{}'), ('model.safetensors', b'test-weights'))]


class Deployment(unittest.TestCase):
    def test_tokenizer_and_contract_assets_are_pinned(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = root / 'tokenizer.json'
            p.write_text('{}')
            expected = {'tokenizer.json': install005.digest(p)}
            install005.check_assets(root, expected)
            p.write_text('different-tokenizer')
            with self.assertRaises(ValueError):
                install005.check_assets(root, expected)

    def test_install_only_merged_is_idempotent_and_preserves_old_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'weights.tar.gz'
            archive_at(archive, model_entries() + [('experiment/adapter/adapter_model.safetensors', b'large-adapter')])
            output = root / 'models/005'
            expected = hashlib.sha256(b'test-weights').hexdigest()
            install005.install(archive, output, expected=expected)
            install005.install(archive, output, expected=expected)
            self.assertFalse((output / 'adapter').exists())
            self.assertEqual((output / 'model.safetensors').read_bytes(), b'test-weights')
            with self.assertRaises(ValueError):
                install005.install(archive, output, expected='0' * 64)
            (output / 'config.json').write_text('changed')
            with self.assertRaises(ValueError):
                install005.install(archive, output, expected=expected)

    def test_ambiguous_wrong_hash_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'weights.tar.gz'
            archive_at(archive, model_entries('a') + model_entries('b'))
            with self.assertRaises(ValueError):
                install005.install(archive, root / 'ambiguous')
            self.assertFalse((root / 'ambiguous').exists())
            archive_at(archive, model_entries() + [('../outside', b'bad')])
            with self.assertRaises(ValueError):
                install005.install(archive, root / 'unsafe')
            self.assertFalse((root / 'outside').exists())
            archive_at(archive, model_entries())
            with self.assertRaises(ValueError):
                install005.install(archive, root / 'wrong')
            self.assertFalse((root / 'wrong').exists())


if __name__ == '__main__':
    unittest.main()
