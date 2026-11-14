"""Release installation is additive, verified, idempotent and does not touch weights."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('v21_release_installer', ROOT / 'docs/evaluations/continue-002-v21/install_release.py')
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class ReleaseTest(unittest.TestCase):
    def package(self, path, payload=b'fixture', unsafe=False):
        files = {'src/fixture.py': payload}
        manifest = {'release_id': 'case-v21-002', 'files': {n: hashlib.sha256(v).hexdigest() for n, v in files.items()}}
        files['RELEASE_MANIFEST.json'] = json.dumps(manifest).encode()
        if unsafe:
            files['../../escape'] = b'not allowed'
        with tarfile.open(path, 'w:gz') as tar:
            for name, value in files.items():
                item = tarfile.TarInfo('case-v21-002/' + name)
                item.size = len(value)
                tar.addfile(item, io.BytesIO(value))
        return installer.digest(path)

    def run_install(self, archive, expected, team):
        with patch.object(sys, 'argv', ['install_release.py', '--archive', str(archive),
                                       '--expected-sha256', expected, '--team-root', str(team)]):
            installer.main()

    def test_verified_release_preserves_old_model_and_rejects_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            team = root / 'team_shpana'
            model = team / 'models/continue-002/model.safetensors'
            model.parent.mkdir(parents=True)
            model.write_bytes(b'original weights')
            archive = root / 'package.tar.gz'
            expected = self.package(archive)
            self.run_install(archive, expected, team)
            self.run_install(archive, expected, team)
            self.assertEqual(model.read_bytes(), b'original weights')
            replacement = root / 'different.tar.gz'
            expected = self.package(replacement, b'changed')
            with self.assertRaises(FileExistsError):
                self.run_install(replacement, expected, team)
            self.assertEqual((team / 'releases/case-v21-002/src/fixture.py').read_bytes(), b'fixture')

    def test_corruption_and_path_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'package.tar.gz'
            expected = self.package(archive, unsafe=True)
            with self.assertRaises(ValueError):
                self.run_install(archive, expected, root / 'team_shpana')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                self.run_install(archive, '0' * 64, root / 'team_shpana')
            self.assertFalse((root / 'escape').exists())
