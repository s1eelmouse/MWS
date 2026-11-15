"""Unified package preparation: checksums, direct/nested archives and no overwrite."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prepare_case import FILES, package_files, prepare_archive
from test_case_schema import CALL


class PackageTest(unittest.TestCase):
    def archive(self, root, *, corrupt=False, duplicate=False, nested=True):
        train = copy.deepcopy(CALL)
        val = {**copy.deepcopy(CALL), 'call_id': 'other-account:call0'}
        data = {name: b'{}' for name in FILES}
        data.update({'train.jsonl': (json.dumps(train) + '\n').encode(),
                     'val.jsonl': (json.dumps(val) + '\n').encode(),
                     'case_metrics.py': b'# fixture scorer\n', 'ANSWERS_v2.md': b'answers'})
        data['CHECKSUMS.txt'] = ('\n'.join(hashlib.sha256(value).hexdigest() + '  ' + name
                 for name, value in data.items()) + '\n' + '0' * 64 + '  CHECKSUMS.txt\n').encode()
        if corrupt:
            data['train.jsonl'] += b' '
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, 'w') as bundle:
            for name, value in data.items():
                bundle.writestr('package/' + name, value)
            if duplicate:
                bundle.writestr('another/train.jsonl', data['train.jsonl'])
        archive = root / 'new.zip'
        if nested:
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('nested/CompactFactExtraction_participants.zip', inner.getvalue())
        else:
            archive.write_bytes(inner.getvalue())
        return archive, data

    def test_direct_and_nested_verified_packages_and_existing_data(self):
        for nested in (False, True):
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive, data = self.archive(root, nested=nested)
                with patch('prepare_case.SCORER_SHA256', hashlib.sha256(data['case_metrics.py']).hexdigest()):
                    report = prepare_archive(archive, root / 'prepared')
                    self.assertEqual(report['validation']['train']['invalid_calls'], 0)
                    self.assertEqual(report['metrics_version'], '2.1')
                    self.assertEqual(len(report['warnings']), 1)
                    self.assertEqual((root / 'prepared/train.jsonl').read_bytes(), data['train.jsonl'])
                    prepare_archive(archive, root / 'prepared')
                    (root / 'prepared/train.jsonl').write_bytes(b'preserve old data')
                    with self.assertRaisesRegex(ValueError, 'Destination differs'):
                        prepare_archive(archive, root / 'prepared')
                    self.assertEqual((root / 'prepared/train.jsonl').read_bytes(), b'preserve old data')

    def test_corrupted_or_unknown_packages_are_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, data = self.archive(root, corrupt=True)
            with patch('prepare_case.SCORER_SHA256', hashlib.sha256(data['case_metrics.py']).hexdigest()):
                with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                    prepare_archive(archive, root / 'prepared')
            self.assertFalse((root / 'prepared').exists())
            archive, data = self.archive(root)
            with self.assertRaisesRegex(ValueError, 'Unexpected scorer'):
                package_files(archive)
            archive, data = self.archive(root, duplicate=True)
            with self.assertRaisesRegex(ValueError, 'Ambiguous archive member'):
                package_files(archive)
