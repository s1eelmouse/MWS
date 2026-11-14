"""Archive checks and prepared data protection without organizer data in Git."""
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prepare_case import FILES
from prepare_v21 import package_files, prepare_archive
from test_case_schema import CALL


class PackageTest(unittest.TestCase):
    def archive(self, root, *, corrupt=False, duplicate=False):
        train = copy.deepcopy(CALL)
        val = {**copy.deepcopy(CALL), 'call_id': 'other-account:call0'}
        data = {name: b'{}' for name in FILES}
        data.update({'train.jsonl': (json.dumps(train)+'\n').encode(),
                     'val.jsonl': (json.dumps(val)+'\n').encode(),
                     'case_metrics.py': b'# test scorer bytes\n', 'ANSWERS_v2.md': b'answers'})
        data['CHECKSUMS.txt'] = ('\n'.join(
            hashlib.sha256(value).hexdigest()+'  '+name for name, value in data.items())
            +'\n'+'0'*64+'  CHECKSUMS.txt\n').encode()
        if corrupt:
            data['train.jsonl'] += b' '
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, 'w') as z:
            for name, value in data.items():
                z.writestr('package/'+name, value)
            if duplicate:
                z.writestr('another/train.jsonl', data['train.jsonl'])
        archive = root/'new.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('CompactFactExtraction_participants.zip', inner.getvalue())
        return archive, data

    def test_prepares_verified_bytes_and_preserves_existing_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            archive, data = self.archive(root)
            with patch('prepare_v21.SCORER_SHA256', hashlib.sha256(data['case_metrics.py']).hexdigest()):
                report=prepare_archive(archive, root/'prepared')
                self.assertEqual(report['validation']['train']['invalid_calls'], 0)
                self.assertEqual(report['metrics_version'], '2.1')
                self.assertEqual(len(report['warnings']), 1)
                self.assertEqual((root/'prepared/train.jsonl').read_bytes(), data['train.jsonl'])
                prepare_archive(archive, root/'prepared')  # idempotent preparation
                (root/'prepared/train.jsonl').write_bytes(b'preserve old data')
                with self.assertRaisesRegex(ValueError, 'Destination differs'):
                    prepare_archive(archive, root/'prepared')
                self.assertEqual((root/'prepared/train.jsonl').read_bytes(), b'preserve old data')

    def test_rejects_bad_checksum_before_writing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            archive,data=self.archive(root,corrupt=True)
            with patch('prepare_v21.SCORER_SHA256',hashlib.sha256(data['case_metrics.py']).hexdigest()):
                with self.assertRaisesRegex(ValueError,'Checksum mismatch'):
                    prepare_archive(archive,root/'prepared')
            self.assertFalse((root/'prepared').exists())

    def test_rejects_unknown_scorer_and_ambiguous_members(self):
        with tempfile.TemporaryDirectory() as temp:
            archive,data=self.archive(Path(temp))
            with self.assertRaisesRegex(ValueError,'Unexpected scorer'):
                package_files(archive)
            archive,data=self.archive(Path(temp),duplicate=True)
            with self.assertRaisesRegex(ValueError,'Ambiguous archive member'):
                package_files(archive)
