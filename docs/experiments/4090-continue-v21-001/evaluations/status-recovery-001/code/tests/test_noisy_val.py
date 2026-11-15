"""Noisy val copy: deterministic, labels intact, original untouched."""
import copy
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from case_io import validate_gold
from make_noisy_val import corrupt_text, make_noisy
from test_case_schema import CALL

TEXT = "Интернет со вторника пропадает, роутер перезагружал уже раза три — не помогает."


class NoisyValTest(unittest.TestCase):
    def rows(self):
        row = copy.deepcopy(CALL)
        row["turns"] = [{"idx": 7, "speaker": "client", "text": TEXT},
                        {"idx": 8, "speaker": "operator", "text": "Понял, сейчас проверю линию."}]
        row["facts"][0]["source"]["span"] = [0, 8]
        return [row]

    def test_same_seed_gives_same_copy_and_original_is_untouched(self):
        rows = self.rows()
        before = copy.deepcopy(rows)
        first, second = make_noisy(rows, .5, 42), make_noisy(rows, .5, 42)
        self.assertEqual(first, second)
        self.assertEqual(rows, before)
        self.assertNotEqual(make_noisy(rows, .5, 7)[0]["turns"], first[0]["turns"])

    def test_labels_and_turn_structure_survive(self):
        rows = self.rows()
        noisy = make_noisy(rows, .6, 1)[0]
        self.assertEqual([t["idx"] for t in noisy["turns"]], [7, 8])
        self.assertEqual([t["speaker"] for t in noisy["turns"]], ["client", "operator"])
        self.assertNotEqual(noisy["turns"][0]["text"], TEXT)
        fact, original = noisy["facts"][0], rows[0]["facts"][0]
        for key in ("object", "attribute", "value", "polarity", "valid_from", "valid_to"):
            self.assertEqual(fact[key], original[key])
        self.assertEqual(fact["source"], {"turn_idx": 7, "span": None, "quote": None})
        self.assertEqual(noisy["noise_level"], .85)
        validate_gold(noisy)

    def test_zero_level_keeps_text(self):
        self.assertEqual(make_noisy(self.rows(), 0, 42)[0]["turns"][0]["text"], TEXT)

    def test_text_never_becomes_empty(self):
        for seed in range(50):
            self.assertTrue(corrupt_text("да нет", 1.0, random.Random(seed)))


class NoisyCatalogTest(unittest.TestCase):
    def test_catalog_is_preserved_and_conflicting_destination_is_rejected(self):
        import json
        import tempfile
        from unittest.mock import patch
        import make_noisy_val
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {'call_id': 'synthetic:call0', 'started_at': '2026-11-13T12:00:00',
                   'turns': [{'idx': 0, 'speaker': 'client', 'text': 'вчера перезагрузил роутер'}], 'facts': []}
            (root / 'val.jsonl').write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
            (root / 'catalog.json').write_text('{"actions": ["перезагрузка роутера"]}', encoding='utf-8')
            output = root / 'noisy/val.jsonl'
            argv = ['make_noisy_val.py', '--input', str(root / 'val.jsonl'), '--output', str(output)]
            with patch.object(sys, 'argv', argv):
                make_noisy_val.main()
            self.assertEqual((output.parent / 'catalog.json').read_bytes(), (root / 'catalog.json').read_bytes())
            manifest = json.loads(output.with_suffix('.manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(manifest['catalog_next_to_output'])
            self.assertIsNotNone(manifest['catalog_sha256'])
            (output.parent / 'catalog.json').write_text('{}', encoding='utf-8')
            argv[-1] = str(output.parent / 'other.jsonl')
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
                make_noisy_val.main()
            self.assertFalse((output.parent / 'other.jsonl').exists())


if __name__ == "__main__":
    unittest.main()
