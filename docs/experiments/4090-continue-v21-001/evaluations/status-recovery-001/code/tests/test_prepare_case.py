"""prepare_case refuses a package that does not match the organizer CHECKSUMS.txt."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_case import verify_checksums


class ChecksumTest(unittest.TestCase):
    def test_listed_files_are_verified_and_the_list_itself_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(verify_checksums(root), 0)  # packages before v2 have no list
            (root / "val.jsonl").write_bytes(b"{}\n")
            good = hashlib.sha256(b"{}\n").hexdigest()
            (root / "CHECKSUMS.txt").write_text(f"{good}  val.jsonl\n{'0' * 64}  CHECKSUMS.txt\n", encoding="utf-8")
            self.assertEqual(verify_checksums(root), 1)
            (root / "val.jsonl").write_bytes(b"{} \n")
            with self.assertRaisesRegex(ValueError, "val.jsonl"):
                verify_checksums(root)
            (root / "CHECKSUMS.txt").write_text(f"{good}  missing.jsonl\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing.jsonl"):
                verify_checksums(root)


if __name__ == "__main__":
    unittest.main()
