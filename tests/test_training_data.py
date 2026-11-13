import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training_data import check_output_directory, encode_row, ensure_training_splits, save_manifest
from test_case_schema import CALL


class Tokenizer:
    eos_token = "<END>"

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False)

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}


class EncodingTest(unittest.TestCase):
    def test_only_target_tokens_have_loss(self):
        encoded = encode_row(Tokenizer(), CALL, 50000)
        count = encoded["labels"].count(-100)
        self.assertGreater(count, 0)
        self.assertEqual(encoded["input_ids"][count:], encoded["labels"][count:])
        self.assertIn(ord("—"), encoded["labels"][count:])

    def test_overflow_fails_instead_of_dropping_labels(self):
        with self.assertRaisesRegex(ValueError, "NOT truncated"):
            encode_row(Tokenizer(), CALL, 20)

    def test_input_budget_is_also_checked(self):
        with patch("training_data.MAX_INPUT_TOKENS", 10):
            with self.assertRaises(ValueError):
                encode_row(Tokenizer(), CALL, 50000)

    def test_target_cannot_exceed_generation_budget(self):
        with patch('training_data.MAX_NEW_TOKENS', 10):
            with self.assertRaises(ValueError):
                encode_row(Tokenizer(), CALL, 50000)

    def test_account_leakage_is_rejected(self):
        other = {**copy.deepcopy(CALL), "call_id": "synthetic:call0001"}
        with self.assertRaisesRegex(ValueError, "same account"):
            ensure_training_splits([CALL], [other])

    def test_existing_results_need_explicit_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "weights").write_text("preserve")
            with self.assertRaises(ValueError):
                check_output_directory(temp)

    def test_resume_must_be_a_real_checkpoint_of_this_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoints" / "checkpoint-10"
            checkpoint.mkdir(parents=True)
            (root / 'experiment.json').write_text('{}')
            with self.assertRaises(ValueError):
                check_output_directory(root, checkpoint)
            (checkpoint / "trainer_state.json").write_text("{}")
            self.assertEqual(check_output_directory(root, checkpoint), root)

    def test_manifest_mismatch_cannot_overwrite_previous_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "experiment.json"
            save_manifest(path, {"model": "A"})
            save_manifest(path, {"model": "A"})
            with self.assertRaises(ValueError):
                save_manifest(path, {"model": "B"})
            self.assertEqual(json.loads(path.read_text()), {"model": "A"})


if __name__ == "__main__":
    unittest.main()
