"""Targeted retry only for calls that hit max_new_tokens."""
import contextlib
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extractor import Extractor
from test_case_schema import CALL


class Tokens:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key):
        return Tokens([row[key[1]] for row in self.rows])

    def tolist(self):
        return self.rows


class Inputs(dict):
    def to(self, device):
        return self


def make_backend(rows, raws):
    class Tokenizer:
        pad_token_id = 0

        def __call__(self, prompts, **kwargs):
            return ({"input_ids": [1, 3]} if isinstance(prompts, str)
                    else Inputs(input_ids=Tokens([[1, 3]] * len(prompts))))

        def batch_decode(self, tokens, **kwargs):
            return [next(raws) for _ in tokens.rows]

    backend = Extractor.__new__(Extractor)
    backend.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    backend.device, backend.with_schema = "cpu", False
    backend.tok, backend.lock = Tokenizer(), threading.Lock()
    backend.constrained_json = False
    backend.model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=100),
        generation_config=SimpleNamespace(eos_token_id=2),
        generate=Mock(side_effect=[Tokens([row]) for row in rows]))
    return backend


class RetryOnTokenLimitTest(unittest.TestCase):
    def test_retry_recovers_a_truncated_answer(self):
        backend = make_backend(
            rows=[[1, 3, 6, 7, 8], [1, 3, 6, 2]],
            raws=iter(['{"facts":[broken]}', '{"facts":[]}']))
        details = []
        with patch("extractor.prompt_for", return_value="prompt"), \
                patch("extractor.MAX_NEW_TOKENS", 3), \
                patch("extractor.PROMPT_LOOKUP_TOKENS", 0), \
                patch("extractor.RETRY_ON_TOKEN_LIMIT", True), \
                patch("extractor.RETRY_REPETITION_PENALTY", 1.2):
            answers = backend.extract([CALL], 1, details)
        self.assertEqual(answers[0], {"facts": []})
        self.assertIsNone(details[0]["retry"]["error"])
        self.assertEqual(details[0]["retry"]["initial_output_tokens"], 3)
        self.assertEqual(details[0]["retry"]["output_tokens"], 2)
        self.assertEqual(details[0]["retry"]["repetition_penalty"], 1.2)
        self.assertEqual(details[0]["output_tokens"], 2)
        self.assertFalse(details[0]["generation_limit_reached"])
        calls = backend.model.generate.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertNotIn("repetition_penalty", calls[0].kwargs)
        self.assertEqual(calls[1].kwargs["repetition_penalty"], 1.2)

    def test_failed_retry_keeps_the_explicit_limit_error(self):
        backend = make_backend(
            rows=[[1, 3, 6, 7, 8], [1, 3, 6, 7, 8]],
            raws=iter(['{"facts":[broken]}', '{"facts":[broken]}']))
        details = []
        with patch("extractor.prompt_for", return_value="prompt"), \
                patch("extractor.MAX_NEW_TOKENS", 3), \
                patch("extractor.PROMPT_LOOKUP_TOKENS", 0), \
                patch("extractor.RETRY_ON_TOKEN_LIMIT", True), \
                patch("extractor.RETRY_REPETITION_PENALTY", 1.2):
            answers = backend.extract([CALL], 1, details)
        self.assertEqual(answers[0], {"error": "output_token_limit"})
        self.assertEqual(details[0]["retry"]["error"], "output_token_limit")
        self.assertEqual(details[0]["diagnostic"]["retry_from"], "output_token_limit")
        self.assertEqual(details[0]["diagnostic"]["retry_repetition_penalty"], 1.2)


if __name__ == "__main__":
    unittest.main()
