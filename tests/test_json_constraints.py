"""JSON-schema constrained decoding: compile once, mask every generate() step."""
import contextlib
import importlib.util
import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extractor import Extractor
from json_constraints import GrammarLogitsProcessor, compile_json_grammar
from test_case_schema import CALL

HAS_XGRAMMAR = importlib.util.find_spec("xgrammar") is not None
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


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


class JsonConstraintTest(unittest.TestCase):
    def test_compact_grammar_uses_the_facts_schema(self):
        calls = []

        class TokenizerInfo:
            @staticmethod
            def from_huggingface(tokenizer, vocab_size):
                calls.append(("tokenizer", tokenizer, vocab_size))
                return "info"

        class GrammarCompiler:
            def __init__(self, info, max_threads):
                calls.append(("compiler", info, max_threads))

            def compile_json_schema(self, schema, **options):
                calls.append(("schema", schema, options))
                return "compiled"

        fake = SimpleNamespace(TokenizerInfo=TokenizerInfo, GrammarCompiler=GrammarCompiler)
        tokenizer, config = object(), SimpleNamespace(vocab_size=1234)
        with patch.dict(sys.modules, {"xgrammar": fake}):
            result = compile_json_grammar(tokenizer, config, max_threads=3)
        self.assertEqual(result, "compiled")
        self.assertEqual(calls[0], ("tokenizer", tokenizer, 1234))
        self.assertEqual(calls[1], ("compiler", "info", 3))
        schema = json.loads(calls[2][1])
        self.assertIn("facts", schema["properties"])
        self.assertEqual(calls[2][2], {
            "any_whitespace": False, "separators": (",", ":"), "strict_mode": True})

    def test_grammar_processor_is_passed_to_generate(self):
        processors = []

        class Processor:
            def __init__(self, grammar, backend):
                processors.append((grammar, backend))

        captured = {}

        class Tokenizer:
            pad_token_id = 0

            def __call__(self, prompts, **kwargs):
                return ({"input_ids": [1, 3]} if isinstance(prompts, str)
                        else Inputs(input_ids=Tokens([[1, 3]] * len(prompts))))

            def batch_decode(self, tokens, **kwargs):
                return ['{"facts":[]}'] * len(tokens.rows)

        def generate(**kwargs):
            captured.update(kwargs)
            return Tokens([[1, 3, 6, 2]])

        backend = Extractor.__new__(Extractor)
        backend.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        backend.device, backend.with_schema = "cpu", False
        backend.tok, backend.lock = Tokenizer(), threading.Lock()
        backend.constrained_json = True
        backend.compiled_json_grammar = "compiled"
        backend.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=100),
            generation_config=SimpleNamespace(eos_token_id=2), generate=generate)
        details = []
        calls = [CALL, {**CALL, "call_id": "synthetic:call2"}]
        with patch("extractor.GrammarLogitsProcessor", Processor), \
                patch("extractor.prompt_for", return_value="prompt"), \
                patch("extractor.MAX_NEW_TOKENS", 3), \
                patch("extractor.PROMPT_LOOKUP_TOKENS", 10):
            answers = backend.extract(calls, 1, details)
        self.assertEqual([answer["facts"] for answer in answers], [[], []])
        self.assertEqual([processor[0] for processor in processors], ["compiled", "compiled"])
        self.assertEqual(captured["logits_processor"][0].__class__, Processor)
        self.assertNotIn("prompt_lookup_num_tokens", captured)

    def test_processor_masks_cpu_and_advances_on_the_next_token(self):
        applied, filled = [], []

        class Mask:
            def __init__(self, batch, vocab):
                self.batch, self.vocab = batch, vocab

            def to(self, device):
                return self

        class Matcher:
            def __init__(self, grammar):
                self.grammar = grammar
                self.accepted = []
                self.terminated = False

            def is_terminated(self):
                return self.terminated

            def accept_token(self, token):
                self.accepted.append(token)
                return True

            def fill_next_token_bitmask(self, bitmask, index):
                filled.append(index)

        fake = SimpleNamespace(
            GrammarMatcher=Matcher,
            allocate_token_bitmask=lambda batch, vocab: Mask(batch, vocab),
            apply_token_bitmask_inplace=lambda scores, mask, backend=None:
                applied.append((scores, mask, backend)))
        processor = GrammarLogitsProcessor(SimpleNamespace(
            tokenizer_info=SimpleNamespace(vocab_size=64)))

        class Token:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class Ids:
            def __init__(self, rows):
                self.rows = rows
                self.shape = (len(rows), len(rows[0]))

            def __getitem__(self, index):
                return [Token(value) for value in self.rows[index]]

        class Scores:
            device = SimpleNamespace(type="cpu")

            def to(self, device):
                return self

        with patch.dict(sys.modules, {"xgrammar": fake}):
            processor(Ids([[1, 3]]), Scores())
            processor(Ids([[1, 3, 6]]), Scores())
        self.assertEqual(filled, [0, 0])
        self.assertEqual([matcher.accepted for matcher in processor.matchers], [[6]])
        self.assertEqual([call[2] for call in applied], ["cpu", "cpu"])

    def test_windows_cuda_uses_the_portable_mask_kernel(self):
        applied = []

        class Mask:
            def to(self, device):
                return self

        class Matcher:
            def __init__(self, grammar):
                pass

            def is_terminated(self):
                return False

            def accept_token(self, token):
                return True

            def fill_next_token_bitmask(self, bitmask, index):
                pass

        fake = SimpleNamespace(
            GrammarMatcher=Matcher,
            allocate_token_bitmask=lambda batch, vocab: Mask(),
            apply_token_bitmask_inplace=lambda scores, mask, backend=None:
                applied.append(backend))

        class Token:
            def item(self):
                return 1

        class Ids:
            shape = (1, 2)

            def __getitem__(self, index):
                return [Token()]

        class Scores:
            device = SimpleNamespace(type="cuda")

        with patch.dict(sys.modules, {"xgrammar": fake}), \
                patch("json_constraints.platform.system", return_value="Windows"):
            processor = GrammarLogitsProcessor(SimpleNamespace(
                tokenizer_info=SimpleNamespace(vocab_size=64)))
            processor(Ids(), Scores())
        self.assertEqual(applied, ["torch_native"])

    @unittest.skipUnless(HAS_XGRAMMAR and HAS_TOKENIZERS and HAS_TRANSFORMERS,
                         "xgrammar/tokenizers/transformers are optional for CPU contract tests")
    def test_current_schema_compiles_and_accepts_the_training_target(self):
        import string
        import xgrammar as xgr
        from schema import to_answer
        from test_case_schema import FACT
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast

        alphabet = (string.printable
                    + 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
                    + 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ—«»')
        vocab = {"[UNK]": 0, "<pad>": 1, "</s>": 2}
        for char in alphabet:
            vocab.setdefault(char, len(vocab))
        raw = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw, unk_token="[UNK]", pad_token="<pad>", eos_token="</s>")
        compiled = compile_json_grammar(tokenizer, SimpleNamespace(vocab_size=len(vocab)))
        self.assertIsNotNone(compiled)
        matcher = xgr.GrammarMatcher(compiled)
        answer = to_answer([FACT])
        for position, char in enumerate(answer):
            self.assertIn(char, vocab, f"test vocabulary is missing {char!r} at {position}")
            self.assertTrue(matcher.accept_token(vocab[char]),
                            f"grammar rejected training target at {position}: {char!r}")
        self.assertTrue(matcher.accept_token(vocab["</s>"]))
        self.assertTrue(matcher.is_terminated())
        # v2.1 permits unnamed objects; real compiled grammar must accept them.
        unnamed = {**FACT, 'object': {**FACT['object'], 'name': ''}}
        empty_matcher = xgr.GrammarMatcher(compiled)
        for char in to_answer([unnamed]):
            self.assertTrue(empty_matcher.accept_token(vocab[char]))
        self.assertTrue(empty_matcher.accept_token(vocab['</s>']))
        self.assertTrue(empty_matcher.is_terminated())

        def accepted_until_rejection(text):
            candidate = xgr.GrammarMatcher(compiled)
            for position, char in enumerate(text):
                if not candidate.accept_token(vocab[char]):
                    return position
            return None

        wrapper = '{"facts":[{"fact":{"object":{"name":"x"}}}]}'
        self.assertIsNotNone(accepted_until_rejection(wrapper),
                             "grammar must reject the nested fact wrapper seen in raw failures")
        extra_colon = '{"facts":[{"object":{"name":"x"},"attribute":"action":"x"}]}'
        self.assertIsNotNone(accepted_until_rejection(extra_colon),
                             "grammar must reject the extra-colon malformation")


if __name__ == "__main__":
    unittest.main()
