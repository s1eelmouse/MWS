"""Case contract, label isolation, complete transcripts and explicit source errors."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from case_io import prediction_row, read_jsonl, validate_gold, write_jsonl
from schema import CallInput, Facts, as_call, build_messages, parse_answer, source_problems, to_answer

FACT = {
    "object": {"id": "synthetic:dev0", "type": "device", "name": "роутер"},
    "attribute": "action", "value": "перезагрузка роутера — не помогло", "polarity": 1,
    "valid_from": "2026-11-12T00:00:00", "valid_to": None,
    "source": {"turn_idx": 7, "span": None, "quote": "чистая цитата до ASR"},
    "speaker": "client", "confirmed": True,
}
CALL = {
    "call_id": "synthetic:call0000", "started_at": "2026-11-13T12:00:00",
    "turns": [{"idx": 7, "speaker": "client", "text": "Вчера перезагрузил роутер — не помогло."}],
    "facts": [FACT], "series": None, "noise_level": 0.25,
}


class CaseContractTest(unittest.TestCase):
    def test_organizer_gold_is_accepted(self):
        call, facts = validate_gold(CALL)
        self.assertEqual(call.call_id, CALL["call_id"])
        self.assertEqual(len(facts.facts), 1)

    def test_targets_preserve_null_end_and_drop_noisy_evidence(self):
        fact = json.loads(to_answer(CALL["facts"]))["facts"][0]
        self.assertIsNone(fact["valid_to"])
        self.assertNotIn("id", fact["object"])
        self.assertEqual(fact["source"], {"turn_idx": 7})
        self.assertNotIn("confirmed", fact)

    def test_minimal_prediction_contract(self):
        self.assertIsNotNone(parse_answer(to_answer([FACT])))
        self.assertIsNotNone(parse_answer('{"facts":[]}'))

    def test_legacy_format_and_missing_facts_are_rejected(self):
        for raw in ['{}', '{"product":"роутер","category":"сбой"}', '{"facts":{}}']:
            self.assertIsNone(parse_answer(raw))

    def test_gold_never_enters_prompt(self):
        messages = build_messages(CALL)
        payload = json.loads(messages[1]["content"])
        self.assertNotIn("facts", payload)
        self.assertNotIn("чистая цитата", messages[1]["content"])
        self.assertEqual(payload["started_at"], CALL["started_at"])
        self.assertEqual(payload["turns"][0]["idx"], 7)

    def test_long_transcript_is_preserved(self):
        row = copy.deepcopy(CALL)
        row["turns"][0]["text"] = "я" * 22000 + "КОНЕЦ"
        self.assertIn("КОНЕЦ", build_messages(row)[1]["content"])

    def test_character_guard_rejects_instead_of_truncating(self):
        with patch("schema.MAX_TEXT_CHARS", 5):
            with self.assertRaisesRegex(ValueError, "refusing to truncate"):
                build_messages(CALL)

    def test_unknown_attribute_is_rejected(self):
        fact = copy.deepcopy(FACT)
        fact["attribute"] = "urgency"
        with self.assertRaises(ValueError):
            to_answer([fact])

    def test_polarity_must_be_an_integer_sign(self):
        for value in (True, "1", 0, 2):
            fact = copy.deepcopy(FACT)
            fact["polarity"] = value
            with self.assertRaises(ValueError):
                to_answer([fact])

    def test_invalid_date_and_reversed_interval_are_rejected(self):
        for changes in [{"valid_from": "вчера"}, {"valid_to": "2026-11-11T00:00:00"}]:
            with self.assertRaises(ValueError):
                to_answer([{**copy.deepcopy(FACT), **changes}])

    def test_other_type_from_actual_gold_is_supported(self):
        fact = copy.deepcopy(FACT)
        fact["object"]["type"] = "other"
        self.assertIn('"other"', to_answer([fact]))

    def test_reasoning_and_markdown_are_not_treated_as_facts(self):
        raw = '<think>{"incorrect":true}</think>\n' + chr(96)*3 + 'json\n' + to_answer([FACT]) + '\n' + chr(96)*3
        self.assertEqual(len(parse_answer(raw).facts), 1)

    def test_duplicate_turn_indices_are_rejected(self):
        row = copy.deepcopy(CALL)
        row["turns"] *= 2
        with self.assertRaises(ValueError):
            as_call(row)

    def test_source_is_idx_not_list_position(self):
        facts = Facts.model_validate({"facts": [FACT]})
        self.assertEqual(source_problems(facts, as_call(CALL)), [])
        facts.facts[0].source.turn_idx = 0
        self.assertTrue(source_problems(facts, as_call(CALL)))

    def test_speaker_mismatch_is_rejected(self):
        fact = copy.deepcopy(FACT)
        fact["speaker"] = "operator"
        with self.assertRaises(ValueError):
            validate_gold({**CALL, "facts": [fact]})

    def test_out_of_bounds_span_is_rejected(self):
        fact = copy.deepcopy(FACT)
        fact["source"]["span"] = [0, 10000]
        with self.assertRaises(ValueError):
            validate_gold({**CALL, "facts": [fact]})

    def test_nonliteral_gold_quote_is_not_a_blocking_error(self):
        validate_gold(CALL)

    def test_http_contract_does_not_accept_gold_labels(self):
        with self.assertRaises(ValueError):
            CallInput.model_validate(CALL)

    def test_prediction_keeps_call_id_and_failure_status(self):
        output = prediction_row(CALL, {"error": "invalid_json"})
        self.assertEqual(output, {"call_id": CALL["call_id"], "facts": [], "error": "invalid_json"})

    def test_duplicate_call_ids_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            write_jsonl(path, [CALL, CALL])
            with self.assertRaisesRegex(ValueError, "duplicate call_id"):
                read_jsonl(path)

    def test_outputs_are_not_overwritten_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            write_jsonl(path, [CALL])
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_jsonl(path, [])
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
