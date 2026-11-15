"""Prediction normalization: diarization speaker and date sanity, without weakening gold."""
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prediction import parse_prediction, salvage_sources, sanitize_interval
from schema import Facts, as_call, source_problems
from test_case_schema import CALL, FACT


class PredictionNormalizationTest(unittest.TestCase):
    def parse(self, *facts):
        return parse_prediction(json.dumps({"facts": list(facts)}), as_call(CALL))

    def test_speaker_comes_from_the_source_turn(self):
        fact = copy.deepcopy(FACT)
        fact["speaker"] = "operator"  # turn 7 is the client
        facts, error, summary = self.parse(fact)
        self.assertIsNone(error)
        self.assertEqual(facts.facts[0].speaker, "client")
        self.assertEqual(summary, {"speaker_from_turn": 1})

    def test_missing_speaker_is_filled_from_diarization(self):
        fact = copy.deepcopy(FACT)
        fact.pop("speaker")
        facts, error, summary = self.parse(fact)
        self.assertIsNone(error)
        self.assertEqual(facts.facts[0].speaker, "client")
        self.assertEqual(summary, {"speaker_from_turn": 1})

    def test_invalid_calendar_date_is_dropped_not_the_call(self):
        fact = copy.deepcopy(FACT)
        fact["valid_from"] = "2026-02-29T00:00:00"  # 2026 is not a leap year
        facts, error, summary = self.parse(fact)
        self.assertIsNone(error)
        self.assertIsNone(facts.facts[0].valid_from)
        self.assertEqual(summary, {"invalid_dates_dropped": 1})

    def test_reversed_interval_is_opened(self):
        fact = copy.deepcopy(FACT)
        fact["valid_to"] = "2026-11-11T00:00:00"
        facts, error, summary = self.parse(fact)
        self.assertIsNone(error)
        self.assertIsNone(facts.facts[0].valid_to)
        self.assertEqual(summary, {"reversed_intervals_opened": 1})

    def test_exact_duplicates_are_removed(self):
        facts, error, summary = self.parse(FACT, FACT)
        self.assertIsNone(error)
        self.assertEqual(len(facts.facts), 1)
        self.assertEqual(summary, {"exact_duplicates_removed": 1})

    def test_duplicates_are_removed_after_speaker_and_date_normalization(self):
        wrong_speaker = copy.deepcopy(FACT)
        wrong_speaker["speaker"] = "operator"
        facts, error, summary = self.parse(FACT, wrong_speaker)
        self.assertIsNone(error)
        self.assertEqual(len(facts.facts), 1)
        self.assertEqual(summary, {"speaker_from_turn": 1,
                                   "exact_duplicates_removed": 1})

    def test_different_values_are_not_deduplicated(self):
        other = copy.deepcopy(FACT)
        other["value"] = "другое действие"
        facts, error, summary = self.parse(FACT, other)
        self.assertIsNone(error)
        self.assertEqual(len(facts.facts), 2)
        self.assertIsNone(summary)

    def test_valid_fact_is_unchanged(self):
        facts, error, summary = self.parse(FACT)
        self.assertIsNone(error)
        self.assertIsNone(summary)
        self.assertEqual(facts.facts[0].valid_from, FACT["valid_from"])
        self.assertIsNone(facts.facts[0].valid_to)

    def test_unknown_turn_stays_an_explicit_source_problem(self):
        fact = copy.deepcopy(FACT)
        fact["source"]["turn_idx"] = 999
        facts, error, _ = self.parse(fact)
        self.assertIsNone(error)
        self.assertTrue(source_problems(facts, as_call(CALL)))

    def test_partial_invalid_sources_drop_only_the_bad_fact(self):
        call = as_call(CALL)
        good = copy.deepcopy(FACT)
        bad = copy.deepcopy(FACT)
        bad["source"]["turn_idx"] = 999
        bad["value"] = "другой факт"
        salvaged, dropped = salvage_sources(Facts.model_validate({"facts": [good, bad]}), call)
        self.assertEqual(len(salvaged.facts), 1)
        self.assertEqual(salvaged.facts[0].value, good["value"])
        self.assertEqual(dropped, 1)
        self.assertIsNone(salvage_sources(Facts.model_validate({"facts": [bad]}), call)[0])

    def test_missing_required_field_is_still_invalid_schema(self):
        facts, error, summary = parse_prediction('{"facts":[{}]}', as_call(CALL))
        self.assertIsNone(facts)
        self.assertEqual(error["error"], "invalid_schema")
        self.assertIsNone(summary)

    def test_json_syntax_error_is_still_explicit(self):
        facts, error, summary = parse_prediction('{"facts":[broken]}', as_call(CALL))
        self.assertIsNone(facts)
        self.assertEqual(error["reason"], "json_syntax")
        self.assertIsNone(summary)

    def test_sanitize_interval_handles_timezone_mismatch(self):
        start, end, changes = sanitize_interval("2026-11-12T00:00:00Z",
                                                "2026-11-13T00:00:00")
        self.assertEqual(start, "2026-11-12T00:00:00Z")
        self.assertIsNone(end)
        self.assertEqual(changes, {"date_timezone_mismatch": 1})


if __name__ == "__main__":
    unittest.main()
