"""Cascade simulation: routing signals read no gold; mixes are scored by the official scorer."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cascade_simulation import as_markdown, call_signals, simulate
from catalog_hints import build_hints
from scoring import Scorer

CATALOG = {"tariffs": ["Смарт"], "symptoms": ["нет гудков"], "actions": ["перезагрузка роутера"]}


def fact(attribute, value, turn=0):
    return {"object": {"name": "клиент", "type": "client"}, "attribute": attribute, "value": value, "polarity": 1,
            "valid_from": "2026-11-13T12:00:00", "valid_to": None, "source": {"turn_idx": turn}}


def call(call_id, texts, facts):
    return {"call_id": call_id, "started_at": "2026-11-13T12:00:00",
            "turns": [{"idx": i, "speaker": "client", "text": t} for i, t in enumerate(texts)], "facts": facts}


class CascadeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        path = Path(cls.directory.name) / "catalog.json"
        path.write_text('{"tariffs": ["Смарт"], "symptoms": ["нет гудков"], "actions": ["перезагрузка роутера"]}',
                        encoding="utf-8")
        cls.scorer = Scorer("2.1", path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_signals_use_transcript_and_model_output_only(self):
        hints = build_hints(CATALOG)
        spoken = call("u1:call0000", ["у меня тариф Смарт, и гудков нет"], [fact("tariff", "Смарт")])
        args = (CATALOG, hints, self.scorer, 0.5)
        self.assertEqual(call_signals(spoken, {"facts": []}, None, *args),
                         {"tariff_named_no_fact", "symptom_named_no_state"})
        answered = {"facts": [fact("tariff", "Смарт"), fact("service_state", "нет гудков")]}
        self.assertEqual(call_signals(spoken, answered, None, *args), set())
        self.assertEqual(call_signals({**spoken, "facts": []}, {"facts": []}, None, *args),
                         call_signals(spoken, {"facts": []}, None, *args))  # gold facts are not read
        other = {"facts": [fact("action", "перезагрузка роутера")]}
        self.assertEqual(call_signals(spoken, answered, other, *args), {"checkpoints_disagree"})

    def test_mix_is_scored_and_bounds_are_reported(self):
        gold = [call("u1:call0000", ["у меня тариф Смарт"], [fact("tariff", "Смарт")]),
                call("u2:call0000", ["перезагрузил роутер"], [fact("action", "перезагрузка роутера")])]
        for row in gold:  # organizer gold carries object ids; predictions do not
            for item in row["facts"]:
                item["object"] = {**item["object"], "id": row["call_id"] + ":client"}
        small = [{"call_id": "u1:call0000", "facts": []},
                 {"call_id": "u2:call0000", "facts": [fact("action", "перезагрузка роутера")]}]
        large = [{"call_id": "u1:call0000", "facts": [fact("tariff", "Смарт")]},
                 {"call_id": "u2:call0000", "facts": []}]
        report = simulate(gold, small, large, CATALOG, self.scorer)
        rows = report["policies"]
        self.assertEqual(rows["small_only"]["fact_recall"], 0.5)
        self.assertEqual(rows["large_only"]["fact_recall"], 0.5)
        self.assertEqual((rows["tariff_named_no_fact"]["calls_to_large"], rows["tariff_named_no_fact"]["fact_recall"]),
                         (1, 1.0))
        self.assertEqual((rows["oracle"]["uses_gold"], rows["oracle"]["fact_recall"]), (True, 1.0))
        self.assertFalse(any(row["uses_gold"] for name, row in rows.items() if name != "oracle"))
        self.assertNotIn("checkpoints_disagree", rows)
        self.assertIn("| `tariff_named_no_fact` | 50.0% |", as_markdown(report))


if __name__ == "__main__":
    unittest.main()
