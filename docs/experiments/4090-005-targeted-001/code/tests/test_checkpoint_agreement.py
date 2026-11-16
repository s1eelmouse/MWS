"""Agreement of two checkpoints: buckets, measurement against gold, pseudo-labels without gold."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from checkpoint_agreement import agreement_buckets, analyse
from scoring import Scorer


def fact(attribute, value, turn=0, valid_from="2026-11-13T12:00:00", valid_to=None, name="роутер"):
    return {"object": {"name": name, "type": "device"}, "attribute": attribute, "value": value, "polarity": 1,
            "valid_from": valid_from, "valid_to": valid_to, "source": {"turn_idx": turn}}


class AgreementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        catalog = Path(cls.directory.name) / "catalog.json"
        catalog.write_text('{"tariffs": ["Смарт"], "actions": ["перезагрузка роутера"]}', encoding="utf-8")
        cls.scorer = Scorer("2.1", catalog)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_buckets(self):
        a = [fact("action", "перезагрузка роутера"), fact("tariff", "Смарт", turn=1),
             fact("service_state", "нет гудков", valid_from="2026-11-10T00:00:00"), fact("action", "проверка кабеля")]
        b = [fact("action", "перезагрузка роутера"), fact("tariff", "Смарт", turn=2),
             fact("service_state", "нет гудков", valid_from="2026-11-11T00:00:00"), fact("agreement", "выезд специалиста")]
        self.assertEqual(agreement_buckets(a, b, self.scorer),
                         {"strict": [0], "dates_only": [1], "value_only": [2], "a_only": [3], "b_only": [3]})

    def test_measurement_and_pseudo_labels(self):
        right = fact("action", "перезагрузка роутера")
        wrong = fact("action", "замена роутера")
        gold = [{"call_id": "u1:call0000", "facts": [right, fact("tariff", "Смарт")]},
                {"call_id": "u2:call0000", "facts": [right]}]
        a = [{"call_id": "u1:call0000", "facts": [copy.deepcopy(right), copy.deepcopy(wrong)]},
             {"call_id": "u2:call0000", "facts": [copy.deepcopy(right)]},
             {"call_id": "u3:call0000", "facts": [copy.deepcopy(right)]}]
        b = [{"call_id": "u1:call0000", "facts": [copy.deepcopy(right)]},
             {"call_id": "u2:call0000", "facts": [copy.deepcopy(right)]}]
        report, pseudo = analyse(a, b, self.scorer, gold)
        self.assertEqual((report["calls"], report["calls_missing_in_b"]), (2, 1))
        self.assertEqual(report["buckets"]["strict"], {"facts": 2, "correct": 2, "precision": 1.0})
        self.assertEqual(report["buckets"]["a_only"], {"facts": 1, "correct": 0, "precision": 0.0})
        self.assertEqual(report["model_a"], {"precision": 0.6667, "recall": 0.6667})
        self.assertEqual(report["pseudo_labels"]["precision"], 1.0)
        # u1 agrees on one fact of two: a partial label, not written as a training row
        self.assertEqual([row["call_id"] for row in pseudo], ["u2:call0000"])
        self.assertEqual(report["pseudo_label_calls"]["calls"], 1)
        self.assertFalse(report["gold_used_for_pseudo_labels"])

    def test_gold_changes_only_the_measurement(self):
        a = [{"call_id": "u1:call0000", "facts": [fact("action", "перезагрузка роутера")]}]
        b = copy.deepcopy(a)
        with_gold, labels = analyse(a, b, self.scorer, [{"call_id": "u1:call0000", "facts": []}])
        without_gold, same_labels = analyse(a, b, self.scorer)
        self.assertEqual(labels, same_labels)
        self.assertNotIn("precision", without_gold["buckets"]["strict"])
        self.assertEqual(with_gold["buckets"]["strict"]["precision"], 0.0)


if __name__ == "__main__":
    unittest.main()
