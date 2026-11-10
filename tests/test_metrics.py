"""Проверка формулы F1. Запуск из корня: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import items, score  # noqa: E402


class ItemsTest(unittest.TestCase):
    def test_empty_values_give_empty_set(self):
        for value in (None, "", "   ", [], [""], [None]):
            self.assertEqual(items(value), set())

    def test_case_and_spaces_are_ignored(self):
        self.assertEqual(items("  Личный   Кабинет "), {"личный кабинет"})

    def test_list_becomes_set_of_items(self):
        self.assertEqual(items(["ДГ-1", "дг-1", " INC-2 "]), {"дг-1", "inc-2"})

    def test_number_is_compared_as_text(self):
        self.assertEqual(items(502), {"502"})


class ScoreTest(unittest.TestCase):
    def test_exact_match(self):
        gold = {"product": "VPN", "ids": ["A-1", "B-2"], "urgency": "высокая"}
        micro = score([(gold, dict(gold))])["micro"]
        self.assertEqual((micro["tp"], micro["fp"], micro["fn"]), (4, 0, 0))
        self.assertEqual(micro["f1"], 1.0)

    def test_wrong_value_is_both_extra_and_missed(self):
        micro = score([({"urgency": "высокая"}, {"urgency": "низкая"})])["micro"]
        self.assertEqual((micro["tp"], micro["fp"], micro["fn"]), (0, 1, 1))
        self.assertEqual(micro["f1"], 0.0)

    def test_missing_and_extra_fields(self):
        gold = {"product": "VPN", "error_code": None}
        pred = {"product": None, "error_code": "502"}
        fields = score([(gold, pred)])["fields"]
        self.assertEqual((fields["product"]["fp"], fields["product"]["fn"]), (0, 1))
        self.assertEqual((fields["error_code"]["fp"], fields["error_code"]["fn"]), (1, 0))

    def test_field_absent_in_gold_but_predicted_counts_as_extra(self):
        micro = score([({"urgency": "низкая"}, {"urgency": "низкая", "product": "почта"})])["micro"]
        self.assertEqual((micro["tp"], micro["fp"], micro["fn"]), (1, 1, 0))

    def test_list_is_scored_per_item(self):
        fields = score([({"ids": ["A-1", "B-2"]}, {"ids": ["b-2", "C-3"]})])["fields"]
        ids = fields["ids"]
        self.assertEqual((ids["tp"], ids["fp"], ids["fn"]), (1, 1, 1))
        self.assertAlmostEqual(ids["f1"], 0.5)

    def test_empty_answer_misses_everything(self):
        gold = {"product": "VPN", "ids": ["A-1"], "urgency": "высокая"}
        micro = score([(gold, {})])["micro"]
        self.assertEqual((micro["tp"], micro["fp"], micro["fn"]), (0, 0, 3))

    def test_both_empty_counts_nothing(self):
        micro = score([({"product": None, "ids": []}, {"product": None, "ids": []})])["micro"]
        self.assertEqual((micro["tp"], micro["fp"], micro["fn"]), (0, 0, 0))

    def test_micro_average_over_several_rows(self):
        pairs = [
            ({"urgency": "высокая", "product": "VPN"}, {"urgency": "высокая", "product": "VPN"}),
            ({"urgency": "низкая", "product": "почта"}, {"urgency": "средняя"}),
        ]
        micro = score(pairs)["micro"]
        self.assertEqual((micro["tp"], micro["fp"], micro["fn"]), (2, 1, 2))
        self.assertAlmostEqual(micro["precision"], 2 / 3)
        self.assertAlmostEqual(micro["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
