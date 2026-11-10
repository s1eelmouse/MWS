"""Проверка схемы и разбора ответа модели. Запуск из корня: python -m unittest discover tests"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import MAX_TEXT_CHARS  # noqa: E402
from schema import SYSTEM, Facts, build_messages, parse_answer, to_answer  # noqa: E402

FULL = {
    "product": "личный кабинет", "category": "сбой", "problem": "не открывается",
    "request": None, "ids": ["ДГ-104552"], "error_code": "502", "urgency": "высокая",
}


class AnswerTest(unittest.TestCase):
    def test_answer_is_compact_and_skips_empty_fields(self):
        answer = to_answer(FULL)
        self.assertNotIn(" :", answer)
        self.assertNotIn(", ", answer.replace("не открывается", ""))
        self.assertNotIn("request", answer)
        self.assertEqual(list(json.loads(answer)), ["product", "category", "problem", "ids", "error_code", "urgency"])

    def test_answer_keeps_cyrillic_readable(self):
        self.assertIn("личный кабинет", to_answer(FULL))

    def test_minimal_answer_has_only_required_fields(self):
        answer = to_answer({"category": "другое", "urgency": "низкая"})
        self.assertEqual(answer, '{"category":"другое","urgency":"низкая"}')

    def test_round_trip(self):
        self.assertEqual(parse_answer(to_answer(FULL)).model_dump(), FULL)

    def test_unknown_category_is_rejected(self):
        with self.assertRaises(ValueError):
            to_answer({"category": "прочее", "urgency": "низкая"})


class ParseTest(unittest.TestCase):
    def test_text_around_json_is_ignored(self):
        raw = 'Вот ответ:\n```json\n{"category":"вопрос","urgency":"низкая"}\n```'
        self.assertEqual(parse_answer(raw).category, "вопрос")

    def test_garbage_gives_none(self):
        for raw in ("", "нет данных", "{", '{"category":"вопрос"', '{"category":"нет","urgency":"низкая"}'):
            self.assertIsNone(parse_answer(raw))

    def test_missing_required_field_gives_none(self):
        self.assertIsNone(parse_answer('{"category":"вопрос"}'))

    def test_numbers_become_strings(self):
        facts = parse_answer('{"category":"сбой","urgency":"средняя","error_code":502,"ids":[4410027]}')
        self.assertEqual((facts.error_code, facts.ids), ("502", ["4410027"]))

    def test_empty_strings_become_none(self):
        facts = parse_answer('{"category":"сбой","urgency":"средняя","product":"  ","ids":["", "A-1"]}')
        self.assertIsNone(facts.product)
        self.assertEqual(facts.ids, ["A-1"])


class MessagesTest(unittest.TestCase):
    def test_short_prompt_by_default(self):
        messages = build_messages("текст")
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM})
        self.assertEqual(messages[1], {"role": "user", "content": "текст"})

    def test_schema_prompt_lists_fields(self):
        system = build_messages("текст", with_schema=True)[0]["content"]
        for name in Facts.model_fields:
            self.assertIn(name, system)

    def test_long_text_is_cut(self):
        messages = build_messages("я" * (MAX_TEXT_CHARS + 500))
        self.assertEqual(len(messages[1]["content"]), MAX_TEXT_CHARS)


if __name__ == "__main__":
    unittest.main()
