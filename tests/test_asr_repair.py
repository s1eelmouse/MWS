"""Noise-tolerant recovery: repairs ASR damage, leaves real words and inflections alone."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_repair import repair_call, repair_text, snap_values, word_matches
from fact_recovery import recover_facts
from schema import Facts, as_call, source_problems

TARIFFS = ["Пятёрочка", "Мини", "Простой", "Орбита", "Безлимит", "Гелиос+", "Гелиос"]


def call(*turns):
    return as_call({"call_id": "synthetic:asr", "started_at": "2026-07-08T12:00:00",
                    "turns": [{"idx": i, "speaker": s, "text": t} for i, (s, t) in enumerate(turns)]})


class RepairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.catalog = str(Path(cls.directory.name) / "catalog.json")
        Path(cls.catalog).write_text(json.dumps({"tariffs": TARIFFS}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_typical_damage_is_repaired(self):
        for damaged, target in (("сийчас", "сейчас"), ("заяфка", "заявка"), ("рабате", "работе"),
                                ("выизд", "выезд"), ("выполнина", "выполнена"), ("рабте", "работе"),
                                ("прията", "принята")):
            self.assertTrue(word_matches(damaged, target), damaged)
        self.assertEqual(repair_text("заявка вработе, уменя всё", (), tariff_context=False),
                         "заявка в работе, у меня всё")

    def test_real_words_and_inflections_are_not_changed(self):
        for word, target in (("придет", "принят"), ("висит", "выезд"), ("работы", "работе"),
                             ("работает", "работе"), ("принято", "принята")):
            self.assertFalse(word_matches(word, target), word)
        text = "мастер придет, всё висит, работы много"
        self.assertEqual(repair_text(text, (), tariff_context=False), text)

    def test_tariff_names_only_as_pure_confusion_and_only_in_tariff_context(self):
        repaired = repair_call(call(("client", "у меня тариф Пятерачка"), ("client", "обеда не было")),
                               self.catalog)
        self.assertEqual(repaired.turns[0].text, "у меня тариф пятерочка")
        self.assertEqual(repaired.turns[1].text, "обеда не было")  # not «Орбита», no context
        no_context = repair_call(call(("client", "пятерачка рядом с домом")), self.catalog)
        self.assertEqual(no_context.turns[0].text, "пятерачка рядом с домом")
        mine = repair_call(call(("client", "у меня тариф, у меня прасто вопрос")), self.catalog)
        self.assertEqual(mine.turns[0].text, "у меня тариф, у меня прасто вопрос")  # not «Мини», «Простой»

    def test_snap_values(self):
        facts = [{"attribute": "tariff", "value": "Пятерачка"}, {"attribute": "ticket_status", "value": "в рабате"},
                 {"attribute": "tariff", "value": "Гелиос+"}, {"attribute": "action", "value": "рабате"}]
        snapped, changes = snap_values(facts, self.catalog)
        self.assertEqual([f["value"] for f in snapped], ["Пятёрочка", "в работе", "Гелиос+", "рабате"])
        self.assertEqual(len(changes), 2)
        self.assertEqual(facts[0]["value"], "Пятерачка")  # input not mutated

    def test_recovery_finds_damaged_status_and_tariff_only_with_the_flag(self):
        c = call(("operator", "Заяфка 735 сийчас вработе."),
                 ("operator", "У вас тариф Пятерачка?"), ("client", "Да, Пятерачка, всё верно."))
        base, _ = recover_facts([], c, self.catalog, noise_robust=False)
        fixed, logs = recover_facts([], c, self.catalog, noise_robust=True)
        values = {(f["attribute"], f["value"]) for f in fixed}
        self.assertIn(("ticket_status", "в работе"), values)
        self.assertIn(("tariff", "Пятёрочка"), values)
        self.assertLess(len(base), len(fixed))
        self.assertEqual(source_problems(Facts.model_validate({"facts": fixed}), c), [])
        self.assertEqual(recover_facts(fixed, c, self.catalog, noise_robust=True)[0], fixed)  # idempotent
        self.assertTrue(all(log["policy"].endswith("+asr-repair-v1") for log in logs))

    def test_flag_off_is_the_unified_policy(self):
        from unified_recovery import recover_facts as unified
        c = call(("operator", "Заявка 735 сейчас в работе."))
        self.assertEqual(recover_facts([], c, self.catalog, noise_robust=False), unified([], c, self.catalog))

    def test_turns_keep_indices_and_speakers(self):
        c = call(("operator", "заяфка вработе"), ("client", "угу"))
        repaired = repair_call(c, self.catalog)
        self.assertEqual([(t.idx, t.speaker) for t in repaired.turns], [(t.idx, t.speaker) for t in c.turns])
        self.assertEqual(c.turns[0].text, "заяфка вработе")


if __name__ == "__main__":
    unittest.main()
