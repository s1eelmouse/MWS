"""Training variants keep gold facts intact; catalog hints read the transcript only."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from case_io import validate_gold
from catalog_hints import build_hints, coverage, find_hints
from make_train_variants import (build_focus, build_variants, evidence_safe_noisy_copies,
                                 focus_reasons, resampling_reasons, _evidence_kept)

CATALOG = {"tariffs": ["Стандарт", "Смарт"], "symptoms": ["нет гудков", "слабый сигнал вай-фай"],
           "statuses": ["принята", "в работе"], "actions": ["перезагрузка роутера"],
           "agreements": ["выезд специалиста"], "services": ["домашний интернет"], "devices": ["роутер"]}


def fact(attribute, value, turn=0, valid_to=None):
    return {"object": {"name": "роутер", "type": "device"}, "attribute": attribute, "value": value,
            "polarity": 1, "valid_from": "2026-11-01T00:00:00" if valid_to else "2026-11-13T12:00:00",
            "valid_to": valid_to,
            "source": {"turn_idx": turn, "quote": "текст"}}


def call(call_id, texts, facts):
    return {"call_id": call_id, "started_at": "2026-11-13T12:00:00",
            "turns": [{"idx": i, "speaker": "client", "text": t} for i, t in enumerate(texts)], "facts": facts}


PLAIN = call("u1:call0000", ["здравствуйте, вопрос по счёту за прошлый месяц"], [fact("action", "проверка счёта")])
BUSY = call("u2:call0000",
            ["вчера перезагрузил роутер и проверил кабель, не помогло, гудков нет совсем",
             "может, тариф Смарт подключить вместо моего"],
            [fact("action", "перезагрузка роутера"), fact("action", "проверка кабеля"),
             fact("service_state", "работает нормально", valid_to="2026-11-12T00:00:00"),
             fact("service_state", "нет гудков")])


class VariantTest(unittest.TestCase):
    def test_reasons_target_measured_weaknesses(self):
        self.assertEqual(resampling_reasons(PLAIN, CATALOG["tariffs"]), [])
        self.assertEqual(set(resampling_reasons(BUSY, CATALOG["tariffs"])),
                         {"state_change", "many_actions", "tariff_named_but_not_a_fact"})
        connected = copy.deepcopy(BUSY)
        connected["facts"].append(fact("tariff", "Смарт", turn=1))
        self.assertNotIn("tariff_named_but_not_a_fact", resampling_reasons(connected, CATALOG["tariffs"]))

    def test_variants_keep_facts_accounts_and_unique_ids(self):
        rows = [PLAIN, BUSY]
        before = copy.deepcopy(rows)
        variants, _ = build_variants(rows, CATALOG["tariffs"], seed=7)
        self.assertEqual(rows, before)
        self.assertEqual({name: len(items) for name, items in variants.items()},
                         {"train_noise.jsonl": 4, "train_resampled.jsonl": 4, "train_noise_resampled.jsonl": 6})
        for items in variants.values():
            ids = [item["call_id"] for item in items]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual({cid.split(":")[0] for cid in ids}, {"u1", "u2"})
            for item in items:
                validate_gold(item)
                original = next(r for r in rows if item["call_id"].startswith(r["call_id"]))
                strip = lambda fs: [{k: v for k, v in f.items() if k != "source"} for f in fs]
                self.assertEqual(strip(item["facts"]), strip(original["facts"]))
                self.assertEqual([f["source"]["turn_idx"] for f in item["facts"]],
                                 [f["source"]["turn_idx"] for f in original["facts"]])
        noisy = variants["train_noise.jsonl"][3]
        self.assertNotEqual([t["text"] for t in noisy["turns"]], [t["text"] for t in BUSY["turns"]])
        self.assertEqual(variants, build_variants(rows, CATALOG["tariffs"], seed=7)[0])  # reproducible


class FocusTest(unittest.TestCase):
    def test_focus_reasons_are_about_current_states(self):
        self.assertEqual(focus_reasons(PLAIN, CATALOG["tariffs"], CATALOG["symptoms"]), [])
        found = set(focus_reasons(BUSY, CATALOG["tariffs"], CATALOG["symptoms"]))
        self.assertIn("state_change", found)
        self.assertNotIn("many_actions", found)  # actions are not scored by the two targets
        chain = call("u5:call0000", ["заявка принята, потом в работе"],
                     [fact("ticket_status", "принята"), fact("ticket_status", "в работе")])
        self.assertIn("ticket_status_chain", focus_reasons(chain, [], []))
        free = call("u6:call0000", ["интернет отваливается каждый вечер"],
                    [fact("service_state", "отваливается каждый вечер")])
        self.assertEqual(focus_reasons(free, [], CATALOG["symptoms"]), ["free_form_current_service_state"])
        fixed = call("u7:call0000", ["у меня тариф не Стандарт, а Смарт"], [fact("tariff", "Смарт")])
        self.assertIn("tariff_correction", focus_reasons(fixed, CATALOG["tariffs"], []))

    def test_noise_never_removes_the_basis_of_a_fact(self):
        self.assertTrue(_evidence_kept("гудков нет совсем", "гутков нед савсем", "нет гудков"))
        self.assertTrue(_evidence_kept("гудков нет совсем", "гудковнет", "нет гудков"))   # glued words
        self.assertFalse(_evidence_kept("гудков нет совсем", "совсем —", "нет гудков"))
        self.assertTrue(_evidence_kept("телефон молчит", "—", "нет гудков"))  # paraphrase in gold
        rows = [call(f"u{i}:call0000", ["вчера перезагрузил роутер, гудков нет совсем уже неделю"],
                     [fact("service_state", "гудков нет")]) for i in range(40)]
        copies, _stats = evidence_safe_noisy_copies(rows, 3, "f")
        for original, noisy in zip(rows, copies):
            self.assertTrue(_evidence_kept(original["turns"][0]["text"], noisy["turns"][0]["text"], "гудков нет"))
            self.assertEqual(noisy["facts"][0]["value"], "гудков нет")

    def test_focus_set_fits_the_budget_and_keeps_gold(self):
        rows = [PLAIN, BUSY] + [call(f"u{i}:call0000", ["заявка принята, потом в работе"],
                                     [fact("ticket_status", "принята"), fact("ticket_status", "в работе")])
                                for i in range(10, 16)]
        selected, info = build_focus(rows, CATALOG["tariffs"], CATALOG["symptoms"], seed=7, target=9)
        self.assertEqual((len(selected), info["calls"], info["problem_calls"]), (9, 9, 7))
        ids = [row["call_id"] for row in selected]
        self.assertEqual(len(ids), len(set(ids)))
        source = {row["call_id"]: row for row in rows}
        for row in selected:
            validate_gold(row)
            original = source[row["call_id"].split("~")[0]]
            self.assertEqual([(f["attribute"], f["value"], f["valid_from"], f["valid_to"], f["source"]["turn_idx"])
                              for f in row["facts"]],
                             [(f["attribute"], f["value"], f["valid_from"], f["valid_to"], f["source"]["turn_idx"])
                              for f in original["facts"]])
        self.assertEqual(selected, build_focus(rows, CATALOG["tariffs"], CATALOG["symptoms"], seed=7, target=9)[0])


class TrainOptionTest(unittest.TestCase):
    def test_train_builds_the_variant_in_memory(self):
        import contextlib
        import io
        import json
        import tempfile
        from unittest.mock import patch
        import train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write = lambda name, rows: (root / name).write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            write("train.jsonl", [PLAIN, BUSY])
            write("val.jsonl", [call("u9:call0000", ["спасибо"], [])])
            (root / "catalog.json").write_text(json.dumps(CATALOG, ensure_ascii=False), encoding="utf-8")
            for variant, expected in (("plain", 2), ("noise", 4), ("resampled", 4), ("noise_resampled", 6),
                                      ("focus", 2)):
                out = io.StringIO()
                argv = ["train.py", "--check-data", "--train", str(root / "train.jsonl"),
                        "--val", str(root / "val.jsonl"), "--catalog", str(root / "catalog.json"),
                        "--train-variant", variant]
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
                    train.main()
                report = json.loads(out.getvalue())
                self.assertEqual((report["train_source_calls"], report["train_calls"]), (2, expected))
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["catalog.json", "train.jsonl", "val.jsonl"])


class HintTest(unittest.TestCase):
    def setUp(self):
        self.table = build_hints(CATALOG, [BUSY])

    def test_hints_come_from_the_transcript_and_survive_asr_damage(self):
        hints = find_hints(BUSY, self.table)
        self.assertEqual(hints["tariff"], ["Смарт"])
        self.assertEqual(hints["action"], ["перезагрузка роутера"])
        self.assertEqual(hints["service_state"], ["нет гудков"])
        self.assertEqual(hints["agreement"], [])
        damaged = call("u3:call0000", ["перезогрузил роутер, гутков нет"], [])
        self.assertEqual(find_hints(damaged, self.table)["action"], ["перезагрузка роутера"])
        without_facts = {**BUSY, "facts": []}
        self.assertEqual(find_hints(without_facts, self.table), hints)  # gold is not read

    def test_table_and_coverage(self):
        entry = next(e for e in self.table["action"] if e["value"] == "перезагрузка роутера")
        self.assertEqual((entry["train_count"], entry["keys"]), (1, ["перезагрузка", "роутера"]))
        report = coverage([BUSY], self.table)
        self.assertEqual(report["action"]["recall"], 1.0)
        self.assertEqual(report["service_state"]["gold_catalog_values"], 2)


if __name__ == "__main__":
    unittest.main()
