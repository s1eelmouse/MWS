"""Date binding for organizer scorer v2.x: layers, cues, noise; no gold labels involved."""
import copy
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from schema import CallInput
from temporal import normalize_fact_dates
from temporal_case_v2 import ValueLayers, day_cues, repair_case_v2_dates

CATALOG = {
    "symptoms": ["нет гудков", "слабый сигнал вай-фай"], "tariffs": ["Смарт"],
    "statuses": ["принята", "в работе", "назначен выезд специалиста", "выполнена"],
    "agreements": ["выезд специалиста", "подтверждение заявки по смс"],
}
LAYERS = ValueLayers(CATALOG)
STARTED = "2026-11-13T12:00:00"  # Friday


def call(*texts):
    return CallInput(call_id="synthetic:call0000", started_at=STARTED,
                     turns=[{"idx": i, "speaker": "client", "text": t} for i, t in enumerate(texts)])


def fact(attribute, value, turn=0, valid_from=STARTED, valid_to=None, name="объект", polarity=1):
    return {"object": {"name": name}, "attribute": attribute, "value": value, "polarity": polarity,
            "valid_from": valid_from, "valid_to": valid_to, "source": {"turn_idx": turn}}


def repaired(item, *texts, **options):
    return repair_case_v2_dates([item], call(*texts), LAYERS, **options)[0][0]


class ValueLayerTest(unittest.TestCase):
    def test_exact_catalog_free_and_paraphrase(self):
        self.assertEqual(LAYERS.layer("agreement", "Выезд специалиста"), "catalog")
        self.assertEqual(LAYERS.layer("service_state", "работает нормально"), "catalog")
        self.assertEqual(LAYERS.layer("agreement", "клиент просит не звонить рано утром"), "free")
        # Paraphrases of catalog entries match catalog gold in the scorer: dates stay with the model.
        self.assertEqual(LAYERS.layer("agreement", "подтверждение по смс"), "unsure")
        self.assertEqual(LAYERS.layer("service_state", "слабый сигнал Wi-Fi"), "catalog")  # transliteration
        self.assertEqual(LAYERS.layer("service_state", "очень слабый сигнал вай-фай в спальне"), "unsure")
        self.assertEqual(LAYERS.layer("ticket_status", "заявка 5401 в работе"), "unsure")
        self.assertEqual(LAYERS.layer("action", "перезагрузка роутера"), "unsure")


class CueTest(unittest.TestCase):
    def days(self, text, **options):
        return sorted((day - date(2026, 11, 13)).days for day, _role, _pos in
                      day_cues(text, date(2026, 11, 13), **options))

    def test_scorer_bank_single_days(self):
        self.assertEqual(self.days("позавчера пропало"), [-2])
        self.assertEqual(self.days("пару дней назад и дней пять назад"), [-5, -2])
        self.assertEqual(self.days("неделю назад"), [-7])
        self.assertEqual(self.days("со вторника не работает"), [-3])
        self.assertEqual(self.days("в пятницу сломалось"), [-7])  # the call is on Friday
        self.assertEqual(self.days("приедем в среду, через неделю или на это воскресенье", future=True),
                         [2, 5, 7])
        self.assertEqual(self.days("в пятницу приедем", future=True), [7])

    def test_periods_are_not_reduced_to_a_day(self):
        self.assertEqual(self.days("на прошлой неделе, месяц назад, в прошлом месяце"), [])

    def test_direction_comes_from_the_caller(self):
        self.assertEqual(self.days("вчера обещали на завтра"), [-1])
        self.assertEqual(self.days("вчера обещали на завтра", future=True), [1])

    def test_typical_asr_variants(self):
        self.assertEqual(self.days("фчера всё пропало"), [-1])
        self.assertEqual(self.days("приедем завдра", future=True), [1])


class BindingTest(unittest.TestCase):
    def test_multiple_objects_do_not_share_one_date_without_object_binding(self):
        first = fact('agreement', 'выезд специалиста', name='роутер 1')
        second = fact('agreement', 'выезд специалиста', name='роутер 2')
        output, audit = repair_case_v2_dates([first, second], call('выезд специалиста завтра'), LAYERS)
        self.assertEqual(output, [first, second])
        self.assertTrue(all(d['reason'] == 'multiple_objects_same_evidence' for d in audit))

    def test_question_and_hypothesis_do_not_rewrite_catalog_agreement(self):
        item = fact('agreement', 'выезд специалиста', valid_from='2026-11-17T00:00:00')
        for text in ('Выезд специалиста завтра?', 'Возможно, выезд специалиста завтра.'):
            self.assertEqual(repaired(copy.deepcopy(item), text), item)

    def test_free_formulation_is_dated_by_the_call_and_open(self):
        item = fact("agreement", "мастер позвонит за полчаса до приезда",
                    valid_from="2026-11-14T00:00:00", valid_to="2026-11-15T00:00:00")
        result = repaired(item, "приедет завтра, только пусть позвонит за полчаса")
        self.assertEqual((result["valid_from"], result["valid_to"]), (STARTED, None))

    def test_catalog_agreement_takes_the_scheduled_day(self):
        result = repaired(fact("agreement", "выезд специалиста"), "могу поставить выезд специалиста на завтра")
        self.assertEqual(result["valid_from"], "2026-11-14T00:00:00")
        result = repaired(fact("agreement", "выезд специалиста", turn=1),
                          "ближайшее окно через неделю", "да, выезд специалиста подходит")
        self.assertEqual(result["valid_from"], "2026-11-20T00:00:00")

    def test_paraphrase_and_negative_and_action_are_left_to_the_model(self):
        for item in (fact("agreement", "подтверждение по смс", valid_from="2026-11-17T00:00:00"),
                     fact("service_state", "нет гудков", polarity=-1, valid_from="2026-11-01T00:00:00"),
                     fact("action", "перезагрузка роутера", valid_from="2026-11-10T00:00:00")):
            self.assertEqual(repaired(copy.deepcopy(item), "вчера и завтра"), item)

    def test_symptom_onset_and_end_of_normal_state(self):
        text = "до вторника всё работало нормально, а со вторника нет гудков"
        symptom = repaired(fact("service_state", "нет гудков", valid_to="2026-11-12T00:00:00"), text)
        self.assertEqual((symptom["valid_from"], symptom["valid_to"]), ("2026-11-10T00:00:00", None))
        normal = repaired(fact("service_state", "работает нормально", valid_from="2026-05-01T00:00:00"), text)
        self.assertEqual((normal["valid_from"], normal["valid_to"]),
                         ("2026-05-01T00:00:00", "2026-11-10T00:00:00"))

    def test_status_start_end_and_still(self):
        started = repaired(fact("ticket_status", "принята"), "четыре дня назад заявка была принята")
        self.assertEqual(started["valid_from"], "2026-11-09T00:00:00")
        ended = repaired(fact("ticket_status", "в работе", valid_from="2026-11-05T00:00:00"),
                         "заявка находилась в работе до вторника")
        self.assertEqual((ended["valid_from"], ended["valid_to"]),
                         ("2026-11-05T00:00:00", "2026-11-10T00:00:00"))
        item = fact("ticket_status", "принята", valid_from="2026-11-06T00:00:00")
        self.assertEqual(repaired(copy.deepcopy(item), "вчера ещё была принята"), item)

    def test_cue_belongs_to_the_nearest_status(self):
        text = "четыре дня назад она была принята, поком назначен выезд специалиста"  # «потом» lost to ASR
        item = fact("ticket_status", "назначен выезд специалиста", valid_from="2026-11-12T00:00:00")
        self.assertEqual(repaired(copy.deepcopy(item), text), item)

    def test_chain_closure_is_opt_in_and_never_uses_a_future_start(self):
        facts = [fact("ticket_status", "принята", valid_from="2026-11-09T00:00:00", name="заявка 1"),
                 fact("ticket_status", "в работе", valid_from="2026-11-11T00:00:00", name="заявка 1")]
        plain, _ = repair_case_v2_dates(facts, call("статус"), LAYERS)
        self.assertIsNone(plain[0]["valid_to"])
        closed, _ = repair_case_v2_dates(facts, call("статус"), LAYERS, close_chains=True)
        self.assertEqual(closed[0]["valid_to"], "2026-11-11T00:00:00")
        facts[1]["valid_from"] = "2026-11-20T00:00:00"
        future, _ = repair_case_v2_dates(facts, call("статус"), LAYERS, close_chains=True)
        self.assertIsNone(future[0]["valid_to"])

    def test_result_always_satisfies_the_interval_contract(self):
        from schema import Facts
        # «до сегодняшнего дня» ends the status on the call day, after a noon start timestamp.
        item = fact("ticket_status", "принята", valid_from=STARTED)
        result = repaired(item, "заявка была принята и до сегодняшнего дня так и числилась")
        Facts.model_validate({"facts": [result]})
        self.assertEqual(result["valid_to"][:10], "2026-11-13")

    def test_negated_day_and_idiom_are_not_dates(self):
        # Guards ported from fix/temporal-evidence-scoping.
        result = repaired(fact("agreement", "выезд специалиста"), "выезд специалиста не завтра, а в среду")
        self.assertEqual(result["valid_from"], "2026-11-18T00:00:00")
        item = fact("service_state", "нет гудков", valid_from="2026-11-01T00:00:00")
        self.assertEqual(repaired(copy.deepcopy(item), "нет гудков, я не вчера родилась"), item)

    def test_status_of_another_ticket_question_and_negation_set_no_date(self):
        mine = fact("ticket_status", "принята", valid_from="2026-11-06T00:00:00", name="заявка 5001")
        for text in ("заявка 7002 была принята вчера",
                     "заявка 5001 и заявка 7002 были приняты вчера",
                     "заявка вчера была принята?",
                     "вчера заявка не была принята",
                     "возможно, вчера заявка была принята"):
            self.assertEqual(repaired(copy.deepcopy(mine), text), mine, text)
        self.assertEqual(repaired(copy.deepcopy(mine), "заявка 5001 была принята вчера")["valid_from"],
                         "2026-11-12T00:00:00")

    def test_call_utc_offset_is_kept(self):
        from schema import Facts
        aware = CallInput(call_id="synthetic:call0000", started_at="2026-11-13T12:00:00+03:00",
                          turns=[{"idx": 0, "speaker": "client", "text": "до вторника заявка была в работе"}])
        item = fact("ticket_status", "в работе", valid_from="2026-11-05T09:00:00+03:00")
        result = repair_case_v2_dates([item], aware, LAYERS)[0][0]
        self.assertEqual(result["valid_to"], "2026-11-10T00:00:00+03:00")
        Facts.model_validate({"facts": [result]})

    def test_nothing_is_dropped_reordered_or_mutated(self):
        facts = [fact("agreement", "выезд специалиста"), fact("tariff", "Смарт"), fact("action", "что-то")]
        before = copy.deepcopy(facts)
        output, decisions = normalize_fact_dates(facts, call("завтра"), mode="case_v2", layers=LAYERS)
        self.assertEqual(facts, before)
        self.assertEqual([f["value"] for f in output], [f["value"] for f in before])
        self.assertEqual([d["fact_index"] for d in decisions], [0, 1, 2])
        self.assertEqual(decisions[0]["status"], "changed")

    def test_mode_requires_the_catalog(self):
        with self.assertRaises(ValueError):
            normalize_fact_dates([fact("tariff", "Смарт")], call("текст"), mode="case_v2")


if __name__ == "__main__":
    unittest.main()
