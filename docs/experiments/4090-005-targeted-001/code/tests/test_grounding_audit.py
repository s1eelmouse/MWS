import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from grounding_audit import diagnose_grounding, compare_grounding
from scoring import Scorer
from case_evaluation import evaluate_predictions


class GroundingAuditTests(unittest.TestCase):
    def setUp(self):
        self.scorer = Scorer('2.1')
        self.fact = {'object':{'name':'Клиент','type':'client','id':None},
                     'attribute':'agreement','value':'выезд специалиста', 'polarity':1,
                     'valid_from':'2026-07-05T12:00:00', 'valid_to':None,
                     'source':{'turn_idx':10}}
        self.gold = {'call_id':'synthetic:grounding','started_at':'2026-07-05T12:00:00',
                     'turns':[{'idx':10,'speaker':'client','text':'Выезд специалиста запланирован.'},
                              {'idx':20,'speaker':'operator','text':'Да, подходит.'},
                              {'idx':30,'speaker':'client','text':'Остался ещё вопрос.'}],
                     'facts':[self.fact]}
        self.pred = {'call_id':self.gold['call_id'], 'facts':[copy.deepcopy(self.fact)]}

    def test_valid_existing_index_can_still_be_wrong_source(self):
        self.pred['facts'][0]['source']['turn_idx']=20
        report, errors=diagnose_grounding([self.gold],[self.pred],scorer=self.scorer,details=True)
        self.assertEqual(report['wrong_source_turn'],1)
        bucket=report['per_attribute']['agreement']
        self.assertEqual(bucket['adjacent_turn'],1)  # positions, not numeric idx difference
        self.assertEqual(bucket['pred_source_is_short_confirmation'],1)
        self.assertEqual(bucket['different_speaker'],1)
        self.assertEqual(errors[0]['gold_source_turn'],10)

    def test_unmatched_predictions_do_not_enter_grounding_denominator(self):
        extra=copy.deepcopy(self.fact);extra['value']='совсем другое действие по договору'
        self.pred['facts'].append(extra)
        report,_=diagnose_grounding([self.gold],[self.pred],scorer=self.scorer)
        self.assertEqual(report['matched_facts'],1)
        self.assertEqual(report['per_attribute']['agreement']['unmatched_predictions'],1)
        self.assertEqual(report['grounding_turn'],1)

    def test_invalid_source_and_missing_prediction_are_counted_separately(self):
        self.pred['facts'][0]['source']['turn_idx']=99
        report,_=diagnose_grounding([self.gold],[self.pred],scorer=self.scorer)
        self.assertEqual(report['per_attribute']['agreement']['missing_or_invalid_source_turn'],1)
        empty,_=diagnose_grounding([self.gold],[],scorer=self.scorer)
        self.assertIsNone(empty['grounding_turn'])

    def test_failed_prediction_does_not_contribute_even_if_it_has_facts(self):
        self.pred['error']='invalid_json'
        report,_=diagnose_grounding([self.gold],[self.pred],scorer=self.scorer)
        self.assertEqual(report['matched_facts'],0)

    def test_mutation_is_forbidden(self):
        before=copy.deepcopy((self.gold,self.pred))
        diagnose_grounding([self.gold],[self.pred],scorer=self.scorer,details=True)
        self.assertEqual((self.gold,self.pred),before)

    def test_equal_aggregate_result_does_not_hide_individual_regressions(self):
        second=copy.deepcopy(self.gold);second['call_id']='synthetic:second'
        a=[copy.deepcopy(self.pred),copy.deepcopy(self.pred)];a[1]['call_id']=second['call_id']
        a[0]['facts'][0]['source']['turn_idx']=20
        b=copy.deepcopy(a);b[0]['facts'][0]['source']['turn_idx']=10;b[1]['facts'][0]['source']['turn_idx']=20
        report=compare_grounding([self.gold,second],a,b,scorer=self.scorer)
        self.assertEqual((report['fixes'],report['regressions']),(1,1))
        self.assertTrue(report['only_sources_and_speakers_changed'])

    def test_changed_date_cannot_be_reported_as_source_only(self):
        after=copy.deepcopy(self.pred);after['facts'][0]['valid_from']=None
        report=compare_grounding([self.gold],[self.pred],[after],scorer=self.scorer)
        self.assertFalse(report['only_sources_and_speakers_changed'])

    def test_duplicate_ids_and_unrelated_calls_are_rejected(self):
        with self.assertRaises(ValueError):
            diagnose_grounding([self.gold],[self.pred,self.pred],scorer=self.scorer)
        bad=copy.deepcopy(self.pred);bad['call_id']='other'
        with self.assertRaises(ValueError):
            diagnose_grounding([self.gold],[bad],scorer=self.scorer)

    def test_official_metrics_and_predictions_are_unchanged_by_extra_diagnostics(self):
        original=copy.deepcopy(self.pred)
        result,_=evaluate_predictions([self.gold],[self.pred],scorer=self.scorer)
        expected=self.scorer.evaluate({self.gold['call_id']:self.gold},{self.pred['call_id']:self.pred})
        self.assertEqual(result['official'],expected)
        self.assertEqual(result['grounding_diagnostics']['grounding_turn'],expected['grounding_turn'])
        self.assertEqual(self.pred,original)


if __name__=='__main__':
    unittest.main()
