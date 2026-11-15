"""Semantic evidence probes independent of organizer val."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from schema import as_call,Facts,source_problems
from fact_recovery import recover_facts

class UnifiedTariffs(unittest.TestCase):
    def run_case(self,texts,facts=()):
        if isinstance(texts,str):texts=[('client',texts)]
        c=as_call({'call_id':'synthetic:unified','started_at':'2026-07-08T12:00:00',
                  'turns':[{'idx':i,'speaker':s,'text':t} for i,(s,t) in enumerate(texts)]})
        client={'object':{'name':'Клиент','type':'client','id':None},'attribute':'agreement','value':'перезвонить',
            'polarity':1,'valid_from':None,'valid_to':None,'source':{'turn_idx':0},'speaker':texts[0][0]}
        source=[client,*copy.deepcopy(facts)];old=copy.deepcopy(source)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'catalog.json';p.write_text(json.dumps({'tariffs':['Пульс','Пульс+','Двухсотый','Мой онлайн+','Мини']}),encoding='utf-8')
            result,audit=recover_facts(source,c,str(p))
            self.assertEqual(source,old)
            self.assertEqual(recover_facts(result,c,str(p))[0],result)
        self.assertFalse(source_problems(Facts.model_validate({'facts':result}),c))
        return [f for f in result if f['attribute']=='tariff'],audit
    def test_condition_request_and_bare_question_are_not_tariffs(self):
        for t in ['Если вы перешли на Пульс, надо проверить оплату.',
                  'Хочу, чтобы вы перевели меня на Пульс.','У меня тариф Пульс?',
                  'А вы случайно не переходили на Мой онлайн+',
                  'Завтра я перейду на Пульс.','Давайте подключим тариф Пульс.']:
            with self.subTest(t=t):self.assertFalse(self.run_case(t)[0])
    def test_switch_target_is_grammatical_not_last_mention(self):
        fs,_=self.run_case('Я перешла на Двухсотый, а Пульс мне только предлагали.')
        self.assertEqual([f['value'] for f in fs],['Двухсотый'])
    def test_missing_switch_target_is_not_replaced_with_old_tariff(self):
        self.assertFalse(self.run_case('Пульс был раньше, затем перешла на другой неизвестный тариф.')[0])
    def test_title_my_does_not_prove_possession(self):
        self.assertFalse(self.run_case('А Мой онлайн+ вам случайно не предлагали')[0])
    def test_tag_question_preserves_prior_assertion(self):
        self.assertEqual(self.run_case('У меня тариф Пульс — это не влияет?')[0][0]['value'],'Пульс')
    def test_confirmation_uses_previous_question(self):
        fs,_=self.run_case([('operator','У вас тариф Пульс?'),('client','Да, Пульс, всё верно.')])
        self.assertEqual(fs[0]['value'],'Пульс');self.assertEqual(fs[0]['source']['turn_idx'],1)
    def test_agreeing_to_offer_is_not_an_active_tariff(self):
        self.assertFalse(self.run_case([('operator','Предлагаю перейти на Пульс?'),('client','Да, Пульс.')])[0])
    def test_explicit_transition_dates_both_sides(self):
        old={'object':{'name':'Клиент','type':'client','id':None},'attribute':'tariff','value':'Пульс',
             'polarity':1,'valid_from':'2026-06-01T00:00:00','valid_to':None,'source':{'turn_idx':0},'speaker':'client'}
        fs,_=self.run_case('Раньше был Пульс, а вчера я перешла на Двухсотый.',[old])
        self.assertEqual(fs[0]['valid_to'],'2026-07-07T00:00:00');self.assertEqual(fs[-1]['valid_from'],'2026-07-07T00:00:00')
    def test_unknown_boundary_retains_supported_history(self):
        old={'object':{'name':'Клиент','type':'client','id':None},'attribute':'tariff','value':'Пульс',
             'polarity':1,'valid_from':'2026-06-01T00:00:00','valid_to':None,'source':{'turn_idx':0},'speaker':'client'}
        fs,audit=self.run_case('Раньше был Пульс, а теперь я перешла на Двухсотый.',[old])
        self.assertEqual(fs,[old]);self.assertEqual(audit[-1]['action'],'abstained')
    def test_plus_is_a_distinct_tariff(self):
        self.assertEqual(self.run_case('У меня тариф Пульс+, сейчас подключён.')[0][0]['value'],'Пульс+')

if __name__=='__main__':unittest.main()
