import copy
from pathlib import Path
import sys,tempfile,unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from build_targeted_train import build
from schema import to_answer

def example(cid,agreement=True):
    return {'call_id':cid,'started_at':'2026-07-05T12:00:00','noise_level':.25,
      'turns':[{'idx':0,'speaker':'operator','text':'Предлагаю выезд специалиста.'},
               {'idx':3,'speaker':'client','text':'Да, подходит.'},
               {'idx':9,'speaker':'operator','text':'Остались вопросы?'},
               {'idx':12,'speaker':'client','text':'До свидания.'}],
      'facts':[{'object':{'name':'Клиент','type':'client','id':None},
                'attribute':'agreement' if agreement else 'service_state','value':'выезд специалиста' if agreement else 'не работает',
                'polarity':1,'valid_from':'2026-07-05T12:00:00','valid_to':None,'source':{'turn_idx':3}}]}

class TargetedTrainTests(unittest.TestCase):
    def fixture(self):
        original=[example('train-a:call0'),example('train-b:call0',False)]
        focus=[copy.deepcopy(original[0])];focus[0]['call_id']+='~f20'
        focus[0]['turns'][1]['text']='дпддд';focus[0]['turns'][3]['text']='noise'
        return original,focus,[example('val-c:call0')]

    def test_complete_labels_background_and_source_neighborhood_are_preserved(self):
        original,focus,val=self.fixture();before=copy.deepcopy((original,focus,val))
        result,info=build(original,focus,val,source_replay=1)
        self.assertEqual(len(result),3)
        self.assertTrue(info['all_original_calls_represented'])
        row=next(r for r in result if r['call_id'].endswith('~f20'))
        self.assertEqual(row['turns'][1]['text'],'Да, подходит.')
        self.assertEqual(row['turns'][2]['text'],'Остались вопросы?')
        self.assertEqual(row['turns'][3]['text'],'noise')
        self.assertEqual(to_answer(row['facts']),to_answer(original[0]['facts']))
        self.assertEqual((original,focus,val),before)

    def test_deterministic_and_val_labels_cannot_select_training_examples(self):
        original,focus,val=self.fixture()
        a,info=build(original,focus,val,source_replay=1)
        val[0]['facts']=[]
        b,again=build(original,focus,val,source_replay=1)
        self.assertEqual(a,b);self.assertEqual(info,again)

    def test_foreign_rows_changed_labels_and_val_accounts_are_rejected(self):
        original,focus,val=self.fixture();focus[0]['facts'][0]['value']='подмена'
        with self.assertRaises(ValueError):build(original,focus,val)
        original,focus,val=self.fixture();focus[0]['call_id']='foreign:call0~f'
        with self.assertRaises(ValueError):build(original,focus,val)
        original,focus,val=self.fixture();val[0]['call_id']='train-a:another'
        with self.assertRaises(ValueError):build(original,focus,val)

    def test_no_duplicate_output_ids(self):
        original,focus,val=self.fixture();focus.append(copy.deepcopy(focus[0]))
        with self.assertRaises(ValueError):build(original,focus,val)

    def test_checkpoint_option_does_not_change_default_behavior(self):
        import subprocess
        path=Path(__file__).resolve().parents[1]/'src/train.py'
        result=subprocess.run([sys.executable,str(path),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0);self.assertIn('--save-total-limit',result.stdout)

if __name__=='__main__':unittest.main()
