from pathlib import Path
import copy,json,tempfile,unittest
import eval_targeted_b4 as evaluation

class EvaluationQueueTests(unittest.TestCase):
    def test_waits_until_complete_and_requires_merged_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(evaluation.training_state(root),'waiting')
            (root/'pipeline-status.json').write_text('{"status":"complete"}')
            with self.assertRaises(RuntimeError):evaluation.training_state(root)
            (root/'merged').mkdir();(root/'merged/model.safetensors').write_bytes(b'test')
            self.assertEqual(evaluation.training_state(root),'ready')

    def test_failed_training_cannot_start_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'pipeline-status.json').write_text('{"status":"failed"}')
            with self.assertRaises(RuntimeError):evaluation.training_state(root)

    def test_probe_uses_longest_prompts_and_does_not_send_gold_to_prompt_builder(self):
        rows=[{'call_id':str(i),'turns':['x'*i],'facts':['secret']} for i in range(12)]
        before=copy.deepcopy(rows)
        def boundary(row):return {k:v for k,v in row.items() if k!='facts'}
        def prompt(tokenizer,call):
            self.assertNotIn('facts',call)
            return call['turns'][0]
        def tokenizer(text,**kwargs):return {'input_ids':list(text)}
        selected,lengths=evaluation.longest_calls(rows,tokenizer,prompt,boundary)
        self.assertEqual([r['call_id'] for r in selected],list(map(str,range(11,3,-1))))
        self.assertEqual(lengths[0]['input_tokens'],11)
        self.assertEqual(rows,before)

if __name__=='__main__':unittest.main()
