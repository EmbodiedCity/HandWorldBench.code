import json
import re
import tempfile
import types
import unittest
from pathlib import Path
from test_shared_hands import function

SOURCE=Path(__file__).resolve().parents[1]/'handworldbench/_vendor/engine/engines/new_aux/handworldbench_aux/judge.py'
class Issue(Exception):
    def __init__(self,status,message,details=None):
        super().__init__(message);self.status=status

class JudgeFormatRetryTests(unittest.TestCase):
    def setup_judge(self,answers):
        import torch
        traces=[];messages=[];budgets=[]
        ns=dict(re=re,json=json,MetricIssue=Issue,digest=lambda x:'hash',write_json=lambda p,x:traces.append(x))
        for name in ['parse_score','parse_binary','retry_prompt','LocalJudge']:function(SOURCE,name,ns)
        class Inputs(dict):
            def to(self,device):return self
        class Processor:
            def apply_chat_template(self,value,**kwargs):
                messages.append(value);return Inputs(input_ids=torch.tensor([[1,2]]))
            def batch_decode(self,*args,**kwargs):return [answers.pop(0)]
        class Model:
            device='cpu'
            def generate(self,**kwargs):budgets.append(kwargs['max_new_tokens']);return torch.tensor([[1,2,3]])
        judge=ns['LocalJudge']({'image_encoder_cache':False},Path('fixture'))
        judge._load=lambda:None;judge.model=Model();judge.processor=Processor();judge.path=Path('fixture-weights')
        return judge,ns,traces,messages,budgets

    def test_valid_first_answer_keeps_prompt_and_budget(self):
        judge,ns,traces,messages,budgets=self.setup_judge(['Reason: one issue\nScore: 2'])
        result=judge.ask('original rubric',[],ns['parse_score'])
        self.assertEqual(result['score'],2);self.assertEqual(budgets,[2048])
        self.assertEqual(messages[0][-1]['content'][-1]['text'],'original rubric')

    def test_invalid_numeric_reply_uses_one_line_retry(self):
        judge,ns,traces,messages,budgets=self.setup_judge(['Reason: repeated text','Score: 1'])
        result=judge.ask('unchanged rubric',[],ns['parse_score'])
        self.assertEqual(result['score'],1);self.assertEqual(budgets,[2048,256])
        self.assertTrue(messages[1][-1]['content'][-1]['text'].startswith('unchanged rubric'))
        self.assertIn('exactly one line',messages[1][-1]['content'][-1]['text'])
        self.assertEqual([t['status'] for t in traces],['parse_error','ok'])

    def test_exhausted_format_retries_remain_execution_error(self):
        judge,ns,traces,messages,budgets=self.setup_judge(['no score','Score: 8'])
        with self.assertRaises(Issue) as err:judge.ask('rubric',[],ns['parse_score'])
        self.assertEqual(err.exception.status,'runtime_error');self.assertEqual(len(traces),2)
        self.assertTrue(all('parsed' not in t for t in traces))

    def test_binary_retry_is_unchanged(self):
        judge,ns,traces,messages,budgets=self.setup_judge(['invalid','Answer: Yes'])
        self.assertTrue(judge.ask('binary rubric',[],ns['parse_binary'])['answer'])
        self.assertEqual(budgets,[2048,2048]);self.assertNotIn('exactly one line',messages[1][-1]['content'][-1]['text'])

    def test_unknown_is_explicit_and_ambiguous_scores_rejected(self):
        judge,ns,*_=self.setup_judge([])
        self.assertEqual(ns['parse_score']('Score: unknown'),{'score':0,'unknown':True})
        for raw in ['Score: 1\nScore: 2','Reason: 0','Score: -1']:
            with self.assertRaises(ValueError):ns['parse_score'](raw)
