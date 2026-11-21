import sys
import types
import unittest
from unittest.mock import patch
from handworldbench.observations.vision import Observer


class VisualBatchTests(unittest.TestCase):
    def requests(self,count=5):
        return [dict(images=[],prompt=str(i),path=f'frame_{i}.json',tokens=100) for i in range(count)]

    def torch(self):
        class AllocationError(RuntimeError):pass
        return types.SimpleNamespace(cuda=types.SimpleNamespace(OutOfMemoryError=AllocationError,empty_cache=lambda:None))

    def test_batch_order_and_single_retry(self):
        observer=Observer({},'models/vision','prediction',4)
        groups=[];retries=[]
        def batch(requests):
            groups.append(len(requests))
            return ['bad' if r['prompt']=='2' else '{"id":'+r['prompt']+'}' for r in requests]
        def retry(images,prompt,path,*args):
            retries.append(str(path));return '{"id":2}'
        observer._batch_raw=batch;observer.ask_raw=retry
        with patch.dict(sys.modules,{'torch':self.torch()}):result=observer.ask_batch(self.requests())
        self.assertEqual([r['id'] for r in result],list(range(5)))
        self.assertEqual(groups,[4,1]);self.assertEqual(retries,['frame_2.1.json'])

    def test_allocation_failure_reduces_batch(self):
        torch=self.torch();models={};observer=Observer(models,'models/vision','prediction',4);groups=[]
        def batch(requests):
            groups.append(len(requests))
            if len(requests)>2:raise torch.cuda.OutOfMemoryError()
            return ['{"id":'+r['prompt']+'}' for r in requests]
        observer._batch_raw=batch
        with patch.dict(sys.modules,{'torch':torch}):result=observer.ask_batch(self.requests())
        self.assertEqual([r['id'] for r in result],list(range(5)))
        self.assertEqual(groups,[4,2,2,1])
        self.assertEqual(models[('vision_batch_limit','models/vision')],2)

    def test_default_keeps_scalar_calls(self):
        observer=Observer({},'models/vision','reference')
        observer.ask=lambda images,prompt,*args: {'id':int(prompt)}
        self.assertEqual([r['id'] for r in observer.ask_batch(self.requests())],list(range(5)))

    def test_repeated_bad_json_is_an_execution_error(self):
        observer=Observer({},'models/vision','prediction',4)
        observer._batch_raw=lambda requests:['bad']*len(requests)
        observer.ask_raw=lambda *args:'still bad'
        with patch.dict(sys.modules,{'torch':self.torch()}),self.assertRaises(ValueError):
            observer.ask_batch(self.requests(1))
