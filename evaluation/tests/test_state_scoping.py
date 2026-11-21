import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class StateScopingTests(unittest.TestCase):
    def evaluator(self, metrics, folder):
        source=Path(__file__).resolve().parents[1]/'handworldbench/_vendor/engine/engines/state_four/runtime/state_eval/state_score.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        definition=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StateEvaluator')
        namespace={'__file__':str(source),'Path':Path,'sha':lambda _: 'hash', 'immutable':lambda *args: None,
                   'evaluate_progress':lambda *args,**kwargs: {'score':.5,'checkpoints':[]}}
        exec(compile(ast.Module(body=[definition],type_ignores=[]),str(source),'exec'),namespace)
        cls=namespace['StateEvaluator'];obj=object.__new__(cls)
        obj.context=SimpleNamespace(metrics=metrics,output_root=Path(folder))
        obj.out=Path(folder);obj.spec_path=Path(folder)/'spec.json';obj.protocol_path=Path(folder)/'protocol.json'
        obj.evidence_path=Path(folder)/'evidence.json';obj.binary_choose=None
        obj.fingerprint=lambda _: {};obj.prepare=lambda *args: {}
        obj.query=lambda *args,**kwargs: self.fail('Unrequested physical judgment executed')
        return obj

    def test_progress_does_not_run_physics_or_retention(self):
        with tempfile.TemporaryDirectory() as folder:
            evaluator=self.evaluator(['state_progression'],folder)
            evaluator.retention=lambda *args:self.fail('Unrequested retention judgment executed')
            result=evaluator.evaluate(None,{'case_id':'case'}, {'applicability':{'state_progression':{'status':'applicable'}}})
            self.assertEqual(set(result['metrics']),{'state_progression'})
            self.assertEqual(result['metrics']['state_progression']['score'],.5)

    def test_regulation_keeps_progress_gate_without_physics(self):
        with tempfile.TemporaryDirectory() as folder:
            evaluator=self.evaluator(['state_regulation'],folder)
            evaluator.retention=lambda judge,case,spec,evidence,parts: {'score':.75,'progress_parts':parts}
            spec={'applicability':{'state_progression':{'status':'applicable'},'state_regulation':{'status':'applicable'}}}
            result=evaluator.evaluate(None,{'case_id':'case'},spec)
            self.assertEqual(set(result['metrics']),{'state_progression','state_regulation'})
            self.assertEqual(result['metrics']['state_regulation']['score'],.75)


if __name__=='__main__':unittest.main()
