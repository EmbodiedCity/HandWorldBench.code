"""CPU regression checks for formal conditional retention scoring and cache scope."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1]
RUNTIME = SOURCE / 'handworldbench/_vendor/engine/engines/state_four/runtime/state_eval'


def load_plain(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


components = load_plain('native_components', RUNTIME / 'component_scores.py')
normalization = load_plain('native_normalization', RUNTIME / 'score_normalization.py')
legal_tree = ast.parse((RUNTIME / 'legal_termination.py').read_text())
legal_ns = {'math': __import__('math'), 'hashlib': hashlib, 'json': __import__('json')}
legal_nodes = [n for n in legal_tree.body if isinstance(n, (ast.Assign, ast.FunctionDef))]
exec(compile(ast.Module(body=legal_nodes, type_ignores=[]), 'legal_termination.py', 'exec'), legal_ns)


class LegalTerminationTests(unittest.TestCase):
    def evaluator(self, choices):
        tree = ast.parse((RUNTIME / 'state_score.py').read_text())
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'grade_value'
                       or isinstance(n, ast.ClassDef) and n.name == 'StateEvaluator']
        namespace = {'Path': Path, 'copy': copy, 'PREFIX': '', 'END': '',
                     'aggregate': components.aggregate, 'publish': normalization.publish,
                     'LEGAL_TERMINATION_PROTOCOL': legal_ns['PROTOCOL'],
                     'legal_termination_prompt': legal_ns['prompt_for'],
                     'legal_termination_evidence': lambda *args: (args[-2], args[-1], {'fixed': True}),
                     'read': lambda _: {'persistence_checkpoint_map': {'property': 'checkpoint'}}}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), 'state_score.py', 'exec'), namespace)
        obj = object.__new__(namespace['StateEvaluator'])
        obj.protocol_path = 'protocol'
        obj.calls = []
        answers = iter(choices)
        def query(*args, **kwargs):
            obj.calls.append((args, kwargs))
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer
        obj.query = query
        return obj

    def evidence(self, checks=1):
        unit = {'id': 'u', 'property_id': 'property', 'state_description': 'new placement',
                'allowed_reversals': ['deliberate removal is exempt'], 'established': [1, 2]}
        return {'retention': [{'unit': unit, 'before': [{'source_index': 0}],
                               'established': [{'source_index': 1}],
                               'checks': [{'check': {'id': 'w' + str(i), 'description': 'retain placement', 'window': [2, 3]},
                                           'frames': [{'source_index': 2 + i}]} for i in range(checks)]}]}

    def test_normal_grades_preserve_previous_scores_without_followup(self):
        for grade in ('0', '1', '2', '3', '4', '5'):
            obj = self.evaluator(['1', grade])
            score = obj.retention(None, {}, {}, self.evidence(), None)
            self.assertEqual(score['score'], 0 if grade == '0' else (int(grade) - 1) / 4)
            self.assertTrue(score['applicable'])
            self.assertEqual(len(obj.calls), 2)
            self.assertEqual(score['units'][0]['windows'][0]['grade'], grade)

    def test_legal_termination_is_scored_and_remains_applicable(self):
        obj = self.evaluator(['1', '6', '3'])
        result = obj.retention(None, {}, {}, self.evidence(), None)
        self.assertEqual(result['score'], .5)
        self.assertTrue(result['applicable'])
        window = result['units'][0]['windows'][0]
        self.assertEqual(window['original_observation']['grade'], '6')
        self.assertFalse(window['exempt'])
        self.assertTrue(obj.calls[-1][0][2].endswith('_legal_termination_v1'))
        self.assertEqual(obj.calls[-1][0][-1], ('0', '1', '2', '3', '4', '5'))

    def test_mixed_windows_keep_fixed_denominator(self):
        obj = self.evaluator(['1', '5', '6', '2'])
        result = obj.retention(None, {}, {}, self.evidence(2), None)
        self.assertEqual(result['score'], .625)
        self.assertEqual(result['units'][0]['required'], 2)
        self.assertEqual(result['components'], result['units'])

    def test_failure_before_establishment_stays_zero(self):
        obj = self.evaluator(['2'])
        result = obj.retention(None, {}, {}, self.evidence(), None)
        self.assertEqual(result['score'], 0)
        self.assertEqual(len(obj.calls), 1)

    def test_linked_zero_progress_does_not_trigger_new_query(self):
        obj = self.evaluator([])
        parts = [{'id': 'checkpoint', 'value': 0, 'published_score': 0}]
        result = obj.retention(None, {}, {}, self.evidence(), parts)
        self.assertEqual(result['score'], 0)
        self.assertEqual(len(obj.calls), 0)

    def test_followup_unknown_uses_existing_missing_penalty(self):
        obj = self.evaluator(['1', '6', '0'])
        result = obj.retention(None, {}, {}, self.evidence(), None)
        self.assertEqual(result['score'], 0)
        self.assertTrue(result['zero_imputed'])
        self.assertTrue(result['applicable'])

    def test_followup_execution_error_is_not_zero_or_na(self):
        obj = self.evaluator(['1', '6', RuntimeError('judge failed')])
        with self.assertRaisesRegex(RuntimeError, 'judge failed'):
            obj.retention(None, {}, {}, self.evidence(), None)

    def test_only_regulation_protocol_changes(self):
        from handworldbench.registry import METRICS
        self.assertEqual(METRICS['state_regulation'].protocol, legal_ns['PROTOCOL'])
        for metric in ('state_progression', 'evolution_correctness', 'memory_consistency'):
            self.assertEqual(METRICS[metric].protocol, 'shared_dino_v1')

    def test_cache_signature_change_is_conditional(self):
        from handworldbench.registry import METRICS
        tree = ast.parse((SOURCE / 'handworldbench/runtime/prepare.py').read_text())
        blocks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and any(isinstance(n, ast.Constant) and n.value == 'state_regulation_protocol' for n in ast.walk(node))
                  and len(node.body) == 1 and isinstance(node.body[0], ast.Assign)]
        self.assertEqual(len(blocks), 1)
        block = compile(ast.Module(body=blocks, type_ignores=[]), 'cache_test', 'exec')
        for name, metrics, expected in [('state_evidence', ['state_regulation'], True),
                                        ('state_evidence', ['state_progression'], False),
                                        ('memory_evidence', ['memory_consistency'], False)]:
            ns = {'name': name, 'requested_metrics': metrics, 'METRICS': METRICS, 'signature': {}}
            exec(block, ns)
            self.assertEqual('state_regulation_protocol' in ns['signature'], expected)

    def test_sampling_matches_approved_twelve_frame_rule(self):
        self.assertEqual(legal_ns['uniform_indices'](30, 180, 3.5, 4.57),
                         [105, 108, 111, 114, 117, 120, 122, 125, 128, 131, 134, 137])

    def test_native_evidence_cache_binds_video_roi_interval_and_target(self):
        identity = legal_ns['evidence_identity']
        original = ['video-A', [0, 0, 20, 20], [1, 2], [30, 45, 59], 'u', 'w']
        first = identity(*original)
        self.assertEqual(first, identity(*copy.deepcopy(original)))
        variants = ['video-B', [1, 1, 20, 20], [1, 3], [30, 50, 59], 'other-u', 'other-w']
        for index, value in enumerate(variants):
            changed = copy.deepcopy(original)
            changed[index] = value
            self.assertNotEqual(first, identity(*changed))


if __name__ == '__main__':
    unittest.main()
