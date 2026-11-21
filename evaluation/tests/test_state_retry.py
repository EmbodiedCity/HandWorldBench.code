import copy, importlib.util, pathlib, tempfile, unittest
from handworldbench.runtime.attempts import state_attempt_config
path=pathlib.Path(__file__).resolve().parents[1]/'handworldbench/_vendor/engine/engines/state_four/runtime/state_eval/progress_frames.py'
spec=importlib.util.spec_from_file_location('progress_frames',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class StateRetry(unittest.TestCase):
    def test_seven_frames_unchanged(self):
        windows={f'state:{k}':{'source_indices':v} for k,v in dict(before=[0,1,2],process=[3,4,5],after=[6,7,8]).items()}
        self.assertEqual(mod.progress_indices(windows),[0,3,4,5,6,7,8])
    def test_short_window_actual_frames(self):
        windows={f'state:{k}':{'source_indices':v} for k,v in dict(before=[0,8,17],process=[18,156,294],after=[312]).items()}
        self.assertEqual(mod.progress_indices(windows),[0,18,156,294,312])
    def test_missing_window_error(self):
        with self.assertRaises(ValueError):mod.progress_indices({f'state:{k}':{'source_indices':v} for k,v in dict(before=[0],process=[],after=[3]).items()})
    def test_attempt_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output = root / 'observations'
            video = root / 'input.mp4'
            before = {
                'output_root': str(output / 'state'),
                'cache_root': str(root / 'cache'),
                'metrics': [],
                'cases': [{
                    'video': str(video),
                    'state_evidence': str(output / 'state' / 'evidence' / 'a.json'),
                    'state_spec': str(output / 'state_spec.json'),
                }],
            }
            original = copy.deepcopy(before)
            one, first_path = state_attempt_config(original, output, 1, video, ['state_progression'])
            two, second_path = state_attempt_config(original, output, 2, video, ['state_progression'])
            self.assertNotEqual(one['output_root'], two['output_root'])
            self.assertNotEqual(one['cache_root'], two['cache_root'])
            self.assertNotEqual(first_path, second_path)
            expected = pathlib.Path(one['output_root']) / 'evidence' / 'a.json'
            self.assertEqual(one['cases'][0]['state_evidence'], str(expected))
            self.assertEqual(one['cases'][0]['state_spec'], str(output / 'state_spec.json'))
            self.assertEqual(original, before)
            self.assertEqual(state_attempt_config(original, output, 1, video, ['state_progression'])[0], one)
