import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from handworldbench.backends.dispatch import observe
from handworldbench.backends.scoring import calculate_photometric
from handworldbench.runtime.execute import execute
from handworldbench.runtime.prepare import prepare
from handworldbench.runtime.storage import read


class ReleaseExecutionTests(unittest.TestCase):
    def test_narrow_observations_have_separate_cache_keys(self):
        pose = 'pose_control,keypoint_coverage'
        entity = 'entity_consistency'
        trajectory = 'trajectory_accuracy'
        metrics = f'{pose},{entity},{trajectory}'.split(',')
        reference = {'sha256': 'reference'}
        row = dict(case_id='example', model='model', category='human',
                   reference=reference, video_sha256='prediction', metrics=metrics,
                   time_mapping={'kind': 'identity'}, spatial_mapping={'kind': 'identity'})
        item = dict(geometry={'path': 'geometry.npz'}, annotations={},
                    applicability=dict.fromkeys(metrics, 'applicable'))
        dataset = {'runtime': {'cache_root': 'unused'}}

        def observation_tasks(root, run_name, selection):
            run = root / run_name
            args = SimpleNamespace(dataset=str(root), model='model',
                                   run_dir=str(run), config=None, gpus='0',
                                   metrics=selection)
            prepare(args)
            return {task['observation']: task for task in read(run / 'configs/tasks.json')
                    if task['kind'] == 'observe'}

        with (tempfile.TemporaryDirectory() as folder,
              patch('handworldbench.runtime.prepare.load_dataset',
                    return_value=(Path(folder), dataset, {'example': item})),
              patch('handworldbench.runtime.prepare.normalize',
                    return_value=(dataset, {'example': item}, [row])),
              patch('handworldbench.runtime.prepare.select_devices', return_value=[])):
            root = Path(folder)
            full = observation_tasks(root, 'full', ','.join(metrics))
            pose_only = observation_tasks(root, 'pose_only', pose)
            entity_only = observation_tasks(root, 'entity_only', entity)

        self.assertNotIn('include_reference_detections', full['hand_detection'])
        self.assertFalse(pose_only['hand_detection']['include_reference_detections'])
        self.assertNotEqual(full['hand_detection']['cache_key'],
                            pose_only['hand_detection']['cache_key'])
        self.assertEqual(full['hand_detection']['cache_key'],
                         entity_only['hand_detection']['cache_key'])
        self.assertNotIn('include_trajectory_score', full['object_tracks'])
        self.assertFalse(entity_only['object_tracks']['include_trajectory_score'])
        self.assertNotEqual(full['object_tracks']['cache_key'],
                            entity_only['object_tracks']['cache_key'])

    def test_pose_only_skips_reference_detector(self):
        reference = {'sha256': 'reference'}
        row = dict(category='human', reference=reference, video_sha256='prediction')
        dependencies = {
            'frames_example': {'frames': ['prediction-frame']},
            'reference_geometry_example': {'reference_images': ['reference-frame']},
        }
        detector = Mock(side_effect=lambda frames, *args, **kwargs: (frames, {}))
        with (tempfile.TemporaryDirectory() as folder,
              patch('handworldbench.observations.video.images', side_effect=lambda frames: frames),
              patch('handworldbench.observations.shared_hands.predict', detector)):
            task = dict(kind='observe', observation='hand_detection', row=row,
                        dependency_results=dependencies, config={'cache_root': folder})
            result = execute(dict(task, cache_key='narrow', include_reference_detections=False),
                             Path(folder), {})
            self.assertEqual([call.kwargs['role'] for call in detector.call_args_list],
                             ['prediction'])
            self.assertNotIn('reference', result)
            detector.reset_mock()
            result = execute(dict(task, cache_key='full'), Path(folder), {})
            self.assertEqual([call.kwargs['role'] for call in detector.call_args_list],
                             ['prediction', 'reference'])
            self.assertEqual(result['reference'], ['reference-frame'])

    def test_tracking_only_dispatch_has_no_trajectory_score(self):
        row = dict(case_id='example', model='model', category='human', video='prediction.mp4',
                   spatial_mapping={'kind': 'identity'},
                   reference={'video': 'reference.mp4', 'sha256': 'reference'})
        task = dict(config={'models': {}}, reference={'annotations': {}}, row=row,
                    observation='object_tracks', cache_key='tracking_only',
                    include_trajectory_score=False)
        inference = ModuleType('handworldbench.backends.inference')
        inference.run_extended = Mock(return_value={'shared_objects': {'entities': ['object']}})
        with (tempfile.TemporaryDirectory() as folder,
              patch('handworldbench.backends.dispatch.initialize'),
              patch.dict('sys.modules', {'handworldbench.backends.inference': inference})):
            result = observe(task, {}, Path(folder))
        forwarded = inference.run_extended.call_args.args[0]
        self.assertEqual(forwarded['metrics'], [])
        self.assertFalse(forwarded['include_trajectory_score'])
        self.assertEqual(result['metrics'], {})
        self.assertEqual(result['shared_objects'], {'entities': ['object']})

    def test_photometric_projection_is_shared_for_first_target(self):
        import numpy as np

        background = ModuleType('handworldbench_aux.background')
        background.clouds = Mock(return_value=[(0,), (1,), (2,)])

        def projection(_arrays, target, source):
            if target == 1:
                return {'visible_pixels': 99}
            return dict(visible_pixels=100, source_colors=np.array([255 * source]),
                        target_colors=np.array([0]))

        background.project = Mock(side_effect=projection)
        with patch.dict('sys.modules', {'handworldbench_aux.background': background}):
            result = calculate_photometric('photometric_consistency', {})
        self.assertEqual([call.args[1:] for call in background.project.call_args_list],
                         [(1, 0), (2, 1), (2, 0)])
        self.assertEqual(result['expected_units'], 4)
        self.assertEqual(result['penalized_units'], 2)
        first_pair = result['details']['coverage'][0]['branches']
        self.assertEqual(first_pair['previous'], first_pair['first'])
        self.assertAlmostEqual(result['score'], (1 + 1 / 101) / 4)
