import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from handworldbench.data.inputs import normalize, validate_output_protocol
from handworldbench.data.submission import PROTOCOL, complete_mappings
from handworldbench.data.timeline import Timeline


NATIVE = dict(name='reference_native', time_mapping={'kind': 'identity'},
              spatial_mapping={'kind': 'identity'},
              match_reference_dimensions=True, match_reference_timeline=True)


def metadata(width=640, height=480, fps=30, count=31):
    return dict(width=width, height=height, frame_count=count,
                times=[i / fps for i in range(count)])


class SubmissionInputTests(unittest.TestCase):
    def row(self, **extra):
        row = dict(case_id='case', video='prediction.mp4',
                   reference=dict(video='reference.mp4', width=640, height=480,
                                  fps=30, frame_count=31))
        row.update(extra)
        return row

    def complete(self, row, prediction, reference=None, protocol=None):
        def probe(path):
            return copy.deepcopy(prediction if path == row['video'] else reference or metadata())
        with patch('handworldbench.observations.video.probe', side_effect=probe):
            return complete_mappings(row, protocol)

    def test_explicit_mappings_keep_priority_without_probe(self):
        row = self.row(time_mapping={'kind': 'affine', 'scale': .3},
                       spatial_mapping={'kind': 'opencv_calibrated', 'spec': {'existing': True}})
        original = copy.deepcopy(row)
        with patch('handworldbench.observations.video.probe', side_effect=AssertionError('must not probe')):
            self.assertIs(complete_mappings(row, NATIVE), row)
        self.assertEqual(row, original)

    def test_noninteger_resize_uses_pixel_centers(self):
        row = self.row(time_mapping={'kind': 'identity'})
        result = self.complete(row, metadata(width=854, height=479))
        matrix = result['spatial_mapping']['matrix']
        sx, sy = 854 / 640, 479 / 480
        self.assertEqual(matrix, [[sx, 0., .5*sx-.5], [0., sy, .5*sy-.5], [0., 0., 1.]])
        self.assertAlmostEqual(matrix[0][0] * 639.5 + matrix[0][2], 853.5)
        self.assertAlmostEqual(matrix[1][0] * 0 + matrix[1][1] * 479.5 + matrix[1][2], 478.5)
        self.assertNotIn('spatial_mapping', row)
        self.assertEqual(result['time_mapping'], row['time_mapping'])
        self.assertEqual(result['output_protocol']['expected_dimensions'], [854, 479])

    def test_different_fps_same_decoded_span_is_identity(self):
        for fps, count in ((15, 16), (24, 25), (60, 61)):
            with self.subTest(fps=fps):
                result = self.complete(self.row(), metadata(fps=fps, count=count))
                self.assertEqual(result['time_mapping'], {'kind': 'identity'})

    def test_uniform_speed_uses_decoded_endpoints_not_frame_count(self):
        result = self.complete(self.row(), metadata(fps=30, count=16))
        time = Timeline(result['time_mapping'])
        self.assertEqual(time.at(0), 0)
        self.assertEqual(time.at(1), .5)
        self.assertEqual(time.inverse().at(.25), .5)
        # Nonuniform frame timestamps still bind the complete presentation span.
        pred = metadata(count=3)
        pred['times'] = [0., .01, 2.5]
        result = self.complete(self.row(), pred)
        self.assertEqual(result['time_mapping']['scale'], 2.5)

    def test_partial_mapping_only_fills_missing_axis(self):
        spatial = dict(kind='affine', matrix=[[1, 0, -30], [0, 1, 0], [0, 0, 1]])
        row = self.row(spatial_mapping=spatial)
        result = self.complete(row, metadata(width=320, height=240, fps=15, count=16))
        self.assertEqual(result['spatial_mapping'], spatial)
        self.assertEqual(result['time_mapping'], {'kind': 'identity'})
        self.assertNotIn('expected_dimensions', result['output_protocol'])
        row = self.row(time_mapping={'kind': 'affine', 'scale': .7, 'offset': .1})
        # With explicit time mapping a single decoded frame is sufficient for resize.
        result = self.complete(row, metadata(width=320, height=240, count=1))
        self.assertEqual(result['time_mapping'], row['time_mapping'])
        self.assertNotIn('expected_time_span_s', result['output_protocol'])

    def test_custom_preset_is_authoritative_without_probe(self):
        protocol = copy.deepcopy(NATIVE)
        protocol['spatial_mapping'] = {'kind': 'affine', 'matrix': [[2, 0, 9], [0, 2, 8], [0, 0, 1]]}
        protocol['time_mapping'] = {'kind': 'affine', 'scale': .2}
        row = self.row(time_mapping={'kind': 'affine', 'scale': .9})
        with patch('handworldbench.observations.video.probe', side_effect=AssertionError('must not probe')):
            result = complete_mappings(row, protocol)
        self.assertEqual(result['time_mapping'], row['time_mapping'])
        self.assertEqual(result['spatial_mapping'], protocol['spatial_mapping'])
        self.assertNotIn('match_reference_timeline', result['output_protocol'])
        self.assertTrue(result['output_protocol']['match_reference_dimensions'])

    def test_native_preset_remains_compatible(self):
        result = self.complete(self.row(), metadata(), protocol=NATIVE)
        self.assertEqual(result['output_protocol'], NATIVE)
        self.assertEqual(result['time_mapping'], {'kind': 'identity'})
        self.assertEqual(result['spatial_mapping'], {'kind': 'identity'})
        validate_output_protocol(result, {'metadata': metadata()})
        result = self.complete(self.row(), metadata(width=320, height=240, fps=30, count=16), protocol=NATIVE)
        self.assertEqual(result['output_protocol']['name'], PROTOCOL)
        self.assertEqual(result['time_mapping']['scale'], .5)
        validate_output_protocol(result, {'metadata': metadata(width=320, height=240, fps=30, count=16)})
        # A small span change inside the old native tolerance still uses the
        # actual decoded spans rather than silently retaining identity.
        changed = metadata()
        changed['times'] = [t*.99 for t in changed['times']]
        result = self.complete(self.row(), changed, protocol=NATIVE)
        self.assertAlmostEqual(result['time_mapping']['scale'], .99)
        self.assertEqual(result['output_protocol']['name'], PROTOCOL)

    def test_invalid_timestamps_are_rejected(self):
        for times in ([0.], [0., 0.], [0., -.1], [0., float('nan')], [0., float('inf')]):
            with self.subTest(times=times):
                prediction = dict(width=640, height=480, frame_count=len(times), times=times)
                with self.assertRaisesRegex(ValueError, 'timestamps'):
                    self.complete(self.row(), prediction)
        invalid_reference = metadata(count=1)
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            self.complete(self.row(), metadata(), invalid_reference)

    def test_frozen_automatic_protocol_is_revalidated(self):
        pred = metadata(width=320, height=240, fps=30, count=16)
        row = self.complete(self.row(), pred)
        validate_output_protocol(row, {'metadata': pred})
        for field, value in (('width', 321), ('frame_count', 17), ('times', [i/20 for i in range(16)])):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Input binding error'):
                changed = copy.deepcopy(pred)
                changed[field] = value
                validate_output_protocol(row, {'metadata': changed})

    def setup_dataset(self, root, protocol=None):
        config = {'catalog': 'catalog.json'}
        if protocol is not None:
            config['standard_output_protocol'] = protocol
        (root/'dataset.json').write_text(json.dumps(config))
        (root/'catalog.json').write_text(json.dumps([
            dict(case_id='case', category='human', reference=dict(video='reference.mp4', width=640,
                 height=480, fps=30, frame_count=31))]))
        (root/'reference.mp4').touch()
        folder = root/'videos'
        folder.mkdir()
        (folder/'case.mp4').write_bytes(b'video')
        return folder

    def args(self, root, folder, predictions=None):
        return SimpleNamespace(dataset=str(root), cases=None, model='Model', metrics='pose_control',
                               predictions=str(predictions) if predictions else None,
                               use_reference_videos=False, videos=str(folder))

    def test_directory_and_manifest_share_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = self.setup_dataset(root, NATIVE)
            path = root/'predictions.json'
            path.write_text(json.dumps([dict(case_id='case', video='videos/case.mp4')]))
            def probe(video):
                return metadata(width=854, height=479, fps=30, count=16) if Path(video).name == 'case.mp4' else metadata()
            with patch('handworldbench.observations.video.probe', side_effect=probe):
                directory = normalize(self.args(root, folder))[2][0]
                manifest = normalize(self.args(root, folder, path))[2][0]
            self.assertNotIn('input_error', directory)
            self.assertEqual(directory, manifest)
            self.assertEqual(directory['time_mapping']['scale'], .5)
            self.assertEqual(directory['spatial_mapping']['output_wh'], [854, 479])

    def test_manifest_without_preset_gets_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = self.setup_dataset(root)
            path = root/'predictions.json'
            path.write_text(json.dumps([dict(case_id='case', video='videos/case.mp4')]))
            with patch('handworldbench.observations.video.probe', return_value=metadata()):
                row = normalize(self.args(root, folder, path))[2][0]
            self.assertNotIn('input_error', row)
            self.assertEqual(row['output_protocol']['name'], PROTOCOL)

    def test_unreadable_video_is_per_input_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = self.setup_dataset(root)
            with patch('handworldbench.observations.video.probe', side_effect=ValueError('bad video')):
                row = normalize(self.args(root, folder))[2][0]
            self.assertIn('Cannot read video', row['input_error'])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg and ffprobe required')
    def test_real_ffmpeg_resize_and_fps_compression(self):
        from handworldbench.observations.video import probe
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = self.setup_dataset(root)
            commands = [
                ['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=red:s=640x480:r=30',
                 '-frames:v', '31', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(root/'reference.mp4')],
                ['ffmpeg', '-v', 'error', '-y', '-i', str(root/'reference.mp4'), '-vf',
                 'scale=854:478,setpts=0.5*PTS', '-fps_mode', 'passthrough', '-enc_time_base', '1/60',
                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(folder/'case.mp4')],
            ]
            for command in commands:
                subprocess.run(command, check=True, capture_output=True)
            row = normalize(self.args(root, folder))[2][0]
            self.assertNotIn('input_error', row)
            self.assertAlmostEqual(row['time_mapping']['scale'], .5)
            self.assertEqual(row['spatial_mapping']['output_wh'], [854, 478])
            pred = probe(folder/'case.mp4')
            self.assertEqual(pred['frame_count'], 31)
            self.assertAlmostEqual(Timeline(row['time_mapping']).at(1), pred['times'][-1])
            validate_output_protocol(row, {'metadata': pred})

    @unittest.skipUnless(shutil.which('ffprobe'), 'ffprobe required')
    def test_prepare_reuses_video_metadata_in_later_observations(self):
        from handworldbench.observations.video import probe, stage
        from handworldbench.runtime.devices import Device
        from handworldbench.runtime.prepare import prepare

        example = Path(__file__).resolve().parents[1] / 'examples/dataset'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / 'runtime.json'
            config.write_text(json.dumps({'cache_root': 'cache'}))
            args = self.args(example, example / 'predictions')
            args.config = str(config)
            args.run_dir = str(root / 'run')
            args.gpus = '0'

            device = Device(0, 0, 'test-gpu', 100, 100)
            with patch('handworldbench.runtime.prepare.select_devices', return_value=[device]), patch(
                'handworldbench.observations.video.probe', wraps=probe
            ) as video_probe:
                receipt = prepare(args)
                self.assertEqual(receipt['input_errors'], [])
                row, = json.loads((root / 'run/configs/predictions.json').read_text())
                self.assertEqual(row['time_mapping']['scale'], 0.5)
                self.assertEqual(row['spatial_mapping']['output_wh'], [64, 44])
                stage(row['video'], row['video_sha256'], root / 'cache')
                reference = row['reference']
                stage(reference['video'], reference['sha256'], root / 'cache')
                self.assertEqual(video_probe.call_count, 2)


if __name__ == '__main__':
    unittest.main()
