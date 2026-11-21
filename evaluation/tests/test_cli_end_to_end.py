"""End-to-end CLI checks on the bundled synthetic dataset (no GPU or model weights needed)."""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from handworldbench.cli.main import main
from handworldbench.runtime.devices import Device
from handworldbench.runtime.storage import read

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
DEVICES = [Device(0, 0, 'GPU-test', 100000, 100000)]


def run_cli(argv):
    out = io.StringIO()
    with patch('handworldbench.runtime.prepare.select_devices', return_value=DEVICES), \
            contextlib.redirect_stdout(out):
        main(argv)
    return out.getvalue()


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'dataset'
        shutil.copytree(EXAMPLES / 'dataset', self.dataset)
        self.config = self.root / 'runtime.json'
        self.config.write_text(json.dumps({'cache_root': 'cache'}))

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, run, *extra):
        argv = ['prepare', '--dataset', str(self.dataset), '--config', str(self.config),
                '--metrics', 'pose_control,keypoint_coverage', '--run-dir', str(self.root / run), *extra]
        return json.loads(run_cli(argv).strip().splitlines()[-1])

    def test_reference_videos_prepare_and_status(self):
        receipt = self.prepare('reference', '--use-reference-videos', '--cases', str(EXAMPLES / 'cases.json'))
        self.assertEqual(receipt['model'], 'reference')
        self.assertEqual(receipt['videos'], 1)
        self.assertGreater(receipt['tasks'], 0)
        self.assertEqual(receipt['input_errors'], [])
        run = self.root / 'reference'
        config = read(run / 'configs/run.json')
        self.assertEqual(config['model'], 'reference')
        self.assertNotIn('baseline', config)
        slots = read(run / 'configs/slots.json')
        self.assertEqual({s['model'] for s in slots}, {'reference'})
        self.assertEqual({s['status'] for s in slots}, {'pending'})
        rows = read(run / 'configs/predictions.json')
        self.assertEqual(rows[0]['video'], str((self.dataset / 'gt/example.mp4').resolve()))
        status = json.loads(run_cli(['status', '--run-dir', str(run)]))
        self.assertGreater(sum(status['tasks'].values()), 0)

    def test_reference_videos_accept_explicit_model_name(self):
        receipt = self.prepare('named', '--use-reference-videos', '--model', 'GT')
        self.assertEqual(receipt['model'], 'GT')

    def test_prediction_manifest_with_model(self):
        manifest = self.root / 'predictions.json'
        manifest.write_text(json.dumps([{'case_id': 'example', 'video': 'predictions/example.mp4'}]))
        receipt = self.prepare('model', '--model', 'MyModel', '--predictions', str(manifest))
        self.assertEqual(receipt['model'], 'MyModel')
        self.assertEqual(receipt['input_errors'], [])

    def test_video_directory_with_model(self):
        receipt = self.prepare('folder', '--model', 'MyModel', '--videos', str(self.dataset / 'predictions'))
        self.assertEqual(receipt['model'], 'MyModel')
        self.assertEqual(receipt['videos'], 1)

    def test_new_run_requires_model_name(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.prepare('missing', '--videos', str(self.dataset / 'predictions'))

    def test_legacy_baseline_flag_is_rejected(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.prepare('legacy', '--baseline', 'MyModel', '--videos', str(self.dataset / 'predictions'))

    def test_manifest_row_for_another_model_is_rejected(self):
        manifest = self.root / 'predictions.json'
        manifest.write_text(json.dumps([{'case_id': 'example', 'model': 'Other', 'video': 'predictions/example.mp4'}]))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.prepare('mixed', '--model', 'MyModel', '--predictions', str(manifest))


if __name__ == '__main__':
    unittest.main()
