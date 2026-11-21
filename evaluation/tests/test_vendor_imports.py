"""Every backend module the dispatcher loads must import under the dispatcher's search path."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / 'handworldbench'
ENGINE = PACKAGE / '_vendor/engine'
SEARCH = [PACKAGE / 'backends', PACKAGE / 'backends/regions_support', ENGINE / 'engines/classic/metric_suite',
          ENGINE / 'engines/classic', ENGINE, ENGINE / 'engines/new_aux', ENGINE / 'engines/state_four/runtime']
MODULES = ['handworldbench_paths', 'runtime_paths',
           'handworldbench_aux.core', 'handworldbench_aux.annotations', 'handworldbench_aux.background',
           'handworldbench_aux.judge', 'handworldbench_aux.vlm_metrics', 'handworldbench_aux.automatic',
           'handworldbench_aux.video',
           'state_eval.io', 'state_eval.memory_transport', 'state_eval.memory_protocol', 'state_eval.pipeline',
           'state_eval.state_score', 'state_eval.legal_termination',
           'handworldbench.backends.dispatch', 'handworldbench.backends.inference', 'handworldbench.backends.scoring',
           'handworldbench.backends.video_models', 'handworldbench.backends.regions']


class VendorImportTests(unittest.TestCase):
    def test_backend_modules_import(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'runtime.json'
            config.write_text(json.dumps({'runtime': {'cache_root': folder, 'models_root': folder}}))
            code = ('import importlib,sys\n'
                    f'sys.path[:0]={[str(p) for p in SEARCH]!r}\n'
                    'failed=[]\n'
                    f'for name in {MODULES!r}:\n'
                    '    try: importlib.import_module(name)\n'
                    '    except ModuleNotFoundError as exc:\n'
                    '        top=(exc.name or "").split(".")[0]\n'
                    '        if "bench" in top or top.startswith(("state_eval","runtime_paths")): failed.append((name,str(exc)))\n'
                    'print(failed); sys.exit(1 if failed else 0)\n')
            env = dict(os.environ, HANDWORLDBENCH_RUNTIME_CONFIG=str(config),
                       PYTHONPATH=str(PACKAGE.parent) + os.pathsep + os.environ.get('PYTHONPATH', ''))
            result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr[-2000:])


if __name__ == '__main__':
    unittest.main()
