import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from handworldbench.metrics.errors import execution_failures
from handworldbench.runtime.prepare import prepare
from handworldbench.runtime.storage import read, write
from handworldbench.runtime.devices import Device


class RuntimeContractTests(unittest.TestCase):
    def test_resume_preserves_attempt_history(self):
        from handworldbench.runtime.queue import connect,enqueue,retry_failed,ready_jobs
        with tempfile.TemporaryDirectory() as folder:
            with connect(folder) as connection:
                tasks=[dict(id='observation',backend='frames',dependencies=[]),
                       dict(id='score',backend='score',dependencies=['observation'])]
                enqueue(connection,tasks)
                failure=json.dumps({'status':'input_binding_error','reason':'mapping'})
                connection.execute("UPDATE jobs SET state='failed',attempt=3,result=? WHERE id='observation'",(failure,))
                connection.commit();ready_jobs(connection,'evaluate')
                response=json.loads(connection.execute("SELECT result FROM jobs WHERE id='score'").fetchone()[0])
                self.assertEqual(response['status'],'input_binding_error')
                self.assertEqual(set(retry_failed(connection)),{'observation','score'})
                row=connection.execute("SELECT * FROM jobs WHERE id='observation'").fetchone()
                self.assertEqual(row['attempt'],3)
                self.assertEqual(json.loads(row['payload'])['retry_ceiling'],6)

    def test_remaining_path_uses_downstream_work(self):
        from handworldbench.runtime.queue import remaining_paths
        tasks=[dict(id='decode',backend='frames',dependencies=[],estimated_seconds=2),
               dict(id='keypoints',backend='hands',dependencies=['decode'],estimated_seconds=5),
               dict(id='judge',backend='judge',dependencies=['decode'],estimated_seconds=20)]
        states={task['id']:'pending' for task in tasks}
        self.assertEqual(remaining_paths(tasks,states,{'judge':30})['decode'],32)
        states['judge']='done'
        self.assertEqual(remaining_paths(tasks,states,{'judge':30})['decode'],7)

    def test_dataset_relative_assets(self):
        from handworldbench.data.inputs import bind_assets
        root=Path.cwd() / 'dataset-fixture'
        value={'reference':{'video':'videos/a.mp4'},'geometry':{'path':'geometry/a.npz'},
               'views':[{'state_protocol':'annotations/p.json'}],
               'provenance':{'path':'archive_ref:abc/annotation.json'}}
        result=bind_assets(value,root)
        self.assertEqual(result['reference']['video'],str(root/'videos/a.mp4'))
        self.assertEqual(result['views'][0]['state_protocol'],str(root/'annotations/p.json'))
        self.assertEqual(result['provenance']['path'],'archive_ref:abc/annotation.json')

    def test_musiq_xla_flags_round_trip_with_spaces(self):
        from handworldbench.runtime.config import relative_runtime_config, resolve_runtime_config

        with tempfile.TemporaryDirectory(prefix='HandWorldBench moved ') as folder:
            root = Path(folder).resolve()
            toolkit = root / 'CUDA toolkit'
            # Older generated files shell-quoted the entire toolkit flag.
            sibling_flag = '--xla_dump_to="trace folder"'
            old = f"'--xla_gpu_cuda_data_dir={toolkit}' {sibling_flag}"
            config = {'backend_env': {'musiq': {'XLA_FLAGS': old}}}
            base = root / 'config'
            expected = f'--xla_gpu_cuda_data_dir="{toolkit}" {sibling_flag}'

            runtime = resolve_runtime_config(config, base)
            self.assertEqual(runtime['backend_env']['musiq']['XLA_FLAGS'], expected)

            portable = relative_runtime_config(runtime, base)
            self.assertEqual(
                portable['backend_env']['musiq']['XLA_FLAGS'],
                f'--xla_gpu_cuda_data_dir="../CUDA toolkit" {sibling_flag}',
            )
            self.assertEqual(resolve_runtime_config(portable, base), runtime)
            self.assertEqual(config['backend_env']['musiq']['XLA_FLAGS'], old)

    def test_runtime_builder_preserves_musiq_xla_toolkit_path_with_spaces(self):
        from handworldbench.runtime.config import resolve_runtime_config

        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='HandWorldBench release ') as folder:
            root = Path(folder).resolve()
            toolkit = root / 'CUDA toolkit'
            runtime = root / 'config' / 'runtime.json'
            subprocess.run([
                sys.executable, '-B', str(source / 'scripts/build_runtime_config.py'),
                '--models-root', str(root / 'models'), '--cache-root', str(root / 'cache'),
                '--main-python', sys.executable, '--musiq-python', sys.executable,
                '--musiq-cuda-root', str(toolkit), '--output', str(runtime),
            ], check=True, capture_output=True, text=True)

            generated = read(runtime)
            self.assertEqual(
                generated['backend_env']['musiq']['XLA_FLAGS'],
                '--xla_gpu_cuda_data_dir="../CUDA toolkit"',
            )
            bound = resolve_runtime_config(generated, runtime.parent)
            self.assertEqual(
                bound['backend_env']['musiq']['XLA_FLAGS'],
                f'--xla_gpu_cuda_data_dir="{toolkit}"',
            )

    def test_directory_protocol_does_not_guess_mapping(self):
        from handworldbench.data.inputs import validate_output_protocol
        row={'output_protocol':{'match_reference_dimensions':True,'match_reference_timeline':True},
             'reference':{'width':640,'height':480,'fps':30,'frame_count':31}}
        observed={'metadata':{'width':640,'height':480,'frame_count':31,'times':[0.,1.]}}
        validate_output_protocol(row,observed)
        observed['metadata']['width']=320
        with self.assertRaisesRegex(ValueError,'Input binding error'):
            validate_output_protocol(row,observed)

    def test_execution_error_is_distinct_from_unknown(self):
        self.assertEqual(execution_failures({'status': 'partial_evaluation_failure', 'units': [
            {'status': 'insufficient_evidence', 'score': 0., 'choice': 0}]}), [])
        errors = execution_failures({'score': 0., 'units': [
            {'observations': {'return': {'status': 'runtime_error', 'error': 'CUDA OOM'}}}]})
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['reason'], 'CUDA OOM')

    def test_annotation_changes_invalidate_only_dependent_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root/'video.mp4'; video.touch()
            reference = dict(video=str(video), sha256='reference', fps=30, frame_count=30)
            item = dict(case_id='case', category='human', reference=reference,
                        geometry={'path': 'geometry.npz'},
                        applicability={'pose_control': 'applicable', 'memory_consistency': 'applicable'},
                        annotations={'supplement': {'memory_consistency': {'inputs': [{'criterion': 'A'}]}}})
            write(root/'catalog.json', [item])
            write(root/'dataset.json', {'catalog': 'catalog.json', 'runtime': {'cache_root': str(root/'cache')}})
            write(root/'predictions.json', [dict(case_id='case', video=str(video), video_sha256='prediction',
                  time_mapping={'kind': 'identity'}, spatial_mapping={'kind': 'identity'})])
            args = SimpleNamespace(dataset=str(root), model='Model', predictions=str(root/'predictions.json'),
                                   videos=None, use_reference_videos=False, cases=None,
                                   metrics='pose_control,memory_consistency', config=None,
                                   gpus='0', run_dir=str(root/'first'))
            with patch('handworldbench.runtime.prepare.select_devices', return_value=[Device(0, 0, 'GPU-test', 100000, 100000)]):
                prepare(args)
                first = {r['backend']: r['id'] for r in read(root/'first/configs/tasks.json') if r['kind']=='observe'}
                item['annotations']['supplement']['memory_consistency']['inputs'][0]['criterion'] = 'B'
                write(root/'catalog.json', [item]); args.run_dir = str(root/'second')
                prepare(args)
                second = {r['backend']: r['id'] for r in read(root/'second/configs/tasks.json') if r['kind']=='observe'}
            self.assertNotEqual(first['memory_evidence'], second['memory_evidence'])
            for name in first.keys()-{'memory_evidence'}:
                self.assertEqual(first[name], second[name], name)


if __name__ == '__main__':
    unittest.main()
