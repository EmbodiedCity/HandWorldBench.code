"""Bind a HandWorldBench runtime template to locally installed environments and assets."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
from string import Template
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from handworldbench.runtime.config import relative_runtime_config, resolve_runtime_config, xla_cuda_data_dir_flag

MODEL_DIRECTORIES = {'hand_detector', 'vision', 'judge', 'state_judge', 'da3', 'pavrm', 'sam3.source'}
RELEASE_PROCESS_CACHES = {
    'TMPDIR': 'tmp',
    'XDG_CACHE_HOME': 'xdg',
    'HF_HOME': 'huggingface',
    'TORCH_HOME': 'torch',
    'TRITON_CACHE_DIR': 'triton',
    'TORCHINDUCTOR_CACHE_DIR': 'torchinductor',
    'TORCH_EXTENSIONS_DIR': 'torch_extensions',
    'CUDA_CACHE_PATH': 'cuda',
    'NUMBA_CACHE_DIR': 'numba',
    'MPLCONFIGDIR': 'matplotlib',
}


def path_string(path):
    # Keep virtual-environment interpreter symlinks in the virtual environment.
    return str(Path(path).expanduser().absolute())


def expand(value, bindings):
    if isinstance(value, dict):
        return {key: expand(item, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, bindings) for item in value]
    if isinstance(value, str):
        return Template(value).substitute(bindings)
    return value


def override_asset(models, binding):
    name, separator, value = binding.partition('=')
    if not separator or not name or not value:
        raise ValueError(f'Expected --asset KEY=PATH, received {binding!r}')
    parts = name.split('.')
    target = models
    for part in parts[:-1]:
        target = target.get(part)
        if not isinstance(target, dict):
            raise ValueError(f'Unknown model asset {name!r}')
    if not isinstance(target.get(parts[-1]), str):
        raise ValueError(f'Unknown model asset {name!r}')
    target[parts[-1]] = path_string(value)


def check_paths(config):
    errors = []
    interpreters = {config['default_python'], *config.get('python', {}).values()}
    for python in sorted(interpreters):
        if not Path(python).is_file() or not os.access(python, os.X_OK):
            errors.append(f'Missing executable interpreter {python}')
    for program in ('ffmpeg', 'ffprobe'):
        if shutil.which(program) is None:
            errors.append(f'{program} is missing from PATH')
    for key in ('models_root', 'classic_data'):
        if not Path(config[key]).is_dir():
            errors.append(f'{key} is not a directory: {config[key]}')
    for group in ('python_paths', 'musiq_python_paths'):
        for path in config.get(group, []):
            if not Path(path).is_dir():
                errors.append(f'{group} entry is not a directory: {path}')
    for scope in (config.get('env', {}), *config.get('backend_env', {}).values()):
        for path in scope.get('LD_LIBRARY_PATH', '').split(os.pathsep):
            if path and not Path(path).is_dir():
                errors.append(f'LD_LIBRARY_PATH entry is not a directory: {path}')
        cuda_root = scope.get('CUDA_HOME')
        if cuda_root and not Path(cuda_root).is_dir():
            errors.append(f'CUDA_HOME is not a directory: {cuda_root}')
        for flag in shlex.split(scope.get('XLA_FLAGS', '')):
            if flag.startswith('--xla_gpu_cuda_data_dir=') and not Path(flag.partition('=')[2]).is_dir():
                errors.append(f'XLA CUDA toolkit is not a directory: {flag.partition("=")[2]}')
    models = config['models']
    paths = {key: value for key, value in models.items() if isinstance(value, str)}
    for key, value in models.items():
        if isinstance(value, dict):
            paths.update({f'{key}.{child}': path for child, path in value.items()})
    for name, path in paths.items():
        exists = Path(path).is_dir() if name in MODEL_DIRECTORIES else Path(path).is_file()
        if not exists:
            errors.append(f'Missing model asset {name}: {path}')
    if errors:
        raise ValueError('Runtime inputs are incomplete:\n  ' + '\n  '.join(errors))


def bind_release(args):
    release = args.release_root.resolve()
    interpreters = release / '.envs'
    args.models_root = args.models_root or release / 'models'
    args.cache_root = args.cache_root or release / 'cache'
    args.main_python = args.main_python or interpreters / 'main/bin/python'
    args.sam3_python = args.sam3_python or interpreters / 'sam3/bin/python'
    args.metrics_python = args.metrics_python or interpreters / 'metrics/bin/python'
    args.musiq_python = args.musiq_python or interpreters / 'musiq/bin/python'
    args.main_cuda_root = args.main_cuda_root or release / 'cuda/12.8'
    args.musiq_cuda_root = args.musiq_cuda_root or args.main_cuda_root
    args.musiq_library_dir = args.musiq_library_dir or [release / 'cudnn8/lib', release / 'cuda/11/lib64']
    return {
        'PYTHONUNBUFFERED': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
        **{name: str(release / 'runtime_cache' / subdir) for name, subdir in RELEASE_PROCESS_CACHES.items()},
    }


def build(args):
    release_env = bind_release(args) if args.profile == 'release' else None
    models_root = path_string(args.models_root)
    bindings = {
        'SOURCE_ROOT': path_string(args.source_root or ROOT),
        'MODELS_ROOT': models_root,
        'CLASSIC_DATA': path_string(args.classic_data or Path(models_root) / 'classic'),
        'CACHE_ROOT': path_string(args.cache_root),
        'MAIN_PYTHON': path_string(args.main_python),
        'SAM3_PYTHON': path_string(args.sam3_python or args.main_python),
        'METRICS_PYTHON': path_string(args.metrics_python or args.main_python),
        'MUSIQ_PYTHON': path_string(args.musiq_python),
    }
    config = expand(json.loads(args.template.read_text(encoding='utf-8')), bindings)
    if release_env:
        config['env'].update(release_env)
        config.pop('cpu_workers', None)
        config.pop('backend_resources', None)
    for asset in args.asset:
        override_asset(config['models'], asset)
    for name, paths in (('python_paths', args.main_python_path), ('musiq_python_paths', args.musiq_python_path)):
        if paths:
            config[name] = [path_string(path) for path in paths]
    if args.main_library_dir:
        config.setdefault('env', {})['LD_LIBRARY_PATH'] = os.pathsep.join(map(path_string, args.main_library_dir))
    if args.main_cuda_root:
        config.setdefault('env', {})['CUDA_HOME'] = path_string(args.main_cuda_root)
    musiq = config.setdefault('backend_env', {}).setdefault('musiq', {})
    if args.musiq_library_dir:
        libraries = list(map(path_string, args.musiq_library_dir))
        if release_env:
            libraries.extend(map(path_string, args.main_library_dir))
        musiq['LD_LIBRARY_PATH'] = os.pathsep.join(libraries)
    if args.musiq_cuda_root:
        musiq['XLA_FLAGS'] = xla_cuda_data_dir_flag(path_string(args.musiq_cuda_root))
    return relative_runtime_config(config, args.output.resolve().parent)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('custom', 'release'), default='custom')
    parser.add_argument('--release-root', type=Path, help='Repository root containing models/ and .envs/')
    parser.add_argument('--template', type=Path, default=ROOT / 'configs/runtime.example.json')
    parser.add_argument('--source-root', type=Path, help='Checkout or install root containing handworldbench/')
    parser.add_argument('--models-root', type=Path)
    parser.add_argument('--classic-data', type=Path, help='Defaults to <models-root>/classic')
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--main-python', type=Path)
    parser.add_argument('--sam3-python', type=Path, help='Segmentation and background workers')
    parser.add_argument('--metrics-python', type=Path, help='JEPA and hand morphology workers')
    parser.add_argument('--musiq-python', type=Path)
    parser.add_argument('--main-python-path', type=Path, action='append', default=[])
    parser.add_argument('--musiq-python-path', type=Path, action='append', default=[])
    parser.add_argument('--main-library-dir', type=Path, action='append', default=[])
    parser.add_argument('--main-cuda-root', type=Path, help='CUDA toolkit root for main workers')
    parser.add_argument('--musiq-library-dir', type=Path, action='append', default=[])
    parser.add_argument('--musiq-cuda-root', type=Path, help='CUDA toolkit root passed to XLA')
    parser.add_argument('--asset', metavar='KEY=PATH', action='append', default=[])
    parser.add_argument('--check', action='store_true', help='Check files, interpreter paths and video commands before writing')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.profile == 'release' and args.release_root is None:
        parser.error('--profile release requires --release-root')
    if args.profile == 'custom':
        required = ('models_root', 'cache_root', 'main_python', 'musiq_python')
        missing = ['--' + name.replace('_', '-') for name in required if getattr(args, name) is None]
        if missing:
            parser.error('Missing required arguments: ' + ', '.join(missing))
    try:
        config = build(args)
        if args.check:
            check_paths(resolve_runtime_config(config, args.output.resolve().parent))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(args.output)


if __name__ == '__main__':
    main()
