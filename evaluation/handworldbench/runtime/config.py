"""Resolve portable runtime paths relative to the configuration file."""

import copy
import json
import os
from pathlib import Path
import shlex


PATH_LISTS = ('PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'HANDWORLDBENCH_LAB_LIBRARY_PREFIX')
PATH_VARIABLES = (
    'CUDA_HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TORCH_HOME',
    'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_EXTENSIONS_DIR',
    'CUDA_CACHE_PATH', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR',
    'HANDWORLDBENCH_CLASSIC_DATA_ROOT', 'HANDWORLDBENCH_MODEL_ROOT', 'HANDWORLDBENCH_ENV_ROOT',
)
XLA_CUDA_DATA_DIR = '--xla_gpu_cuda_data_dir='


def xla_cuda_data_dir_flag(value):
    """Quote the path value for XLA, which parses flags without a shell."""
    return _format_xla_flag(XLA_CUDA_DATA_DIR + str(value))


def _format_xla_flag(flag):
    name, separator, value = flag.partition('=')
    if not separator or not name.startswith('--'):
        return shlex.quote(flag)
    if any(character.isspace() or character in ('"', "'", '\\') for character in value):
        value = json.dumps(value, ensure_ascii=False)
    return name + separator + value


def resolve_runtime_config(config, base_dir):
    """Keep absolute paths and resolve declared relative paths from base_dir."""
    base = Path(base_dir).expanduser().absolute()

    def path(value):
        if not value:
            return value
        target = Path(value).expanduser()
        if target.is_absolute():
            return str(target)
        # Preserve the requested virtual-environment launcher.
        return os.path.abspath(base / target)

    result = _map_paths(config, path)
    result.get("metric_defaults", {}).pop("diagnostic", None)
    for options in result.get("observation_options", {}).values():
        options.pop("diagnostic", None)
    return result


def relative_runtime_config(config, base_dir):
    """Express absolute runtime paths as references to base_dir."""
    base = Path(base_dir).expanduser().absolute()

    def path(value):
        if not value or not Path(value).is_absolute():
            return value
        target = os.path.relpath(value, base)
        return target if '/' in target or target in ('.', '..') else './' + target

    return _map_paths(config, path)


def _map_paths(config, path):
    result = copy.deepcopy(config)

    def command(value):
        return path(value) if value and ('/' in value or value.startswith('~')) else value

    def model_paths(models):
        for key, value in models.items():
            if isinstance(value, dict):
                model_paths(value)
            elif isinstance(value, str):
                models[key] = path(value)

    def environment(env):
        for key in env:
            if key in PATH_VARIABLES or key.startswith('HANDWORLDBENCH_ENV_'):
                env[key] = path(env[key])
        if 'HANDWORLDBENCH_HAND_FINDER_PYTHON' in env:
            env['HANDWORLDBENCH_HAND_FINDER_PYTHON'] = command(env['HANDWORLDBENCH_HAND_FINDER_PYTHON'])
        for key in PATH_LISTS:
            if key in env:
                env[key] = os.pathsep.join(path(item) for item in env[key].split(os.pathsep))
        if XLA_CUDA_DATA_DIR in env.get('XLA_FLAGS', ''):
            parts = shlex.split(env['XLA_FLAGS'])
            env['XLA_FLAGS'] = ' '.join(
                xla_cuda_data_dir_flag(path(part[len(XLA_CUDA_DATA_DIR):]))
                if part.startswith(XLA_CUDA_DATA_DIR) else _format_xla_flag(part)
                for part in parts
            )

    for key in ('cache_root', 'models_root', 'classic_data'):
        if key in result:
            result[key] = path(result[key])
    for key in ('python_paths', 'musiq_python_paths'):
        if key in result:
            result[key] = [path(item) for item in result[key]]
    if 'default_python' in result:
        result['default_python'] = command(result['default_python'])
    for name, value in result.get('python', {}).items():
        result['python'][name] = command(value)
    model_paths(result.get('models', {}))
    environment(result.get('env', {}))
    for env in result.get('backend_env', {}).values():
        environment(env)
    return result
