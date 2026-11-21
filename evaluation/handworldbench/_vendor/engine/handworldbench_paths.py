"""Configured resource and cache roots."""
from pathlib import Path
import json, os
ROOT=Path(__file__).resolve().parent

def hb_path(relative):
    relative=str(relative)
    exact={'runtime_data/classic':'HANDWORLDBENCH_CLASSIC_DATA_ROOT','resources/models':'HANDWORLDBENCH_MODEL_ROOT','.venvs':'HANDWORLDBENCH_ENV_ROOT'}
    if relative in exact:return str(Path(os.environ.get(exact[relative],ROOT/relative)))
    if relative.startswith('.venvs/'):
        suffix=relative[len('.venvs/'):]
        name, _, rest=suffix.partition('/')
        configured=os.environ.get('HANDWORLDBENCH_ENV_'+name.upper().replace('-','_'))
        return str((Path(configured) if configured else Path(os.environ.get('HANDWORLDBENCH_ENV_ROOT',ROOT/'.venvs'))/name)/rest)
    if relative.startswith('runtime_data/classic/'):
        suffix=relative[len('runtime_data/classic/'):]
        return str(Path(os.environ.get('HANDWORLDBENCH_CLASSIC_DATA_ROOT',ROOT/'runtime_data/classic'))/suffix)
    if relative.startswith('resources/models/'):
        suffix=relative[len('resources/models/'):]
        return str(Path(os.environ.get('HANDWORLDBENCH_MODEL_ROOT',ROOT/'resources/models'))/suffix)
    if Path(relative).is_absolute() or '..' in Path(relative).parts:raise ValueError('Unsafe release-relative path '+relative)
    return str(ROOT/relative)

def resolve(value):
    if isinstance(value,dict):return {k:resolve(v) for k,v in value.items()}
    if isinstance(value,list):return [resolve(v) for v in value]
    if isinstance(value,str) and value.startswith('@HANDWORLDBENCH_ROOT@/'):
        return hb_path(value[len('@HANDWORLDBENCH_ROOT@/'):])
    return value

def loads(text,*args,**kwargs):
    return resolve(json.loads(text,*args,**kwargs))

def load(file,*args,**kwargs):
    return resolve(json.load(file,*args,**kwargs))
