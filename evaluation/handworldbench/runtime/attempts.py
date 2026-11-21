"""Isolate native state outputs and evidence bindings between retries."""
import copy
from pathlib import Path

def state_attempt_config(config, output, attempt, video, metrics):
    if not isinstance(attempt,int) or attempt < 1:
        raise ValueError('Attempt must be a positive integer')
    folder=Path(output)/'state_attempts'/str(attempt)
    previous=Path(config['output_root'])
    def rebind(value):
        if isinstance(value,dict):return {k:rebind(v) for k,v in value.items()}
        if isinstance(value,list):return [rebind(v) for v in value]
        if isinstance(value,str):
            path=Path(value)
            if path.is_absolute() and path.is_relative_to(previous):
                return str(folder/'output'/path.relative_to(previous))
        return value
    result=rebind(copy.deepcopy(config))
    result['output_root']=str(folder/'output')
    result['cache_root']=str(folder/'evidence_cache')
    result['metrics']=list(metrics)
    for case in result['cases']:case['video']=str(video)
    return result,folder/'config.json'
