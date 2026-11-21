"""Scheduling only: no quality decision is reused from an upper bound."""
from .storage import read_json,sha256_file

def prioritize(files,cfg):
    setting=cfg.get('runtime',{}).get('source_priority')
    if not setting:return files
    if sha256_file(setting['path'])!=setting['sha256']:raise ValueError('priority index changed')
    bounds=read_json(setting['path'])
    def tier(path):
        bound=bounds.get(path,0)
        return next((i for i,n in enumerate((1121,801,561,321,161,81)) if bound>=n),6)
    # Stable within each length tier: retain existing session round-robin order.
    return sorted(files,key=tier)
