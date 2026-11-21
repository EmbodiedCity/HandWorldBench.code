"""Immutable run context. Paths and task scope are configuration, not module globals."""
from dataclasses import dataclass
from pathlib import Path
from .io import read,sha
METRICS=('memory_consistency','state_progression','evolution_correctness','state_regulation')
@dataclass(frozen=True)
class EvaluationContext:
    config_path:Path
    config:dict
    output_root:Path
    cache_root:Path
    model_config_path:Path
    device:str
    @classmethod
    def load(cls,path,device=None):
        path=Path(path).resolve();cfg=read(path)
        def resolve(v):
            p=Path(v);return (p if p.is_absolute() else path.parent/p).resolve()
        metrics=cfg.get('metrics',list(METRICS))
        if not metrics or set(metrics)-set(METRICS):raise ValueError('Only the four declared metrics are supported')
        cases=cfg.get('cases')
        if not isinstance(cases,list) or not cases:raise ValueError('Configuration requires explicit nonempty cases')
        ids=[c['case_id'] for c in cases]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate case IDs')
        for c in cases:
            if Path(c['case_id']).name!=c['case_id'] or c['case_id'] in ('','.','..'):raise ValueError('Unsafe case ID')
            if not c.get('model',c.get('model_id')):raise ValueError('Case requires model identity')
        return cls(path,cfg,resolve(cfg['output_root']),resolve(cfg['cache_root']),resolve(cfg['model_config']),device or cfg.get('device','cuda:0'))
    def path(self,value):
        p=Path(value);return (p if p.is_absolute() else self.config_path.parent/p).resolve()
    def field_path(self,row,key):
        if not row.get(key):raise ValueError('Missing configured '+key+' for '+row['case_id'])
        return self.path(row[key])
    def case(self,row):
        c={k:row[k] for k in ('case_id','clip_id','video_sha256') if k in row}
        c.update(video=str(self.field_path(row,'video')),model_id=row.get('model',row.get('model_id')),
                 status=row.get('status','available'))
        for k in ('reference_video','reference_sha256'):
            if k in row:c[k]=str(self.path(row[k])) if k.endswith('video') else row[k]
        return c
    def state_spec(self,row):
        value=read(self.field_path(row,'state_spec'))
        if 'tasks' in value:
            values=[x for x in value['tasks'] if x['clip_id']==row['clip_id']]
            if len(values)!=1:raise ValueError('State spec must resolve uniquely')
            return values[0]
        if value.get('clip_id')!=row['clip_id']:raise ValueError('State spec clip differs')
        return value
    def memory_annotation(self,row):
        value=read(self.field_path(row,'memory_annotation'))
        if value.get('clip_id')!=row['clip_id']:raise ValueError('Memory annotation clip differs')
        return value
    @property
    def metrics(self):return tuple(self.config.get('metrics',METRICS))
    @property
    def fingerprint(self):return sha(self.config_path)

    def model_binding(self,verify_contents=False):
        from functools import lru_cache
        return model_binding(str(self.model_config_path))

_MODEL_BINDINGS={}
def model_binding(path):
    cfg=read(path);root=Path(cfg['model_manifest']['path']).resolve();key=str(root)
    if key not in _MODEL_BINDINGS:
        index=root/'model.safetensors.index.json'
        names=set(read(index)['weight_map'].values()) if index.is_file() else {'model.safetensors'}
        names.add('config.json');files={}
        for name in sorted(names):
            file=root/name
            if not file.is_file():raise FileNotFoundError(file)
            stat=file.stat();files[name]={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        _MODEL_BINDINGS[key]={'model_path':key,'files':files}
    return dict(_MODEL_BINDINGS[key],config_sha256=sha(path))
