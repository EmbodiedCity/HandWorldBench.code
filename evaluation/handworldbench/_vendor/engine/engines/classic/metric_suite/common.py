"""Shared model source lookup, numeric summaries and atomic outputs."""
from pathlib import Path
import ast,hashlib,importlib.util,json,math,sys
from functools import lru_cache
from runtime_paths import FINAL_ENGINE,FINAL_DATA,FINAL_RESULTS,FINAL_SUITE
from handworldbench.runtime.storage import write as _write
CODE=FINAL_SUITE
DATA=FINAL_DATA/'metric_suite'
RESULTS=FINAL_RESULTS
@lru_cache(maxsize=4096)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def clean(value):
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if hasattr(value,'tolist'):return clean(value.tolist())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value
def write_json(path,value):_write(path,clean(value))
def official(name,relative):
    path=CODE/'sources'/name/relative
    if not path.is_file():raise FileNotFoundError(path)
    return path
def definitions(name,relative,names,namespace):
    path=official(name,relative);tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Missing model source definition')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace
def provenance(names):
    metadata=json.loads((CODE/'source_versions.json').read_text())
    return [dict(model=name,**metadata.get(name,{})) for name in names]
def sampler():
    key='_handworldbench_video_sampling'
    if key not in sys.modules:
        path=FINAL_ENGINE/'tools/video_sampling.py'
        spec=importlib.util.spec_from_file_location(key,path);module=importlib.util.module_from_spec(spec)
        sys.modules[key]=module;spec.loader.exec_module(module)
    return sys.modules[key]
def statistics(values):
 import numpy as np
 a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
 if not len(a):return {'count':0,'mean':None}
 return {'count':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
         'std':float(a.std()),'min':float(a.min()),'max':float(a.max()),
         'p05':float(np.percentile(a,5)),'p95':float(np.percentile(a,95))}
