from __future__ import annotations

# Resolve release-local configuration from this file, independently of the working directory.
import sys as _hb_sys
from pathlib import Path as _HbPath
_hb_root = next(p for p in _HbPath(__file__).resolve().parents if (p / "handworldbench_paths.py").is_file())
if str(_hb_root) not in _hb_sys.path: _hb_sys.path.insert(0, str(_hb_root))
from handworldbench_paths import hb_path as _hb_path, loads as _hb_loads, load as _hb_load
import hashlib,json,os,sys,importlib.util,types,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from handworldbench.backends.context import CONFIG
DATA=Path(CONFIG['cache_root'])/'judgments'
RUNTIME=DATA
from handworldbench.registry import METRICS as REGISTRY
METRICS={m:(REGISTRY[m].name,'WBench',REGISTRY[m].protocol) for m in ('photometric_consistency','action_adherence','event_adherence','causal_plausibility','visual_plausibility')}
ANNOTATED={'action_adherence','event_adherence','state_progression','evolution_correctness','memory_consistency','state_regulation'}
BACKGROUND={'photometric_consistency'}
VLM={'action_adherence','event_adherence','causal_plausibility'}
PHYSICS_IDS={'collision','surface_interaction','deformation','wind_weather','reflection','human_physics','fluid_dynamics'}
class MetricIssue(Exception):
 def __init__(self,status,reason,details=None):super().__init__(reason);self.status=status;self.reason=reason;self.details=details or {}
def read_json(path):
 with Path(path).open(encoding='utf-8-sig') as f:return _hb_load(f)
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def writable(path):
 p=Path(path).resolve()
 return p
def write_json(path,value):
 p=writable(path);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def configure_runtime():
 sys.dont_write_bytecode=True
 for key,rel in [('HF_HOME','huggingface'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg'),('TRITON_CACHE_DIR','triton'),('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TORCH_EXTENSIONS_DIR','extensions'),('CUDA_CACHE_PATH','cuda'),('HF_DATASETS_CACHE','datasets')]:
  p=RUNTIME/'cache'/rel;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
 import tempfile
 temp=RUNTIME/'tmp';temp.mkdir(parents=True,exist_ok=True);os.environ['TMPDIR']=str(temp);tempfile.tempdir=str(temp)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
def load_vendor(relative,name):
 path=ROOT/'vendor'/relative
 if not path.is_file():raise MetricIssue('missing_dependency','Missing vendored source '+str(path))
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def extract_symbols(relative,names):
 # Load the model preprocessing functions without initializing API clients.
 import ast
 path=ROOT/'vendor'/relative;tree=ast.parse(path.read_text());nodes=[]
 for node in tree.body:
  bindings=[]
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)):bindings=[node.name]
  elif isinstance(node,ast.Assign):bindings=[t.id for t in node.targets if isinstance(t,ast.Name)]
  if any(n in names for n in bindings):nodes.append(node)
 selected=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
 ns={};exec(compile(ast.fix_missing_locations(selected),str(path),'exec'),ns)
 return ns

def resolve_config(path):
 p=Path(path).resolve();c=read_json(p)
 if c.get('schema_version')!=1:raise MetricIssue('invalid_input','Run config schema_version must be 1')
 base=p.parent
 for key in ['video','reference_video','annotation','output_root']:
  if c.get(key):c[key]=str((base/Path(c[key])).resolve())
 for key in ['masks_manifest','depth_manifest']:
  if c.get('background',{}).get(key):c['background'][key]=str((base/Path(c['background'][key])).resolve())
 for k,v in c.get('models',{}).items():
  if isinstance(v,str):c['models'][k]=str((base/Path(v)).resolve())
 if not c.get('clip_id') or not c.get('model_id') or not c.get('video'):raise MetricIssue('invalid_input','clip_id, model_id and video are required')
 if not Path(c['video']).is_file():raise MetricIssue('invalid_input','Video missing: '+c['video'])
 c['_config_path']=str(p)
 return c

def model_path(cfg,key):
 defaults={'qwen':'Qwen3.5-9B','da3':'DA3-GIANT-1.1','amt':'amt/amt-s.pth','vp':'qwen3vl-a3b-visual-plausibility'}
 path=Path(cfg.get('models',{}).get(key,DATA/'weights'/defaults[key]))
 if not path.exists():raise MetricIssue('missing_dependency','Missing model '+str(path))
 if path.is_dir():
  idx=path/'model.safetensors.index.json'
  required=set(read_json(idx)['weight_map'].values()) if idx.exists() else {'model.safetensors'}
  required.add('config.json')
  if key in ['qwen','vp']:required.update(['tokenizer_config.json','preprocessor_config.json','chat_template.jinja'])
  missing=[n for n in required if not (path/n).is_file()]
  if missing:raise MetricIssue('missing_dependency','Incomplete model weights',{'path':str(path),'missing':missing})
 manifest=ROOT/'audit/weight_manifest_planned.json'
 if manifest.is_file():
  for item in read_json(manifest):
   local=Path(item['local'])
   if (local==path or (path.is_dir() and local.parent==path)) and item.get('size') and (not local.is_file() or local.stat().st_size!=item['size']):raise MetricIssue('missing_dependency','Model file not complete: '+str(local))
 return path

def result(metric,status='ok',score=None,details=None,reason=None):
 name,source,protocol=METRICS[metric]
 if score is not None and (not isinstance(score,(int,float)) or not math.isfinite(score)):raise ValueError('Score must be finite or null')
 return {'metric':metric,'name':name,'source':source,'protocol':protocol,'status':status,'score':score,'direction':'higher','reason':reason,'details':details or {}}


_VERIFIED_MODELS={}
def stage_model(cfg,key):
 """Validate a configured model once per resident process."""
 source=model_path(cfg,key).resolve()
 _VERIFIED_MODELS.setdefault(str(source),True)
 return source


def audit_imports():
 paths={name:str(Path(module.__file__).resolve()) for name,module in list(sys.modules.items()) if getattr(module,'__file__',None)}
 vendored=('sam3','depth_anything_3','wbench')
 forbidden={k:v for k,v in paths.items() if k.split('.')[0] in vendored and not Path(v).is_relative_to(ROOT)}
 if forbidden:raise MetricIssue('runtime_error','Vendored module resolved outside release',{'forbidden_imports':forbidden})
 return {'forbidden_imports':forbidden,'module_paths':paths,'executable':sys.executable,'sys_path':sys.path,'python_dont_write_bytecode':sys.dont_write_bytecode}
