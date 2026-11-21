
# Resolve release-local configuration from this file, independently of the working directory.
import sys as _hb_sys
from pathlib import Path as _HbPath
_hb_root = next(p for p in _HbPath(__file__).resolve().parents if (p / "handworldbench_paths.py").is_file())
if str(_hb_root) not in _hb_sys.path: _hb_sys.path.insert(0, str(_hb_root))
from handworldbench_paths import hb_path as _hb_path, loads as _hb_loads, load as _hb_load
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return _hb_loads(Path(p).read_text())
from functools import lru_cache
def sha(p):
 p=Path(p);stat=p.stat();return _cached_sha(str(p),stat.st_size,stat.st_mtime_ns)

@lru_cache(maxsize=8192)
def _cached_sha(p,size,mtime):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def save(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(p)
