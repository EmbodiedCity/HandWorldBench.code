"""Atomic JSON records and content-addressed cache entries."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
from functools import lru_cache

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.part")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    os.replace(temp, path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()

def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

@lru_cache(maxsize=8192)
def _artifact_digest(path,version):
    return file_digest(path)

def artifact_digest(path):
    """Reuse a generated artifact digest until its filesystem version changes."""
    path=Path(path).resolve();stat=path.stat()
    version=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    return _artifact_digest(str(path),version)

@contextlib.contextmanager
def cache_lock(root, key=''):
    import fcntl
    folder = Path(root) / key
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield folder

