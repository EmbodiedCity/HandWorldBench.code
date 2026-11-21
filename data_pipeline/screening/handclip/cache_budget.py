"""FIFO, process-safe weighted scratch admission. Waiting never rejects data."""
import contextlib,json,os,shutil,subprocess,sys,tempfile,time,uuid
from pathlib import Path

def process_identity(pid):
    if sys.platform == 'darwin':
        stat = subprocess.run(['ps','-p',str(pid),'-o','lstart='],
                              capture_output=True,text=True,check=False)
        return stat.stdout.strip() or None
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None

@contextlib.contextmanager
def _locked(root):
    import fcntl
    root.mkdir(parents=True,exist_ok=True)
    with (root/'lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'reservations.json'
        state=json.loads(path.read_text()) if path.exists() else {'entries':[]}
        state['entries']=[e for e in state['entries'] if process_identity(e['pid'])==e['process_identity']]
        try:yield state
        finally:
            fd,name=tempfile.mkstemp(prefix='state-',dir=root)
            try:
                with os.fdopen(fd,'w') as stream:json.dump(state,stream);stream.flush();os.fsync(stream.fileno())
                os.replace(name,path)
            finally:
                if os.path.exists(name):os.unlink(name)
            fcntl.flock(lock,fcntl.LOCK_UN)

@contextlib.contextmanager
def reserve(cfg,requested_bytes):
    runtime=cfg.get('runtime',{})
    if not runtime.get('weighted_cache_admission'):
        yield
        return
    capacity=int(runtime['max_cache_gb']*1024**3)
    if type(requested_bytes) is not int or not 0<requested_bytes<=capacity:
        raise ValueError('invalid weighted scratch request')
    root=Path(runtime['cache_budget_root']);guard=int(runtime.get('cache_min_free_gib',16)*1024**3)
    token=uuid.uuid4().hex;pid=os.getpid();identity=process_identity(pid)
    if identity is None:raise RuntimeError('Cannot establish scratch owner identity')
    entry=dict(token=token,pid=pid,process_identity=identity,bytes=requested_bytes,status='waiting',queued_at=time.time())
    with _locked(root) as state:state['entries'].append(entry)
    try:
        while True:
            with _locked(root) as state:
                mine=next(e for e in state['entries'] if e['token']==token)
                waiting=[e for e in state['entries'] if e['status']=='waiting']
                used=sum(e['bytes'] for e in state['entries'] if e['status']=='active')
                # Conservatively count existing reservations again against free
                # space, covering allocations admitted but not written yet.
                free=shutil.disk_usage(root).free
                admitted=waiting[0]['token']==token and used+requested_bytes<=capacity and free>=guard+used+requested_bytes
                if admitted:mine.update(status='active',admitted_at=time.time())
            if admitted:break
            time.sleep(.2)
        yield
    finally:
        with _locked(root) as state:state['entries']=[e for e in state['entries'] if e['token']!=token]
