"""Mandatory recording-level benchmark isolation for batch 2 future searches."""
import hashlib,json
from pathlib import Path

DEFAULT_POLICY=Path(__file__).resolve().parents[2]/'recording_isolation/CONFIRMED_POLICY.json'

def load_policy(path=None):
    path=DEFAULT_POLICY if path is None else Path(path)
    policy=json.loads(path.read_text(encoding='utf-8'))
    if policy.get('scope')!='whole_recording' or policy.get('user_confirmed') is not True:
        raise ValueError('Recording-level isolation has not been confirmed')
    raw=(path.parent/Path(policy['recordings_file'])).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=policy['recordings_sha256']:
        raise ValueError('Recording isolation input checksum mismatch')
    rows=json.loads(raw.decode('utf-8'))
    if not isinstance(rows,list) or not rows or not all(isinstance(x,str) and x for x in rows) or len(rows)!=len(set(rows)):
        raise ValueError('Invalid isolated recording IDs')
    return frozenset(rows),policy['policy_id']

def rejection(row,blocked):
    return 'benchmark_recording_isolated' if row.get('group') in blocked else None
