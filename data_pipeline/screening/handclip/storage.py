import hashlib
import contextlib
import json
import os
import tempfile
import socket
import sys
import time
from pathlib import Path

# Bound retries when a mounted filesystem exposes incomplete content after rename.
RETRY_DELAYS_SECONDS = (.1, .2, .4, .8, 1.6, 3.2, 5., 5., 5., 5.)
RETRY_TIMEOUT_SECONDS = 30.0


class StorageVisibilityError(OSError):
    """A published file did not become byte-exact/readable within the retry budget."""


def _retry(operation, description):
    deadline = time.monotonic() + RETRY_TIMEOUT_SECONDS
    failure = None
    for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
        try:
            return operation()
        except (FileNotFoundError, StorageVisibilityError) as exc:
            failure = exc
        if attempt == len(RETRY_DELAYS_SECONDS):
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(RETRY_DELAYS_SECONDS[attempt], remaining))
    raise StorageVisibilityError(f'{description}: visibility retry budget exhausted: {failure}') from failure


def _read_visible_bytes(path, allow_empty=False):
    path = Path(path)
    def read():
        value = path.read_bytes()
        if not value and not allow_empty:
            raise StorageVisibilityError(f'{path}: temporarily empty file')
        if b'\0' in value:
            raise StorageVisibilityError(f'{path}: transient NUL bytes ({value.count(bytes([0]))}/{len(value)})')
        return value
    return _retry(read, f'read {path}')


def _verify_bytes(path, expected):
    path = Path(path)
    def verify():
        actual = path.read_bytes()
        if actual != expected:
            raise StorageVisibilityError(
                f'{path}: byte mismatch; expected {len(expected)} bytes/{hashlib.sha256(expected).hexdigest()}, '
                f'got {len(actual)} bytes/{hashlib.sha256(actual).hexdigest()}')
    _retry(verify, f'validate {path}')


def mirror_path(path):
    """Map an absolute target under an optional, independent recovery mirror.

    The mirror is evidence of intended bytes, never an implicit successful main
    publication and never substituted by read_json/read_jsonl.
    """
    configured = os.environ.get('HANDCLIP_STATE_MIRROR_DIR')
    if not configured:
        return None
    root = Path(configured)
    if not root.is_absolute():
        raise ValueError('HANDCLIP_STATE_MIRROR_DIR must be an absolute local path')
    source = Path(os.path.abspath(path))
    anchor = source.anchor
    if anchor in ('/', '\\'):
        prefix = 'posix'
    elif len(source.drive) == 2 and source.drive[1] == ':':
        prefix = 'drive_' + source.drive[0].upper()
    else:
        prefix = 'anchor_' + hashlib.sha256(anchor.encode()).hexdigest()[:16]
    mapped = root / prefix / Path(*source.parts[1:])
    if os.path.abspath(mapped) == os.path.abspath(source):
        raise ValueError('Recovery mirror cannot be the canonical destination')
    return mapped


def _replace_with_retry(source, destination):
    """Retry transient FSX EPERM without weakening publication validation."""
    deadline = time.monotonic() + RETRY_TIMEOUT_SECONDS
    for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            # Keep the original errno and both filenames on permanent failure.
            # Other OS errors are not evidence of the observed transient EPERM.
            remaining = deadline - time.monotonic()
            if attempt == len(RETRY_DELAYS_SECONDS) or remaining <= 0:
                raise
            time.sleep(min(RETRY_DELAYS_SECONDS[attempt], remaining))


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.partial', dir=path.parent)
    renamed = False
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _verify_bytes(temporary, data)
        _replace_with_retry(temporary, path)
        renamed = True
        _verify_bytes(path, data)
    finally:
        # The rename may already have succeeded; only this unique temporary path
        # is eligible for cleanup. Never remove/rewrite the destination on error.
        if not renamed and os.path.exists(temporary):
            os.unlink(temporary)


def atomic_bytes(path, payload: bytes):
    """Mirror intended bytes, then publish only after exact remote readback."""
    if not isinstance(payload, bytes):
        raise TypeError('atomic_bytes payload must be bytes')
    recovery = mirror_path(path)
    if recovery is not None:
        _atomic_bytes(recovery, payload)
    _atomic_bytes(path, payload)


def json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _json_bytes(data):
    # Serialize once, before changing either location. No NaN or partial JSON is
    # ever intentionally published, and both copies receive identical UTF-8.
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       default=json_default) + '\n').encode('utf-8')


def atomic_json(path, data):
    atomic_bytes(path, _json_bytes(data))


def publish_progress(path, data):
    """Publish only the non-authoritative progress snapshot, warning on I/O failure.

    Critical ledgers/manifests must continue to use atomic_json/atomic_bytes.
    The intended snapshot is mirrored before canonical publication as usual; a
    mirror is reported as retained only if its bytes are independently checked.
    Invalid payloads/configuration still raise instead of being called I/O loss.
    """
    path = Path(path)
    if path.name != 'progress.json':
        raise ValueError('publish_progress is restricted to progress.json')
    encoded = _json_bytes(data)
    recovery = mirror_path(path)
    try:
        atomic_bytes(path, encoded)
    except OSError as exc:
        mirrored = False
        if recovery is not None:
            try:
                mirrored = recovery.read_bytes() == encoded
            except OSError:
                pass
        warning = dict(event='progress_publication_failed', level='warning',
                       path=str(path), error_type=type(exc).__name__, error=str(exc),
                       errno=exc.errno, mirror_path=str(recovery) if recovery else None,
                       mirror_verified=mirrored, updated_at=utc_now(), pid=os.getpid())
        print(json.dumps(warning, ensure_ascii=False, allow_nan=False), file=sys.stderr, flush=True)
        return False
    return True


def read_json(path):
    raw = _read_visible_bytes(path)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f'{path}: damaged JSON; refusing silent recovery') from exc


def _parse_jsonl(raw, path):
    records = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f'{path}:{number}: damaged ledger; refusing silent recovery') from exc
        if not isinstance(value, dict):
            raise ValueError(f'{path}:{number}: ledger record must be an object')
        records.append(value)
    return records


def append_jsonl(path, record):
    path = Path(path)
    if not isinstance(record, dict):
        raise ValueError('ledger record must be an object')
    return append_segment(path, record)


def append_segment(path, payload):
    """Append one immutable object/batch segment using an exclusive index claim."""
    path = Path(path)
    records = payload if isinstance(payload, list) else [payload]
    if not isinstance(payload, (dict, list)) or any(not isinstance(item, dict) for item in records):
        raise ValueError('ledger segment must be an object or a list of objects')
    if path.exists():
        # Preserve an existing legacy ledger. Validate its complete original
        # contents and the exact appended bytes; do not migrate or truncate it.
        original = _read_visible_bytes(path, allow_empty=True)
        _parse_jsonl(original, path)
        encoded = ''.join(json.dumps(record, ensure_ascii=False, allow_nan=False, default=json_default) + '\n'
                          for record in records).encode('utf-8')
        addition = (b'\n' if original and not original.endswith(b'\n') else b'') + encoded
        expected = original + addition
        recovery = mirror_path(path)
        if recovery is not None:
            _atomic_bytes(recovery, expected)
        with path.open('ab') as stream:
            stream.write(addition)
            stream.flush()
            os.fsync(stream.fileno())
        _verify_bytes(path, expected)
        return
    # New ledgers use separately validated immutable JSON records. Persist the
    # exclusive index claim even after success, so a stale directory listing or
    # interrupted publication cannot cause a later append to overwrite a record.
    parts = path.with_suffix('.parts')
    parts.mkdir(parents=True, exist_ok=True)
    indices = []
    for item in parts.iterdir():
        stem = item.name.split('.')[0]
        if stem.isdigit():
            indices.append(int(stem))
    index = max(indices, default=-1) + 1
    while True:
        target = parts / f'{index:06d}.json'
        claim = parts / f'{index:06d}.claim'
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            index += 1
            continue
        os.close(fd)
        if target.exists():
            index += 1
            continue
        atomic_json(target, payload)
        return


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        parts = path.with_suffix('.parts')
        if not parts.exists():
            return []
        records = []
        files = list(parts.glob('*.json'))
        if any(not item.stem.isdigit() for item in files):
            raise ValueError(f'{parts}: unexpected ledger segment filename')
        for part in sorted(files, key=lambda item: int(item.stem)):
            value = read_json(part)
            # _batch historically writes an atomic batch list. New append_jsonl
            # writes one object per segment; both preserve their complete order.
            batch = value if isinstance(value, list) else [value]
            if any(not isinstance(record, dict) for record in batch):
                raise ValueError(f'invalid ledger segment: {part}')
            records.extend(batch)
        return records
    return _parse_jsonl(_read_visible_bytes(path, allow_empty=True), path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def clip_id(candidate, confighash):
    key = f"{candidate['source_path']}:{candidate['start_frame']}:{candidate['end_frame']}:{confighash}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def utc_now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


@contextlib.contextmanager
def exclusive_writer(run):
    """Share the production runner's atomic lock for review/report mutations."""
    lock = Path(run) / 'writer.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f'writer lock exists: {lock}; inspect recorded host/pid before recovery') from exc
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(dict(pid=os.getpid(), host=socket.gethostname(), started_at=utc_now()), stream)
        yield
    finally:
        lock.unlink(missing_ok=True)
