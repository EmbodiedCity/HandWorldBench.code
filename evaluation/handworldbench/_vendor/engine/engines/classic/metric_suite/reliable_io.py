"""Finite retries for transient OBS/FUSE I/O; never converts a failed read to success."""

# Resolve release-local configuration from this file, independently of the working directory.
import sys as _hb_sys
from pathlib import Path as _HbPath
_hb_root = next(p for p in _HbPath(__file__).resolve().parents if (p / "handworldbench_paths.py").is_file())
if str(_hb_root) not in _hb_sys.path: _hb_sys.path.insert(0, str(_hb_root))
from handworldbench_paths import hb_path as _hb_path, loads as _hb_loads, load as _hb_load
import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

RETRYABLE = {errno.EFAULT, errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.EAGAIN, errno.EBUSY}


def retry_io(operation, attempts=5):
    for index in range(attempts):
        try:
            return operation()
        except OSError as error:
            if error.errno not in RETRYABLE or index + 1 == attempts:
                raise
            time.sleep(min(0.25 * (2 ** index), 2.0))


def read_json(path):
    return _hb_loads(retry_io(lambda: Path(path).read_bytes()))


def is_file(path):
    return retry_io(lambda: Path(path).is_file())


def publish_bytes(destination, data):
    """Sequential write + readback hash + rename. Existing output is preserved on failure."""
    destination = Path(destination)
    retry_io(lambda: destination.parent.mkdir(parents=True, exist_ok=True))
    expected = hashlib.sha256(data).digest()
    temporary = destination.with_name(destination.name + '.part-' + uuid.uuid4().hex)
    def save():
        with temporary.open('wb') as stream:
            stream.write(data)
            stream.flush()
        if hashlib.sha256(temporary.read_bytes()).digest() != expected:
            raise OSError(errno.EIO, 'Published bytes failed readback verification', str(temporary))
        temporary.replace(destination)
    try:
        retry_io(save)
    finally:
        try:
            retry_io(lambda: temporary.unlink(missing_ok=True))
        except OSError:
            pass


def local_tmp_root():
    """Temporary files only; persistent weights, inputs, and final outputs retain their configured paths."""
    root = Path(os.environ.get('HANDWORLDBENCH_SCRATCH', str(Path(__import__('tempfile').gettempdir())/'handworldbench')))
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_image(path, mode=None):
    """Read the entire image with retries before Pillow decodes local bytes."""
    from io import BytesIO
    from PIL import Image
    data = retry_io(lambda: Path(path).read_bytes())
    with Image.open(BytesIO(data)) as image:
        return image.convert(mode) if mode is not None else image.copy()


def save_image(image, path, **options):
    """Use the same Pillow encoder/options, then publish sequential bytes."""
    from io import BytesIO
    from PIL import Image
    suffix = Path(path).suffix.lower()
    Image.init()
    image_format = options.pop('format', None) or Image.registered_extensions()[suffix]
    buffer = BytesIO()
    image.save(buffer, format=image_format, **options)
    data = buffer.getvalue()
    publish_bytes(path, data)
    return data
