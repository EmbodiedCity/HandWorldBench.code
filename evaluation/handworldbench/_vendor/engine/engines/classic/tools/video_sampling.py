"""Shared, versioned video sampling for image-quality metrics.

The module deliberately imports no model framework and supports Python 3.8+.
Only this module defines video decoding, time sampling and PNG encoding.
"""

# Resolve release-local configuration from this file, independently of the working directory.
import sys as _hb_sys
from pathlib import Path as _HbPath
_hb_root = next(p for p in _HbPath(__file__).resolve().parents if (p / "handworldbench_paths.py").is_file())
if str(_hb_root) not in _hb_sys.path: _hb_sys.path.insert(0, str(_hb_root))
from handworldbench_paths import hb_path as _hb_path, loads as _hb_loads, load as _hb_load
import hashlib
import errno
import tempfile
import time
import uuid
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

PROTOCOL_ID = 'handworldbench_uniform_source_png_v1'
DEFAULT_SAMPLE_COUNT = 16
TOOLS_ROOT = Path(_hb_path('engines/classic/MUSIQ/tools'))


def video_tool(name):
    path = TOOLS_ROOT / name
    if path.is_file() and os.access(str(path), os.X_OK):
        return str(path)
    located = shutil.which(name)
    if located:
        return located
    raise RuntimeError('Cannot find {}; ffmpeg/ffprobe are required.'.format(name))


def command(argv):
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError('{} failed:\n{}'.format(Path(argv[0]).name, result.stderr[-2500:]))
    return result


def probe_video(video):
    result = command([video_tool('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
                      '-show_entries', 'stream=width,height,duration,nb_frames,avg_frame_rate,start_time:format=duration',
                      '-of', 'json', str(video)])
    metadata = _hb_loads(result.stdout)
    if not metadata.get('streams'):
        raise RuntimeError('The input file has no readable video frames.')
    stream = metadata['streams'][0]
    duration = None
    for value in [stream.get('duration'), metadata.get('format', {}).get('duration')]:
        try:
            candidate = float(value)
            if math.isfinite(candidate) and candidate > 0:
                duration = candidate
                break
        except (TypeError, ValueError):
            pass
    if duration is None:
        raise RuntimeError('Cannot read the video duration; uniform sampling is impossible.')
    try:
        frame_count = int(stream.get('nb_frames', 0))
    except (TypeError, ValueError):
        frame_count = 0
    return {'duration_seconds': duration, 'width': stream.get('width'), 'height': stream.get('height'),
            'declared_frame_count': frame_count if frame_count > 0 else None,
            'avg_frame_rate': stream.get('avg_frame_rate'), 'original_start_time': stream.get('start_time')}


def _extract_frames_local(video, directory, requested=DEFAULT_SAMPLE_COUNT, metadata=None):
    from PIL import Image
    video, directory = Path(video), Path(directory)
    if requested < 1:
        raise ValueError('The number of sampled frames must be greater than zero.')
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.glob('frame_*.png')):
        raise RuntimeError('The frame directory already contains frames; samples from two runs cannot be mixed.')
    metadata = metadata if metadata is not None else probe_video(video)
    declared = metadata['declared_frame_count']
    limit = min(requested, declared) if declared else requested
    interval = metadata['duration_seconds'] / limit
    expression = "setpts=PTS-STARTPTS,select='gte(t,selected_n*{:.17g})',showinfo".format(interval)
    result = command([video_tool('ffmpeg'), '-hide_banner', '-nostdin', '-loglevel', 'info',
                      '-threads', '2', '-i', str(video), '-map', '0:v:0', '-an', '-sn', '-dn',
                      '-vf', expression, '-frames:v', str(limit), '-fps_mode', 'vfr',
                      '-c:v', 'png', '-pix_fmt', 'rgb24', '-compression_level', '3', '-threads', '2',
                      str(directory / 'frame_%06d.png')])
    paths = sorted(directory.glob('frame_*.png'))
    timestamps = [float(match.group(1)) for match in re.finditer(r'\bn:\s*\d+\s+pts:\s*\S+\s+pts_time:(\S+)', result.stderr)]
    if not paths:
        raise RuntimeError('No usable frames were decoded from the video.')
    if len(timestamps) < len(paths):
        raise RuntimeError('Cannot verify the sampled frame timestamps; scores are not saved.')
    timestamps = timestamps[:len(paths)]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise RuntimeError('Sampled source-frame timestamps are not increasing; scores are not saved.')
    frames = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            frames.append({'sample_index': index, 'path': path, 'frame_file': path.name,
                           'timestamp_seconds': timestamps[index], 'width': rgb.width, 'height': rgb.height,
                           'decoded_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()})
    sampling = {'protocol_id': PROTOCOL_ID,
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'requested_frame_count': requested, 'actual_frame_count': len(frames),
                'sampling_interval_seconds': interval,
                'rule': 'First decoded frame at or after each uniform time boundary, starting at 0; no duplicated source frames.',
                'frame_format': 'lossless PNG RGB24', 'resize': None, 'crop': None,
                'timestamps_relative_to': 'first decoded video frame', 'ffmpeg_filter': expression,
                'implementation_file': str(Path(__file__).resolve())}
    return frames, sampling


def _retry_storage(operation):
    retryable = {errno.EFAULT, errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.EAGAIN, errno.EBUSY}
    for attempt in range(5):
        try:
            return operation()
        except OSError as error:
            if error.errno not in retryable or attempt == 4:
                raise
            time.sleep(min(.25 * 2 ** attempt, 2.0))


def _publish_frame(source, destination):
    # Encode on local scratch, then sequentially publish each completed PNG.
    data = source.read_bytes()
    expected = hashlib.sha256(data).digest()
    partial = destination.with_name(destination.name + '.part-' + uuid.uuid4().hex)
    def publish():
        with partial.open('wb') as stream:
            stream.write(data)
            stream.flush()
        if hashlib.sha256(partial.read_bytes()).digest() != expected:
            raise OSError(errno.EIO, 'PNG readback hash mismatch', str(partial))
        partial.replace(destination)
    try:
        _retry_storage(publish)
    finally:
        try:
            _retry_storage(lambda: partial.unlink(missing_ok=True))
        except OSError:
            pass


def extract_frames(video, directory, requested=DEFAULT_SAMPLE_COUNT, metadata=None):
    # Sampling, RGB encoding and timestamps are unchanged. Only temporary I/O
    # moves off OBS/FUSE, which does not reliably support encoder access.
    directory = Path(directory)
    _retry_storage(lambda: directory.mkdir(parents=True, exist_ok=True))
    if _retry_storage(lambda: any(directory.glob('frame_*.png'))):
        raise RuntimeError('The frame directory already contains frames; samples from two runs cannot be mixed.')
    scratch = Path(os.environ.get('HANDWORLDBENCH_SCRATCH', str(Path(__import__('tempfile').gettempdir())/'handworldbench')))
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sample-', dir=scratch) as temporary:
        frames, sampling = _extract_frames_local(video, Path(temporary) / 'frames', requested, metadata)
        for frame in frames:
            destination = directory / frame['frame_file']
            _publish_frame(Path(frame['path']), destination)
            frame['path'] = destination
    sampling['encoding_io'] = 'local scratch encoding; sequential verified PNG publication'
    return frames, sampling
