"""Validate this exporter's non-fragmented, single-video MP4 sample timeline.

Only headers and the bounded moov box are read; mdat payloads are skipped. RGB
decoding and source-pixel equality remain independent validation requirements.
Nonzero composition offsets/B-frame reordering and nontrivial edit lists are
deliberately unsupported, rather than inferred from nominal frame-rate metadata.
"""
from fractions import Fraction
import hashlib
import math
from numbers import Integral
from pathlib import Path
import struct


MAX_MOOV_BYTES = 32 * 1024 * 1024


class MediaTimingError(ValueError):
    """Malformed, unsupported or temporally inconsistent MP4 export."""


def _error(message):
    raise MediaTimingError(message)


def _uint(data, offset, size=4, signed=False):
    if offset < 0 or offset + size > len(data):
        _error('truncated MP4 timing field')
    return int.from_bytes(data[offset:offset + size], 'big', signed=signed)


def _boxes(data, label):
    """Yield bounded immediate-child payload views; no media decoding."""
    position = 0
    while position < len(data):
        if len(data) - position < 8:
            _error(f'{label}: truncated atom header')
        size = _uint(data, position)
        kind = bytes(data[position + 4:position + 8])
        header = 8
        if size == 1:
            size = _uint(data, position + 8, 8)
            header = 16
        elif size == 0:
            size = len(data) - position
        if size < header or position + size > len(data):
            _error(f'{label}: invalid or out-of-bounds {kind!r} atom size')
        yield kind, data[position + header:position + size]
        position += size


def _one(children, name, label, optional=False):
    matches = [payload for kind, payload in children if kind == name]
    if len(matches) != 1:
        if optional and not matches:
            return None
        _error(f'{label}: expected exactly one {name.decode("ascii")} atom, found {len(matches)}')
    return matches[0]


def _fullbox(data, label, versions=(0,)):
    if len(data) < 4:
        _error(f'{label}: truncated full-box header')
    version = data[0]
    if version not in versions or bytes(data[1:4]) != b'\0\0\0':
        _error(f'{label}: unsupported version or flags')
    return version


def _clock(data, label):
    version = _fullbox(data, label, (0, 1))
    minimum = (100 if version == 0 else 112) if label == 'mvhd' else (24 if version == 0 else 36)
    if len(data) < minimum:
        _error(f'{label}: truncated clock header')
    offset = 12 if version == 0 else 20
    scale = _uint(data, offset)
    width = 4 if version == 0 else 8
    duration = _uint(data, offset + 4, width)
    if scale <= 0 or duration <= 0 or duration == (1 << (width * 8)) - 1:
        _error(f'{label}: zero/unknown timescale or duration')
    return scale, duration


def _read_moov(path):
    moov = None
    media_count = 0
    has_ftyp = False
    with Path(path).open('rb') as stream:
        stream.seek(0, 2)
        total = stream.tell()
        position = 0
        while position < total:
            stream.seek(position)
            header = stream.read(8)
            if len(header) != 8:
                _error('truncated top-level MP4 atom header')
            size, kind = struct.unpack('>I4s', header)
            header_size = 8
            if size == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    _error('truncated extended MP4 atom size')
                size = struct.unpack('>Q', extended)[0]
                header_size = 16
            elif size == 0:
                size = total - position
            if size < header_size or position + size > total:
                _error(f'invalid or out-of-bounds top-level {kind!r} atom size')
            if kind == b'moof':
                _error('fragmented MP4/moof is unsupported')
            if kind == b'ftyp':
                if has_ftyp or size - header_size < 8:
                    _error('missing contents or duplicate ftyp atom')
                has_ftyp = True
            elif kind == b'mdat':
                media_count += 1
                if size == header_size:
                    _error('empty mdat media payload')
            elif kind == b'moov':
                if moov is not None:
                    _error('duplicate moov atom')
                if size > MAX_MOOV_BYTES:
                    _error('moov exceeds bounded metadata read limit')
                moov = stream.read(size - header_size)
                if len(moov) != size - header_size:
                    _error('truncated moov payload')
            position += size
    if not has_ftyp or moov is None or media_count == 0:
        _error('MP4 requires ftyp, moov and nonempty mdat')
    return moov


def _run_table(payload, label, signed_second=False):
    version = _fullbox(payload, label, (0, 1) if label == 'ctts' else (0,))
    count = _uint(payload, 4)
    if count == 0 or len(payload) != 8 + count * 8:
        _error(f'{label}: empty, truncated or inconsistent entry count')
    entries = []
    for index in range(count):
        samples = _uint(payload, 8 + index * 8)
        value = _uint(payload, 12 + index * 8, signed=signed_second and version == 1)
        if samples == 0:
            _error(f'{label}: zero sample count in run')
        entries.append((samples, value))
    return entries


def _near(actual, expected, tolerance, label):
    if abs(actual - expected) > tolerance:
        _error(f'{label}: {float(actual):.12g}s differs from expected '
               f'{float(expected):.12g}s (tolerance {float(tolerance):.12g}s)')


def validate_video_timing(path, expected_frames, expected_fps):
    """Return JSON-safe timing evidence or fail closed with MediaTimingError.

    Export media timing may differ by at most one media clock tick, capped at
    1% of a frame period (rejecting insufficiently precise timebases). Movie/edit
    declarations use their own clock's one-tick rounding allowance. Frame count
    and positive sample durations are exact requirements, without tolerances.
    """
    if isinstance(expected_frames, bool) or not isinstance(expected_frames, Integral) or expected_frames < 1:
        _error('expected_frames must be a positive integer')
    expected_frames = int(expected_frames)
    try:
        if isinstance(expected_fps, bool):
            raise ValueError('boolean fps')
        fps = Fraction(str(expected_fps))
        if fps <= 0 or not math.isfinite(float(fps)):
            raise ValueError('nonpositive/nonfinite fps')
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise MediaTimingError('expected_fps must be a finite positive number or rational') from exc
    period, expected_duration = 1 / fps, expected_frames / fps
    raw = _read_moov(path)
    moov = list(_boxes(memoryview(raw), 'moov'))
    if any(kind == b'mvex' for kind, _ in moov):
        _error('fragmented MP4/mvex is unsupported')
    movie_scale, movie_duration = _clock(_one(moov, b'mvhd', 'moov'), 'mvhd')
    tracks = [payload for kind, payload in moov if kind == b'trak']
    if len(tracks) != 1:
        _error('only one video track with no additional tracks is supported')
    track = list(_boxes(tracks[0], 'trak'))
    tkhd = _one(track, b'tkhd', 'trak')
    # tkhd uses enabled/in-movie flags, unlike other timing full boxes.
    if len(tkhd) < 4 or tkhd[0] not in (0, 1):
        _error('tkhd: unsupported version or truncated header')
    version = tkhd[0]
    if len(tkhd) < (84 if version == 0 else 96):
        _error('tkhd: truncated track header')
    if _uint(tkhd, 1, 3) not in (1, 3, 7):
        _error('tkhd: disabled track or unsupported flags')
    track_id = _uint(tkhd, 12 if version == 0 else 20)
    track_duration = _uint(tkhd, 20 if version == 0 else 28, 4 if version == 0 else 8)
    if track_id <= 0 or track_duration <= 0:
        _error('tkhd: zero track ID or duration')
    mdia = list(_boxes(_one(track, b'mdia', 'trak'), 'mdia'))
    hdlr = _one(mdia, b'hdlr', 'mdia')
    _fullbox(hdlr, 'hdlr')
    if len(hdlr) < 24 or bytes(hdlr[8:12]) != b'vide':
        _error('only a video handler is supported')
    media_scale, media_duration = _clock(_one(mdia, b'mdhd', 'mdia'), 'mdhd')
    minf = list(_boxes(_one(mdia, b'minf', 'mdia'), 'minf'))
    stbl = list(_boxes(_one(minf, b'stbl', 'minf'), 'stbl'))
    stts = _run_table(_one(stbl, b'stts', 'stbl'), 'stts')
    stsz = _one(stbl, b'stsz', 'stbl')
    _fullbox(stsz, 'stsz')
    sample_size, sample_count = _uint(stsz, 4), _uint(stsz, 8)
    if len(stsz) != 12 + (0 if sample_size else sample_count * 4):
        _error('stsz: truncated or inconsistent sample sizes')
    if sample_size == 0 and any(_uint(stsz, 12 + i * 4) <= 0 for i in range(sample_count)):
        _error('stsz: zero-sized video sample')
    if sample_count != expected_frames or sum(count for count, _ in stts) != expected_frames:
        _error('sample count mismatch between stts/stsz and expected_frames')
    ctts = _one(stbl, b'ctts', 'stbl', optional=True)
    if ctts is not None:
        composition = _run_table(ctts, 'ctts', signed_second=True)
        if sum(count for count, _ in composition) != expected_frames:
            _error('ctts sample count mismatch')
        if any(offset != 0 for _, offset in composition):
            _error('nonzero ctts/B-frame composition reordering is unsupported')
    tolerance = min(Fraction(1, media_scale), period / 100)
    clock, frame_index = 0, 0
    max_pts_error = Fraction(0)
    frame_duration_errors = []
    for count, duration in stts:
        if duration <= 0:
            _error('stts: every sample duration must be positive, including the last sample')
        delta = Fraction(duration, media_scale)
        _near(delta, period, tolerance, 'nonuniform sample duration')
        frame_duration_errors.append(abs(delta - period))
        # Within each constant-duration run, PTS error is linear. Both endpoints
        # bound every interior frame, without allocating per-frame arrays.
        for within in (0, count - 1):
            actual = Fraction(clock + within * duration, media_scale)
            expected = (frame_index + within) / fps
            error = abs(actual - expected)
            _near(actual, expected, tolerance, 'nonuniform presentation timestamps')
            max_pts_error = max(max_pts_error, error)
        clock += count * duration
        frame_index += count
    actual_duration = Fraction(clock, media_scale)
    _near(actual_duration, expected_duration, tolerance, 'sample presentation duration')
    if media_duration != clock:
        _error('mdhd duration disagrees with the complete stts sample timeline')
    movie_tolerance = Fraction(1, movie_scale)
    _near(Fraction(movie_duration, movie_scale), expected_duration, movie_tolerance, 'mvhd movie duration')
    _near(Fraction(track_duration, movie_scale), expected_duration, movie_tolerance, 'tkhd track duration')
    if track_duration != movie_duration:
        _error('single-video movie and track durations disagree')
    edts = _one(track, b'edts', 'trak', optional=True)
    edit = None
    if edts is not None:
        elst = _one(list(_boxes(edts, 'edts')), b'elst', 'edts')
        version = _fullbox(elst, 'elst', (0, 1))
        entry_size = 12 if version == 0 else 20
        if _uint(elst, 4) != 1 or len(elst) != 8 + entry_size:
            _error('only one identity edit-list segment is supported')
        width = 4 if version == 0 else 8
        edit_duration = _uint(elst, 8, width)
        media_time = _uint(elst, 8 + width, width, signed=True)
        rate_integer = _uint(elst, 8 + 2 * width, 2, signed=True)
        rate_fraction = _uint(elst, 10 + 2 * width, 2, signed=True)
        if media_time != 0 or rate_integer != 1 or rate_fraction != 0:
            _error('nonzero/empty edit offsets or non-unit playback rates are unsupported')
        if edit_duration != track_duration:
            _error('elst duration disagrees with the declared track duration')
        _near(Fraction(edit_duration, movie_scale), expected_duration, movie_tolerance, 'elst duration')
        edit = dict(segment_duration_ticks=edit_duration, media_time_ticks=media_time, rate=1)
    last_duration = Fraction(stts[-1][1], media_scale)
    return dict(schema_version=1, status='ok', container='nonfragmented_single_video_mp4',
                expected_frames=expected_frames, sample_count=sample_count,
                expected_fps=float(fps), expected_fps_fraction=str(fps),
                actual_average_fps=float(expected_frames / actual_duration),
                actual_average_fps_fraction=str(expected_frames / actual_duration),
                expected_duration_s=float(expected_duration), presentation_duration_s=float(actual_duration),
                first_pts_s=0., last_pts_s=float(actual_duration - last_duration),
                last_sample_duration_s=float(last_duration), min_sample_duration_s=min(d for _, d in stts) / media_scale,
                max_sample_duration_s=max(d for _, d in stts) / media_scale,
                max_pts_error_s=float(max_pts_error), max_sample_duration_error_s=float(max(frame_duration_errors)),
                media_tolerance_s=float(tolerance), movie_tolerance_s=float(movie_tolerance),
                media_timescale=media_scale, media_duration_ticks=media_duration,
                movie_timescale=movie_scale, movie_duration_ticks=movie_duration,
                track_duration_ticks=track_duration, track_id=track_id, stts_entry_count=len(stts),
                composition_offsets='all_zero' if ctts is not None else 'absent', edit_list=edit,
                moov_sha256=hashlib.sha256(raw).hexdigest())
