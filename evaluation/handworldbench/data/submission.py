"""Bind missing submission mappings using full-frame resize and uniform retiming."""
import copy
import math
import subprocess


PROTOCOL = 'full_frame_resize_uniform_time_v1'


def _metadata(path, sha256, cache_root):
    from handworldbench.observations.video import probe, stage
    try:
        if cache_root is not None:
            from handworldbench.runtime.storage import artifact_digest
            return stage(path, sha256 or artifact_digest(path), cache_root)[1]
        return probe(path)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        raise ValueError('Cannot read video for automatic input alignment: ' + str(path)) from exc


def _dimensions(meta):
    values = [meta.get('width'), meta.get('height')]
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError('Automatic resize requires positive integer video dimensions')
    return values


def _span(meta):
    times = meta.get('times', [])
    if (len(times) < 2 or len(times) != meta.get('frame_count') or
            any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in times) or
            any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError('Automatic time alignment requires at least two strictly ordered frame timestamps')
    return float(times[-1] - times[0])


def _native_preset(protocol):
    """Only the existing reference-native default may use the new fallback."""
    return (protocol is not None and
            protocol.get('time_mapping') == {'kind': 'identity'} and
            protocol.get('spatial_mapping') == {'kind': 'identity'} and
            protocol.get('match_reference_dimensions') is True and
            protocol.get('match_reference_timeline') is True)


def complete_mappings(row, protocol=None, *, cache_root=None):
    """Map the reference timestamp span onto the decoded prediction span."""
    missing = {field for field in ('spatial_mapping', 'time_mapping') if not row.get(field)}
    if not missing:
        return row
    row = copy.deepcopy(row)

    if protocol is not None and not _native_preset(protocol):
        for field in missing:
            if not protocol.get(field):
                raise ValueError('Declared output protocol is missing ' + field)
            row[field] = copy.deepcopy(protocol[field])
        binding = copy.deepcopy(protocol)
        if 'spatial_mapping' not in missing:
            binding.pop('match_reference_dimensions', None)
        if 'time_mapping' not in missing:
            binding.pop('match_reference_timeline', None)
        row['output_protocol'] = binding
        return row

    reference = row['reference']
    observed = _metadata(row['video'], row.get('video_sha256'), cache_root)
    source_wh = _dimensions(reference) if 'spatial_mapping' in missing else None
    target_wh = _dimensions(observed) if 'spatial_mapping' in missing else None
    binding = {'name': PROTOCOL}

    if 'spatial_mapping' in missing:
        if source_wh == target_wh:
            row['spatial_mapping'] = {'kind': 'identity'}
        else:
            sx, sy = (target_wh[j] / source_wh[j] for j in range(2))
            row['spatial_mapping'] = {
                'kind': 'affine',
                'matrix': [[sx, 0., .5 * sx - .5],
                           [0., sy, .5 * sy - .5],
                           [0., 0., 1.]],
                'resize_coordinate_convention': 'pixel_centers',
                'source_wh': source_wh, 'output_wh': target_wh,
                'provenance': PROTOCOL,
            }
        binding['expected_dimensions'] = target_wh

    native_timeline = False
    if 'time_mapping' in missing:
        source = _metadata(reference['video'], reference.get('sha256'), cache_root)
        source_span, target_span = _span(source), _span(observed)
        equal_span = math.isclose(source_span, target_span, rel_tol=1e-9, abs_tol=1e-9)
        if protocol is not None:
            expected_last = (reference['frame_count'] - 1) / reference['fps']
            native_timeline = (equal_span and observed['frame_count'] == reference['frame_count'] and
                               abs(observed['times'][-1] - expected_last) <= .5 / reference['fps'])
        if equal_span:
            row['time_mapping'] = {'kind': 'identity'}
        else:
            row['time_mapping'] = {
                'kind': 'affine', 'scale': target_span / source_span,
                'offset': 0., 'provenance': PROTOCOL,
            }
        binding['expected_frame_count'] = observed['frame_count']
        binding['expected_time_span_s'] = target_span

    # Preserve the native protocol for a matching viewport and timeline.
    native_dimensions = source_wh == target_wh
    if protocol is not None and all(
            native_dimensions if field == 'spatial_mapping' else native_timeline
            for field in missing):
        binding = copy.deepcopy(protocol)
        if 'spatial_mapping' not in missing:
            binding.pop('match_reference_dimensions', None)
        if 'time_mapping' not in missing:
            binding.pop('match_reference_timeline', None)
    row['output_protocol'] = binding
    return row


def validate_automatic_protocol(protocol, meta):
    """Confirm decoding still matches metadata used to freeze the fallback."""
    if protocol.get('name') != PROTOCOL:
        return
    if 'expected_dimensions' in protocol and _dimensions(meta) != protocol['expected_dimensions']:
        raise ValueError('Input binding error: video dimensions changed after automatic alignment')
    if 'expected_time_span_s' in protocol:
        if (meta['frame_count'] != protocol['expected_frame_count'] or
                not math.isclose(_span(meta), protocol['expected_time_span_s'], rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError('Input binding error: video timeline changed after automatic alignment')
