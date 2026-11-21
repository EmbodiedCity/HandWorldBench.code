"""One strict, non-mutating frame-length contract for every production gate.

Configuration is never rounded or coerced here. Intervals are half-open source
frame indices; a fixed length is an additional constraint, not a max override.
"""
from collections.abc import Mapping


class LengthContractError(ValueError):
    """A configuration or candidate violates the requested frame lengths."""


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise LengthContractError(f'{name} must be an integer >= {minimum}; bool/float/string are not accepted')
    return value


def validate_selection(selection):
    """Reject invalid lengths without changing the supplied configuration."""
    if not isinstance(selection, Mapping):
        raise LengthContractError('selection must be a mapping')
    multiple = _integer(selection.get('frame_multiple', 81), 'frame_multiple')
    offset = _integer(selection.get('frame_offset', 0), 'frame_offset', 0)
    if offset >= multiple:
        raise LengthContractError('frame_offset must be smaller than frame_multiple')
    minimum = _integer(selection.get('min_frames', multiple + offset), 'min_frames')
    if (minimum - offset) % multiple:
        raise LengthContractError('min_frames minus frame_offset must be divisible by frame_multiple')
    maximum, exact = selection.get('max_frames'), selection.get('exact_frames')
    for name, value in (('max_frames', maximum), ('exact_frames', exact)):
        if value is None:
            continue
        _integer(value, name)
        if (value - offset) % multiple:
            raise LengthContractError(f'{name} minus frame_offset must be divisible by frame_multiple')
        if value < minimum:
            raise LengthContractError(f'{name} must be >= min_frames')
    if exact is not None and maximum is not None and exact > maximum:
        raise LengthContractError('exact_frames exceeds max_frames')


def validate_interval(candidate, selection, *, source_frames=None):
    """Require a legal [start,end) and all min/max/multiple/exact constraints.

    An explicit source_frames is authoritative; otherwise an available source
    n field provides an extra bound. Missing source n does not fabricate one.
    """
    validate_selection(selection)
    if not isinstance(candidate, Mapping):
        raise LengthContractError('candidate must be a mapping')
    start = _integer(candidate.get('start_frame'), 'start_frame', 0)
    end = _integer(candidate.get('end_frame'), 'end_frame')
    length = _integer(candidate.get('n_frames'), 'n_frames')
    if end - start != length:
        raise LengthContractError('end_frame - start_frame must equal n_frames')
    multiple = selection.get('frame_multiple', 81)
    if (length - selection.get('frame_offset', 0)) % multiple:
        raise LengthContractError('n_frames minus frame_offset must be divisible by frame_multiple')
    if length < selection.get('min_frames', multiple):
        raise LengthContractError('n_frames is smaller than min_frames')
    if selection.get('max_frames') is not None and length > selection['max_frames']:
        raise LengthContractError('n_frames exceeds max_frames')
    if selection.get('exact_frames') is not None and length != selection['exact_frames']:
        raise LengthContractError('n_frames differs from exact_frames')
    total = candidate.get('n') if source_frames is None else source_frames
    if total is not None:
        _integer(total, 'source_frames', 0)
        if end > total:
            raise LengthContractError('end_frame exceeds source_frames')
