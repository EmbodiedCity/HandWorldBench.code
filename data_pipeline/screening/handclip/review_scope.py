"""Limit a negative review to the interval or defect it actually establishes.

Alternative windows remain unreviewed. No negative review can approve them.
Coordinates in a defect record are local to its hash-bound reviewed artifact.
"""


def validate_scope(review, frame_count):
    scope = review.get('rejection_scope', 'exact_window')
    if scope not in ('exact_window', 'first_frame', 'frame_ranges', 'overlap'):
        raise ValueError('Unknown rejection_scope')
    ranges = review.get('bad_frame_ranges', [])
    if not isinstance(ranges, list):
        raise ValueError('bad_frame_ranges must be a list')
    if scope == 'frame_ranges':
        if not ranges:
            raise ValueError('frame_ranges requires explicit bad frame intervals')
        for span in ranges:
            if (not isinstance(span, list) or len(span) != 2
                    or any(type(v) is not int for v in span)
                    or not 0 <= span[0] < span[1] <= frame_count):
                raise ValueError('Bad frame interval outside reviewed artifact')
    elif ranges:
        raise ValueError('bad_frame_ranges requires frame_ranges scope')
    return scope, ranges


def excludes(candidate, reviewed_candidate, review):
    scope, ranges = validate_scope(review, reviewed_candidate['n_frames'])
    if candidate['recording_id'] != reviewed_candidate['recording_id']:
        return False
    a = candidate['source_frame_offset'] + candidate['start_frame']
    b = candidate['source_frame_offset'] + candidate['end_frame']
    x = reviewed_candidate['source_frame_offset'] + reviewed_candidate['start_frame']
    y = reviewed_candidate['source_frame_offset'] + reviewed_candidate['end_frame']
    # Always avoid retrying the exact reviewed interval, including across MCAPs.
    if (a, b) == (x, y):
        return True
    if scope == 'first_frame':
        return a == x
    if scope == 'overlap':
        return a < y and x < b
    if scope == 'frame_ranges':
        return any(a < x+v and x+u < b for u, v in ranges)
    return False
