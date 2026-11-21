"""Select actual in-window observations for checkpoint grading."""
def progress_indices(windows):
    before=windows['state:before']['source_indices']
    process=windows['state:process']['source_indices']
    after=windows['state:after']['source_indices']
    if not before or not process or not after:
        raise ValueError('State evidence requires nonempty before/process/after windows')
    # Preserve the initial image and the existing ordered union of later images.
    # Short windows can supply fewer distinct frames than the requested sample count.
    return [before[0]] + sorted(set(process + after))
