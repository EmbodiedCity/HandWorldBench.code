"""Aggregate published component scores without discarding valid partial scores.

Inputs are score_normalization.publish() records (or aggregates from this module).
These outputs are already published: do not pass them through publish() again.
"""
from copy import deepcopy
import math

POLICY_VERSION = 'component_status_v1'


def _check(record):
    if type(record['applicable']) is not bool:
        raise ValueError('applicable must be an explicit boolean')
    if type(record['zero_imputed']) is not bool:
        raise ValueError('zero_imputed must be an explicit boolean')
    for field in ('score', 'raw_score'):
        value = record[field]
        if value is not None and (type(value) not in (int, float)
                                  or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(field + ' must be finite and in [0,1]')
    if record['applicable']:
        if record['score'] is not None and not record['zero_imputed'] and record['raw_score'] is None:
            raise ValueError('Valid inputs must retain their raw score')
    elif record['score'] is not None or record['raw_score'] is not None:
        raise ValueError('Inapplicable inputs cannot carry a score')


def _result(score, raw_score, applicable, failed):
    return dict(score=score, raw_score=raw_score, applicable=applicable,
                original_status=('not_applicable' if not applicable else
                                 'partial_failure' if failed else 'ok'),
                zero_imputed=bool(applicable and failed), score_normalization=POLICY_VERSION)


def aggregate(values):
    """Equal mean over applicable components; execution errors remain unscored.

    Empty/all-NA input is NA. Counts refer to direct components; components
    preserves every original value and status without mutating the caller.
    """
    components = deepcopy(list(values))
    for record in components:
        _check(record)
    active = [record for record in components if record['applicable']]
    if any(record['score'] is None for record in active):
        return dict(score=None,raw_score=None,applicable=True,original_status='execution_error',zero_imputed=False,score_normalization=POLICY_VERSION,components=components)
    failed = sum(record['zero_imputed'] for record in active)
    score = sum(record['score'] for record in active) / len(active) if active else None
    raw = (sum(record['raw_score'] for record in active) / len(active)
           if active and not failed else None)
    return dict(**_result(score, raw, bool(active), failed),
                planned=len(components), required=len(active),
                valid=len(active) - failed, failed=failed,
                exempt=len(components) - len(active), components=components)


def gated(progress, raw_correctness):
    """Multiply published progress and correctness; either NA makes the gate NA.

    Semantic unknown is zero; execution failure leaves the product unscored.
    """
    progress, raw_correctness = deepcopy(progress), deepcopy(raw_correctness)
    _check(progress)
    _check(raw_correctness)
    applicable = progress['applicable'] and raw_correctness['applicable']
    if applicable and (progress['score'] is None or raw_correctness['score'] is None):
        return dict(score=None,raw_score=None,applicable=True,original_status='execution_error',zero_imputed=False,score_normalization=POLICY_VERSION,progress=progress,raw_correctness=raw_correctness)
    failed = progress['zero_imputed'] or raw_correctness['zero_imputed']
    score = progress['score'] * raw_correctness['score'] if applicable else None
    raw = (progress['raw_score'] * raw_correctness['raw_score']
           if applicable and not failed else None)
    return dict(**_result(score, raw, applicable, failed),
                progress=progress, raw_correctness=raw_correctness)
