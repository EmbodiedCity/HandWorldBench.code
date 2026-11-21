"""Published score normalization, separate from immutable evaluator observations."""
import math
POLICY_VERSION = 'semantic_unknown_zero_v1'
def publish(raw_score, status, *, applicable=True):
    if not isinstance(applicable, bool):
        raise ValueError('applicable must be an explicit boolean')
    if raw_score is not None and (type(raw_score) not in (int, float) or not math.isfinite(raw_score) or not 0 <= raw_score <= 1):
        raise ValueError('Score must be finite and in [0,1]')
    if status in ('execution_error','invalid_output','runtime_error','invalid_input','missing_dependency','missing_annotation','missing_reference'):
        return dict(score=None,raw_score=raw_score,applicable=applicable,original_status=status,zero_imputed=False,score_normalization=POLICY_VERSION)
    if not applicable:
        if raw_score is not None:
            raise ValueError('Inapplicable metric cannot carry a score')
        return dict(score=None, raw_score=None, applicable=False, original_status=status,
                    zero_imputed=False, score_normalization=POLICY_VERSION)
    # Task applicability comes from the frozen manifest, not from an evaluator failure.
    failed = raw_score is None or status != 'ok'
    return dict(score=0.0 if failed else float(raw_score), raw_score=raw_score,
                applicable=True, original_status=status, zero_imputed=failed,
                score_normalization=POLICY_VERSION)
def summarize(records):
    active = [r for r in records if r['applicable']]
    return dict(planned=len(records), applicable=len(active),
                valid=sum(not r['zero_imputed'] for r in active),
                zero_imputed=sum(r['zero_imputed'] for r in active),
                mean=sum(r['score'] for r in active)/len(active) if active and all(r['score'] is not None for r in active) else None,
                score_normalization=POLICY_VERSION)
