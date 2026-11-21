"""Pure rendering of explicit R-only full-establishment normalization. No inference/import side effects.

Task-specific semantics belong to the unit's establishment annotation. Existing
P/C-linked and partial-new-state paths must not call this renderer implicitly.
Whitespace is preserved so a migrated frozen protocol can retain identical input.
"""
from collections.abc import Mapping

TEMPLATE_VERSION = 'retention-full-establishment-neutral-labels-v1'
HEAD = ('Judge only visible evidence. The requirement states what to evaluate, not what happened. '
        'Do not infer success from familiar hand activity or hidden geometry. '
        'The S0, S1, ... images are observations in actual chronological order. S0 is earlier context; '
        'later images are NOT labelled as successful establishment. '
        'The following is the required state to test, NOT a description of what has already happened.\n'
        'Required state: ')
TAIL = ('Reply with one digit only: 1=yes, the full required state is visibly established; '
        '2=no, the full required state is not established; '
        '0=essential evidence cannot determine whether it is established. Do not explain.')
TASK_FIELDS = ('comparison_question', 'partial_completion_boundary', 'relation_instruction')


def validate_establishment(normalization, checkpoints, *, retention_only):
    """Reject absent/implicit policies, non-R-only use, and divergent source conditions."""
    if retention_only is not True:
        raise ValueError('Full-establishment renderer requires explicit R-only context')
    if not isinstance(normalization, Mapping):
        raise ValueError('An explicit establishment normalization is required')
    if normalization.get('level') != 'full_condition':
        raise ValueError('Expected full_condition establishment level')
    if normalization.get('template_version') != TEMPLATE_VERSION:
        raise ValueError('Unsupported establishment template version')
    for name in ('source_checkpoint_id', 'full_condition') + TASK_FIELDS:
        value = normalization.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Missing nonempty establishment field: ' + name)
        if '\x00' in value:
            raise ValueError('NUL is not allowed in establishment fields')
    if not isinstance(checkpoints, (tuple, list)):
        raise ValueError('Explicit checkpoint list is required')
    identifiers = [cp.get('id') for cp in checkpoints if isinstance(cp, Mapping)]
    if len(identifiers) != len(checkpoints) or len(set(identifiers)) != len(identifiers):
        raise ValueError('Invalid or duplicate checkpoint records')
    matches = [cp for cp in checkpoints if cp.get('id') == normalization['source_checkpoint_id']]
    if len(matches) != 1:
        raise ValueError('Unknown source checkpoint')
    if matches[0].get('full_condition') != normalization['full_condition']:
        raise ValueError('Full condition differs from the frozen source checkpoint')
    return dict(normalization)


def render_gate(normalization, checkpoints, *, retention_only):
    """Return prompt text only; caller owns image bindings, decoder, scores and cache."""
    normalization = validate_establishment(normalization, checkpoints, retention_only=retention_only)
    return HEAD + normalization['full_condition'] + '\n' + ''.join(normalization[k] for k in TASK_FIELDS) + TAIL


def validate_state_policies(spec):
    retention_only = spec['applicability']['state_progression']['status'] != 'applicable'
    for unit in spec.get('persistence_units', []):
        normalization = unit.get('establishment')
        if normalization is None or normalization.get('level') == 'partial_condition':
            continue
        validate_establishment(normalization, spec.get('checkpoints', []), retention_only=retention_only)
