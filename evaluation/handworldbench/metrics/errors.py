"""Keep model execution failures distinct from semantic unknown answers."""
FAILURES = frozenset({
    'runtime_error', 'execution_error', 'invalid_input', 'invalid_output',
    'missing_dependency', 'missing_annotation', 'missing_reference',
})


def execution_failures(value, path='result'):
    failures = []
    if isinstance(value, dict):
        status = value.get('status', value.get('original_status'))
        if status in FAILURES or value.get('error_observations', 0):
            failures.append({'path': path, 'status': status or 'invalid_observation',
                             'reason': str(value.get('error') or value.get('reason') or status)})
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                failures.extend(execution_failures(child, path + '.' + str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(execution_failures(child, path + '[' + str(index) + ']'))
    return failures
