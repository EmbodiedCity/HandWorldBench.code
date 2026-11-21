"""Resolve task-scoped inference settings."""
from contextlib import contextmanager
import os

THINKING_ENV = 'HANDWORLDBENCH_STATE_THINKING_TOKENS'


def thinking_tokens(config, observation):
    value = config.get('observation_options', {}).get(observation, {}).get('thinking_tokens')
    if value is None:
        value = config.get('env', {}).get(THINKING_ENV)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError('thinking_tokens must be a nonnegative integer')
    number = int(value)
    if number < 0 or str(number) != str(value):
        raise ValueError('thinking_tokens must be a nonnegative integer')
    return number


@contextmanager
def thinking_environment(config, observation):
    """Restore worker state when a model call finishes or raises."""
    value = thinking_tokens(config, observation)
    previous = os.environ.get(THINKING_ENV)
    try:
        if value is None:
            os.environ.pop(THINKING_ENV, None)
        else:
            os.environ[THINKING_ENV] = str(value)
        yield
    finally:
        if previous is None:
            os.environ.pop(THINKING_ENV, None)
        else:
            os.environ[THINKING_ENV] = previous
