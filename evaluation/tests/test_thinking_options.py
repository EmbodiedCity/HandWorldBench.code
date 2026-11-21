import os
import unittest
from unittest.mock import patch
from handworldbench.runtime.options import THINKING_ENV, thinking_tokens, thinking_environment


class ThinkingOptionsTests(unittest.TestCase):
    def test_memory_budget_does_not_change_other_state_tasks(self):
        config = {'observation_options': {'memory_evidence': {'thinking_tokens': 256}}}
        self.assertEqual(thinking_tokens(config, 'memory_evidence'), 256)
        self.assertIsNone(thinking_tokens(config, 'state_evidence'))
        with patch.dict(os.environ, {THINKING_ENV: '1024'}):
            with thinking_environment(config, 'memory_evidence'):
                self.assertEqual(os.environ[THINKING_ENV], '256')
            self.assertEqual(os.environ[THINKING_ENV], '1024')
            with thinking_environment(config, 'state_evidence'):
                self.assertNotIn(THINKING_ENV, os.environ)
            self.assertEqual(os.environ[THINKING_ENV], '1024')

    def test_exception_restores_environment(self):
        config = {'env': {THINKING_ENV: '0'}}
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                with thinking_environment(config, 'memory_evidence'):
                    self.assertEqual(os.environ[THINKING_ENV], '0')
                    raise RuntimeError('backend failed')
            self.assertNotIn(THINKING_ENV, os.environ)

    def test_invalid_budget(self):
        for value in (-1, 1.5, True, 'bad'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                thinking_tokens({'observation_options': {'memory_evidence': {'thinking_tokens': value}}}, 'memory_evidence')
