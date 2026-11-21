import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from handworldbench.backends.dispatch import initialize


@unittest.skipUnless(os.name=='posix','Runtime resource links use POSIX semantics')
class ResourceBindingTests(unittest.TestCase):
    def test_changed_resource_root_cannot_silently_reuse_old_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory)
            first=root/'first';second=root/'second'
            first.mkdir();second.mkdir()
            context=types.SimpleNamespace(S=root/'cache')
            original_path=list(sys.path)
            try:
                with patch.dict(sys.modules,{'handworldbench.backends.context':context}):
                    initialize(dict(models={},models_root=str(first)))
                    initialize(dict(models={},models_root=str(first)))
                    with self.assertRaisesRegex(ValueError,'Cache resource binding differs'):
                        initialize(dict(models={},models_root=str(second)))
                self.assertEqual((context.S/'models').resolve(),first.resolve())
            finally:
                sys.path[:]=original_path


if __name__=='__main__':unittest.main()
