import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from handworldbench.runtime.storage import artifact_digest,file_digest


class ArtifactDigestTests(unittest.TestCase):
    def test_reused_artifact_and_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mask.png';path.write_bytes(b'first mask')
            with patch('handworldbench.runtime.storage.file_digest',wraps=file_digest) as digest:
                self.assertEqual(artifact_digest(path),hashlib.sha256(b'first mask').hexdigest())
                self.assertEqual(artifact_digest(path),hashlib.sha256(b'first mask').hexdigest())
                self.assertEqual(digest.call_count,1)
                replacement=path.with_suffix('.part');replacement.write_bytes(b'other mask');replacement.replace(path)
                self.assertEqual(artifact_digest(path),hashlib.sha256(b'other mask').hexdigest())
                self.assertEqual(digest.call_count,2)


if __name__=='__main__':unittest.main()
