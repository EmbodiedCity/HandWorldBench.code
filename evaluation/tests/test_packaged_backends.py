from pathlib import Path
import unittest

import handworldbench


class PackagedBackendTests(unittest.TestCase):
    def test_required_model_implementations_are_packaged(self):
        root=Path(handworldbench.__file__).parent/'_vendor/engine/engines'
        paths=[
            'classic/metric_suite/sources/DINOv3/dinov3/models/__init__.py',
            'classic/metric_suite/sources/VBench2-Hand/models/vision_transformer.py',
        ]
        for relative in paths:
            with self.subTest(path=relative):self.assertTrue((root/relative).is_file())


if __name__=='__main__':unittest.main()
