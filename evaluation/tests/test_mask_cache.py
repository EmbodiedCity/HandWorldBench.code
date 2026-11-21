import pathlib
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from handworldbench.observations.segmentation import hands


class MaskCacheTests(unittest.TestCase):
    def test_cached_masks_do_not_reload_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory)
            path=root/'frame.png';Image.new('RGB',(2,2),'white').save(path)
            frame=dict(path=str(path),sample_index=0,decoded_rgb_sha256='pixels')
            geometry_frame=dict(sample_index=0,width=2,height=2,hands=[dict(side='left',joints_2d=[[.5,.5]]*21,valid_2d=[True]*21)])
            observation=dict(reference_frames=[frame],prediction_frames=[frame],reference=[[dict(box=[0,0,2,2],detection_confidence=.9)]],prediction=[[dict(box=[0,0,2,2],detection_confidence=.9)]],geometry=dict(native_frames=[geometry_frame],frames=[geometry_frame]))
            config=dict(models={'sam3':{'source':'unused','checkpoint':'unused','vocabulary':'unused'}})
            class FakeSegmenter:
                def __init__(self,spec):pass
                def predict(self,image,instances):
                    return [(hand,np.ones((2,2),dtype=bool),.9) for hand in instances]
            with patch('handworldbench.observations.segmentation.Segmenter',FakeSegmenter):
                first=hands(observation,{},config,root)
            with patch('handworldbench.observations.segmentation.Segmenter',side_effect=AssertionError('Model loaded for cached masks')):
                second=hands(observation,{},config,root)
            self.assertEqual(first['prediction'],second['prediction'])
            self.assertEqual(first['reference'],second['reference'])
            self.assertEqual(second['timing']['cached_frames'],2)
            self.assertEqual(second['timing']['inference_frames'],0)

    def test_empty_detections_do_not_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory)
            path=root/'frame.png';Image.new('RGB',(2,2),'white').save(path)
            frame=dict(path=str(path),sample_index=0,decoded_rgb_sha256='pixels')
            geometry_frame=dict(sample_index=0,width=2,height=2,hands=[])
            observation=dict(reference_frames=[frame],prediction_frames=[frame],reference=[[]],prediction=[[]],geometry=dict(native_frames=[geometry_frame],frames=[geometry_frame]))
            with patch('handworldbench.observations.segmentation.Segmenter',side_effect=AssertionError('Model loaded without detections')):
                result=hands(observation,{},dict(models={'sam3':{}}),root)
            self.assertEqual(result['prediction'],{})
            self.assertEqual(result['reference'],{})
            self.assertEqual(result['timing']['model_load_seconds'],0)


if __name__=='__main__':unittest.main()
