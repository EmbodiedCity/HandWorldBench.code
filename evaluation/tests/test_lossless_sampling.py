import hashlib
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from handworldbench.observations.video import decode


class LosslessSamplingTests(unittest.TestCase):
    def test_sampled_decode_preserves_pixels_and_duplicate_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);video=root/'input.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),12,(64,48))
            self.assertTrue(writer.isOpened())
            for index in range(20):
                image=np.zeros((48,64,3),dtype=np.uint8)
                image[:]=(index*9,index*7,index*11)
                image[10:25,index:index+12]=(255,33,140)
                writer.write(image)
            writer.release()
            expected=[];capture=cv2.VideoCapture(str(video))
            while True:
                ok,frame=capture.read()
                if not ok:break
                expected.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            capture.release()
            indices=[0,3,3,11,19]
            result=decode(video,hashlib.sha256(video.read_bytes()).hexdigest(),root/'cache','prediction',indices=indices)
            for record,index in zip(result['frames'],indices):
                with Image.open(record['path']) as image:actual=np.asarray(image)
                np.testing.assert_array_equal(actual,expected[index])
                self.assertEqual(record['decoded_rgb_sha256'],hashlib.sha256(expected[index].tobytes()).hexdigest())


if __name__=='__main__':unittest.main()
