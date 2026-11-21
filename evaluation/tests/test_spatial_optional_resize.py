import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from handworldbench.backends import spatial as module
from handworldbench.observations.reference import transform

class PixelCoordinateMapping(unittest.TestCase):
 def evaluate(self,pillow):
  with tempfile.TemporaryDirectory() as td:
   source=np.zeros((4,4),np.uint8);source[1,1]=1;p=Path(td)/'mask.png';Image.fromarray(source).save(p)
   spatial=dict(kind='opencv',K=[[2.,0,2.],[0,2.,2.],[0,0,1]],D=[0.]*8,source_wh=[4,4])
   if pillow:spatial['resize']='pillow_pixel_centers'
   geometry={'object_0':{'0':{'width':4,'height':4}}};tracks={'object_0':{'0':{'mask':str(p)}}}
   result=module.trajectory_reference({'reference_image_opencv':spatial},geometry,tracks,[{'width':12,'height':12}])['object_0']['0']
   point=transform([[1.,1.]],spatial,12,12)[0]
   return result,point
 def test_recorded_opencv_integer_warp_without_optional_resize(self):
  result,point=self.evaluate(False)
  self.assertEqual(result['center_xy'],[3.,3.]);self.assertEqual(result['area_pixels'],9)
  self.assertEqual(result['bbox_xyxy'],[2,2,5,5]);np.testing.assert_allclose(point,[3.,3.])
 def test_explicit_pillow_keeps_half_pixel_shift(self):
  result,point=self.evaluate(True)
  self.assertEqual(result['center_xy'],[4.,4.]);self.assertEqual(result['area_pixels'],9)
  self.assertEqual(result['bbox_xyxy'],[3,3,6,6]);np.testing.assert_allclose(point,[4.,4.])

if __name__=='__main__':unittest.main(verbosity=2)
