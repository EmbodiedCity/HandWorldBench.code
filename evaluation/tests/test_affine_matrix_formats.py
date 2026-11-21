import tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from handworldbench.backends import spatial
from handworldbench.data.spatial_mapping import affine_matrix_3x3
from handworldbench.observations.reference import transform

# reference 8x8 -> crop x in [2,8), y in [1,7) -> output 6x6
CROP_2X3=[[1.,0.,-2.],[0.,1.,-1.]]
CROP_3X3=CROP_2X3+[[0.,0.,1.]]


class AffineMatrixFormats(unittest.TestCase):
    def track(self,matrix):
        with tempfile.TemporaryDirectory() as td:
            mask=np.zeros((8,8),np.uint8);mask[3:5,4:6]=1;p=Path(td)/'m.png';Image.fromarray(mask).save(p)
            geometry={'object_0':{'0':{'width':8,'height':8}}};tracks={'object_0':{'0':{'mask':str(p)}}}
            return spatial.trajectory_reference({'reference_image_affine':matrix},geometry,tracks,[{'width':6,'height':6}])['object_0']['0']

    def test_object_track_accepts_2x3_and_3x3_identically(self):
        a,b=self.track(CROP_2X3),self.track(CROP_3X3)
        self.assertEqual(a,b)
        self.assertEqual(a['center_xy'],[2.5,2.5]);self.assertEqual(a['bbox_xyxy'],[2,2,4,4]);self.assertEqual(a['area_pixels'],4)

    def test_hand_points_accept_2x3_and_3x3_identically(self):
        pts=[[4.5,3.5],[2.,1.]]
        a=transform(pts,{'kind':'affine','matrix':CROP_2X3},6,6);b=transform(pts,{'kind':'affine','matrix':CROP_3X3},6,6)
        np.testing.assert_allclose(a,b);np.testing.assert_allclose(a,[[2.5,2.5],[0.,0.]])

    def test_hand_points_and_object_track_agree(self):
        center=self.track(CROP_2X3)['center_xy']
        np.testing.assert_allclose(transform([[4.5,3.5]],{'kind':'affine','matrix':CROP_2X3},6,6)[0],center)

    def test_invalid_shapes_rejected(self):
        for bad in ([[1.,0.],[0.,1.]],[[1.,0.,0.],[0.,1.,0.],[0.,1e-3,1.]],[[1.,0.,float('nan')],[0.,1.,0.]]):
            with self.assertRaises(ValueError):affine_matrix_3x3(bad)


if __name__=='__main__':unittest.main(verbosity=2)
