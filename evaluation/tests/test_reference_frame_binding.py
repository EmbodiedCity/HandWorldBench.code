import unittest
import numpy as np
from handworldbench.observations.reference import geometry_indices

class FrameBindingTests(unittest.TestCase):
    def setUp(self):
        self.data={'time':np.array([100.,100.031,100.062,100.08]),'index':np.array([0,1,2,3])}
        self.geom={'time_key':'time','time_origin':'first_sample','frame_index_key':'index','frame_index_reference_sha256':'video'}
        self.source={'fps':30.,'sha256':'video'}

    def test_frame_identity_ignores_capture_clock_drift(self):
        _,chosen=geometry_indices(self.data,self.geom,self.source,[{'source_frame_index':0},{'source_frame_index':3}],[0.,0.2])
        self.assertEqual(chosen,[0,3])

    def test_noncontiguous_geometry_rows(self):
        self.data['index']=np.array([0,3,6,9])
        _,chosen=geometry_indices(self.data,self.geom,self.source,[{'source_frame_index':6},{'source_frame_index':3}],[0.2,0.1])
        self.assertEqual(chosen,[2,1])

    def test_wrong_video_rejected(self):
        self.geom['frame_index_reference_sha256']='other'
        with self.assertRaisesRegex(ValueError,'another video'):
            geometry_indices(self.data,self.geom,self.source,[{'source_frame_index':0}],[0.])

    def test_missing_frame_is_not_clamped(self):
        with self.assertRaisesRegex(ValueError,'has no geometry'):
            geometry_indices(self.data,self.geom,self.source,[{'source_frame_index':9}],[0.])

    def test_duplicate_frame_rejected(self):
        self.data['index']=np.array([0,1,1,3])
        with self.assertRaisesRegex(ValueError,'invalid geometry frame'):
            geometry_indices(self.data,self.geom,self.source,[{'source_frame_index':0}],[0.])

    def test_timestamp_binding_preserved(self):
        geom={'time_key':'time','time_origin':'first_sample'}
        _,chosen=geometry_indices(self.data,geom,self.source,[{},{}],[0.,0.06])
        self.assertEqual(chosen,[0,2])
        with self.assertRaisesRegex(ValueError,'outside tolerance'):
            geometry_indices(self.data,geom,self.source,[{}],[0.3])

if __name__=='__main__':unittest.main()
