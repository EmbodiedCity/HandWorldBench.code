from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import Mock,patch

from handworldbench.registry import observation_order
from handworldbench.runtime.prepare import reference_issue
from handworldbench.observations.reference import images


class ReferenceScopingTests(unittest.TestCase):
    def test_trajectory_does_not_require_hand_geometry(self):
        graph=observation_order(['trajectory_accuracy'])
        self.assertIn('reference_frames',graph)
        self.assertNotIn('reference_geometry',graph)
        self.assertNotIn('hand_detection',graph)

    def test_pending_calibration_only_blocks_geometry_consumers(self):
        item={'geometry':{'path':'geometry.npz','status':'camera_calibration_pending'}}
        for metric in ('pose_control','keypoint_coverage','entity_consistency'):
            self.assertIn('camera_calibration_pending',reference_issue(item,metric))
        for metric in ('trajectory_accuracy','memory_consistency','event_adherence','musiq'):
            self.assertIsNone(reference_issue(item,metric))

    def test_reference_frames_bind_without_joint_assets(self):
        reference=dict(video='reference.mp4',sha256='reference')
        item={'reference':reference}
        row={'time_mapping':{'kind':'affine','scale':2}}
        frames=[{'timestamp_seconds':0.},{'timestamp_seconds':2.}]
        with patch('handworldbench.observations.reference.stage',return_value=(None,{'times':[0.,.5,1.]})),patch('handworldbench.observations.reference.decode',return_value={'frames':[{'sample_index':0},{'sample_index':1}]}) as decode:
            result=images(item,row,frames,'fixture-cache')
        self.assertEqual(result['source_times'],[0.,1.])
        self.assertEqual(decode.call_args.args[-1],[0,2])
        self.assertEqual(result['source'],reference)

    def test_reference_can_contain_no_manipulated_object(self):
        from PIL import Image
        support=Path(__file__).resolve().parents[1]/'handworldbench/backends/regions_support'
        with patch.object(sys,'path',[str(support),*sys.path]):
            from entity_selection import select_objects
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'frame.png';Image.new('RGB',(16,16)).save(path)
                observer=Mock();observer.ask.return_value={'objects':[],'uncertainties':[]}
                entities,receipt=select_objects(observer,[{'path':str(path),'timestamp_seconds':0.}],directory)
                self.assertEqual(entities,[])
                self.assertEqual(receipt['requests'],1)


if __name__=='__main__':unittest.main()
