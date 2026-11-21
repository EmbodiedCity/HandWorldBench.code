import unittest
import copy
from handworldbench.registry import METRICS, select, observation_order
from handworldbench.runtime.devices import inventory, select_devices, placements
from handworldbench.metrics.keypoints import score
from handworldbench.data.timeline import Timeline

class ContractTests(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(len(METRICS),17)
        with self.assertRaises(ValueError): select("unknown_metric")
        order=observation_order(["pose_control","keypoint_coverage","entity_consistency"])
        self.assertEqual(order.count("hand_detection"),1)
        self.assertEqual(order.count("hand_keypoints"),1)
        self.assertLess(order.index("hand_detection"),order.index("hand_masks"))

    def test_topologies(self):
        for count in (1,2,4,8):
            physical=[dict(index=i,uuid=f"GPU-{i:08d}",total_mib=96000,free_mib=80000) for i in range(count)]
            devices=inventory(physical,visible=",".join(str(i) for i in range(count)))
            self.assertEqual(len(select_devices("auto",devices)),count)
            self.assertEqual(placements(devices,1,30000)[0][1],(0,))
        physical=[dict(index=i,uuid=f"GPU-{i:08d}",total_mib=96000,free_mib=80000) for i in range(8)]
        devices=inventory(physical,visible="5,1,3")
        self.assertEqual([d.physical_index for d in select_devices("0,2",devices)],[5,3])
        self.assertEqual(placements(devices,2,30000,busy=(1,))[0][1],(0,2))
        with self.assertRaises(ValueError):select_devices("0,0",devices)
        with self.assertRaises(ValueError):select_devices("3",devices)

    def frames(self):
        hand=dict(side="left",joints_2d=[[50.,50.]]*21,valid_2d=[True]*21,visible_2d=[True]*21)
        ref=[dict(sample_index=0,width=100,height=100,hands=[hand])]
        pred=copy.deepcopy(ref);pred[0]["hands"][0]["confidence"]=[1.]*21
        return ref,pred

    def test_missing(self):
        r,p=self.frames()
        self.assertEqual(score(r,p,"pose_control")["score_0_1"],1)
        p[0]["hands"][0]["confidence"][0]=.1
        for metric in ("pose_control","keypoint_coverage"):
            result=score(r,p,metric)
            self.assertAlmostEqual(result["score_0_1"],20/21)
            self.assertEqual(result["expected_units"],21)
        p[0]["hands"]=[]
        self.assertEqual(score(r,p,"pose_control")["score_0_1"],0)
        r[0]["hands"][0]["visible_2d"]=[False]*21
        self.assertEqual(score(r,p,"keypoint_coverage")["score_0_1"],0)
        self.assertEqual(score(r,p,"pose_control")["score_0_1"],0)

    def test_coverage_uses_all_valid_points(self):
        r,p=self.frames();r[0]['hands'][0]['visible_2d']=[False]*21
        self.assertEqual(score(r,p,'keypoint_coverage')['score_0_1'],1)
        r[0]['hands'][0]['joints_2d'][0]=[-1,50]
        result=score(r,p,'keypoint_coverage')
        self.assertEqual(result['expected_units'],20)
        self.assertEqual(result['score_0_1'],1)
        self.assertNotIn('reference_visibility',observation_order(['keypoint_coverage']))
        p[0]['hands'][0]['joints_2d']=[[90,90]]*21
        result=score(r,p,'keypoint_coverage')
        self.assertEqual(result['score_0_1'],1)
        self.assertEqual(result['frames'][0]['score'],1)
        self.assertEqual(score(r,p,'pose_control')['score_0_1'],0)

    def test_duplicate_identity(self):
        r,p=self.frames();p[0]["hands"]*=2
        with self.assertRaises(ValueError):score(r,p,"keypoint_coverage")

    def test_offscreen_instance_cannot_take_an_inframe_match(self):
        from handworldbench.observations.hands import associate
        left=dict(side='left',joints_2d=[[80.,50.]]*21,valid_2d=[True]*21,visible_2d=[False]*21)
        right=dict(side='right',joints_2d=[[105.,50.]]*21,valid_2d=[True]*21)
        reference=[dict(sample_index=0,width=100,height=100,hands=[left,right])]
        prediction=[[dict(box=[90.,40.,99.,60.],detection_confidence=.9)]]
        result=associate(reference,prediction)
        self.assertEqual([hand['side'] for hand in result[0]['hands']],['left'])

    def test_timeline(self):
        t=Timeline(dict(kind="points",points=[[0,0],[3,1],[6,4]]))
        self.assertEqual(t.at(4.5),2.5)
        self.assertAlmostEqual(t.inverse().at(2.5),4.5)
        with self.assertRaises(ValueError): t.at(7)
        with self.assertRaises(ValueError): Timeline(dict(kind="affine",scale=0))

    def test_distribution_aggregation(self):
        from handworldbench.metrics.distribution import polynomial_distance
        import numpy as np
        x=np.array([[1,2],[3,4]],dtype=float);y=np.array([[2,2],[4,5]],dtype=float)
        self.assertAlmostEqual(polynomial_distance(x,x),0)
        kernel=lambda a,b:(a@b.T/a.shape[1])**2
        expected=100*(kernel(x,x).mean()+kernel(y,y).mean()-2*kernel(x,y).mean())
        self.assertAlmostEqual(polynomial_distance(x,y),expected)
        self.assertAlmostEqual(polynomial_distance(x,y),polynomial_distance(y,x))

    def test_annotation_mapping(self):
        from handworldbench.data.annotations import convert
        source={'events':[{'before':[3,6],'process':[6,9],'description':'wait 3 seconds'}], 'returns':[{'start_s':6,'end_s':6.3}]}
        bound=convert(source,Timeline(dict(kind='affine',scale=1/3)))
        self.assertEqual(bound['events'][0]['before'],[1,2])
        self.assertEqual(bound['events'][0]['description'],'wait 3 seconds')
        self.assertAlmostEqual(bound['returns'][0]['end_s'],2.1)
        self.assertEqual(source['events'][0]['before'],[3,6])

    def test_bad_mapping_is_per_input_error(self):
        import tempfile,json
        from pathlib import Path
        from types import SimpleNamespace
        from handworldbench.data.inputs import normalize
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'dataset.json').write_text(json.dumps({'catalog':'catalog.json'}))
            items=[{'case_id':c,'category':'human','reference':{'video':str(root/'ref.mp4'),'sha256':'reference'}} for c in ('good','bad')]
            (root/'catalog.json').write_text(json.dumps(items));(root/'good.mp4').write_bytes(b'test')
            rows=[{'case_id':'good','video':'good.mp4','time_mapping':{'kind':'identity'},'spatial_mapping':{'kind':'identity'}},{'case_id':'bad','video':'missing.mp4'}]
            path=root/'predictions.json';path.write_text(json.dumps(rows))
            args=SimpleNamespace(dataset=str(root),cases=None,predictions=str(path),model='Model',metrics='pose_control')
            _,_,normalized=normalize(args)
            self.assertNotIn('input_error',normalized[0]);self.assertIn('input_error',normalized[1])

if __name__=="__main__":unittest.main()
