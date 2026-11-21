import types
import unittest
from pathlib import Path
from PIL import Image
from test_shared_hands import function

ROOT=Path(__file__).resolve().parents[1]/'handworldbench/_vendor/engine/engines/state_four/runtime/state_eval'


class EvidenceBatchTests(unittest.TestCase):
    def test_frame_order_tail_and_crop_coordinates(self):
        calls=[]
        def many(images,concepts):
            calls.append((len(images),concepts))
            return [[dict(pixel=im.getpixel((0,0)),size=im.size)] for im in images]
        localizer=types.SimpleNamespace(batch_size=3,candidates_many=many)
        def image_at(cap,index):
            image=Image.new('RGB',(20,10),(index,0,0))
            image.putpixel((2,1),(index,2,1));return image
        run=function(ROOT/'prepare.py','localized_frames',dict(image_at=image_at))
        indices=[7,2,9,0,1,6,4]
        rows=list(run(None,indices,localizer,['hand'],crop=[2,1,12,9]))
        self.assertEqual([n for n,_ in calls],[3,3,1])
        self.assertEqual([r[0] for r in rows],indices)
        self.assertEqual([r[2][0] for r in rows],
            [dict(pixel=(i,2,1),size=(10,8)) for i in indices])
        self.assertEqual(list(run(None,[],localizer,[])),[])

    def test_localization_rejects_truncated_output(self):
        run=function(ROOT/'prepare.py','localized_frames',dict(image_at=lambda *_:Image.new('RGB',(4,4))))
        localizer=types.SimpleNamespace(batch_size=8,candidates_many=lambda images,concepts:[[]])
        with self.assertRaisesRegex(ValueError,'output count'):
            list(run(None,[0,1],localizer,['hand']))

    def test_hand_instances_stay_with_their_input(self):
        calls=[]
        def predict(images,*args,**kwargs):
            calls.append(len(images))
            return [[dict(box=[i,0,i+2,2],detection_confidence=.9)] if i!=2 else []
                    for i in images],dict(batches=1,inferred_images=len(images),seconds=.1)
        cls=function(ROOT/'hand_bridge.py','HandFinder',dict(predict=predict,selection_binding=lambda _:{}))
        context=types.SimpleNamespace(config=dict(shared_hand_runtime={},hand_finder=dict(category='human',role='prediction',video_sha256='a')))
        finder=cls(context);rows=finder.candidates_many([3,2,5])
        self.assertEqual(calls,[3]);self.assertEqual(rows[1],[])
        self.assertEqual([rows[i][0]['bbox'] for i in [0,2]],[[3,0,5,2],[5,0,7,2]])
        self.assertEqual(finder.stats['batches'],1)

    def test_batch_hand_only_never_initializes_sam(self):
        cls=function(ROOT/'localization.py','Localizer',{})
        obj=object.__new__(cls)
        expected=[[dict(concept='hand',instance_id=i)] for i in range(2)]
        obj.hand_finder=types.SimpleNamespace(candidates_many=lambda _:expected)
        obj._ensure_sam=lambda:self.fail('Hand-only batch initialized SAM')
        self.assertEqual(obj.candidates_many([Image.new('RGB',(4,4))]*2,['hand','']),expected)

    def test_object_localization_receives_corresponding_hand_records(self):
        cls=function(ROOT/'localization.py','Localizer',{})
        obj=object.__new__(cls)
        obj.hand_finder=types.SimpleNamespace(candidates_many=lambda images:[[i] for i in images])
        obj._object_candidates=lambda image,concepts,hands:(image,hands)
        self.assertEqual(obj.candidates_many([8,1,6],['cup']),[(8,[8]),(1,[1]),(6,[6])])


if __name__=='__main__':unittest.main()
