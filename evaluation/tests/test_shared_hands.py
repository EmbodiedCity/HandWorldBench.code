import ast
import contextlib
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from handworldbench.observations import shared_hands as shared

ROOT=Path(__file__).resolve().parents[1]/'handworldbench'

def function(path,name,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

class SharedHandsTests(unittest.TestCase):
    def test_binding_survives_json_receipts(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'config.json').write_text('{}')
            bound=shared.binding({'models':{'hand_detector':folder}},'human')
            self.assertEqual(bound,json.loads(json.dumps(bound)))

    def test_pixel_role_video_prompt_isolation_and_lazy_model(self):
        class Detector:
            calls=[]
            def __init__(self,*args,**kwargs):pass
            def predict(self,images,category):
                self.calls.append((len(images),category))
                return [[dict(box=[0,0,2,2],detection_confidence=.9)] for _ in images],dict(batches=1)
        with tempfile.TemporaryDirectory() as folder:
            config={'cache_root':folder,'models':{'hand_detector':folder}}
            im=Image.new('RGB',(4,4))
            shared._DETECTORS.clear()
            with patch.object(shared,'HandDetector',Detector),patch.object(shared,'model_binding',return_value={'model':'fixed'}):
                a,t=shared.predict([im,im], 'human',config,role='prediction',video_sha256='a')
                self.assertEqual(t['inferred_images'],1)
                shared._DETECTORS.clear()
                with patch.object(shared,'HandDetector',side_effect=AssertionError('Model loaded on cache hit')):
                    b,t=shared.predict([im],'human',config,role='prediction',video_sha256='a')
                self.assertEqual(t['cache_hits'],1);self.assertEqual(a[0],b[0])
                for role,video,category in [('reference','a','human'),('prediction','b','human'),('prediction','a','real_robot')]:
                    _,t=shared.predict([im],category,config,role=role,video_sha256=video)
                    self.assertEqual(t['inferred_images'],1)
                changed=im.copy();changed.putpixel((0,0),(1,0,0))
                _,t=shared.predict([changed],'human',config,role='prediction',video_sha256='a')
                self.assertEqual(t['inferred_images'],1)
        shared._DETECTORS.clear()

    def test_empty_object_prompts_never_load_or_encode_sam(self):
        path=ROOT/'_vendor/engine/engines/state_four/runtime/state_eval/localization.py'
        cls=function(path,'Localizer',{})
        obj=object.__new__(cls)
        expected=[dict(concept='hand',instance_id=0,bbox=[0,0,2,2])]
        obj.hand_finder=types.SimpleNamespace(candidates=lambda _:expected)
        obj._ensure_sam=lambda:self.fail('Unnecessary SAM model load')
        self.assertEqual(obj.candidates(Image.new('RGB',(4,4)),['hand','']),expected)
        self.assertEqual(obj.candidates(Image.new('RGB',(4,4)),[]),expected)

    def test_morphology_batches_across_frames_preserve_bindings(self):
        import numpy as np
        path=ROOT/'_vendor/engine/engines/classic/metric_suite/hand_morphology.py'
        calls=[];saved={}
        def predict(_,images):
            calls.append(len(images));return [[float(x)/10,1-float(x)/10] for x in images]
        analyzer=types.SimpleNamespace(smart_cut=lambda im,box:float(box[0]),predict_batch=predict)
        namespace=dict(np=np,math=__import__('math'),cv2=types.SimpleNamespace(imread=lambda _:np.zeros((100,100,3))),
            torch=types.SimpleNamespace(cuda=types.SimpleNamespace(OutOfMemoryError=MemoryError)),
            load_vbench=lambda *args:(analyzer,None),write_json=lambda p,v:saved.update({p.name:v}),
            statistics=lambda xs:{'mean':sum(xs)/len(xs) if xs else None})
        run=function(path,'vbench_result',namespace)
        frames=[dict(sample_index=i,timestamp_seconds=i,path='unused',decoded_rgb_sha256=str(i)) for i in range(5)]
        hands=[dict(sample_index=i,hands=[dict(box=[i,0,i+2,2],side='left')]) for i in range(5)]
        run(frames,None,None,dict(shared_hand_frames=hands,morphology_batch=4),None,Path('fixture'))
        self.assertEqual(calls,[4,1])
        rows=saved['vbench2_hand_per_crop.json']
        self.assertEqual([r['sample_index'] for r in rows],list(range(5)))
        self.assertEqual([r['p_normal'] for r in rows],[1-i/10 for i in range(5)])

if __name__=='__main__':unittest.main()
