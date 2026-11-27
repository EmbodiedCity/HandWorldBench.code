import json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from handworldbench.backends.regions import precomputed_reference_objects
from handworldbench.data.inputs import load_dataset

SHA='a'*64


def make(root,mask='hw500_0001/m.png'):
    (root/'reference_objects/hw500_0001').mkdir(parents=True)
    Image.fromarray(np.ones((4,4),np.uint8)).save(root/'reference_objects/hw500_0001/m.png')
    doc=dict(case_id='hw500_0001',reference_sha256=SHA,
             entities=[dict(id='object_1',type='object',description='red cup',weight=1.0,role='target')],
             tracks={'object_1':{'0':{'mask':mask,'area_pixels':16}}},
             geometry={'object_1':{'0':{'center_xy':[1.5,1.5],'geometry_status':'valid'}}})
    p=root/'reference_objects/hw500_0001.json';p.write_text(json.dumps(doc));return p


class PrecomputedReferenceObjects(unittest.TestCase):
    def test_loads_and_resolves_masks(self):
        with tempfile.TemporaryDirectory() as td:
            p=make(Path(td));data=precomputed_reference_objects(p,SHA)
            mask=Path(data['tracks']['object_1']['0']['mask'])
            self.assertTrue(mask.is_absolute() and mask.is_file())
            self.assertEqual([e['id'] for e in data['entities']],['object_1'])
            self.assertEqual(data['geometry']['object_1']['0']['center_xy'],[1.5,1.5])

    def test_rejects_other_reference_video(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):precomputed_reference_objects(make(Path(td)),'b'*64)

    def test_rejects_missing_or_escaping_mask(self):
        for bad in ('hw500_0001/none.png','../../outside.png'):
            with tempfile.TemporaryDirectory() as td:
                with self.assertRaises(ValueError):precomputed_reference_objects(make(Path(td),bad),SHA)

    def test_catalog_path_is_bound_to_dataset_root(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);make(root)
            (root/'dataset.json').write_text(json.dumps({'catalog':'catalog.json'}))
            (root/'catalog.json').write_text(json.dumps([dict(case_id='hw500_0001',category='human',
                reference_objects=dict(path='reference_objects/hw500_0001.json',sha256=SHA))]))
            _,_,catalog=load_dataset(root)
            self.assertEqual(Path(catalog['hw500_0001']['reference_objects']['path']),(root/'reference_objects/hw500_0001.json').resolve())


if __name__=='__main__':unittest.main(verbosity=2)
