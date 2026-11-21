import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch,Mock

import handworldbench
sys.path.insert(0,str(Path(handworldbench.__file__).resolve().parent/'backends/regions_support'))
from handworldbench.backends import regions

def declaration():
    return dict(reference_sha256='refsha',entities=[dict(id='white_spoon',type='object',description='single white spoon',weight=1.)],
                localization=dict(source_frame_indices=[0,10,20],evidence='frame.jpg'))

class ReferenceEntityTests(unittest.TestCase):
    def test_absent_override_preserves_selector_path(self):
        self.assertIsNone(regions.declared_reference_entities({},'refsha'))

    def test_wrong_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            regions.declared_reference_entities({'background':{'reference_entities':declaration()}},'different')

    def test_review_evidence_is_required(self):
        d=declaration();d['localization'].pop('source_frame_indices')
        with self.assertRaisesRegex(ValueError,'source-frame evidence'):
            regions.declared_reference_entities({'background':{'reference_entities':d}},'refsha')

    def test_duplicate_identity_is_rejected(self):
        d=declaration();d['entities']*=2
        with self.assertRaisesRegex(ValueError,'unique'):
            regions.declared_reference_entities({'background':{'reference_entities':d}},'refsha')

    def test_hand_cannot_be_declared_as_object_override(self):
        d=declaration();d['entities'][0]['type']='hand'
        with self.assertRaisesRegex(ValueError,'only object'):
            regions.declared_reference_entities({'background':{'reference_entities':d}},'refsha')

    def test_declaration_is_copied(self):
        d=declaration();entities,selection=regions.declared_reference_entities({'background':{'reference_entities':d}},'refsha')
        entities[0]['description']='mutated'
        self.assertEqual(d['entities'][0]['description'],'single white spoon')
        self.assertEqual(selection['localization']['source_frame_indices'],[0,10,20])
        self.assertEqual(selection['requests'],0)

    def test_both_roles_localize_fresh_and_inventory_changes_cache(self):
        import region_observations,entity_selection,region_scoring
        import handworldbench.backends.normalization,handworldbench.backends.spatial
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=root/'config.json'
            frames=[dict(sample_index=0,decoded_rgb_sha256='pixels',width=10,height=10)]
            task=dict(config=str(cfg),row={'case_id':'case'},dependency_results={
                'frames_a':{'frames':frames},'reference_frames_a':{'reference_images':frames,'source':{'sha256':'refsha'}}})
            context=types.ModuleType('handworldbench.backends.context');context.CONFIG=dict(cache_root=str(root/'cache'),models={'vision':'model','sam3':{}})
            calls=[]
            def locate(observer,frames,entities,folder,**kwargs):
                calls.append((kwargs['role'],entities[0]['description']))
                return entities,{},{}
            def clean(raw,entities,frames,role,geometry_out):
                geometry_out.update({e['id']:{'0':dict(geometry_status='valid',center_xy=[1.,1.])} for e in entities})
                return {e['id']:{} for e in entities}
            with patch.dict(sys.modules,{'handworldbench.backends.context':context}),patch.object(regions,'Observer',return_value=Mock()),patch.object(regions,'track',return_value={}),patch.object(region_observations,'locate_objects',side_effect=locate),patch.object(region_observations,'clean_tracks',side_effect=clean),patch.object(entity_selection,'select_objects',side_effect=AssertionError('selector must not run')),patch.object(region_scoring,'evaluate_geometric_trajectory',return_value={'score':1.}) as score_trajectory,patch.object(handworldbench.backends.normalization,'trajectory_predictions',return_value=({},{})) as interpolate,patch.object(handworldbench.backends.spatial,'trajectory_reference',side_effect=lambda cfg,g,t,f:g) as map_reference:
                d=declaration();cfg.write_text(json.dumps({'background':{'reference_entities':d}}))
                first=regions.objects(task,{},root/'out1')
                self.assertEqual([x[0] for x in calls],['reference','prediction'])
                self.assertEqual(first['shared_objects']['entities'][0]['id'],'white_spoon')
                d['entities'][0]['description']='single white plastic spoon, location independent'
                cfg.write_text(json.dumps({'background':{'reference_entities':d}}))
                full = regions.objects(task, {}, root/'out2')
                self.assertEqual([x[0] for x in calls],['reference','prediction','reference','prediction'])
                score_trajectory.reset_mock()
                interpolate.reset_mock()
                map_reference.reset_mock()
                tracking_only = regions.objects(dict(task, include_trajectory_score=False), {}, root/'out3')
                self.assertEqual(tracking_only['shared_objects'], full['shared_objects'])
                self.assertNotIn('metrics', tracking_only)
                score_trajectory.assert_not_called()
                interpolate.assert_not_called()
                map_reference.assert_not_called()
                self.assertEqual(len(list((root/'cache/reference_objects').iterdir())),2)
                self.assertEqual(len(list((root/'cache/reference_objects').glob('*/selection/reference_inventory.json'))),2)

if __name__=='__main__':unittest.main(verbosity=2)
