import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import handworldbench
sys.path.insert(0,str(Path(handworldbench.__file__).resolve().parent/'backends/regions_support'))
import entity_selection
import region_observations

class IdentityRecoveryTests(unittest.TestCase):
    def test_success_never_reselects(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(entity_selection,'select_objects',return_value=([{'id':'one'}],{'requests':1})) as select, patch.object(region_observations,'locate_objects',return_value=([],{},{})) as locate:
            result=entity_selection.select_and_localize(object(),[],Path(tmp))
            self.assertEqual(select.call_count,1)
            self.assertEqual(locate.call_count,1)
            self.assertEqual(result[1]['requests'],1)
            self.assertFalse((Path(tmp)/'identity_recovery.json').exists())

    def test_identity_conflict_reselects_reference_once(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(entity_selection,'select_objects',side_effect=[([{'id':'group'}],{'requests':1}),([{'id':'left'},{'id':'right'}],{'requests':1})]) as select, patch.object(region_observations,'locate_objects',side_effect=[region_observations.LocalizationIdentityError('duplicate'),([],{}, {'valid':True})]) as locate:
            result=entity_selection.select_and_localize(object(),[],Path(tmp))
            self.assertEqual([x['id'] for x in result[0]],['left','right'])
            self.assertTrue(select.call_args.kwargs['separate_instances'])
            self.assertEqual(result[1]['requests'],2)
            self.assertEqual(result[1]['identity_corrections'],1)
            self.assertTrue(all(c.kwargs['role']=='reference' for c in locate.call_args_list))
            self.assertEqual(json.loads((Path(tmp)/'identity_recovery.json').read_text())['status'],'resolved')

    def test_repeated_conflict_is_error_not_missing_detection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(entity_selection,'select_objects',return_value=([{'id':'one'}],{'requests':1})) as select, patch.object(region_observations,'locate_objects',side_effect=region_observations.LocalizationIdentityError('duplicate')) as locate:
            with self.assertRaises(region_observations.LocalizationIdentityError):
                entity_selection.select_and_localize(object(),[],Path(tmp))
            self.assertEqual(locate.call_count,2)
            self.assertEqual(select.call_count,2)
            self.assertNotEqual(json.loads((Path(tmp)/'identity_recovery.json').read_text())['status'],'resolved')

    def test_other_errors_do_not_change_entities(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(entity_selection,'select_objects',return_value=([{'id':'one'}],{'requests':1})) as select, patch.object(region_observations,'locate_objects',side_effect=RuntimeError('CUDA failure')):
            with self.assertRaisesRegex(RuntimeError,'CUDA failure'):
                entity_selection.select_and_localize(object(),[],Path(tmp))
            self.assertEqual(select.call_count,1)
