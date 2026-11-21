import unittest

from handworldbench.runtime.scheduler import evictions_for_group


class ModelResidencyTests(unittest.TestCase):
    def setUp(self):
        self.group=[{'uuid':'gpu-a','free_mib':50000},{'uuid':'gpu-b','free_mib':50000}]
        self.idle=[{'id':'detector'},{'id':'segmenter'}]

    def test_free_memory_preserves_other_idle_models(self):
        self.assertEqual(evictions_for_group(self.idle,self.group,{'gpu-a':45000,'gpu-b':42000},40000),[])

    def test_insufficient_device_in_group_releases_idle_models(self):
        self.assertEqual(evictions_for_group(self.idle,self.group,{'gpu-a':45000,'gpu-b':35000},40000),self.idle)

    def test_initial_device_snapshot_is_used_before_live_query(self):
        self.assertEqual(evictions_for_group(self.idle,self.group,{},40000),[])


if __name__=='__main__':unittest.main()
