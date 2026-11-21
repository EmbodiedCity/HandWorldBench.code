import unittest
from handworldbench.metrics.population import reference_hand_sides
from handworldbench.backends.regions_support.region_scoring import aggregate_region_appearance


class EntityPopulationTests(unittest.TestCase):
    def test_offscreen_hand_is_outside_reference_population(self):
        frames=[dict(width=100,height=100,hands=[
            dict(side='left',joints_2d=[[20,30]]*21,valid_2d=[True]*21,visible=[False]*21),
            dict(side='right',joints_2d=[[120,30]]*21,valid_2d=[True]*21)])]
        self.assertEqual(reference_hand_sides(frames),['left'])

    def test_missing_prediction_keeps_its_penalty(self):
        per={'left_hand':dict(type='hand',weight=1.,score=0.,reference_coverage=1.),
             'object_1':dict(type='object',weight=1.,score=.8,reference_coverage=1.)}
        self.assertAlmostEqual(aggregate_region_appearance(per)['score'],.4)
        per['left_hand']['score']=None
        self.assertIsNone(aggregate_region_appearance(per)['score'])


if __name__=='__main__':unittest.main()
