import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from handworldbench.registry import METRICS
from handworldbench.reporting.report import report
from handworldbench.runtime.storage import read


class HWBScoreTests(unittest.TestCase):
    def comparison(self, output):
        values = dict(visual_quality=0.1, hand_fidelity=0.4, appearance_consistency=0.6,
                      instruction_following=0.8, physical_interaction=1.0,
                      temporal_coherence=0.4)
        feature_file = output/'features.npy'
        np.save(feature_file, np.ones((1, 2)), allow_pickle=False)
        records = []
        for model in ('A', 'B'):
            for metric, spec in METRICS.items():
                row = dict(model=model, case_id='test_case', category='real_robot',
                           metric_id=metric, metric_protocol=spec.protocol,
                           applicability='applicable', status='ok', score_0_1=values[spec.category])
                if metric in ('hand_morphology', 'memory_consistency'):
                    row.update(applicability='not_applicable', status='not_applicable',
                               score_0_1=None, metric_protocol=f'not_used_{model}')
                if metric == 'jepa_similarity':
                    feature = dict(feature_file=str(feature_file))
                    row.update(status='observed', score_0_1=None,
                               features=dict(reference=feature, prediction=feature))
                records.append(row)
        return records

    def exported(self, output, records, metrics=None):
        with patch('handworldbench.reporting.report.collect', return_value=records):
            report(['example-run'], output=output, metrics=metrics)
        return read(output/'hwbscore.json')

    def test_weights_groups_equally_and_omits_only_inapplicable_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            records = self.comparison(output)
            totals = self.exported(output, records)
            self.assertEqual(read(output/'scores.json'), records)
            metric_summary = read(output/'summary.json')

        self.assertEqual({(row['model'], row['category']) for row in totals},
                         {('A', 'all'), ('B', 'all'), ('A', 'real_robot'), ('B', 'real_robot')})
        for row in totals:
            self.assertAlmostEqual(row['HWBScore'], 60.0)
            self.assertEqual(row['status'], 'ok')
            self.assertEqual(len(row['group_scores']), 6)
            self.assertEqual(row['not_applicable_metrics'], ['hand_morphology', 'memory_consistency'])
            self.assertEqual(row['mixed_protocol_metrics'], [])
        self.assertEqual(next(row['score_0_1'] for row in metric_summary
                              if row['metric_id'] == 'jepa_similarity'), 1.0)

    def test_total_remains_null_without_full_common_metric_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            records = self.comparison(output)
            partial = self.exported(output, records, ['musiq'])
            self.assertEqual(len(partial), 4)
            self.assertTrue(all(row['status'] == 'incomplete_metric_selection'
                                and row['HWBScore'] is None for row in partial))

            pose = next(row for row in records if row['model'] == 'B'
                        and row['metric_id'] == 'pose_control')
            pose.update(score_0_1=None, status='missing_reference')
            missing = self.exported(output, records)
            self.assertEqual(len(missing), 4)
            self.assertTrue(all(row['missing_common_metrics'] == ['pose_control']
                                and row['status'] == 'unavailable_common_metrics'
                                and row['HWBScore'] is None for row in missing))
            self.assertIn(pose, read(output/'scores.json'))

            pose.update(metric_protocol='alternate_pose', score_0_1=0.4, status='ok')
            mixed = self.exported(output, records)
            self.assertEqual(len(mixed), 4)
            self.assertTrue(all(row['mixed_protocol_metrics'] == ['pose_control']
                                and row['status'] == 'mixed_protocols'
                                and row['HWBScore'] is None for row in mixed))
