"""The criterion-status table (docs/criteria-status.json) agrees with spec §5.2 and the evaluation's models."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
STATUS = json.loads((ROOT / 'docs/criteria-status.json').read_text())


class Table(unittest.TestCase):
    def test_every_model_has_every_criterion_with_a_known_status_and_a_source(self):
        for model, row in STATUS['models'].items():
            for criterion in STATUS['criteria']:
                entry = row[criterion]
                self.assertIn(entry['status'], STATUS['statuses'], f'{model} {criterion}')
                self.assertTrue(entry['why'] and entry['section'], f'{model} {criterion}')

    def test_the_models_are_those_the_evaluation_scores(self):
        import run_evaluation as rv
        self.assertEqual(set(STATUS['models']), set(rv.H2_MODELS) | {'h0', 'h1'})
        criteria = {c.split('_')[0] for c in STATUS['criteria']}
        self.assertEqual(criteria, set(rv.CRITERIA) | {'S3a', 'S3b'} - {'S3'})

    def test_only_the_upper_limb_tests_are_open_for_h2(self):
        """Spec §5.2, amendment of 10 November 2026: for H2 the informative criteria are the upper-limb half
        of P2 and S2."""
        h2 = STATUS['models']['h2']
        open_ = sorted(c for c in STATUS['criteria'] if h2[c]['status'] == 'open')
        self.assertEqual(open_, sorted(STATUS['open_for_h2']))
        self.assertEqual(open_, ['P2_UL', 'S2'])
        self.assertIn('upper-limb half of P2 and S2', (ROOT / 'docs/analysis-spec.md').read_text())
        for c in ('P1', 'P3', 'S1', 'S3b'):
            self.assertEqual(h2[c]['status'], 'built_in', c)
        self.assertEqual(h2['P2_FB']['status'], 'calibrated')
        self.assertEqual(h2['S3a']['status'], 'previewed')

    def test_s3_is_scored_for_h2_only(self):
        for model, row in STATUS['models'].items():
            if model != 'h2':
                self.assertEqual({row['S3a']['status'], row['S3b']['status']}, {'not_scored'}, model)


if __name__ == '__main__':
    unittest.main()
