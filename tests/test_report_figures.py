"""The report's figures and tables run end to end on a synthetic result.json (final-phase plan, task A1).

The result is made by the evaluation's own scoring (`run_evaluation.report`) from the small untrained networks
of tests/test_run_evaluation.py, written under every encoder and as the all-context configuration."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for folder in ('model', 'tests', 'analysis'):
    sys.path.insert(0, str(ROOT / folder))

try:
    import torch  # noqa: F401
    import snntorch  # noqa: F401
    import matplotlib  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch, snntorch and matplotlib needed')
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import criteria as cr
        import run_evaluation as rv
        import test_run_evaluation as tre
        tre.EndToEnd.setUpClass()
        h2, h0 = tre.EndToEnd.h2, tre.EndToEnd.h0
        old = cr.human_intervals
        cr.human_intervals = lambda: tre.INTERVALS
        try:
            with tempfile.TemporaryDirectory() as d:
                folders = {f'{h}_{e}_snn': records for e in ('openclip', 'dinov2_cls', 'dinov2_patchmean')
                           for h, records in (('h2', h2), ('h0', h0), ('h1', [dict(r, hypothesis='h1', labels={
                               k.replace('h0|', 'h1|'): v for k, v in r['labels'].items()}) for r in h0]))}
                folders['h2_openclip_snn_allctx'] = h2
                for folder, records in folders.items():
                    (Path(d) / folder).mkdir()
                    for i, r in enumerate(records):
                        (Path(d) / folder / f'{i}.json').write_text(json.dumps(r))
                cls.result = json.loads(json.dumps(rv.report(Path(d), samples=50)))
        finally:
            cr.human_intervals = old

    def test_the_result_carries_cells_for_every_model(self):
        models = self.result['main']['openclip']['analyses']['remove']['models']
        for m in ('h2', 'ungated', 'h0', 'h1'):
            cells = models[m]['cells']
            self.assertEqual(cells['order'][0], ['full_body', 'upper_limb', 'detached_hand_silhouette'])
            self.assertEqual((len(cells['mean']), len(cells['mean'][0]), len(cells['mean'][0][0])), (3, 2, 2))
        self.assertIn('decision', self.result)

    def test_every_figure_and_table_is_written(self):
        import report_figures as rf
        with tempfile.TemporaryDirectory() as d:
            written = rf.write_all(self.result, d)
            names = {Path(p).name for p in written}
            for name in ('cells.png', 'effects.png', 'diets.png', 'controls.png', 'criteria.csv', 'effects.csv',
                         'robustness.csv', 'over_driven.csv', 'tables.md'):
                self.assertIn(name, names)
                self.assertGreater((Path(d) / name).stat().st_size, 0)
            text = (Path(d) / 'tables.md').read_text()
            self.assertIn('Decision rule', text)

    def test_criterion_table_joins_verdicts_and_status(self):
        import report_figures as rf
        table = rf.criterion_table(self.result)
        h2 = table[table.model == rf.MODEL_STYLE['h2'][2]].set_index('criterion')
        self.assertEqual(h2.loc['P2', 'status'], 'FB calibrated, UL open')
        self.assertEqual(h2.loc['S2', 'status'], 'open')
        self.assertEqual(h2.loc['P1', 'status'], 'built in')
        self.assertTrue(set(table.verdict) <= {'met', 'not met', 'undetermined'})

    def test_robustness_table_lists_encoders_repetitions_and_configurations(self):
        import report_figures as rf
        table = rf.robustness_table(self.result)
        self.assertIn('main, openclip', set(table.scored))
        self.assertIn('configuration snn_allctx', set(table.scored))
        self.assertIn('rectangle, openclip', set(table.scored))

    def test_human_inputs_have_the_paper_shape(self):
        import report_figures as rf
        mean, low, high = rf.human_cells()
        self.assertEqual(mean.shape, (3, 2, 2))
        self.assertTrue(((low <= mean) & (mean <= high)).all())
        self.assertAlmostEqual(rf.human_contrasts()['C_FDI'][0], 0.283, places=3)


if __name__ == '__main__':
    unittest.main()
