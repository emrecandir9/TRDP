"""Checks for the Gate 2 control predictions and probe sweep (synthetic data, small networks)."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch
    import snntorch  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch and snntorch not installed')
class Predictions(unittest.TestCase):
    def test_prediction_vector(self):
        import gate2 as g2
        cells = np.zeros((6, 2, 2))
        cells[g2.CONTROLS.index('hand_stub'), 1, 0] = 1.
        cells[g2.CONTROLS.index('window_arm'), 1, 0] = .4
        v = g2.prediction_vector(cells)
        self.assertEqual(len(v), len(g2.QUANTITIES))
        self.assertEqual(v[g2.QUANTITIES.index('hand_stub-hand_gap_stub')], 1.)
        self.assertAlmostEqual(v[g2.QUANTITIES.index('window_arm-window_hand')], .4)

    def test_bootstrap_p_and_distinctness(self):
        import criteria as cr
        import gate2 as g2
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(20), 10)
        sets = cr.draws(groups, 400)
        self.assertLess(g2.bootstrap_p(rng.normal(.5, .3, 200), sets), .01)
        self.assertGreater(g2.bootstrap_p(rng.normal(0, .3, 200), sets), .05)
        a = rng.normal(0, .3, (200, 8))
        b = a + rng.normal(0, .3, (200, 8))
        self.assertFalse(g2.distinct(a, b, groups, sets)['distinct'])
        b[:, 6] += 1.
        self.assertTrue(g2.distinct(a, b, groups, sets)['distinct'])

    def test_probe_strengths_apply_per_copy(self):
        """The sweep runs several probe strengths in one batch: each copy gets its own amplitude, so
        the mean readout rises with strength while the pre-probe rates stay identical."""
        import evaluate as ev
        import network as nw
        import test_evaluate as te
        import trials
        b, net = te.bank(), te.network()
        threshold = nw.threshold(net)
        with te.ShortSession():
            s = trials.session(te.FAMILIES, 3, 2, 0, ('full_body',))
            tables = ev.simulate_grid(net, [ev.Model('ungated')] * 3, b, b.scaler(te.FAMILIES), s,
                                      [k * threshold for k in (1.0, 1.25, 1.5)], task_only=True)
        means = [t.fdi.mean() for t in tables]
        self.assertTrue(means[0] < means[1] < means[2])
        self.assertTrue(all(np.array_equal(t.pre_fdi_hz, tables[0].pre_fdi_hz) for t in tables))


if __name__ == '__main__':
    unittest.main()
