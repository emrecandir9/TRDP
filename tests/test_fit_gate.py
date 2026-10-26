"""Checks for the fit of c (small network, synthetic features; no real source)."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch
    import snntorch  # noqa: F401
except ImportError:
    torch = None


class Choice(unittest.TestCase):
    def test_closest_ratio_ties_and_undefined(self):
        import fit_gate as fg
        rows = [(0., -.05, 1.), (.05, -.15, 1.), (.1, -.1, -.2), (.15, .3, 1.)]
        self.assertEqual(fg.choose(rows), 0.)                     # -0.05 and -0.15 tie at 0.05: the smaller c; c = 0.1 has delta FB <= 0
        self.assertIsNone(fg.choose([(0., np.nan, 1.), (.5, -.1, 0.)]))
        self.assertEqual(fg.choose([(0., .2, 1.), (.5, -.09, 1.)]), .5)

    def test_curve_uses_sources_with_both_deltas(self):
        import fit_gate as fg
        rows = fg.curve({0.: [(1., -.1, False, False), (np.nan, .3, True, False), (2., -.2, False, True)]})
        c, ratio, fb, dh, used, over_fb, over_dh = rows[0]
        self.assertEqual(used, 2)
        self.assertAlmostEqual(ratio, -.15 / 1.5)
        self.assertAlmostEqual(over_fb, 1 / 3)


@unittest.skipIf(torch is None, 'torch and snntorch not installed')
class Fit(unittest.TestCase):
    def setUp(self):
        import bank as bk
        import trials
        self.families = [f'r0{s}_v{i}_s9_neutral' for s in (1, 2, 3) for i in range(2)]
        rows = [(f, c, r) for f in self.families for c in trials.BODIES for r in ('static', 'movement')]
        x = np.random.default_rng(0).normal(0, 1, (len(rows), 8))
        self.bank = bk.Bank('r', arrays=dict(r=x), index=pd.DataFrame(rows, columns=['family', 'condition', 'frame']),
                            black=np.zeros(8))

    def test_full_body_readouts_are_identical_across_c_when_its_gate_is_one(self):
        """a(FB) = 1 gives g = 1 on full-body trials for every c: with common random numbers their raw
        readouts are identical along the grid, and only the detached hand changes."""
        import evaluate as ev
        import fit_gate as fg
        import network as nw
        import test_evaluate as te
        import trials
        net = te.network()
        a = lambda f, d: 1. if d == 'full_body' else 0.
        models = [ev.Model('gated', c=c, attachment=a) for c in fg.GRID]
        with te.ShortSession():
            s = trials.session(self.families[2:4], 2, 1, 0, fg.DISPLAYS, fg.KIND_CODE)
            tables = ev.simulate_grid(net, models, self.bank, self.bank.scaler(self.families), s,
                                      [1.2 * nw.threshold(net)] * len(models), task_only=True)
        fb = [t[t.display == 'full_body'][['fdi', 'adm', 'pre_fdi_hz']].to_numpy() for t in tables]
        dh = [t[t.display != 'full_body'][['fdi', 'adm', 'pre_fdi_hz']].to_numpy() for t in tables]
        self.assertTrue(all(np.array_equal(f, fb[0]) for f in fb))
        self.assertFalse(np.array_equal(dh[0], dh[-1]))
        self.assertTrue(all((t.kind != 'fixation').all() for t in tables))

    def test_fit_is_reproducible_and_complete(self):
        import fit_gate as fg
        import test_evaluate as te
        net = te.network()
        attachments = dict(continuity=lambda f, d: 1. if d == 'full_body' else 0., area=lambda f, d: .5)
        ranges = dict(fdi=(99., 99.), adm=(99., 99.))
        with te.ShortSession():
            one = fg.fit(net, self.bank.scaler(self.families), self.bank, dict(seed=0), attachments, ranges, [2, 3])
            two = fg.fit(net, self.bank.scaler(self.families), self.bank, dict(seed=0), attachments, ranges, [2, 3])
        self.assertEqual(one, two)
        amplitude, fits = one
        for gate in attachments:
            for t in ('raw', 'remove', 'maximum'):
                self.assertEqual(len(fits[gate][t]['curve']), len(fg.GRID))
                self.assertTrue(fits[gate][t]['c'] is None or fits[gate][t]['c'] in fg.GRID)


if __name__ == '__main__':
    unittest.main()
