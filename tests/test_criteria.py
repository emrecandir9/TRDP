"""Checks for the criteria code (synthetic readouts with known answers; no real source)."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BODIES = ('full_body', 'upper_limb', 'detached_hand_silhouette')
RANGES = dict(fdi=(8., 30.), adm=(8., 30.))
INTERVALS = dict(S1=(-.67, .38), S2=(.34, 1.78))


def table(rng, effect=dict(full_body=1., upper_limb=1., detached_hand_silhouette=0.), pre=None, n=30):
    """Trials with FDI movement effect `effect` per display; pre[(display, kind)] = pre-probe FDI rate."""
    rows = []
    for display in BODIES:
        for kind in ('static', 'movement'):
            for _ in range(n):
                rows.append(dict(display=display, kind=kind, fdi=rng.normal(10 + effect[display] * (kind == 'movement'), 1),
                                 adm=rng.normal(9, 1), pre_fdi_hz=(pre or {}).get((display, kind), 1.), pre_adm_hz=1.))
    rows.append(dict(display='black', kind='fixation', fdi=10., adm=9., pre_fdi_hz=1., pre_adm_hz=1.))
    return pd.DataFrame(rows)


class Treatments(unittest.TestCase):
    def test_flags_and_both_treatments(self):
        import criteria as cr
        t = table(np.random.default_rng(0), pre={('detached_hand_silhouette', 'movement'): 12.})
        raw, flags = cr.treated_cells(t, BODIES, RANGES)
        removed, _ = cr.treated_cells(t, BODIES, RANGES, 'remove')
        maximum, _ = cr.treated_cells(t, BODIES, RANGES, 'maximum')
        self.assertTrue(flags[2, 1, 0] and flags.sum() == 1)                 # only DH movement FDI is over-driven
        self.assertTrue(np.isnan(removed[2, 1, 0]) and np.isfinite(removed).sum() == 11)
        self.assertGreater(maximum[2, 1, 0], raw[2, 1, 0])                    # set to the readout maximum (30 > ~10)
        self.assertTrue(np.isfinite(raw).all())

    def test_trimming_keeps_identical_values(self):
        import criteria as cr
        t = table(np.random.default_rng(1), pre={('full_body', 'movement'): 12.})
        cells, _ = cr.treated_cells(t, BODIES, RANGES, 'maximum')
        self.assertTrue(np.isfinite(cells).all())                             # a cell of equal values is not trimmed away


class Scores(unittest.TestCase):
    def cells(self, rng, n_sources=20, n_init=10, dh=0.):
        out, groups, inits = [], [], []
        for s in range(n_sources):
            for i in range(n_init):
                c = rng.normal(0, .1, (3, 2, 2))
                c[0, 1, 0] += .5
                c[1, 1, 0] += .4
                c[2, 1, 0] += dh
                out.append(c); groups.append(s); inits.append(i)
        return np.array(out), np.array(groups), np.array(inits)

    def test_human_pattern_meets_p_and_s(self):
        import criteria as cr
        cells, groups, inits = self.cells(np.random.default_rng(2))
        r = cr.score(cells, groups, inits, cr.draws(groups, 300), INTERVALS)
        self.assertEqual(r['criteria'], dict(P1=True, P2=True, P3=True, S1=True, S2=True))
        self.assertAlmostEqual(r['S2_ratio'], .8, delta=.1)

    def test_detached_hand_facilitation_fails_p1_and_s1(self):
        import criteria as cr
        cells, groups, inits = self.cells(np.random.default_rng(3), dh=.6)
        r = cr.score(cells, groups, inits, cr.draws(groups, 300), INTERVALS)
        self.assertFalse(r['criteria']['P1'])
        self.assertFalse(r['criteria']['S1'])
        self.assertTrue(r['criteria']['P2'])

    def test_complete_cases_per_contrast(self):
        import criteria as cr
        cells, groups, inits = self.cells(np.random.default_rng(4))
        cells[0, 2, 1, 0] = np.nan                                           # one instance lost its DH movement cell
        r = cr.score(cells, groups, inits, cr.draws(groups, 100), INTERVALS)
        self.assertEqual(r['instances']['C_FDI'], len(cells) - 1)
        self.assertEqual(r['instances']['S2'], len(cells))                   # S2 needs no detached hand
        self.assertAlmostEqual(r['delta_FB']['estimate'], np.mean(cells[:, 0, 1, 0] - cells[:, 0, 0, 0]))

    def test_ratio_with_non_positive_denominator_is_not_met(self):
        import criteria as cr
        cells, groups, inits = self.cells(np.random.default_rng(5))
        cells[:, 0, 1, 0] = cells[:, 0, 0, 0] - .5                            # negative full-body effect
        r = cr.score(cells, groups, inits, cr.draws(groups, 100), INTERVALS)
        self.assertFalse(r['criteria']['S2'])
        self.assertTrue(np.isnan(r['S2_ratio']))


class Verdicts(unittest.TestCase):
    def test_combination(self):
        import criteria as cr
        self.assertEqual(cr.verdict(dict(remove=True, maximum=True)), 'met')
        self.assertEqual(cr.verdict(dict(remove=False, maximum=False)), 'not met')
        self.assertEqual(cr.verdict(dict(remove=True, maximum=False)), 'undetermined')

    def test_decision_rule(self):
        import criteria as cr
        all_met = dict.fromkeys(('P1', 'P2', 'P3', 'S1', 'S2', 'S3'), 'met')
        self.assertEqual(cr.decision(all_met, dict(dinov2_cls='met', dinov2_patchmean='met')), 'met')
        self.assertEqual(cr.decision(all_met, dict(dinov2_cls='met', dinov2_patchmean='undetermined')), 'undetermined')
        self.assertEqual(cr.decision(dict(all_met, S2='not met'), dict(dinov2_cls='undetermined', dinov2_patchmean='met')), 'not met')

    def test_paired_difference_and_control_pairs(self):
        import criteria as cr
        rng = np.random.default_rng(6)
        groups = np.repeat(np.arange(20), 5)
        a, b = rng.normal(.5, .2, 100), rng.normal(.1, .2, 100)
        a[3] = np.nan
        d = cr.paired_difference(a, b, cr.draws(groups, 300))
        self.assertTrue(d['exceeds'] and d['instances'] == 99)
        displays = ('hand_stub', 'hand_gap_stub', 'window_hand', 'window_arm')
        cells = np.zeros((2, 4, 2, 2))
        cells[:, 0, 1, 0] = 1.                                                # stub facilitates, gap stub does not
        cells[:, 3, 1, 0] = .5                                                # arm window facilitates, hand window does not
        effects = cr.pair_effects(cells, displays)
        self.assertTrue(np.allclose(effects['hand_stub-hand_gap_stub'], 1.) and np.allclose(effects['window_arm-window_hand'], .5))

    def test_align_matches_instances(self):
        import criteria as cr
        common, a, b = cr.align([(1, 0), (2, 0), (3, 0)], [(3, 0), (1, 0)])
        self.assertEqual((common, a, b), ([(1, 0), (3, 0)], [0, 2], [1, 0]))


try:
    import torch  # noqa: F401
    import snntorch  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch and snntorch not installed')
class EndToEnd(unittest.TestCase):
    def test_simulated_session_scores(self):
        import criteria as cr
        import evaluate as ev
        import network as nw
        import test_evaluate as te
        import trials
        b, net = te.bank(), te.network()
        with te.ShortSession():
            s = trials.session(te.FAMILIES, 3, 2, 0, trials.BODIES)
            records = ev.simulate_trialwise(net, ev.Model('ungated'), b, b.scaler(te.FAMILIES), s, 1.2 * nw.threshold(net))
        for treatment in (None,) + cr.TREATMENTS:
            cells, flags = cr.treated_cells(records, trials.BODIES, dict(fdi=(-1., 99.), adm=(99., 99.)), treatment)
            self.assertEqual(cells.shape, (3, 2, 2))
            self.assertTrue(flags[..., 0].all() and not flags[..., 1].any())   # every FDI rate (>= 0) is above -1 Hz
            if treatment == 'remove':
                self.assertTrue(np.isnan(cells[..., 0]).all() and np.isfinite(cells[..., 1]).all())


if __name__ == '__main__':
    unittest.main()
