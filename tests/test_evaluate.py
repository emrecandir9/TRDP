"""CPU checks for the evaluation simulator (small networks, synthetic features; no real source)."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

try:
    import torch
    import snntorch  # noqa: F401
except ImportError:   # the review environment needs no torch
    torch = None
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))

FAMILIES = [f'r03_v{i}_s9_neutral' for i in range(4)]
SHORT = dict(fixation_trials=2, trials_per_type=4, familiarisation_ms=500, first_min_ms=400, first_max_ms=500)


def bank(dim=8, seed=0):
    import bank as bk
    import trials
    rows = [(f, c, r) for f in FAMILIES for c in trials.BODIES + ('hand_only',) for r in ('static', 'movement')]
    index = pd.DataFrame(rows, columns=['family', 'condition', 'frame'])
    x = np.random.default_rng(seed).normal(0, 1, (len(rows), dim))
    return bk.Bank('r', arrays=dict(r=x), index=index, black=np.zeros(dim))


class ShortSession:
    """Shorter trials and blocks, so the continuous session runs quickly on a CPU."""
    def __enter__(self):
        import trials
        self.saved = dict(trials.TIMING)
        trials.TIMING.update(SHORT)

    def __exit__(self, *_):
        import trials
        trials.TIMING.clear()
        trials.TIMING.update(self.saved)


def network(spiking=True, membrane_sd=None, seed=0):
    import network as nw
    config = dict(nw.CONFIG, n_visual=32, n_motor=16)
    if membrane_sd is not None:
        config['membrane_sd'] = membrane_sd
    return nw.Network(8, config, spiking=spiking, seed=seed)


@unittest.skipIf(torch is None, 'torch and snntorch not installed')
class Models(unittest.TestCase):
    def test_inputs_and_gates_per_hypothesis(self):
        import evaluate as ev
        import trials
        key = ('r03_v0_s9_neutral', 'upper_limb', 'movement')
        static = ('r03_v0_s9_neutral', 'upper_limb', 'static')
        gated = ev.Model('gated', c=.2, attachment=lambda f, d: .5)
        self.assertAlmostEqual(gated.gate(key), .2 + .8 * .5)
        self.assertAlmostEqual(gated.gate(static), gated.gate(key))       # a is a property of the display
        self.assertAlmostEqual(gated.gate(trials.BLACK), .2)               # fixation: no forearm, a = 0
        h1 = ev.Model('h1', salience=lambda f, d: 3., mean_salience=2.)
        self.assertEqual((h1.gate(static), h1.gate(key), h1.gate(trials.BLACK)), (1., 2.5, 1.))
        h0 = ev.Model('h0')
        self.assertEqual(h0.input_key(key), ('r03_v0_s9_neutral', 'hand_only', 'movement'))
        self.assertEqual(h0.gate(key), 1.)
        self.assertEqual(ev.Model('ungated').input_key(key), key)
        with self.assertRaises(ValueError):
            ev.Model('gated', c=.2)


@unittest.skipIf(torch is None, 'torch and snntorch not installed')
class Simulation(unittest.TestCase):
    def session(self, displays=None):
        import trials
        return trials.session(FAMILIES, 3, 2, 0, displays or trials.BODIES)

    def scaler(self, b):
        return b.scaler(FAMILIES)

    def test_trialwise_records_every_trial_and_is_reproducible(self):
        import evaluate as ev
        b, net = bank(), network()
        with ShortSession():
            s = self.session()
            one = ev.simulate_trialwise(net, ev.Model('ungated'), b, self.scaler(b), s, 3.)
            two = ev.simulate_trialwise(net, ev.Model('ungated'), b, self.scaler(b), s, 3.)
        self.assertEqual(len(one), len(s.trials))
        self.assertEqual(set(one.kind), {'fixation', 'static', 'movement'})
        pd.testing.assert_frame_equal(one, two)

    def test_common_random_numbers_across_models(self):
        """A gated model with c = 1 gives g = 1 everywhere: exactly the ungated network's run."""
        import evaluate as ev
        import network as nw
        b, net = bank(), network()
        probe = ev.PROBE_FACTOR * nw.threshold(net)
        with ShortSession():
            s = self.session()
            run = lambda model: ev.simulate_trialwise(net, model, b, self.scaler(b), s, probe)
            ungated = run(ev.Model('ungated'))
            gated = run(ev.Model('gated', c=1., attachment=lambda f, d: 0.))
            closed = run(ev.Model('gated', c=0., attachment=lambda f, d: 0.))
        pd.testing.assert_frame_equal(ungated, gated)
        task = closed.kind != 'fixation'
        self.assertFalse(np.allclose(closed.loc[task, 'fdi'], ungated.loc[task, 'fdi']))   # c = 0 cuts the visual input

    def test_continuous_session_equals_trialwise_for_the_feedforward_twin(self):
        """Without noise the feedforward state at the probe depends on the second frame alone (§6.4),
        so both modes give the same readout and background on every trial."""
        import evaluate as ev
        b, net = bank(), network(spiking=False, membrane_sd=0.)
        with ShortSession():
            s = self.session()
            a = ev.simulate_trialwise(net, ev.Model('ungated'), b, self.scaler(b), s, 3.)
            c = ev.simulate_continuous(net, ev.Model('ungated'), b, self.scaler(b), s, 3.)
        np.testing.assert_allclose(a[['fdi', 'adm', 'pre_fdi_hz', 'pre_adm_hz']], c[['fdi', 'adm', 'pre_fdi_hz', 'pre_adm_hz']],
                                   rtol=1e-4, atol=1e-4)

    def test_continuous_probe_twin_shares_noise(self):
        """With noise, a zero-amplitude probe gives exactly zero readout: the twin branches see the
        same noise, and the session continues identically."""
        import evaluate as ev
        b, net = bank(), network()
        with ShortSession():
            out = ev.simulate_continuous(net, ev.Model('ungated'), b, self.scaler(b), self.session(), 0.)
        self.assertTrue((out[['fdi', 'adm']].to_numpy() == 0).all())

    def test_held_out_source_is_refused(self):
        import evaluate as ev
        b, net = bank(), network()
        report = dict(held_out_source_not_used=3, seed=0)
        with self.assertRaises(ValueError):
            ev.simulate_instance(net, ev.Model('ungated'), b, self.scaler(b), report, 3)


class Readouts(unittest.TestCase):
    def table(self, rng):
        import trials
        rows = []
        for display in trials.BODIES:
            for kind in ('static', 'movement'):
                for i in range(30):
                    rows.append(dict(trial=len(rows), display=display, kind=kind, fdi=rng.normal(10 + (kind == 'movement'), 1),
                                     adm=rng.normal(8, 1), pre_fdi_hz=1., pre_adm_hz=1.))
        return pd.DataFrame(rows)

    def test_trimming_and_normalisation(self):
        import evaluate as ev
        import trials
        table = self.table(np.random.default_rng(0))
        table.loc[0, 'fdi'] = 100.                                     # one outlier in the first cell
        cells, rates, removed = ev.instance_cells(table, trials.BODIES)
        self.assertEqual(cells.shape, (3, 2, 2))
        self.assertGreaterEqual(removed, 1)
        _, _, kept_all = ev.instance_cells(table, trials.BODIES, trim=False)
        self.assertEqual(kept_all, 0)
        self.assertTrue(np.allclose(rates, 1.))
        self.assertGreater(cells[0, 1, 0], cells[0, 0, 0])            # movement above static for FDI

    def test_summary_uses_the_human_contrasts(self):
        import evaluate as ev
        import human_target as ht
        rng = np.random.default_rng(1)
        cells = rng.normal(0, .1, (40, 3, 2, 2))
        cells[:, :2, 1, 0] += .5                                       # FDI facilitation for FB and UL only
        groups, inits = np.repeat(np.arange(20), 2), np.tile([0, 1], 20)
        table, per_init = ev.summarise(cells, groups, inits, samples=500)
        row = table.set_index('name').loc['C_FDI']
        self.assertAlmostEqual(row.estimate, ht.contrasts()[0][2](cells.mean(0)))
        self.assertTrue(row.ci_low < row.estimate < row.ci_high and row.ci_low > 0)
        self.assertEqual(set(per_init), {0, 1})

    def test_bootstrap_resamples_groups_then_instances(self):
        import evaluate as ev
        cells = np.repeat(np.arange(5.), 3)[:, None]                  # identical instances within each group
        groups = np.repeat(np.arange(5), 3)
        boot = ev.hierarchical_bootstrap(cells, groups, samples=200, seed=0)
        again = ev.hierarchical_bootstrap(cells, groups, samples=200, seed=0)
        self.assertTrue(np.array_equal(boot, again))
        self.assertGreater(boot.std(), 0)                              # between-group variation is resampled
        within = ev.hierarchical_bootstrap(np.zeros((15, 1)) + np.tile([0., 1., 2.], 5)[:, None], groups, samples=200)
        self.assertGreater(within.std(), 0)                            # and so is variation within groups


if __name__ == '__main__':
    unittest.main()
