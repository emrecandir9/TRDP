"""Checks for the evaluation driver (small untrained networks, synthetic features; no real source)."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

try:
    import torch
    import snntorch  # noqa: F401
except ImportError:
    torch = None
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))

INTERVALS = dict(S1=(-.667, .379), S2=(.342, 1.781), DH_FB=(-.672, .347))
SOURCES, SEEDS = (1, 2, 3), (0, 1)


class Lock(unittest.TestCase):
    def test_evaluation_opens_only_with_an_unchanged_freeze_record(self):
        import run_evaluation as rv
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'code.py').write_text('a = 1\n')
            record = root / 'freeze.json'
            with self.assertRaisesRegex(ValueError, 'not open'):
                rv.opened(record, root)
            rv.freeze(record, root, ('*.py',), date='2026-12-11')
            self.assertEqual(rv.opened(record, root)['date'], '2026-12-11')
            with self.assertRaisesRegex(ValueError, 'already frozen'):
                rv.freeze(record, root, ('*.py',))
            (root / 'code.py').write_text('a = 2\n')
            with self.assertRaisesRegex(ValueError, 'Changed since the freeze'):
                rv.opened(record, root)

    def test_the_real_evaluation_is_closed(self):
        """Until the freeze of spec §8 there is no freeze record, so `simulate` cannot run."""
        import run_evaluation as rv
        if not rv.FREEZE.exists():
            with self.assertRaisesRegex(ValueError, 'not open'):
                rv.opened()


    def test_a_network_needs_the_open_evaluation_for_its_held_out_source(self):
        """simulate_network checks the freeze record itself, and a rehearsal can use neither the held-out
        source nor the upper limb."""
        import json
        import run_evaluation as rv
        if torch is None or rv.FREEZE.exists():
            self.skipTest('needs torch and a closed evaluation')
        import test_evaluate as te
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'health_r_snn_fold3_seed0.pt'
            net = te.network()
            torch.save(dict(state=net.state_dict(), readout={}, config=net.config, scaler=[[0.] * 8, [1.] * 8], mean_salience=None), path)
            path.with_suffix('.json').write_text(json.dumps(dict(spiking=True, seed=0, held_out_source_not_used=3, representation='r')))
            with self.assertRaisesRegex(ValueError, 'not open'):
                rv.simulate_network(path)
            with self.assertRaisesRegex(ValueError, 'held out'):
                rv.simulate_network(path, source=3)
            with self.assertRaisesRegex(ValueError, 'upper limb'):
                rv.simulate_network(path, source=4)


class Runs(unittest.TestCase):
    def test_runs_and_labels_follow_the_fit(self):
        import fit_gate as fg
        import run_evaluation as rv
        flat = [(c, .1, 1., .1) for c in fg.GRID]                     # every c inside the human interval
        steep = [(c, 2 * c, 1., 2 * c) for c in fg.GRID]              # inside up to c = 0.15
        fits = dict(continuity={a: dict(c=0., curve=steep) for a in rv.ANALYSES},
                    area=dict(raw=dict(c=.5, curve=flat), remove=dict(c=None, curve=flat), maximum=dict(c=.5, curve=flat)))
        runs, labels = rv.gated_runs(fits, INTERVALS['DH_FB'])
        self.assertEqual(runs['continuity@0.00'], ('continuity', 0.))
        self.assertEqual(labels['h2|remove'], 'continuity@0.00')
        self.assertEqual(labels['h2_cplus|remove'], 'continuity@0.15')
        self.assertEqual(labels['ungated|maximum'], 'ungated')
        self.assertNotIn('area|remove', labels)                        # undefined fit: no run under that analysis
        self.assertEqual(labels['area|maximum'], 'area@0.50')


@unittest.skipIf(torch is None, 'torch and snntorch not installed')
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import bank as bk
        import evaluate as ev
        import network as nw
        import run_evaluation as rv
        import test_evaluate as te
        import trials
        displays = tuple(dict.fromkeys(trials.BODIES + trials.SENSITIVITY_BODIES + trials.CONTROL_DISPLAYS + ('hand_only',)))
        families = [f'r{s:02d}_v{i}_s9_neutral' for s in SOURCES for i in range(3)]
        rows = [(f, c, r) for f in families for c in displays for r in ('static', 'movement')]
        x = np.random.default_rng(1).normal(0, 1, (len(rows), 8))
        bank = bk.Bank('r', arrays=dict(r=x), index=pd.DataFrame(rows, columns=['family', 'condition', 'frame']), black=np.zeros(8))
        connected = ('full_body', 'upper_limb', 'hand_stub', 'window_arm')
        a = lambda family, display: float(display in connected)
        area = lambda family, display: .5
        cls.h2, cls.h0 = [], []
        with te.ShortSession():
            for source in SOURCES:
                for seed in SEEDS:
                    net = te.network(seed=seed)
                    report = dict(seed=seed, held_out_source_not_used=source, representation='r', spiking=True,
                                  readout_range_hz=99., readout_peak=20., adm_readout_range_hz=99., adm_readout_peak=20.,
                                  criteria=dict(task=seed == 0))
                    amplitude = 1.2 * nw.threshold(net)
                    models = {'continuity@0.00': ev.Model('gated', c=0., attachment=a), 'continuity@1.00': ev.Model('gated', c=1., attachment=a),
                              'area@0.00': ev.Model('gated', c=0., attachment=area), 'ungated': ev.Model('ungated')}
                    labels = {f'{n}|{t}': k for t in rv.ANALYSES for n, k in
                              (('h2', 'continuity@0.00'), ('h2_cplus', 'continuity@1.00'), ('area', 'area@0.00'), ('ungated', 'ungated'))}
                    scaler = bank.scaler(families)
                    cls.h2.append(rv.run_instance(net, scaler, bank, report, models, labels, amplitude, source))
                    cls.h0.append(rv.run_instance(net, scaler, bank, dict(report, hypothesis='h0'), {'h0': ev.Model('h0')},
                                                  {f'h0|{t}': 'h0' for t in rv.ANALYSES}, amplitude, source))

    def test_instance_records_hold_every_session_and_model(self):
        import run_evaluation as rv
        r = self.h2[0]
        self.assertEqual(set(r['tables']), set(rv.SESSIONS))
        self.assertEqual(set(r['tables']['primary']), {'continuity@0.00', 'continuity@1.00', 'area@0.00', 'ungated'})
        self.assertEqual(rv.cells_of(r, 'primary', 'h2', 'raw').shape, (3, 2, 2))
        self.assertTrue(np.isnan(rv.cells_of(r, 'primary', 'h4', 'raw')).all())            # no such run here

    def test_an_open_gate_is_the_ungated_network(self):
        """H2 at c = 1 and the ungated network are the same run, so the gate adds exactly nothing."""
        import run_evaluation as rv
        for r in self.h2:
            np.testing.assert_array_equal(rv.cells_of(r, 'primary', 'h2_cplus', 'remove'), rv.cells_of(r, 'primary', 'ungated', 'remove'))
        scored = rv.score_encoder(self.h2, self.h0, [], INTERVALS, samples=200)
        self.assertEqual(scored['analyses']['remove']['necessity_cplus']['estimate'], 0.)
        self.assertEqual(scored['gate_necessary']['adds_to_C_FDI'] in ('met', 'not met', 'undetermined'), True)

    def test_sessions_show_their_displays_and_fixation(self):
        """Every session holds its displays and the fixation trials; with c = 0 a display without a
        connected forearm gets no visuomotor input, so its readout does not depend on which frame is shown:
        the closed-gate run equals itself with the movement frames replaced, i.e. the area run at g = 0.5
        differs from it while the two share every noise draw."""
        import run_evaluation as rv
        for r in self.h2:
            for name, (displays, _) in rv.SESSIONS.items():
                table = pd.DataFrame(r['tables'][name]['continuity@0.00'])
                self.assertEqual(set(table.display), set(displays) | {'black'})
                self.assertEqual(int((table.kind == 'fixation').sum()), 4)
            closed = pd.DataFrame(r['tables']['primary']['continuity@0.00'])
            half = pd.DataFrame(r['tables']['primary']['area@0.00'])
            ungated = pd.DataFrame(r['tables']['primary']['ungated'])
            full = closed.display == 'full_body'
            np.testing.assert_array_equal(closed.fdi[full], ungated.fdi[full])           # a = 1: the gate is open
            self.assertFalse(np.array_equal(half.fdi[full], ungated.fdi[full]))

    def test_scores_verdicts_and_repetitions(self):
        import run_evaluation as rv
        scored = rv.score_encoder(self.h2, self.h0, [], INTERVALS, samples=200)
        self.assertEqual(set(scored['analyses']), set(rv.ANALYSES))
        self.assertEqual(scored['instances'], dict(h2=6, h0=6, h1=0))
        for name in ('h2', 'ungated', 'area', 'h0'):
            self.assertEqual(set(scored['verdicts'][name]) >= set(rv.CRITERIA), True)
            self.assertTrue(all(v in ('met', 'not met', 'undetermined') for v in scored['verdicts'][name].values()))
        self.assertEqual({'S3', 'S3a', 'S3b'} <= set(scored['verdicts']['h2']), True)
        self.assertEqual(set(scored['analyses']['raw']['S3a']['pairs']), {'hand_stub-hand_gap_stub', 'window_arm-window_hand'})
        self.assertEqual(scored['analyses']['raw']['S3b']['instances'], 6)
        self.assertEqual(set(scored['over_driven']['h2']), set(rv.SESSIONS['primary'][0]))
        fewer = rv.score_encoder(self.h2, self.h0, [], INTERVALS, exclude=(3,), samples=100)
        self.assertEqual(fewer['instances']['h2'], 4)
        healthy = rv.score_encoder(self.h2, self.h0, [], INTERVALS, healthy_only=True, samples=100)
        self.assertEqual(healthy['instances']['h2'], 3)
        rectangle = rv.score_encoder(self.h2, self.h0, [], INTERVALS, session='rectangle', samples=100)
        self.assertNotIn('S3a', rectangle['analyses']['raw'])
        by_encoder = {e: scored for e in (rv.PRIMARY_ENCODER,) + rv.ROBUSTNESS_ENCODERS}
        self.assertIn(rv.decide(by_encoder), ('met', 'not met', 'undetermined'))

    def test_scoring_is_reproducible(self):
        import json
        import run_evaluation as rv
        one = rv.score_encoder(self.h2, self.h0, [], INTERVALS, samples=100)
        two = rv.score_encoder(self.h2, self.h0, [], INTERVALS, samples=100)
        self.assertEqual(json.dumps(one, sort_keys=True), json.dumps(two, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
