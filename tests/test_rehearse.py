"""Checks for the training-source rehearsal of the evaluation (small untrained networks, synthetic features)."""
from pathlib import Path
import json
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
ROOT = Path(__file__).resolve().parents[1]
for folder in ('model', 'tests', 'scripts'):
    sys.path.insert(0, str(ROOT / folder))

SOURCES, SEEDS = (1, 2, 3), (0, 1)


class Displays(unittest.TestCase):
    def test_the_rehearsal_never_shows_the_upper_limb_or_the_rectangle(self):
        import rehearse
        shown = {d for displays, _ in rehearse.SESSIONS.values() for d in displays}
        self.assertNotIn('upper_limb', shown)
        self.assertNotIn('detached_hand', shown)
        self.assertIn('detached_hand_silhouette', shown)


@unittest.skipIf(torch is None, 'torch and snntorch not installed')
class Scoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import bank as bk
        import evaluate as ev
        import network as nw
        import rehearse
        import run_evaluation as rv
        import test_evaluate as te
        import trials
        displays = tuple(dict.fromkeys(trials.BODIES + trials.CONTROL_DISPLAYS + ('hand_only',)))
        families = [f'r{s:02d}_v{i}_s9_neutral' for s in SOURCES for i in range(3)]
        rows = [(f, c, r) for f in families for c in displays for r in ('static', 'movement')]
        x = np.random.default_rng(1).normal(0, 1, (len(rows), 8))
        bank = bk.Bank('r', arrays=dict(r=x), index=pd.DataFrame(rows, columns=['family', 'condition', 'frame']), black=np.zeros(8))
        connected = ('full_body', 'upper_limb', 'hand_stub', 'window_arm')
        a = lambda family, display: float(display in connected)
        cls.records = []
        with te.ShortSession():
            for source in SOURCES:
                for seed in SEEDS:
                    net = te.network(seed=seed)
                    report = dict(seed=seed, held_out_source_not_used=source, representation='r', spiking=True,
                                  readout_range_hz=99., readout_peak=20., adm_readout_range_hz=99., adm_readout_peak=20.,
                                  criteria=dict(task=True))
                    models = {'continuity@0.00': ev.Model('gated', c=0., attachment=a), 'continuity@1.00': ev.Model('gated', c=1., attachment=a),
                              'area@0.00': ev.Model('gated', c=0., attachment=lambda f, d: .5), 'ungated': ev.Model('ungated')}
                    labels = {f'{n}|{t}': k for t in rv.ANALYSES for n, k in
                              (('h2', 'continuity@0.00'), ('h2_cplus', 'continuity@1.00'), ('area', 'area@0.00'), ('ungated', 'ungated'))}
                    cls.records.append(dict(rv.run_instance(net, bank.scaler(families), bank, report, models, labels,
                                                            1.2 * nw.threshold(net), source, rehearse.SESSIONS), seconds=1.))

    def test_records_hold_only_the_rehearsal_displays(self):
        import rehearse
        shown = set(self.records[0]['tables']['primary']['ungated']['display'])
        self.assertEqual(shown - {'black'}, set(rehearse.PRIMARY))

    def test_an_open_gate_has_no_effect_on_the_detached_hand(self):
        import rehearse
        r = rehearse.score_configuration(self.records, samples=200)
        self.assertEqual(r['instances'], len(SOURCES) * len(SEEDS))
        for analysis in ('raw', 'remove', 'maximum'):
            gate = r['analyses'][analysis]['gate_on_detached_hand']
            self.assertAlmostEqual(gate['h2_cplus']['estimate'], 0., places=9)   # c = 1: the gate stays open
            self.assertIn('stub', next(iter(r['analyses'][analysis]['control_pairs_h2_minus_area'])))
        self.assertEqual(set(r['over_driven']['h2']), {'primary', 'controls'})

    def test_score_writes_costs_and_configurations(self):
        import rehearse
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / 'h2_r_snn'
            folder.mkdir()
            for i, record in enumerate(self.records):
                (folder / f'health_{i}.json').write_text(json.dumps(record))
            (folder / 'health_9.failed.txt').write_text('RuntimeError: test\n')
            result = rehearse.score(d, samples=100)
            self.assertEqual(result['costs']['networks'], len(self.records))
            self.assertEqual(result['costs']['failures'], ['h2_r_snn/health_9.failed.txt'])
            self.assertTrue((Path(d) / 'rehearsal.json').is_file())


if __name__ == '__main__':
    unittest.main()
