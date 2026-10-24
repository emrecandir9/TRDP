"""CPU checks for the trial simulator and feature bank; the real-data test needs output/features."""
from collections import Counter
from itertools import permutations
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
import bank as bk
import trials as t

FAMILIES = [f'r03_v{k:02d}' for k in range(24)]


def make(initialisation=0, rank=2, displays=t.BODIES, kind_code=0):
    return t.session(FAMILIES, 3, rank, initialisation, displays, kind_code)


class Session(unittest.TestCase):
    def test_counts_and_structure(self):
        s = make()
        kinds = Counter(x['kind'] for x in s.trials)
        self.assertEqual(kinds, Counter(fixation=40, static=90, movement=90))
        self.assertTrue(all(x['kind'] == 'fixation' for x in s.trials[:20] + s.trials[-20:]))
        for block in range(3):
            b = [x for x in s.trials if x['block'] == block]
            self.assertEqual(Counter(x['kind'] for x in b), Counter(static=30, movement=30))
            self.assertEqual(len({x['display'] for x in b}), 1)
        self.assertEqual({x['display'] for x in s.trials if x['block'] >= 0}, set(t.BODIES))

    def test_timing(self):
        s = make()
        for x in s.trials:
            self.assertTrue(2250 <= x['first_ms'] <= 3250)
            self.assertEqual(x['onset2_ms'] - x['start_ms'], x['first_ms'])
            self.assertEqual(x['probe_ms'], x['onset2_ms'] + 250)
            self.assertEqual(x['end_ms'], x['onset2_ms'] + 750)
            self.assertEqual(x['probe_ms'] + t.READOUT_MS[1], x['onset2_ms'] + 305)   # readout ends inside frame 2
        segments = s.segments
        self.assertEqual(segments[0][0], 0)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(segments, segments[1:])))   # contiguous, no gaps
        for block in s.blocks:
            start = block['familiarisation_start_ms']
            segment = next(x for x in segments if x[0] == start)
            self.assertEqual(segment[1] - start, 60000)
            self.assertEqual(segment[2], (block['familiarisation_family'], block['display'], 'static'))
        jitters = [x['first_ms'] for x in s.trials]
        self.assertGreater(len(set(jitters)), 100)                                  # integer ms, spread out

    def test_frames(self):
        s = make()
        by_start = {seg[0]: seg for seg in s.segments}
        for x in s.trials:
            first, second = by_start[x['start_ms']][2], by_start[x['onset2_ms']][2]
            if x['kind'] == 'fixation':
                self.assertEqual(first, t.BLACK)
                self.assertEqual(second, t.BLACK)
            else:
                self.assertEqual(first, (x['family'], x['display'], 'static'))
                self.assertEqual(second, (x['family'], x['display'], x['kind'] if x['kind'] == 'movement' else 'static'))

    def test_variants_cover_the_source(self):
        s = make()
        for block in range(3):
            for kind in ('static', 'movement'):
                used = Counter(x['family'] for x in s.trials if x['block'] == block and x['kind'] == kind)
                self.assertEqual(set(used), set(FAMILIES))
                self.assertLessEqual(max(used.values()), 2)
        self.assertTrue(all(k == t.BLACK or k[0] in FAMILIES for k in s.keys()))   # held-out source only

    def test_orders(self):
        self.assertEqual(sorted(map(tuple, t.block_orders(3))), sorted(permutations(range(3))))
        for n in (3, 4, 6):
            rows = t.williams(n)
            self.assertTrue(all(sorted(r) == list(range(n)) for r in rows))
            pairs = Counter((r[i], r[i + 1]) for r in rows for i in range(n - 1))
            self.assertEqual(len(set(pairs.values())), 1)                            # carry-over balanced
        for rank in range(20):
            self.assertEqual({t.order_index(rank, i, 6) for i in range(10)}, set(range(6)))
        for i in range(10):
            self.assertEqual({t.order_index(rank, i, 6) for rank in range(20)}, set(range(6)))
        self.assertEqual(make(initialisation=0, rank=0).order, [0, 1, 2])
        self.assertEqual(make(initialisation=1, rank=0).order, [0, 2, 1])

    def test_determinism_and_pairing(self):
        a, b = make(), make()
        self.assertEqual(a.segments, b.segments)
        self.assertNotEqual(make(initialisation=1).segments, a.segments)
        rect = make(displays=t.SENSITIVITY_BODIES)
        strip = lambda s: [(x['first_ms'], x['kind'], x['family'], x['noise_seed']) for x in s.trials]
        self.assertEqual(strip(rect), strip(a))                                      # paired sensitivity session
        controls = make(displays=t.CONTROL_DISPLAYS, kind_code=2)
        self.assertEqual(len([x for x in controls.trials if x['block'] >= 0]), 6 * 60)

    def test_steps(self):
        s = make()
        steps = s.steps(1)
        self.assertEqual(len(steps), s.duration_ms)
        keys = s.keys()
        for x in s.trials[::17]:
            self.assertEqual(keys[steps[x['probe_ms']]][2] if x['kind'] != 'fixation' else keys[steps[x['probe_ms']]],
                             'movement' if x['kind'] == 'movement' else ('static' if x['kind'] == 'static' else t.BLACK))
        self.assertEqual(len(s.steps(.5)), 2 * s.duration_ms)
        self.assertTrue(900_000 < s.duration_ms < 1_100_000)                          # about 16-17 minutes

    def test_training_session(self):
        s = t.training_session(FAMILIES[:5], 40, seed=1)
        self.assertEqual(Counter(x['kind'] for x in s.trials), Counter(static=20, movement=20))
        self.assertTrue(all(x['display'] == 'full_body' and x['family'] in FAMILIES[:5] for x in s.trials))


class Bank(unittest.TestCase):
    def synthetic(self):
        rows = [(f, c, r) for f in FAMILIES for c in t.BODIES for r in ('static', 'movement')]
        index = pd.DataFrame(rows, columns=['family', 'condition', 'frame'])
        x = np.arange(len(rows) * 3, dtype=float).reshape(-1, 3)
        return bk.Bank('r', arrays=dict(r=x), index=index, black=np.full(3, -1.))

    def test_lookup_and_black(self):
        b = self.synthetic()
        self.assertTrue(np.array_equal(b.vector(t.BLACK), [-1, -1, -1]))
        self.assertEqual(b.matrix(make().keys()).shape[1], 3)
        with self.assertRaises(ValueError):
            b.vector(('other_source', 'full_body', 'static'))

    def test_scaler_uses_training_full_body_only(self):
        b = self.synthetic()
        mean, sd = b.scaler(FAMILIES[:10])
        x = b.matrix([(f, 'full_body', r) for f in FAMILIES[:10] for r in ('static', 'movement')])
        self.assertTrue(np.allclose(mean, x.mean(0)) and np.allclose(sd, x.std(0)))
        z = b.inputs(make(), (mean, sd))
        self.assertEqual(z.shape, (len(make().keys()), 3))

    @unittest.skipUnless((bk.FEATURES / 'features.npz').is_file(), 'Local features not present')
    def test_real_features_cover_a_session(self):
        families = sorted(f for f in pd.read_csv(bk.FEATURES / 'index.csv').family.unique() if f.startswith('r03_'))
        b = bk.Bank('openclip')
        s = t.session(families, 3, 2, 0)
        keys = [k for k in s.keys() if k != t.BLACK]
        self.assertEqual(b.matrix(keys).shape, (len(keys), 1280))


if __name__ == '__main__':
    unittest.main()
