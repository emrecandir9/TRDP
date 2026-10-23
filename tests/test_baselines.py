"""CPU checks for the Gate 1 baselines; local data only for the rendering test."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import baselines as b
import features as fx

CONDITIONS = len(fx.CONDITIONS)


def synthetic(sources=6, variants=4, dim=5, movement=1., seed=0):
    """Families [family, condition, frame, dim]: source offsets, variant offsets, small noise,
    and a fixed movement vector of the given length added to every movement frame."""
    rng = np.random.default_rng(seed)
    source = np.repeat(np.arange(1, sources + 1), variants)
    variant = np.tile([f'v{k}' for k in range(variants)], sources)
    base = rng.normal(0, 1, (sources, CONDITIONS, dim))[source - 1] + rng.normal(0, .3, (variants, 1, dim))[
        np.tile(np.arange(variants), sources)]
    x = np.stack([base, base], axis=2) + rng.normal(0, .01, (len(source), CONDITIONS, 2, dim))
    direction = np.zeros(dim)
    direction[0] = movement
    x[:, :, 1] += direction
    return x, source, variant


class Signal(unittest.TestCase):
    def test_spread_is_between_source_sd(self):
        static = np.array([[0.], [2.], [4.], [10.], [12.], [14.]])
        variant = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
        self.assertAlmostEqual(b.spread(static, variant), 2.)            # sample SD within each variant
        self.assertTrue(np.allclose(b.spread(np.hstack([static, 3 * static]), variant, True), [2, 6]))

    def test_normalised_signal_is_scale_and_offset_free(self):
        x, _, variant = synthetic()
        one = b.movement_signal(x, variant)
        self.assertTrue(np.allclose(one, b.movement_signal(7 * x + 3, variant)))
        self.assertTrue(np.allclose(b.movement_signal(7 * x, variant, measure='raw'),
                                    7 * b.movement_signal(x, variant, measure='raw')))
        cosine = b.movement_signal(x + 50, variant, measure='cosine')
        self.assertTrue((cosine >= 0).all() and (cosine < .01).all())

    def test_contrasts_and_intervals(self):
        cells = np.ones((20, CONDITIONS))
        cells[:, fx.CONDITIONS.index('full_body')] = 3
        cells[:, fx.CONDITIONS.index('upper_limb')] = 1
        cells[:, fx.CONDITIONS.index('detached_hand_silhouette')] = 4
        rows = {r['statistic']: r for r in b.summarise_signal(cells, b.bootstrap_index(20, 200))}
        self.assertAlmostEqual(rows['C']['estimate'], -2)
        self.assertAlmostEqual(rows['C']['ci_low'], -2)
        self.assertAlmostEqual(rows['ratio_DH_to_connected']['estimate'], 2)
        self.assertAlmostEqual(rows['ratio_UL_to_FB']['estimate'], 1 / 3)
        noisy = cells + np.random.default_rng(1).normal(0, 1, cells.shape)
        noisy[:, fx.CONDITIONS.index('detached_hand_silhouette')] = (
            noisy[:, fx.CONDITIONS.index('full_body')] + noisy[:, fx.CONDITIONS.index('upper_limb')]) / 2
        c = {r['statistic']: r for r in b.summarise_signal(noisy, b.bootstrap_index(20, 500))}['C']
        self.assertAlmostEqual(c['estimate'], 0)

    def test_gate_rule(self):
        rows = []
        for name, (low, high) in dict(a=(.1, .3), b=(-.1, .3), c=(-.4, -.2)).items():
            for statistic, estimate in (('C', (low + high) / 2), ('ratio_DH_to_connected', 0.), ('ratio_UL_to_FB', 3.)):
                rows.append(dict(representation=name, measure='normalised', cohort='all',
                                 detached_hand='detached_hand_silhouette', statistic=statistic, estimate=estimate,
                                 ci_low=low, ci_high=high, n_positive=10))
        g = b.gate1(pd.DataFrame(rows)).set_index('representation')
        self.assertEqual(list(g.meets_P1), [True, False, False])
        self.assertEqual(list(g.reverse_pattern), [False, False, True])
        self.assertTrue(g.S1_compatible.all() and not g.S2_compatible.any())

    def test_variance_shares(self):
        rows = [dict(representation='r', condition='full_body', source=s, desk=d, seed=e, balance=w, signal=float(s))
                for s in range(4) for d in range(2) for e in range(2) for w in range(3)]
        shares = b.variance_shares(pd.DataFrame(rows), 'r', ('full_body',)).iloc[0]
        self.assertAlmostEqual(shares.source, 1)
        self.assertAlmostEqual(shares.residual, 0)


class Decoding(unittest.TestCase):
    def test_paired_accuracy(self):
        self.assertEqual(b.paired_accuracy(np.array([2., 1, 0]), np.array([1., 1, 1])), .5)

    def test_held_out_source_never_trains(self):
        x, source, _ = synthetic(movement=.5)
        seen = []
        original = b.fit_fold
        def spy(xt, y, groups):
            seen.append(set(groups))
            return original(xt, y, groups)
        with patch.object(b, 'fit_fold', spy):
            rows = b.movement_decoding_fold(x, source, 3, 'full_body')
        self.assertNotIn(3, seen[0])
        self.assertEqual(len(rows), CONDITIONS)
        self.assertGreater(min(r['paired'] for r in rows), .95)   # the same movement vector in every condition

    def test_no_movement_is_chance(self):
        x, source, _ = synthetic(movement=0.)
        rows = [r for s in np.unique(source) for r in b.movement_decoding_fold(x, source, s, 'full_body')]
        self.assertLess(abs(np.mean([r['paired'] for r in rows]) - .5), .2)

    def test_attachment_transfer(self):
        x, source, _ = synthetic(movement=0.)
        for c in b.CONNECTED:
            x[:, fx.CONDITIONS.index(c), :, 1] += 3   # one shared 'attachment' axis
        rows = b.attachment_fold(x, source, 2, 'stub_pair')
        self.assertTrue(all(r['paired'] == 1 for r in rows))


class Geometry(unittest.TestCase):
    def test_square_crop_is_centred(self):
        image = np.zeros((50, 60, 3), np.uint8)
        image[10:20, 5:45] = 200
        crop = fx.square_crop(image, [5, 10, 45, 20])
        self.assertEqual(crop.shape, (40, 40, 3))
        self.assertTrue((crop[15:25] == 200).all() and not crop[:15].any() and not crop[25:].any())

    def test_lowlevel_statistics(self):
        black = np.zeros((32, 32, 3), np.uint8)
        self.assertTrue(np.allclose(fx.lowlevel(black), 0))
        grey = np.full((32, 32, 3), 51, np.uint8)
        self.assertTrue(np.allclose(fx.lowlevel(grey), [1, .2, 0, 0]))
        self.assertEqual(fx.pixels(grey).shape, (fx.PIXEL_GRID ** 2,))
        half = black.copy()
        half[:, 16:] = 255
        stats = fx.lowlevel(half)
        self.assertAlmostEqual(stats[0], .5)
        self.assertAlmostEqual(stats[2], .5)
        self.assertGreater(stats[3], 0)

    def test_directions_and_alignment_are_unit(self):
        x, _, variant = synthetic(movement=2.)
        u = b.directions(x, variant, False)
        self.assertTrue(np.allclose(np.linalg.norm(u, axis=-1), 1))
        self.assertGreater((u[:, :, 0]).min(), .99)

    def test_partial_correlation_removes_a_shared_cause(self):
        rng = np.random.default_rng(0)
        cause = rng.normal(size=200)
        y, x = cause + rng.normal(0, .1, 200), cause + rng.normal(0, .1, 200)
        self.assertGreater(np.corrcoef(y, x)[0, 1], .9)
        self.assertLess(abs(b.partial_spearman(y, x, [cause])), .2)


class Data(unittest.TestCase):
    def test_feature_order_is_enforced(self):
        ids = [f['id'] for f in fx.families()[:2]]
        rows = [(f, c, r) for f in ids for c in fx.CONDITIONS for r in fx.FRAMES]
        index = pd.DataFrame(rows, columns=['family', 'condition', 'frame'])
        arrays = dict(lowlevel=np.zeros((len(rows), 4)))
        self.assertEqual(b.Data(arrays=arrays, index=index).x['lowlevel'].shape, (2, CONDITIONS, 2, 4))
        with self.assertRaises(ValueError):
            b.Data(arrays=arrays, index=index.iloc[::-1].reset_index(drop=True))

    @unittest.skipUnless((fx.ROOT / 'data/accepted').is_dir() and (fx.ROOT / 'data/accepted_silhouette').is_dir(),
                         'Local archive not present')
    def test_rendering_matches_the_frozen_conditions(self):
        family = fx.families()[0]
        images = fx.Renderer().render(family)
        self.assertEqual(list(images), list(fx.CONDITIONS))
        full = images['full_body']
        for name in fx.controls.CONTROLS:
            for frame, image in zip(fx.FRAMES, images[name]):
                expected = fx.controls.render(full[fx.FRAMES.index(frame)], family['source'], name, frame)
                self.assertTrue(np.array_equal(image, expected), name)
        side = max(family['source']['used_hand_box'][2] - family['source']['used_hand_box'][0],
                   family['source']['used_hand_box'][3] - family['source']['used_hand_box'][1])
        self.assertEqual(images['hand_only'][0].shape, (side, side, 3))
        self.assertEqual(int(images['hand_only'][0].astype(int).sum()),
                         int(images['detached_hand_silhouette'][0].astype(int).sum()))


if __name__ == '__main__':
    unittest.main()
