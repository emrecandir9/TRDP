"""CPU checks for the hand-centred views of docs/baselines.md, Addendum A; local data only for the
rendering test."""
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import features as fx
import handview as h

BOX = [384, 163, 866, 768]   # used_hand_box of reference 1: 482 x 605 px


class Window(unittest.TestCase):
    def test_side_is_scale_times_longer_side(self):
        for scale in h.SCALES:
            self.assertEqual(h.window(BOX, scale)[2], int(scale * 605))
        self.assertEqual(h.window([0, 0, 50, 10], 2)[2], 100)   # wider than tall

    def test_window_is_centred_on_the_box(self):
        for box in (BOX, [367, 155, 898, 776], [0, 0, 3, 4], [10, 20, 30, 60]):
            for scale in (1, 2, 3, 4):
                left, top, side = h.window(box, scale)
                self.assertLessEqual(abs(left + side / 2 - (box[0] + box[2]) / 2), .5)
                self.assertLessEqual(abs(top + side / 2 - (box[1] + box[3]) / 2), .5)
        self.assertEqual(h.window([10, 20, 30, 60], 2), (-20, 0, 80))
        self.assertEqual(h.window([0, 0, 3, 4], 2), (-3, -2, 8))   # centre 1.5 rounded down to 1

    def test_box_centre_lands_at_the_window_centre(self):
        image = np.zeros((2048, 2048, 3), np.uint8)
        cx, cy = (BOX[0] + BOX[2]) // 2, (BOX[1] + BOX[3]) // 2
        image[cy, cx] = 255
        for scale in h.SCALES:
            out = h.crop(image, h.window(BOX, scale))
            (row, col), = np.argwhere(out.max(axis=2) > 0)
            self.assertLessEqual(abs(row - out.shape[0] / 2), 1)
            self.assertLessEqual(abs(col - out.shape[1] / 2), 1)


class Crop(unittest.TestCase):
    def test_outside_the_image_is_black(self):
        image = np.full((100, 100, 3), 200, np.uint8)
        out = h.crop(image, (-10, 90, 40))   # columns -10..29, rows 90..129
        self.assertEqual(out.shape, (40, 40, 3))
        self.assertTrue((out[:10, 10:] == 200).all())
        self.assertFalse(out[10:].any())
        self.assertFalse(out[:, :10].any())
        self.assertFalse(h.crop(image, (150, 150, 20)).any())   # entirely outside
        big = h.crop(image, (-50, -50, 200))                    # window larger than the image
        self.assertTrue(np.array_equal(big[50:150, 50:150], image))
        self.assertEqual(int((big > 0).all(axis=2).sum()), 100 * 100)

    def test_inside_the_image_is_a_plain_slice(self):
        image = np.random.default_rng(0).integers(0, 256, (64, 64, 3), np.uint8)
        self.assertTrue(np.array_equal(h.crop(image, (5, 7, 30)), image[7:37, 5:35]))


def synthetic_family(rng, size=64):
    """Distinct random images per condition and frame, sharing one marker pixel at (20, 30)."""
    images = {}
    for condition in fx.CONDITIONS:
        pair = []
        for _ in fx.FRAMES:
            image = rng.integers(1, 256, (size, size, 3), np.uint8)
            image[20, 30] = (0, 0, 0)
            pair.append(image)
        images[condition] = tuple(pair)
    return images


class Views(unittest.TestCase):
    def test_same_window_for_every_condition_and_frame(self):
        images = synthetic_family(np.random.default_rng(1))
        source = dict(used_hand_box=[24, 16, 40, 36])   # s = 2 inside the image, s = 4 beyond it
        shown = h.views(images, source)
        self.assertEqual(sorted(shown), sorted(h.SCALES))
        for scale in h.SCALES:
            self.assertEqual(list(shown[scale]), list(h.CONDITIONS))
            left, top, side = win = h.window(source['used_hand_box'], scale)
            marker = (20 - top, 30 - left)
            for condition, pair in shown[scale].items():
                for image, original in zip(pair, images[condition]):
                    self.assertEqual(image.shape, (side, side, 3))
                    self.assertTrue(np.array_equal(image, h.crop(original, win)))
                    self.assertFalse(image[marker].any())   # the shared marker lands at the same place
                    self.assertTrue(image[marker[0], marker[1] + 1].all())

    def test_conditions_are_the_ten_contexts(self):
        self.assertEqual(len(h.CONDITIONS), 10)
        self.assertEqual(set(fx.CONDITIONS) - set(h.CONDITIONS), {'hand_only'})

    def test_identity_check_follows_the_upper_limb_cut(self):
        rng = np.random.default_rng(2)
        fb = tuple(rng.integers(1, 256, (100, 100, 3), np.uint8) for _ in fx.FRAMES)
        ul = tuple(np.where(np.arange(100)[:, None, None] < 70, image, 0).astype(np.uint8) for image in fb)
        above, across = (20, 10, 40), (20, 40, 40)   # ends at row 50 / row 80; the cut is at row 70
        self.assertTrue(h.identical([h.crop(i, above) for i in fb], [h.crop(i, above) for i in ul]))
        self.assertFalse(h.identical([h.crop(i, across) for i in fb], [h.crop(i, across) for i in ul]))


class Analysis(unittest.TestCase):
    def test_tables_follow_the_declared_analysis(self):
        """Synthetic features with a larger movement in the silhouette DH: C < 0 at every scale,
        identical to baselines.summarise_signal on the same cells, and no hand_only rows."""
        families = fx.families()
        source = np.array([f['reference'] for f in families])
        rng = np.random.default_rng(3)
        n, dim = len(families), 6
        base = rng.normal(0, 1, (20, 1, len(h.CONDITIONS), 1, dim))[source - 1]
        x = np.repeat(np.repeat(base, len(h.SCALES), 1), 2, 3) + rng.normal(0, .01, (n, len(h.SCALES), len(h.CONDITIONS), 2, dim))
        x[:, :, :, 1, 0] += .5
        x[:, :, h.CONDITIONS.index('detached_hand_silhouette'), 1, 0] += 1.5
        signal, family, gate = h.tables(families, dict(dinov2_cls=x))
        self.assertNotIn('delta_hand only', set(signal.statistic))
        self.assertNotIn('hand_only', set(family.condition))
        self.assertEqual(sorted(gate.scale), sorted(h.SCALES))
        self.assertTrue((gate.C < 0).all() and gate.reverse_pattern.all())
        reading = h.reading(gate)['dinov2_cls']
        self.assertTrue(reading['reading'].startswith('reverse at every scale'))
        data = h.scale_data(families, dict(dinov2_cls=x), 2)
        cells = h.b.source_means(h.b.movement_signal(data.x['dinov2_cls'], data.variant), data.source)
        expected = {r['statistic']: r for r in h.b.summarise_signal(cells, h.b.bootstrap_index(len(cells)))}['C']
        got = signal.query("scale == 2 and measure == 'normalised' and cohort == 'all' and "
                           "detached_hand == 'detached_hand_silhouette' and statistic == 'C'").iloc[0]
        self.assertAlmostEqual(got.estimate, expected['estimate'])
        self.assertAlmostEqual(got.ci_low, expected['ci_low'])

    def test_reading_rule(self):
        import pandas as pd
        gate = pd.DataFrame([dict(representation='a', scale=2, C_low=-.3, C_high=-.1),
                             dict(representation='a', scale=4, C_low=-.2, C_high=-.05),
                             dict(representation='b', scale=2, C_low=.1, C_high=.3),
                             dict(representation='b', scale=4, C_low=-.2, C_high=-.05),
                             dict(representation='c', scale=2, C_low=-.1, C_high=.3),
                             dict(representation='c', scale=4, C_low=-.2, C_high=-.05)])
        r = h.reading(gate)
        self.assertTrue(r['a']['reading'].startswith('reverse at every scale'))
        self.assertTrue(r['b']['reading'].startswith('positive'))
        self.assertTrue(r['c']['reading'].startswith('other'))


class Rendering(unittest.TestCase):
    @unittest.skipUnless((fx.ROOT / 'data/accepted').is_dir() and (fx.ROOT / 'data/accepted_silhouette').is_dir(),
                         'Local archive not present')
    def test_one_family(self):
        family = fx.families()[0]
        rows, checks = h.prepare(family)
        self.assertEqual(len(rows), len(h.SCALES) * len(h.CONDITIONS) * len(fx.FRAMES))
        self.assertEqual([(r['scale'], r['condition'], r['frame']) for r in rows],
                         [(s, c, f) for s in h.SCALES for c in h.CONDITIONS for f in fx.FRAMES])
        for r in rows:
            self.assertEqual(r['small'].shape, (h.SIZE, h.SIZE, 3))
            self.assertEqual(r['pixels'].shape, (fx.PIXEL_GRID ** 2,))
            self.assertEqual(r['lowlevel'].shape, (len(fx.LOWLEVEL),))
        self.assertEqual(sum('difference' in r for r in rows), len(h.SCALES) * len(h.CONDITIONS))
        self.assertEqual([c['scale'] for c in checks], list(h.SCALES))
        images = fx.Renderer().render(family)
        for scale, shown in h.views(images, family['source']).items():
            side = h.window(family['source']['used_hand_box'], scale)[2]
            for pair in shown.values():
                self.assertTrue(all(image.shape == (side, side, 3) for image in pair))


if __name__ == '__main__':
    unittest.main()
