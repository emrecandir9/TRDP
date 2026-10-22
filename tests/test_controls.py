"""CPU checks for the control stimuli; local data only for the last test."""
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import controls as c


def case():
    """Upright left hand: wrist at (1000, 1200), fingers up, little finger on the left."""
    points = np.zeros((21, 2))
    points[0] = [1000, 1200]
    points[9] = [1000, 1000]
    points[5] = [1060, 1010]
    points[17] = [920, 1030]
    return {'static_points': points.tolist()}


class Controls(unittest.TestCase):
    def test_geometry(self):
        g = c.geometry(case())
        self.assertAlmostEqual(g['palm'], np.hypot(140, 20))
        self.assertAlmostEqual(g['along'][1100, 1000], 100)     # 100 px towards the fingers
        self.assertAlmostEqual(g['along'][1300, 1000], -100)    # 100 px down the arm
        side = g['across'][1000, 900] * g['little_side']
        self.assertGreater(side, 0)                              # x < wrist is the little-finger side

    def test_windows_have_equal_area_and_contain_the_hand(self):
        g = c.geometry(case())
        hand = np.zeros((2048, 2048), bool)
        hand[950:1195, 920:1080] = True
        window_hand, window_arm, sideways = c.windows(g, hand)
        a, b = window_hand.astype(float).sum(), window_arm.astype(float).sum()
        self.assertLess(abs(a - b) / b, c.PARAMETERS['window_area_tolerance'])
        self.assertTrue((window_hand[hand] == 255).all() and (window_arm[hand] == 255).all())
        self.assertFalse(window_hand[1250:1300, 990:1010].any())   # no arm below the wrist
        self.assertTrue((window_arm[1250:1300, 990:1010] == 255).all())
        self.assertTrue(window_hand[1100, 920 - c.PARAMETERS['window_margin_px'] - 10] > 0)   # extends sideways
        self.assertGreater(sideways, 0)

    def test_scramble_moves_blocks_but_keeps_pixels_and_hand(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (2048, 2048, 3), dtype=np.uint8)
        guard = np.zeros((2048, 2048), bool)
        guard[500:600, 500:600] = True
        cutoff = 1250
        table, rows, cols = c.block_permutation(guard, cutoff, 3)
        out = c.scramble(image, table, cutoff)
        np.testing.assert_array_equal(out[480:608, 480:608], image[480:608, 480:608])   # blocks 15-18 touch the guard
        self.assertFalse(out[cutoff:].any())
        size = c.PARAMETERS['block_px']
        top = image[:rows * size].reshape(rows, size, cols, size, 3).swapaxes(1, 2).reshape(-1, size * size * 3)
        moved = out[:rows * size].reshape(rows, size, cols, size, 3).swapaxes(1, 2).reshape(-1, size * size * 3)
        self.assertEqual(sorted(map(bytes, top)), sorted(map(bytes, moved)))
        self.assertGreater((np.any(top != moved, axis=1)).mean(), .9)

    @unittest.skipUnless(c.MANIFEST.is_file() and (c.s.DATA / 'sources').is_dir(), 'Local controls not present')
    def test_frozen_controls_pass_their_checks(self):
        data = c.s.read(c.MANIFEST)
        self.assertEqual(data['silhouettes_sha256'], c.s.sha(c.s.MANIFEST))
        source = c.s.sources()[11]
        record = data['sources'][source['id']]
        c.check(source, record, *c.frames(source))
        self.assertLess(record['stub_area_difference'], 1e-6)
        self.assertAlmostEqual(record['gap_px'], .5 * record['palm_px'])


if __name__ == '__main__':
    unittest.main()
