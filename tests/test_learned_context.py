"""CPU checks for the learned context pathways (docs/model-method.md §7.1)."""
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
import learned_context as lc


def upright():
    """Hand pointing up the image: wrist at (1000, 1200), fingers towards smaller y, palm 100 px."""
    return np.array([1000., 1200.]), np.array([0., -1.]), np.array([1., 0.]), 100.


class Windows(unittest.TestCase):
    def test_band_lies_below_the_wrist_and_hand_window_above(self):
        luma = np.zeros((2048, 2048))
        luma[1200:1300, 950:1050] = 1.                           # a forearm just below the wrist
        band = lc.sample(luma, upright(), 'band')
        hand = lc.sample(luma, upright(), 'hand')
        self.assertGreater(band.mean(), .8)
        self.assertLess(hand.mean(), .1)

    def test_a_gap_below_the_wrist_darkens_the_band(self):
        luma = np.zeros((2048, 2048))
        luma[1250:1400, 950:1050] = 1.                           # forearm starts 0.5 P below the wrist
        self.assertLess(lc.sample(luma, upright(), 'band').mean(), .65)


class Match(unittest.TestCase):
    def test_match_is_one_up_to_the_median_and_zero_beyond_every_calibration_value(self):
        calibration = np.arange(1., 101.)
        m = lc.match(np.array([0., 50., 75., 100., 1000.]), calibration)
        self.assertEqual(m[0], 1.)
        self.assertAlmostEqual(m[1], 1.)
        self.assertAlmostEqual(m[2], .52)
        self.assertAlmostEqual(m[3], .02)
        self.assertEqual(m[4], 0.)

    def test_ridge_recovers_a_linear_map(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 5))
        y = x @ rng.normal(size=(5, 3)) + 1
        fit = lc.ridge_fit(x, y, 1e-6)
        self.assertTrue(np.allclose(fit(x), y, atol=1e-4))


class Familiarity(unittest.TestCase):
    def test_typical_images_match_and_far_images_do_not(self):
        """Own-body features of 6 training sources: images of an unseen source drawn from the same
        distribution get m = 0.75 on average (1 up to the median, then falling evenly to 0); an image
        far outside the distribution gets 0."""
        rng = np.random.default_rng(1)
        n, d = 40, 6
        x = rng.normal(size=(7 * n + 1, d))
        x[-1] += 50
        sources = np.r_[np.repeat(np.arange(1, 8), n), 7]
        conditions = np.array(['full_body'] * (7 * n) + ['detached_hand_silhouette'])
        m, info = lc.familiarity_fold(x, sources, conditions, None, held_out=7)
        self.assertAlmostEqual(m[6 * n:7 * n].mean(), .75, delta=.1)
        self.assertEqual(m[-1], 0.)
        self.assertGreaterEqual(info['components'], 1)


if __name__ == '__main__':
    unittest.main()
