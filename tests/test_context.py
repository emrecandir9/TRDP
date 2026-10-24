"""CPU checks for H2's context pathway (forearm continuity at the wrist)."""
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
import context as cx


def geometry(palm=40):
    """Upright forearm: wrist line at row 100, fingers up; forearm columns 40-60 below the wrist."""
    yy, xx = np.indices((200, 100))
    along = 100. - yy                      # > 0 towards the fingers
    arm = (xx >= 40) & (xx < 60)
    return along, arm, palm


class Continuity(unittest.TestCase):
    def test_full_forearm_reaches_the_cap(self):
        along, arm, palm = geometry()
        self.assertEqual(cx.continuity(np.ones_like(arm), arm, along, palm), 1.0)

    def test_break_at_the_wrist_gives_zero(self):
        along, arm, palm = geometry()
        visible = along >= 0                  # nothing below the wrist line
        self.assertEqual(cx.continuity(visible, arm, along, palm), 0.0)

    def test_gap_stops_the_count_even_if_arm_returns(self):
        """A visible segment after a gap does not count: continuity, not amount."""
        along, arm, palm = geometry()
        visible = ~((along < -10) & (along >= -15))      # 5-px gap starting 10 px below the wrist
        self.assertAlmostEqual(cx.continuity(visible, arm, along, palm), 10 / palm)

    def test_half_visible_cross_section_still_counts(self):
        along, arm, palm = geometry()
        yy, xx = np.indices(arm.shape)
        visible = np.ones_like(arm)
        visible[(along < -20) & (xx < 50)] = False       # left half of the forearm hidden below 20 px
        self.assertEqual(cx.continuity(visible, arm, along, palm), 1.0)   # exactly half visible: counts
        visible[(along < -20) & (xx < 51)] = False       # less than half
        self.assertAlmostEqual(cx.continuity(visible, arm, along, palm), 20 / palm)

    def test_gate(self):
        self.assertTrue(np.allclose(cx.gate([0, .5, 1], 0), [0, .5, 1]))
        self.assertTrue(np.allclose(cx.gate([0, 1], .2), [.2, 1]))

    @unittest.skipUnless(cx.OUTPUT.is_file(), 'Continuity table not built')
    def test_table_separates_the_matched_pairs(self):
        table = cx.load()
        self.assertEqual(sorted(table), list(range(1, 21)))
        for values in table.values():
            self.assertEqual(values['full_body'], 1.0)
            self.assertEqual(values['upper_limb'], 1.0)
            self.assertEqual(values['detached_hand_silhouette'], 0.0)
            self.assertEqual(values['hand_gap_stub'], 0.0)
            self.assertEqual(values['window_hand'], 0.0)
            self.assertGreater(values['hand_stub'], .95)
            self.assertGreater(values['window_arm'], .95)


if __name__ == '__main__':
    unittest.main()
