"""CPU checks for the paper-matched detached-hand silhouettes; local data only for the last test."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import silhouette as s


class Silhouette(unittest.TestCase):
    def test_condition_pixels_are_weighted_full_body_on_black(self):
        full = np.full((4, 4, 3), 200, np.uint8)
        alpha = np.array([[0, 64, 128, 255]] * 4, np.uint8)
        pixels = s.apply(full, alpha)
        np.testing.assert_array_equal(pixels[0, :, 0], [0, 50, 100, 200])

    def test_place_rotates_about_pivot_in_image_coordinates(self):
        mask = np.zeros((101, 101))
        mask[50, 80] = 1  # 30 px right of the pivot (50, 50)
        moved = s.place(mask, np.array([50., 50.]), 90, 0, 0)
        y, x = np.unravel_index(np.argmax(moved), moved.shape)
        # +90 degrees with y pointing down maps (+30, 0) to (0, +30): below the pivot.
        self.assertEqual((x, y), (50, 80))
        shifted = s.place(mask, np.array([50., 50.]), 0, -5, 3)
        self.assertEqual(np.unravel_index(np.argmax(shifted), shifted.shape), (53, 75))

    def test_registration_recovers_a_known_rigid_rotation(self):
        # Index-like geometry: a 30 px wide shaft from 40 to 270 px beyond the knuckle.
        rng = np.random.default_rng(0)
        texture = s.nd.gaussian_filter(rng.random((360, 360, 3)) * 255, (2, 2, 0))
        shaft = np.zeros((360, 360), bool)
        shaft[30:260, 165:195] = True
        pivot = np.array([180., 300.])
        moved = np.stack([s.place(texture[..., c], pivot, 12.3, 1.5, -1) for c in range(3)], -1)
        error, angle, dx, dy = s.register(texture, moved, shaft, pivot, 11.0)
        self.assertAlmostEqual(angle, 12.3, delta=.05)
        self.assertAlmostEqual(dx, 1.5, delta=.2)
        self.assertAlmostEqual(dy, -1, delta=.2)

    def test_wrist_cut_keeps_the_hand_side(self):
        case = {'static_points': [[50, 80]] + [[0, 0]] * 8 + [[50, 20]] + [[0, 0]] * 11}
        weight, line = s.wrist_weight((100, 100), case)
        self.assertEqual((weight[30, 50], weight[90, 50], weight[80, 50]), (1, 0, .5))
        self.assertEqual(line['normal'], [0.0, -1.0])

    @unittest.skipUnless((s.DATA / 'sources').is_dir() and s.MANIFEST.is_file(), 'Local frozen sources not present')
    def test_frozen_mattes_pass_invariants(self):
        manifest = s.read(s.MANIFEST)
        self.assertEqual(s.read(s.ROOT / 'config/recipe.json')['silhouettes_sha256'], s.sha(s.MANIFEST))
        source = s.sources()[11]
        directory = s.DATA / 'sources' / source['source_directory'] / 'full_body'
        static, movement = s.rgb(directory / 'static.png'), s.rgb(directory / 'movement.png')
        alphas = s.check_source(source, static, movement)
        record = manifest['sources'][source['id']]
        self.assertEqual(manifest['hand_segmentation_sha256'], s.sha(s.SEGMENTATION))
        self.assertAlmostEqual(record['movement_to_static_area'], 1, delta=.02)
        self.assertGreater(record['segmentation']['static_iou_with_reviewed_matte'], .95)
        prompt = record['prompt_mask_diagnostics']
        self.assertAlmostEqual(prompt['fitted_degrees'], prompt['recorded_degrees'], delta=.5)
        self.assertTrue((alphas[0] == 0).mean() > .9)
        original = s.read
        with patch.object(s, 'read', side_effect=lambda path: {'sources': {source['id']: {'files': {
                'sam_static.png': '0' * 64}}}} if path == s.SEGMENTATION else original(path)):
            with self.assertRaisesRegex(ValueError, 'changed mask'):
                s.segmentation(source, 'static')

    def test_clean_closes_hairline_seams_but_not_real_gaps(self):
        mask = np.zeros((60, 60), bool)
        mask[10:50, 10:25] = mask[10:50, 27:40] = True   # 2 px seam between touching fingers
        mask[10:50, 48:55] = True                        # finger separated by an 8 px gap
        mask[45:50, 10:55] = True                        # palm joins them
        cleaned = s.clean(mask)
        self.assertTrue(cleaned[20, 25:27].all())
        self.assertFalse(cleaned[20, 41:47].any())


if __name__ == '__main__':
    unittest.main()
