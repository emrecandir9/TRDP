"""CPU checks for the human MEP target; the deposit is needed only for the final reproduction test."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import human_target as h


def synthetic(participants=6, trials=4, seed=0):
    """Known design: +0.2 mV FDI movement effect in full body and upper limb only."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, participants + 1):
        scale = 1 + pid / 10
        for body in h.BODIES:
            for movement in h.MOVEMENTS:
                for muscle in h.MUSCLES:
                    effect = .2 if muscle == 'FDI' and movement == 'movement' and body != 'detached_hand' else 0
                    base = 1.5 if muscle == 'FDI' else 1.
                    for _ in range(trials):
                        rows.append(dict(id=pid, body=body, movement=movement, muscle=muscle,
                                         uv=1000 * scale * (base + effect + rng.normal(0, .01))))
    return pd.DataFrame(rows)


class HumanTarget(unittest.TestCase):
    def test_participant_cells_and_primary_contrast(self):
        ids, x = h.participant_cells(h.scaled(synthetic(), 'raw_mv'))
        self.assertEqual((len(ids), x.shape[1:]), (6, (3, 2, 2)))
        c_fdi = dict((n, f) for n, _, f in h.contrasts())['C_FDI'](x)
        expected = .2 * (1 + np.arange(1, 7) / 10)
        np.testing.assert_allclose(c_fdi, expected, atol=.03)

    def test_standardisation_scopes_and_exclusion_order(self):
        d = synthetic()
        z = h.scaled(d, 'participant_z').groupby('id').response
        np.testing.assert_allclose(z.mean(), 0, atol=1e-12)
        np.testing.assert_allclose(z.std(ddof=1), 1)
        zm = h.scaled(d, 'participant_muscle_z').groupby(['id', 'muscle']).response
        np.testing.assert_allclose(zm.std(ddof=1), 1)
        d.loc[0, 'uv'] = 10
        cleaned = h.scaled(d, 'participant_z_min50')
        self.assertEqual(len(cleaned), len(d) - 1)
        np.testing.assert_allclose(cleaned.groupby('id').response.std(ddof=1), 1)

    def test_missing_cell_breaks_pairing(self):
        d = synthetic()
        d = d[~((d.id == 1) & (d.body == 'detached_hand') & (d.muscle == 'ADM'))]
        with self.assertRaisesRegex(ValueError, 'pairing'):
            h.participant_cells(h.scaled(d, 'raw_mv'))

    def test_bootstrap_is_paired_and_reproducible(self):
        _, x = h.participant_cells(h.scaled(synthetic(), 'raw_mv'))
        a, b = h.bootstrap_means(x, 500, 7), h.bootstrap_means(x, 500, 7)
        np.testing.assert_array_equal(a, b)
        cells, table, ratios = h.summarise(x, a)
        self.assertEqual(len(cells), 12)
        self.assertEqual(table.query("family == 'primary'").contrast.tolist(), ['C_FDI'])
        primary = table.set_index('contrast').loc['C_FDI']
        self.assertTrue(primary.ci_low <= primary.estimate <= primary.ci_high)
        self.assertTrue((table.p_holm >= table.p - 1e-12).all())
        self.assertAlmostEqual(ratios.set_index('ratio').loc['ratio_DH_to_connected_FDI', 'estimate'], 0, places=1)

    def test_holm(self):
        np.testing.assert_allclose(h.holm([.01, .04, .03]), [.03, .06, .06])

    def test_changed_deposit_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'MEPs_amplitude.csv'
            path.write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'Changed human data'):
                h.verified(path, '0' * 64)

    @unittest.skipUnless((h.ROOT / 'data/human/MEPs_amplitude.csv').is_file(), 'Local OSF deposit not present')
    def test_deposit_reproduces_published_statistics_and_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            tables, summary = h.run(Path(temporary), h.read_config())
        self.assertTrue(summary['published_statistics_reproduced'])
        self.assertEqual(summary['audit']['missing_records'], 240)
        primary = tables['contrasts'].query("contrast == 'C_FDI' and scale == 'participant_z'").iloc[0]
        self.assertAlmostEqual(primary.estimate, .2830, places=4)
        self.assertTrue(summary['primary_C_FDI_positive_on_every_scale'])


if __name__ == '__main__':
    unittest.main()
