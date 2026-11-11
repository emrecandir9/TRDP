"""The completeness audit (scripts/audit_outputs.py) on a small fake output tree."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def network(root, configuration, run, fit=True, sweep=True, predictions=True, peak=True):
    folder = root / 'campaign' / configuration
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{run}.pt').write_bytes(b'')
    (folder / f'{run}.json').write_text(json.dumps(dict(adm_readout_peak_found=True, readout_range_hz=7.9,
                                                        adm_readout_range_hz=4.8, criteria=dict(task=True, readout_peak_found=peak))))
    if fit:
        (root / 'gate_fit' / configuration).mkdir(parents=True, exist_ok=True)
        fits = {g: {a: dict(c=0., curve=[]) for a in ('raw', 'remove', 'maximum')} for g in ('continuity', 'area', 'forearm', 'familiarity')}
        (root / 'gate_fit' / configuration / f'{run}.json').write_text(json.dumps(dict(fits=fits)))
    for step, present in (('sweep', sweep), ('predictions', predictions)):
        if present:
            (root / 'gate2' / step / configuration).mkdir(parents=True, exist_ok=True)
            (root / 'gate2' / step / configuration / f'{run}.json').write_text('{}')


class Audit(unittest.TestCase):
    def test_declared_numbers_add_up_to_the_trained_networks(self):
        import audit_outputs as ao
        self.assertEqual(sum(ao.DECLARED.values()), 3220)
        self.assertEqual(sum(ao.DECLARED[c] for c in ao.CAMPAIGN), 2400)
        self.assertEqual(sum(ao.DECLARED[c] for c in ao.PREDICTED), 1800)
        self.assertEqual(sum(n for c, n in ao.DECLARED.items() if c.startswith('h2_')), 2020)

    def test_a_configuration_is_complete_only_with_every_output(self):
        import audit_outputs as ao
        old = dict(ao.DECLARED)
        try:
            ao.DECLARED.clear()
            ao.DECLARED.update({'h2_openclip_snn': 2, 'h0_openclip_snn': 1})
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                network(root, 'h2_openclip_snn', 'health_openclip_snn_fold1_seed0')
                network(root, 'h2_openclip_snn', 'health_openclip_snn_fold1_seed1', predictions=False)
                network(root, 'h0_openclip_snn', 'health_h0_openclip_snn_fold1_seed0', fit=False)
                (root / 'campaign/h2_openclip_snn/health_openclip_snn_fold2_seed0.trained.pt').write_bytes(b'')   # pending: not counted
                result = ao.audit(root)
                h2, h0 = result['configurations']['h2_openclip_snn'], result['configurations']['h0_openclip_snn']
                self.assertEqual((h2['networks'], h2['fit'], h2['predictions']), (2, 2, 1))
                self.assertEqual(h2['missing_predictions'], ['health_openclip_snn_fold1_seed1'])
                self.assertNotIn('fit', h0)
                self.assertFalse(result['complete'])
                network(root, 'h2_openclip_snn', 'health_openclip_snn_fold1_seed1')
                self.assertTrue(ao.audit(root)['complete'])
                network(root, 'h0_openclip_snn', 'health_h0_openclip_snn_fold1_seed0', fit=False, peak=False)
                self.assertFalse(ao.audit(root)['complete'])
        finally:
            ao.DECLARED.clear()
            ao.DECLARED.update(old)


if __name__ == '__main__':
    unittest.main()
