"""CPU-only regression checks; no local dataset, model downloads or GPU required."""
import argparse
import sys
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import dataset as d
import generate as g
import models
import run
import validate_dataset as archive


class Guards(unittest.TestCase):
    def test_grid_and_frozen_recipe(self):
        cfg, sources, jobs = d.plan()
        self.assertEqual((len(sources), len(jobs), len({j['base_id'] for j in jobs})), (20, 480, 160))
        self.assertEqual({j['id'] for j in jobs if j['canary']}, set(cfg['canary']))
        self.assertEqual(d.sha(d.DATA / 'frozen_sources.json'), cfg['frozen_sources_sha256'])

    def test_protected_composition_and_rejection(self):
        a = np.full((64, 64, 3), 100, np.uint8)
        b = a.copy()
        b[:4, :4] = 80
        weight = np.zeros((64, 64), np.uint8)
        weight[10:, 10:] = 255
        new_a, new_b, _ = d.shared_desk(a, b, np.full_like(a, 150), weight)
        np.testing.assert_array_equal(new_a[weight > 0], new_b[weight > 0])
        np.testing.assert_array_equal(new_b[weight == 0], b[weight == 0])
        for gains in d.read(d.RECIPE)['lighting'].values():
            d.audit_pair(d.lighting(new_a, gains), d.lighting(new_b, gains), a, b, weight, gains)
        weight[:4, :4] = 255
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            d.shared_desk(a, b, a, weight)

    def test_exports_and_scoped_verification_preserve_aggregate(self):
        cfg, sources, jobs = d.plan()
        job = jobs[0]
        source = dict(sources[job['source_id']], used_hand_box=[0, 0, 400, 400], upper_limb_cutoff=1200)
        a = np.full((2048, 2048, 3), 100, np.uint8)
        b = a.copy()
        b[20:30, 20:30] = 80
        weight = np.zeros(a.shape[:2], np.uint8)
        weight[100:200, 100:200] = 255
        new_a, new_b, _ = d.shared_desk(a, b, np.full_like(a, 150), weight)
        alpha = np.zeros(a.shape[:2], np.uint8)
        alpha[10:40, 10:40], alpha[9, 10:40] = 255, 128
        with tempfile.TemporaryDirectory() as temporary, patch.object(d.silhouette, 'load', return_value=alpha):
            root = Path(temporary)
            data, output = root / 'data', root / 'output'
            original = data / 'sources' / source['source_directory'] / 'full_body'
            original.mkdir(parents=True)
            Image.fromarray(a).save(original / 'static.png')
            Image.fromarray(b).save(original / 'movement.png')
            inputs = data / 'inputs' / source['id']
            inputs.mkdir(parents=True)
            Image.fromarray(weight).save(inputs / 'desk_weight.png')
            d.write(output / 'provenance.json', {'identity': d.identity()})
            parent = output / 'parents' / job['base_id']
            parent.mkdir(parents=True)
            Image.fromarray(new_a).save(parent / 'static_with_shared_desk.png')
            Image.new('RGB', (1024, 1024)).save(parent / 'raw_1024.png')
            d.write(parent / 'result.json', dict(run_sha256=d.sha(output / 'provenance.json'),
                    files={name: d.sha(parent / name) for name in ('static_with_shared_desk.png', 'raw_1024.png')}))
            provenance = dict(identity=d.identity(), run_sha256=d.sha(output / 'provenance.json'),
                              parent_sha256=d.sha(parent / 'result.json'))
            destination = output / 'candidates' / job['id']
            d.export_family(destination, new_a, new_b, a, b, source, job, weight, [1, 1, 1], provenance)
            with self.assertRaises(FileExistsError):
                d.export_family(destination, new_a, new_b, a, b, source, job, weight, [1, 1, 1], provenance)
            d.write(output / 'verification.json', {'aggregate': 'must survive'})
            with patch.object(d, 'DATA', data), patch.object(d, 'plan', return_value=(cfg, {source['id']: source}, [job])), \
                 patch.object(d, 'check_sources'):
                self.assertEqual(d.verify(output, job['id'])['condition_pngs_checked'], 8)
                silhouette_image = np.array(Image.open(destination / 'detached_hand_silhouette/static.png'))
                np.testing.assert_array_equal(silhouette_image[10:40, 10:40], new_a[10:40, 10:40])
                np.testing.assert_array_equal(silhouette_image[9, 10:40], np.rint(new_a[9, 10:40].astype(float) * 128 / 255))
                self.assertFalse(silhouette_image[41:].any())
                self.assertEqual(d.read(output / 'verification.json'), {'aggregate': 'must survive'})
                with self.assertRaises(FileNotFoundError):
                    d.require_canary(output)
                path = destination / 'detached_hand/static.png'
                image = np.array(Image.open(path))
                image[1000, 1000] = 255
                Image.fromarray(image).save(path)
                with self.assertRaisesRegex(ValueError, 'Changed frozen file'):
                    d.verify_family(output, job, source, cfg)

    def test_missing_and_changed_frozen_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'Missing local file'):
                d.frozen_files(root, {'input.png': 'abc'})
            (root / 'input.png').write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'Changed frozen file'):
                d.frozen_files(root, {'input.png': 'abc'})

    def test_archive_rejects_changed_approval_and_hashes(self):
        cfg, sources, jobs = d.plan()
        job = jobs[0]
        source = sources[job['source_id']]
        family = dict(d.read(archive.MANIFEST)['families'][0])
        audit = dict(id=job['id'], source_id=source['id'], reference=source['reference'],
                     desk_id=job['desk_id'], seed=job['seed'], lighting=job['light_id'],
                     source_lineage_group=source['source_lineage_group'],
                     source_condition_sha256=source['source_condition_sha256'], files=family['condition_png_sha256'])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            d.write(root / 'audit.json', audit)
            with self.assertRaisesRegex(ValueError, 'Missing local file'):
                archive.check_record(root, family, job, source)
            with patch.object(archive, 'frozen_files'):
                self.assertEqual(archive.check_record(root, family, job, source)['condition_pngs_checked'], 6)
                family['decision'] = 'reject'
                with self.assertRaisesRegex(ValueError, 'approval'):
                    archive.check_record(root, family, job, source)

    def test_archive_binds_the_original_guard_to_parent_provenance(self):
        _, sources, jobs = d.plan()
        job = jobs[0]
        source = dict(sources[job['source_id']])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weight = root / '_provenance/inputs' / source['id'] / 'desk_weight.png'
            weight.parent.mkdir(parents=True)
            Image.new('L', (2048, 2048)).save(weight)
            source['mask_coordinate_correction'] = {'previous_input_sha256': {'desk_weight.png': d.sha(weight)}}
            inputs = dict(source['input_sha256'])
            inputs['desk_weight.png'] = d.sha(weight)
            parent = root / '_provenance/diffusion_parents' / job['base_id'] / 'result.json'
            d.write(parent, dict(job=dict(id=job['base_id'], source_id=source['id']), frozen_input_sha256=inputs))
            audit = {'provenance': {'diffusion_parent_result_sha256': d.sha(parent)}}
            self.assertFalse(archive.historical_weight(root, job, source, audit).any())
            Image.new('L', (2048, 2048), 255).save(weight)
            with self.assertRaisesRegex(ValueError, 'Historical desk weight changed'):
                archive.historical_weight(root, job, source, audit)

    def test_incomplete_output_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            g.check_checkpoints(output)
            (output / 'parents' / 'partial').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'Incomplete checkpoint'):
                g.check_checkpoints(output)

    def test_completed_scope_does_not_load_models(self):
        job = dict(id='done', canary=True)
        args = argparse.Namespace(output=Path('/unused'), scope='canary', preflight=False)
        with patch.object(g, 'plan', return_value=({}, {}, [job])), patch.object(g, 'check_sources'), \
             patch.object(g, 'check_checkpoints'), patch.object(Path, 'exists', return_value=True), \
             patch.object(g, 'verify_family'), patch.object(g, 'preflight') as preflight:
            job['source_id'] = 'source'
            with patch.object(g, 'plan', return_value=({}, {'source': {}}, [job])):
                g.run(args)
            preflight.assert_not_called()

    def test_idle_timer_resets_when_peer_appears(self):
        free = dict(name='test', free_mib=16000, other_processes=[])
        busy = dict(free, other_processes=['42, peer, 100'])
        with patch.object(run, 'gpu_state', side_effect=[free, free, busy, free, free, free]), \
             patch.object(run.time, 'monotonic', side_effect=[0, 30, 31, 32, 62, 92]), \
             patch.object(run.time, 'sleep') as sleep, patch.object(run, 'write'):
            run.wait_for_idle(Path('/unused'))
            self.assertEqual(sleep.call_count, 5)

    def test_model_download_is_pinned_and_cache_only_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def snapshot(**kwargs):
                for pattern in kwargs['allow_patterns']:
                    path = root / pattern.replace('*', 'config.json')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                return str(root)
            hub = types.SimpleNamespace(snapshot_download=unittest.mock.Mock(side_effect=snapshot))
            with patch.dict('sys.modules', {'huggingface_hub': hub}):
                models.model_paths()
                calls = hub.snapshot_download.call_args_list
                self.assertEqual(len(calls), 3)
                self.assertTrue(all(c.kwargs['local_files_only'] and len(c.kwargs['revision']) == 40 for c in calls))
                hub.snapshot_download.reset_mock()
                models.model_paths(download=True)
                self.assertTrue(all(not c.kwargs['local_files_only'] for c in hub.snapshot_download.call_args_list))


if __name__ == '__main__':
    unittest.main()
