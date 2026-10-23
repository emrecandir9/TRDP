"""CPU checks for the rater validation (scripts/raters.py); local data only for the last test."""
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import raters as r

PROTOCOL = r.ROOT / 'docs/rater-protocol.md'


def fake_sample(directory, seed=0):
    """A sample directory as `sample` writes it (key.csv, manifest.json), without images."""
    rng = np.random.default_rng(seed)
    assignment, _ = r.assign_variants(list(range(1, 21)), rng)
    families = [r.family_id(ref, assignment[ref]) for ref in range(1, 21)]
    items = [(f, c, fr) for f in families for c in r.CONTEXTS for fr in r.FRAMES]
    codes = {item: f'img_{n + 1:03d}' for item, n in zip(items, rng.permutation(r.IMAGES))}
    orders = {rater: [codes[item] for item in r.presentation_order(families, rng)] for rater in r.RATERS}
    rows = []
    for (family, context, frame), code in sorted(codes.items(), key=lambda x: x[1]):
        reference = int(family[1:3])
        rows.append(dict(code=code, reference=reference, source_id=f'p{reference:02d}_seed9_detailed', family=family,
                         desk='', seed='', balance='', context=context, frame=frame, original_path='', original_sha256='',
                         exported_sha256='', position_A=orders['A'].index(code) + 1,
                         position_B=orders['B'].index(code) + 1))
    paths = r.layout(directory)
    Path(directory).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths['key'], index=False)
    manifest = dict(orders=orders, packet_id='test', files={'key.csv': r.s.sha(paths['key'])})
    paths['manifest'].write_text(json.dumps(manifest))
    return r.read_key_table(paths['key']), orders


def ratings_frame(key, orders, rater, realism, artifact):
    """A complete rating file's rows; realism and artifact map each key row to a value."""
    position = {c: p + 1 for p, c in enumerate(orders[rater])}
    frame = pd.DataFrame(dict(code=key['code'], realism=realism, artifact=artifact, comment='', rater=rater,
                              position=key['code'].map(position), timestamp='2026-11-10T10:00:00.000Z'))
    return frame.sort_values('position').reset_index(drop=True)


class Sample(unittest.TestCase):
    def test_variant_assignment_is_balanced_and_deterministic(self):
        for seed in range(5):
            assignment, _ = r.assign_variants(list(range(1, 21)), np.random.default_rng(seed))
            chosen = list(assignment.values())
            self.assertEqual(r.variant_problems(chosen), [])
            self.assertEqual(len(set(chosen)), 20)
            self.assertEqual(set(Counter(v[0] for v in chosen).values()), {5})
            self.assertEqual(set(Counter(v[1] for v in chosen).values()), {10})
            self.assertEqual(sorted(Counter(v[2] for v in chosen).values()), [6, 7, 7])
        one = r.assign_variants(list(range(1, 21)), np.random.default_rng(3))
        self.assertEqual(one, r.assign_variants(list(range(1, 21)), np.random.default_rng(3)))

    def test_variant_rules_catch_imbalance(self):
        chosen, _ = r.assign_variants(list(range(1, 21)), np.random.default_rng(0))
        chosen = list(chosen.values())
        self.assertTrue(any('twice' in p for p in r.variant_problems(chosen[:19] + chosen[:1])))
        all_beech = [('beech', seed, b) for seed in r.SEEDS for b in r.BALANCES] * 4
        self.assertTrue(any('desks' in p for p in r.variant_problems(all_beech[:20])))
        # Balanced desks and seeds, but seed 9 loses both of its cool variants on two desks.
        dropped = {('beech', 9, 'cool'), ('birch', 9, 'cool'), ('maple', 123, 'warm'), ('oak', 123, 'neutral')}
        problems = r.variant_problems([v for v in r.variants() if v not in dropped])
        self.assertTrue(any('seed_balance' in p for p in problems), problems)


class Orders(unittest.TestCase):
    families = [f'r{i:02d}_beech_s9_warm' for i in range(1, 21)]

    def test_orders_meet_every_rule(self):
        for seed in range(10):
            order = r.presentation_order(self.families, np.random.default_rng(seed))
            self.assertEqual(r.order_problems(order), [])
            for a, b in zip(order, order[1:]):
                self.assertNotEqual(a[0], b[0])      # frames and contexts of a family never adjacent
            last = {}
            for p, (family, _, _) in enumerate(order):
                if family in last:
                    self.assertGreaterEqual(p - last[family], r.MIN_SOURCE_GAP)
                last[family] = p

    def test_rules_catch_violations(self):
        order = r.presentation_order(self.families, np.random.default_rng(1))
        q = next(p for p in range(20) if order[p][0] == order[20][0])
        adjacent = list(order)
        adjacent[q], adjacent[19] = adjacent[19], adjacent[q]   # same family at positions 20 and 21
        self.assertTrue(any('same family' in p for p in r.order_problems(adjacent)))
        canonical = sorted(order)                                # families in blocks of eight
        problems = r.order_problems(canonical)
        self.assertTrue(any('same family' in p for p in problems) and any('round' in p for p in problems))
        self.assertEqual(r.order_problems(order[:-1]), ['not every family x context x frame exactly once'])
        swapped = list(order)                                    # one image's frames shown in the other order
        family, context = order[0][0], order[0][1]
        i, j = swapped.index((family, context, 'static')), swapped.index((family, context, 'movement'))
        swapped[i], swapped[j] = swapped[j], swapped[i]
        problems = r.order_problems(swapped)
        self.assertTrue(any(f'{context}: static frame first' in p for p in problems), problems)
        self.assertTrue(any(f'{family}: static frame first' in p for p in problems), problems)

    def test_orders_are_deterministic_and_rater_specific(self):
        streams = lambda: [np.random.default_rng(x) for x in np.random.SeedSequence(r.SEED).spawn(4)]
        a1, a2 = (r.presentation_order(self.families, streams()[2]) for _ in range(2))
        b = r.presentation_order(self.families, streams()[3])
        self.assertEqual(a1, a2)
        self.assertNotEqual(a1, b)


class Packet(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.key, self.orders = fake_sample(self.directory)
        paths = r.layout(self.directory)
        paths['images'].mkdir(parents=True)
        for code in self.key['code']:
            (paths['images'] / f'{code}.png').write_bytes(b'')
        paths['form'].write_text(r.rate_html(self.orders, 'test'), encoding='utf-8')
        self.paths = paths

    def test_key_stays_outside_the_packet(self):
        self.assertFalse(self.paths['key'].resolve().is_relative_to(self.paths['packet'].resolve()))
        self.assertFalse(self.paths['manifest'].resolve().is_relative_to(self.paths['packet'].resolve()))
        self.assertEqual(r.blinding_problems(self.directory, self.key), [])
        (self.paths['packet'] / '.DS_Store').write_bytes(b'')          # Finder files are harmless
        self.assertEqual(r.blinding_problems(self.directory, self.key), [])
        shutil.copy(self.paths['key'], self.paths['packet'] / 'key.csv')
        self.assertIn('unexpected file in the packet: key.csv', r.blinding_problems(self.directory, self.key))

    def test_missing_images_and_names_in_the_form_are_caught(self):
        (self.paths['images'] / 'img_007.png').unlink()
        html = self.paths['form'].read_text().replace('Image rating', f'Image rating {self.key["family"][0]} static')
        self.paths['form'].write_text(html)
        problems = r.blinding_problems(self.directory, self.key)
        self.assertIn('missing from the packet: images/img_007.png', problems)
        self.assertIn(f'the form contains "{self.key["family"][0]}"', problems)
        self.assertIn('the form contains "static"', problems)

    def test_form_is_offline_neutral_and_carries_the_orders(self):
        html = r.rate_html(self.orders, 'test')
        self.assertNotRegex(html, r'(?i)https?://|<link|@import|<script[^>]+src=|fetch\(|XMLHttpRequest')
        embedded = json.loads(re.search(r'var ORDERS = (\{.*?\});', html).group(1))
        self.assertEqual(embedded, self.orders)
        for text in ('How realistic does this photograph look?', '1 = clearly artificial',
                     '7 = indistinguishable from a real photograph',
                     'Do you see any visible error or artifact (e.g. distorted fingers, smeared skin, '
                     'wrong number of fingers, unnatural edges)?'):
            self.assertIn(text, html)
        self.assertEqual(re.findall(r"var COLUMNS = \[(.*?)\]", html)[0].replace("'", '').split(', '), list(r.COLUMNS))

    @unittest.skipUnless(shutil.which('node'), 'node not installed')
    def test_form_script_parses(self):
        script = re.search(r'<script>(.*)</script>', r.rate_html(self.orders, 'test'), re.S).group(1)
        path = self.directory / 'form.js'
        path.write_text(script, encoding='utf-8')
        result = subprocess.run(['node', '--check', str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(PROTOCOL.is_file(), 'Protocol not present')
    def test_protocol_quotes_the_form(self):
        """The questions, answer hints and every instruction paragraph read aloud equal the form's text."""
        flat = lambda text: ' '.join(text.replace(' ', ' ').split())
        protocol = flat(re.sub(r'(?m)^> ?', '', PROTOCOL.read_text(encoding='utf-8')))
        decode = lambda literal: flat(json.loads('"' + literal + '"'))
        html = r.rate_html(self.orders, 'test')
        for language in ('en', 'fr'):
            block = re.search(language + r': \{(.*?)\n    \}', html, re.S).group(1)
            for name in ('q1', 'q2', 'anchorLow', 'anchorHigh', 'noneHint', 'minorHint', 'majorHint'):
                literal = re.search(name + r": '((?:[^'\\]|\\.)*)'", block).group(1)
                self.assertTrue(decode(literal) in protocol, f'{language} {name} not quoted: {decode(literal)}')
            paragraphs = re.search(r'instructions: \[(.*?)\]', block, re.S).group(1)
            for literal in re.findall(r"'((?:[^'\\]|\\.)*)'", paragraphs):
                text = decode(literal).replace('{total}', str(r.IMAGES))
                self.assertTrue(text in protocol, f'{language} instruction not quoted: {text}')


class Ratings(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.key, self.orders = fake_sample(self.directory)
        self.good = ratings_frame(self.key, self.orders, 'A', 4, 'none')

    def write(self, frame, name='A.csv'):
        path = self.directory / name
        frame.to_csv(path, index=False)
        return path

    def problems(self, frame):
        with self.assertRaises(ValueError) as caught:
            r.read_ratings(self.write(frame), self.key, self.orders)
        return str(caught.exception)

    def test_complete_file_passes(self):
        out = r.read_ratings(self.write(self.good), self.key, self.orders)
        self.assertEqual(len(out), 160)
        self.assertEqual(out['realism'].dtype.kind, 'i')
        with_bom = self.directory / 'bom.csv'
        with_bom.write_bytes(b'\xef\xbb\xbf' + self.write(self.good).read_bytes())
        self.assertEqual(len(r.read_ratings(with_bom, self.key, self.orders)), 160)

    def test_missing_duplicate_and_unknown_codes(self):
        self.assertIn('1 missing codes', self.problems(self.good.drop(index=5)))
        duplicate = self.good.copy()
        duplicate.loc[5, 'code'] = duplicate.loc[6, 'code']
        message = self.problems(duplicate)
        self.assertIn('1 duplicate codes', message)
        self.assertIn('1 missing codes', message)
        unknown = self.good.copy()
        unknown.loc[0, 'code'] = 'img_999'
        self.assertIn('unknown codes: img_999', self.problems(unknown))

    def test_values_out_of_range(self):
        for value in (0, 8, 3.5, '', 'seven'):
            bad = self.good.astype({'realism': object})
            bad.loc[3, 'realism'] = value
            self.assertIn('realism must be an integer 1-7', self.problems(bad))
        bad = self.good.copy()
        bad.loc[3, 'artifact'] = 'severe'
        self.assertIn('artifact must be one of', self.problems(bad))
        bad = self.good.copy()
        bad.loc[3, 'rater'] = 'B'
        self.assertIn('rater must be one of', self.problems(bad))
        bad = self.good.copy()
        bad.loc[[3, 4], 'position'] = bad.loc[[4, 3], 'position'].to_numpy()
        self.assertIn('2 codes not at their position in order A', self.problems(bad))
        self.assertIn('missing columns', self.problems(self.good.drop(columns='artifact')))

    def test_one_file_per_order(self):
        a = self.write(self.good, 'A.csv')
        again = self.write(self.good, 'A2.csv')
        with self.assertRaises(ValueError) as caught:
            r.analyse([a, again], self.directory, self.directory / 'results')
        self.assertIn('Both rating files are from order A', str(caught.exception))


class Statistics(unittest.TestCase):
    def test_unweighted_kappa_textbook(self):
        # 50 items: both yes 20, A yes / B no 5, A no / B yes 10, both no 15; po = .70, pe = .50, kappa = .40.
        a = ['yes'] * 25 + ['no'] * 25
        b = ['yes'] * 20 + ['no'] * 5 + ['yes'] * 10 + ['no'] * 15
        self.assertAlmostEqual(r.cohen_kappa(a, b, ('yes', 'no')), .4)
        self.assertAlmostEqual(r.cohen_kappa(a, a, ('yes', 'no', 'maybe')), 1)

    def test_quadratic_weighted_kappa(self):
        rng = np.random.default_rng(0)
        x = rng.integers(1, 8, 300)
        y = np.clip(x + rng.integers(-2, 3, 300), 1, 7)
        # Exact identity (Cohen 1968; Schuster 2004): 2 cov / (var x + var y + (mean x - mean y)^2), population moments.
        expected = 2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
        self.assertAlmostEqual(r.cohen_kappa(x, y, r.REALISM, 'quadratic'), expected)
        self.assertAlmostEqual(r.cohen_kappa(x, x + 0, r.REALISM, 'quadratic'), 1)
        two = rng.integers(0, 2, 100), rng.integers(0, 2, 100)
        self.assertAlmostEqual(r.cohen_kappa(*two, (0, 1), 'quadratic'), r.cohen_kappa(*two, (0, 1)))
        try:
            from sklearn.metrics import cohen_kappa_score
        except ImportError:
            return
        self.assertAlmostEqual(r.cohen_kappa(x, y, r.REALISM, 'quadratic'), cohen_kappa_score(x, y, weights='quadratic'))

    def test_kappa_undefined_without_variation(self):
        self.assertTrue(np.isnan(r.cohen_kappa(['none'] * 10, ['none'] * 10, r.ARTIFACT)))

    def test_icc_textbook(self):
        # Shrout & Fleiss (1979), Table 2: six targets, four judges; ICC(2,1) = .29.
        x = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]])
        self.assertAlmostEqual(r.icc_2_1(x), .29, places=2)
        self.assertAlmostEqual(r.icc_2_1(x), (11.2417 - 1.0194) / (11.2417 + 3 * 1.0194 + 4 * (32.4861 - 1.0194) / 6),
                               places=4)
        rng = np.random.default_rng(1)
        a = rng.integers(1, 8, 50)
        b = np.clip(a + rng.integers(-1, 2, 50), 1, 7)
        self.assertAlmostEqual(float(r.icc_from_moments(r.moments(a, b))), r.icc_2_1(np.c_[a, b]))
        self.assertAlmostEqual(r.icc_2_1(np.c_[a, a]), 1)

    def test_bootstrap_is_deterministic(self):
        values = np.random.default_rng(2).normal(size=20)
        one = r.estimate_row(values, r.bootstrap_index(20))
        self.assertEqual(one, r.estimate_row(values, r.bootstrap_index(20)))
        self.assertTrue(np.array_equal(r.bootstrap_index(20), r.bootstrap_index(20)))
        self.assertFalse(np.array_equal(r.bootstrap_index(20), r.bootstrap_index(20, seed=1)))
        self.assertEqual(r.bootstrap_index(19).shape, (r.SAMPLES, 19))
        self.assertLess(one['ci_low'], one['estimate'])
        self.assertLess(one['estimate'], one['ci_high'])
        constant = r.estimate_row(np.full(20, 3.), r.bootstrap_index(20, 500))
        self.assertEqual((constant['ci_low'], constant['ci_high']), (3., 3.))
        self.assertEqual(r.interval([np.nan, 1., 2., np.nan])[2], 2)


class Analysis(unittest.TestCase):
    def test_end_to_end_on_a_fake_sample(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        key, orders = fake_sample(directory)
        silhouette_moved = (key['context'] == 'detached_hand_silhouette') & (key['frame'] == 'movement')
        realism = np.where(silhouette_moved, 2, 5)
        artifact = np.where(silhouette_moved, 'major', 'none')
        paths = []
        for rater in r.RATERS:
            path = directory / f'{rater}.csv'
            ratings_frame(key, orders, rater, realism, artifact).to_csv(path, index=False)
            paths.append(path)
        summary = r.analyse(paths[::-1], directory, directory / 'results')
        results = {name: pd.read_csv(directory / 'results' / f'{name}.csv')
                   for name in ('images', 'contexts', 'agreement', 'frames', 'flagged')}
        self.assertEqual(len(results['images']), 160)
        self.assertEqual(summary['flagged_major_either'], 20)
        self.assertEqual(set(results['flagged']['context']), {'detached_hand_silhouette'})
        frames = results['frames'].set_index(['cohort', 'context', 'measure', 'rater'])
        self.assertEqual(frames.loc[('all', 'detached_hand_silhouette', 'realism', 'both'), 'estimate'], -3)
        self.assertEqual(frames.loc[('all', 'detached_hand_silhouette', 'realism', 'both'), 'n_movement_lower'], 20)
        self.assertEqual(frames.loc[('all', 'full_body', 'realism', 'both'), 'estimate'], 0)
        self.assertEqual(frames.loc[('without_r17', 'full_body', 'realism', 'both'), 'n_sources'], 19)
        agreement = results['agreement'].set_index(['cohort', 'subset', 'statistic'])
        self.assertAlmostEqual(agreement.loc[('all', 'all', 'realism_weighted_kappa'), 'estimate'], 1)
        self.assertAlmostEqual(agreement.loc[('all', 'all', 'realism_icc_2_1'), 'estimate'], 1)
        self.assertAlmostEqual(agreement.loc[('all', 'all', 'artifact_percent_agreement'), 'estimate'], 100)
        self.assertTrue(np.isnan(agreement.loc[('all', 'full_body', 'artifact_kappa'), 'estimate']))
        contexts = results['contexts'].set_index(['cohort', 'context', 'statistic', 'rater'])
        self.assertEqual(contexts.loc[('all', 'detached_hand_silhouette', 'realism_mean', 'both'), 'estimate'], 3.5)
        self.assertEqual(contexts.loc[('all', 'detached_hand_silhouette', 'major_artifact_rate', 'A'), 'estimate'], .5)
        self.assertEqual(summary['files'].keys(), {f'{n}.csv' for n in ('images', 'contexts', 'agreement', 'frames',
                                                                         'flagged', 'distributions')})


@unittest.skipUnless((r.s.DATA / 'accepted').is_dir() and (r.s.DATA / 'accepted_silhouette').is_dir(),
                     'Local archive not present')
class FrozenSample(unittest.TestCase):
    def test_sample_writes_160_hash_checked_images(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        manifest = r.sample(directory)
        key = r.read_key_table(r.layout(directory)['key'])
        images = sorted(r.layout(directory)['images'].glob('*.png'))
        self.assertEqual(len(images), 160)
        self.assertEqual(len(key), 160)
        cohort = {f['id']: f for f in r.fx.families()}
        silhouettes = r.s.read(r.s.ARCHIVE_MANIFEST)['families']
        for row in key.itertuples():
            path, expected = r.image_source(cohort[row.family], row.context, row.frame, silhouettes)
            self.assertEqual(row.original_sha256, expected)
            self.assertEqual(r.s.sha(r.layout(directory)['images'] / f'{row.code}.png'), row.exported_sha256)
        with r.Image.open(images[0]) as image:
            self.assertEqual((image.mode, image.size, image.info), ('RGB', (1024, 1024), {}))
        self.assertEqual(Counter(key['context']), {c: 40 for c in r.CONTEXTS})
        self.assertEqual(key.groupby('reference')['family'].nunique().max(), 1)
        r.check(directory)
        frozen = r.layout(r.OUTPUT)['manifest']
        if frozen.is_file():   # the packet handed to raters must be exactly reproducible
            previous = r.s.read(frozen)
            self.assertEqual(previous['sample'], manifest['sample'])
            self.assertEqual(previous['orders'], manifest['orders'])
            self.assertEqual(previous['packet_images'], manifest['packet_images'])


if __name__ == '__main__':
    unittest.main()
