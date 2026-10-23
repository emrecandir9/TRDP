"""Rater validation of a fixed image sample: issue 5 of docs/next-phase-plan.md, protocol in
docs/rater-protocol.md.

Two naive raters score 160 images for realism (1-7) and visible artifacts (none / minor / major):
one family per source, 20 sources x 4 contexts (full body, upper limb, detached-hand silhouette,
detached-hand rectangle) x 2 frames. The families use 20 distinct appearance variants, so every
desk appears 5 times, every seed 10 times and every white balance 6 or 7 times.

  sample   check every archive image against its manifest hash, draw the sample and one
           presentation order per rater, and write the blinded packet
           (output/raters/packet/: images/img_NNN.png at 1024 px, lossless PNG, and rate.html)
           with the hidden key.csv and manifest.json outside the packet
  check    re-hash the packet and the key against manifest.json and scan the packet for
           anything that would unblind it
  analyse  validate two rating files against the key, then write the per-image table,
           per-context summaries, agreement, static vs movement differences and flagged images

Sources are the independent units: intervals are percentile bootstraps over the 20 sources
(10,000 resamples, seed 20261022), and every result is repeated without reference 17.
Ratings are reported only; they never exclude images.
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import platform
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
import features as fx  # noqa: E402

s = fx.s
require = fx.require

OUTPUT = ROOT / 'output/raters'
CONFIG = ROOT / 'config/analysis.json'
SEED, SAMPLES, INTERVAL = 20261022, 10000, .95
CONTEXTS = fx.MAIN            # full_body, upper_limb, detached_hand_silhouette, detached_hand
FRAMES = fx.FRAMES            # static, movement; the movement frame is the edited one
DESKS = ('beech', 'birch', 'maple', 'oak')
SEEDS = (9, 123)
BALANCES = ('cool', 'neutral', 'warm')
SOURCES = 20
IMAGES = SOURCES * len(CONTEXTS) * len(FRAMES)
EXPORT_SIZE = 1024
ROUND = SOURCES               # one image of every source per round
ROUNDS = IMAGES // ROUND      # 8
MIN_SOURCE_GAP = 10           # two images of the same source are at least this many positions apart
RATERS = ('A', 'B')
REALISM = tuple(range(1, 8))
ARTIFACT = ('none', 'minor', 'major')
COLUMNS = ('code', 'realism', 'artifact', 'comment', 'rater', 'position', 'timestamp')
EXCLUDED_REFERENCE = 17       # known clothing issue; every result is repeated without it
COHORTS = ('all', 'without_r17')
# Words that must not appear in the packet's form (case-insensitive, whole words).
FORBIDDEN = ('full_body', 'full body', 'upper_limb', 'upper limb', 'detached', 'detached_hand', 'silhouette',
             'static', 'movement', 'edited', 'abduction', 'beech', 'birch', 'maple', 'oak', 'hypothesis')
IGNORED_FILES = ('.DS_Store', 'Thumbs.db', 'desktop.ini')


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


# ---------- sample: one appearance variant per source ----------

def variants():
    """The 24 appearance variants (desk, seed, white balance)."""
    return [(d, seed, b) for d in DESKS for seed in SEEDS for b in BALANCES]


def variant_problems(chosen):
    """Balance rules for 20 variants, one per source; returns the rules that fail."""
    problems = []
    if len(chosen) != SOURCES:
        problems.append(f'{len(chosen)} variants instead of {SOURCES}')
    if len(set(chosen)) != len(chosen):
        problems.append('a variant is used twice')
    desk, seed, balance = (Counter(v[i] for v in chosen) for i in range(3))
    if any(desk[d] != 5 for d in DESKS):
        problems.append(f'desks not 5 each: {dict(desk)}')
    if any(seed[x] != 10 for x in SEEDS):
        problems.append(f'seeds not 10 each: {dict(seed)}')
    if sorted(balance[b] for b in BALANCES) != [6, 7, 7]:
        problems.append(f'white balances not 7/7/6: {dict(balance)}')
    pairs = dict(desk_seed=((0, 1), DESKS, SEEDS, {2, 3}), desk_balance=((0, 2), DESKS, BALANCES, {1, 2}),
                 seed_balance=((1, 2), SEEDS, BALANCES, {3, 4}))
    for name, ((i, j), first, second, allowed) in pairs.items():
        cells = Counter((v[i], v[j]) for v in chosen)
        if any(cells[(a, b)] not in allowed for a in first for b in second):
            problems.append(f'{name} cells outside {sorted(allowed)}')
    return problems


def assign_variants(references, rng, attempts=100000):
    """Shuffle the 24 variants and give the first 20 to the sources in reference order; keep the
    first shuffle that meets every balance rule. Returns {reference: variant} and the number of draws."""
    pool = variants()
    for draw in range(1, attempts + 1):
        chosen = [pool[i] for i in rng.permutation(len(pool))[:len(references)]]
        if not variant_problems(chosen):
            return dict(zip(references, chosen)), draw
    raise ValueError('No balanced variant assignment found')


def family_id(reference, variant):
    desk, seed, balance = variant
    return f'r{reference:02d}_{desk}_s{seed}_{balance}'


# ---------- presentation orders ----------

def _binary_patterns():
    """4 x 4 binary matrices with two ones in every row, column and wrapped diagonal."""
    bits = ((np.arange(2 ** 16)[:, None] >> np.arange(16)) & 1).reshape(-1, 4, 4)
    ok = (bits.sum(2) == 2).all(1) & (bits.sum(1) == 2).all(1)
    g = np.arange(4)
    for r in range(4):
        ok &= bits[:, g, (g + r) % 4].sum(1) == 2
    return bits[ok]


def _static_first(rng, extra):
    """5 x 4 binary matrix (families of one group x contexts): 1 = static frame shown first.
    Every family has two contexts with the static frame first; column sums are 2 + extra."""
    rows = [np.array(p) for p in itertools.product((0, 1), repeat=4) if sum(p) == 2]
    options = [m for m in (np.array(c) for c in itertools.product(rows, repeat=5))
               if (m.sum(0) == 2 + extra).all()]
    return options[rng.integers(len(options))]


def presentation_order(families, rng, attempts=100000):
    """One rater's order of the 160 images as (family, context, frame).

    Eight rounds of 20 images. The families are split at random into four groups of five; in
    round r, group g shows context (g + r) mod 4 of a randomly ordered context cycle, so every
    round has every family once and every context five times, and every family shows each
    context once in rounds 1-4 and once in rounds 5-8, with the other frame. Which frame comes
    first is balanced: every family has the static frame first in two contexts, every context in
    ten families, and every round has ten static and ten movement images (two or three of each
    per context). Within a round the order is random, redrawn until no family recurs within
    MIN_SOURCE_GAP positions of its previous image.
    """
    require(len(families) == SOURCES and len(set(families)) == SOURCES, 'Twenty distinct families expected')
    cycle = [CONTEXTS[i] for i in rng.permutation(len(CONTEXTS))]
    groups = rng.permutation(SOURCES).reshape(4, 5)
    patterns = _binary_patterns()
    extra = patterns[rng.integers(len(patterns))]
    first = {}
    for g in range(4):
        y = _static_first(rng, extra[g])
        for row, f in enumerate(groups[g]):
            for j in range(4):
                first[(f, cycle[j])] = bool(y[row, j])
    order, last = [], {}
    for r in range(ROUNDS):
        items = []
        for g in range(4):
            context = cycle[(g + r) % 4]
            for f in groups[g]:
                static = first[(f, context)] == (r < ROUNDS // 2)
                items.append((families[f], context, 'static' if static else 'movement'))
        for _ in range(attempts):
            candidate = [items[i] for i in rng.permutation(len(items))]
            if all(len(order) + q - last.get(item[0], -MIN_SOURCE_GAP) >= MIN_SOURCE_GAP
                   for q, item in enumerate(candidate)):
                break
        else:
            raise ValueError('No order within the round met the source gap')
        for item in candidate:
            last[item[0]] = len(order)
            order.append(item)
    return order


def order_problems(order, min_gap=MIN_SOURCE_GAP):
    """Rules every presentation order must meet; returns the rules that fail."""
    problems = []
    families = sorted({f for f, _, _ in order})
    expected = {(f, c, fr) for f in families for c in CONTEXTS for fr in FRAMES}
    if len(families) != SOURCES or len(order) != IMAGES or set(order) != expected:
        return ['not every family x context x frame exactly once']
    for p in range(1, len(order)):
        if order[p][0] == order[p - 1][0]:
            problems.append(f'positions {p} and {p + 1} show the same family')
    seen = {}
    for p, (f, c, fr) in enumerate(order):
        if f in seen and p - seen[f] < min_gap:
            problems.append(f'family {f} recurs after {p - seen[f]} positions')
        seen[f] = p
    position = {item: p for p, item in enumerate(order)}
    for f in families:
        for c in CONTEXTS:
            if abs(position[(f, c, 'static')] - position[(f, c, 'movement')]) < 3 * ROUND:
                problems.append(f'the two frames of {f} {c} are fewer than {3 * ROUND} positions apart')
    for r in range(ROUNDS):
        block = order[r * ROUND:(r + 1) * ROUND]
        contexts, frames = Counter(c for _, c, _ in block), Counter(fr for _, _, fr in block)
        if len({f for f, _, _ in block}) != ROUND:
            problems.append(f'round {r + 1} does not show every family once')
        if any(contexts[c] != ROUND // len(CONTEXTS) for c in CONTEXTS):
            problems.append(f'round {r + 1} contexts unbalanced: {dict(contexts)}')
        if any(frames[fr] != ROUND // 2 for fr in FRAMES):
            problems.append(f'round {r + 1} frames unbalanced: {dict(frames)}')
    first = {(f, c): position[(f, c, 'static')] < position[(f, c, 'movement')] for f in families for c in CONTEXTS}
    for c in CONTEXTS:
        count = sum(first[(f, c)] for f in families)
        if count != SOURCES // 2:
            problems.append(f'{c}: static frame first for {count} of {SOURCES} families')
    for f in families:
        count = sum(first[(f, c)] for c in CONTEXTS)
        if count != len(CONTEXTS) // 2:
            problems.append(f'{f}: static frame first in {count} of {len(CONTEXTS)} contexts')
    return problems


# ---------- packet ----------

def layout(directory):
    """Where sample writes: the packet for the raters, and the key and manifest outside it."""
    directory = Path(directory)
    return dict(packet=directory / 'packet', images=directory / 'packet/images', form=directory / 'packet/rate.html',
                key=directory / 'key.csv', manifest=directory / 'manifest.json')


def blinding_problems(directory, key):
    """Anything in the packet that could unblind a rater: unexpected files (such as the key),
    missing images, or condition, family or source names in the form."""
    paths = layout(directory)
    packet = paths['packet']
    expected = {'rate.html'} | {f'images/{code}.png' for code in key['code']}
    present = {p.relative_to(packet).as_posix() for p in packet.rglob('*')
               if p.is_file() and p.name not in IGNORED_FILES}
    problems = [f'unexpected file in the packet: {name}' for name in sorted(present - expected)]
    problems += [f'missing from the packet: {name}' for name in sorted(expected - present)]
    if not all(re.fullmatch(r'img_\d{3}', code) for code in key['code']):
        problems.append('image codes are not of the form img_NNN')
    if paths['key'].resolve().is_relative_to(packet.resolve()):
        problems.append('the key is inside the packet folder')
    if (packet / 'rate.html').is_file():
        html = (packet / 'rate.html').read_text(encoding='utf-8').lower()
        words = set(FORBIDDEN) | set(key['family']) | set(key['source_id'])
        for word in sorted(words):
            if re.search(r'(?<![a-z0-9])' + re.escape(word.lower()) + r'(?![a-z0-9])', html):
                problems.append(f'the form contains "{word}"')
    return problems


def export_image(path, expected, target):
    """Check the archive file's hash, downscale 2048 -> 1024 by averaging each 2 x 2 block, and
    write a lossless PNG without metadata. Returns the exported file's hash."""
    data = Path(path).read_bytes()
    require(sha_bytes(data) == expected, f'Changed archive image: {path}')
    with Image.open(io.BytesIO(data)) as image:
        require(image.mode == 'RGB' and image.size == (2048, 2048), f'Expected 2048-square RGB: {path}')
        small = image.reduce(2048 // EXPORT_SIZE)
    small.save(target, format='PNG')
    return s.sha(target)


def check_inputs():
    """The manifests themselves must be the frozen ones recorded in config/analysis.json."""
    cohort = s.read(CONFIG)['cohort']
    hashes = dict(accepted_manifest_sha256=s.sha(fx.ACCEPTED), frozen_sources_sha256=s.sha(s.DATA / 'frozen_sources.json'),
                  silhouette_manifest_sha256=s.sha(s.ARCHIVE_MANIFEST))
    require(hashes['accepted_manifest_sha256'] == cohort['accepted_manifest_sha256'], 'Accepted manifest changed')
    require(hashes['frozen_sources_sha256'] == cohort['frozen_sources_sha256'], 'Frozen source manifest changed')
    require(hashes['silhouette_manifest_sha256'] == cohort['detached_hand']['archive_silhouette_manifest_sha256'],
            'Silhouette manifest changed')
    return hashes


def image_source(family, context, frame, silhouettes):
    """Archive path and manifest hash of one condition image."""
    if context == s.CONTEXT:
        path = s.DATA / 'accepted_silhouette' / family['id'] / context / f'{frame}.png'
        return path, silhouettes[family['id']]['files'][f'{context}/{frame}.png']
    return s.DATA / 'accepted' / family['id'] / context / f'{frame}.png', family['hashes'][f'{context}/{frame}.png']


def versions():
    import importlib.metadata as m
    return {name: m.version(name) for name in ('numpy', 'pillow', 'pandas')}


def sample(directory=OUTPUT, force=False):
    paths = layout(directory)
    inputs = check_inputs()
    if paths['manifest'].exists() or paths['packet'].exists():
        require(force, f'{directory} already holds a packet; ratings may have started. Use "check", or --force to rebuild')
        shutil.rmtree(paths['packet'], ignore_errors=True)
        for name in ('key', 'manifest'):
            paths[name].unlink(missing_ok=True)
    cohort = {f['id']: f for f in fx.families()}
    silhouettes = s.read(s.ARCHIVE_MANIFEST)['families']
    sources = sorted(s.sources(), key=lambda x: x['reference'])
    references = [x['reference'] for x in sources]
    variant_rng, code_rng, *order_rngs = [np.random.default_rng(x) for x in np.random.SeedSequence(SEED).spawn(4)]
    assignment, draws = assign_variants(references, variant_rng)
    chosen = []
    for source in sources:
        family = cohort[family_id(source['reference'], assignment[source['reference']])]
        require(family['source_id'] == source['id'], f'Source mismatch for {family["id"]}')
        chosen.append(family)
    items = [(f['id'], c, fr) for f in chosen for c in CONTEXTS for fr in FRAMES]
    codes = {item: f'img_{n + 1:03d}' for item, n in zip(items, code_rng.permutation(IMAGES))}
    orders = {}
    for rater, rng in zip(RATERS, order_rngs):
        order = presentation_order([f['id'] for f in chosen], rng)
        problems = order_problems(order)
        require(not problems, f'Order {rater}: {problems}')
        orders[rater] = [codes[item] for item in order]
    require(orders['A'] != orders['B'], 'The two orders are identical')
    position = {rater: {code: p + 1 for p, code in enumerate(order)} for rater, order in orders.items()}

    paths['images'].mkdir(parents=True)
    by_id = {f['id']: f for f in chosen}
    rows = []
    for item, code in sorted(codes.items(), key=lambda x: x[1]):   # written in code order
        family_name, context, frame = item
        family = by_id[family_name]
        path, expected = image_source(family, context, frame, silhouettes)
        exported = export_image(path, expected, paths['images'] / f'{code}.png')
        desk, seed, balance = assignment[family['reference']]
        rows.append(dict(code=code, reference=family['reference'], source_id=family['source_id'], family=family_name,
                         desk=desk, seed=seed, balance=balance, context=context, frame=frame,
                         original_path=path.relative_to(ROOT).as_posix(), original_sha256=expected,
                         exported_sha256=exported, position_A=position['A'][code], position_B=position['B'][code]))
    with open(paths['key'], 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    packet_id = sha_bytes(json.dumps(orders, sort_keys=True).encode())[:12]
    paths['form'].write_text(rate_html(orders, packet_id), encoding='utf-8')
    key = read_key_table(paths['key'])
    problems = blinding_problems(directory, key)
    require(not problems, f'Packet not blind: {problems}')

    counts = {name: dict(Counter(str(assignment[r][i]) for r in references)) for i, name in
              enumerate(('desk', 'seed', 'balance'))}
    manifest = dict(
        method='docs/rater-protocol.md', issue='docs/next-phase-plan.md section 3, issue 5', code_sha256=s.sha(__file__),
        seed=SEED, random_streams='numpy SeedSequence(seed).spawn(4): variants, codes, order A, order B',
        inputs=inputs, contexts=list(CONTEXTS), frames=list(FRAMES), images=IMAGES,
        sample=[dict(reference=f['reference'], source_id=f['source_id'], family=f['id'], desk=assignment[f['reference']][0],
                     seed=assignment[f['reference']][1], balance=assignment[f['reference']][2]) for f in chosen],
        balance=dict(counts, variants_used=len(set(assignment.values())), variant_draws=draws,
                     rules='each desk 5, each seed 10, white balances 7/7/6; desk x seed 2-3, desk x balance 1-2, '
                           'seed x balance 3-4; 20 distinct variants'),
        export=dict(size=EXPORT_SIZE, method='PIL Image.reduce(2): mean of each 2 x 2 block, rounded',
                    format='PNG (lossless), 8-bit RGB, no metadata', names='img_NNN.png, numbers assigned at random'),
        order_design=dict(rounds=ROUNDS, round_size=ROUND, min_source_gap=MIN_SOURCE_GAP,
                          rules=' '.join(presentation_order.__doc__.strip().split('\n\n')[1].split())),
        orders=orders, packet_id=packet_id, python=platform.python_version(), packages=versions(),
        files={'key.csv': s.sha(paths['key']), 'packet/rate.html': s.sha(paths['form'])},
        packet_images={f'packet/images/{r["code"]}.png': r['exported_sha256'] for r in rows})
    s.write(paths['manifest'], manifest)
    return manifest


def read_key_table(path):
    return pd.read_csv(path, dtype={'code': str, 'family': str, 'source_id': str, 'context': str, 'frame': str})


def load_sample(directory=OUTPUT):
    """Manifest and key, with the key checked against the manifest's hash and orders."""
    paths = layout(directory)
    require(paths['manifest'].is_file(), f'No rater sample in {directory}; run "sample" first')
    manifest = s.read(paths['manifest'])
    require(s.sha(paths['key']) == manifest['files']['key.csv'], 'key.csv does not match manifest.json')
    key = read_key_table(paths['key'])
    require(len(key) == IMAGES and key['code'].is_unique, 'The key must list 160 distinct codes')
    indexed = key.set_index('code')
    for rater in RATERS:
        order = manifest['orders'][rater]
        require(sorted(order) == sorted(key['code']), f'Order {rater} is not a permutation of the key')
        require(indexed.loc[order, f'position_{rater}'].tolist() == list(range(1, IMAGES + 1)),
                f'Key positions differ from order {rater}')
    return manifest, key


def check(directory=OUTPUT):
    manifest, key = load_sample(directory)
    paths = layout(directory)
    require(s.sha(paths['form']) == manifest['files']['packet/rate.html'], 'rate.html changed')
    for name, expected in manifest['packet_images'].items():
        require(s.sha(Path(directory) / name) == expected, f'Changed packet image: {name}')
    require(all(manifest['packet_images'][f'packet/images/{r.code}.png'] == r.exported_sha256
                for r in key.itertuples()), 'Key and manifest disagree on exported hashes')
    problems = blinding_problems(directory, key)
    require(not problems, f'Packet not blind: {problems}')
    return manifest


# ---------- rating files ----------

def read_ratings(path, key, orders):
    """Read one rating file and check it against the key; raises with every problem found."""
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    except Exception as error:   # noqa: BLE001 - report any parsing failure as a validation error
        raise ValueError(f'{path}: not a readable CSV file ({error})')
    frame.columns = [c.strip() for c in frame.columns]
    missing_columns = [c for c in COLUMNS if c not in frame.columns]
    require(not missing_columns, f'{path}: missing columns {missing_columns}')
    frame = frame[list(COLUMNS)].apply(lambda col: col.str.strip() if col.name != 'comment' else col)
    problems = []
    raters = set(frame['rater'])
    rater = next(iter(raters)) if len(raters) == 1 else None
    if rater not in RATERS:
        problems.append(f'rater must be one of {RATERS} in every row, found {sorted(raters)}')
    codes = set(key['code'])
    duplicated = sorted(set(frame['code'][frame['code'].duplicated()]))
    unknown = sorted(set(frame['code']) - codes)
    absent = sorted(codes - set(frame['code']))
    for label, values in (('duplicate codes', duplicated), ('unknown codes', unknown), ('missing codes', absent)):
        if values:
            problems.append(f'{len(values)} {label}: {", ".join(values[:10])}{" ..." if len(values) > 10 else ""}')
    realism = pd.to_numeric(frame['realism'], errors='coerce')
    bad = ~realism.isin(REALISM)
    if bad.any():
        problems.append(f'realism must be an integer 1-7: {format_rows(frame, bad, "realism")}')
    artifact = frame['artifact'].str.lower()
    bad = ~artifact.isin(ARTIFACT)
    if bad.any():
        problems.append(f'artifact must be one of {ARTIFACT}: {format_rows(frame, bad, "artifact")}')
    position = pd.to_numeric(frame['position'], errors='coerce')
    bad = ~position.isin(range(1, IMAGES + 1))
    if bad.any():
        problems.append(f'position must be an integer 1-{IMAGES}: {format_rows(frame, bad, "position")}')
    elif rater in RATERS:
        wrong = [c for c, p in zip(frame['code'], position.astype(int)) if c in codes and orders[rater][p - 1] != c]
        if wrong:
            problems.append(f'{len(wrong)} codes not at their position in order {rater}: {", ".join(wrong[:10])}')
    if problems:
        raise ValueError(f'{path}: ' + '; '.join(problems))
    return frame.assign(realism=realism.astype(int), artifact=artifact, position=position.astype(int))


def format_rows(frame, bad, column):
    shown = [f'{c}={v!r}' for c, v in zip(frame['code'][bad], frame[column][bad])][:10]
    return ', '.join(shown) + (' ...' if bad.sum() > 10 else '')


# ---------- statistics ----------

def bootstrap_index(n, samples=SAMPLES, seed=SEED):
    """Resampled source indices [samples, n], as in analysis/baselines.py."""
    return np.random.default_rng(seed).integers(0, n, (samples, n))


def interval(values):
    """95% percentile interval of the defined bootstrap values, and how many were undefined."""
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan, np.nan, len(values)
    tail = (1 - INTERVAL) / 2
    return float(np.quantile(finite, tail)), float(np.quantile(finite, 1 - tail)), int(len(values) - len(finite))


def kappa_from_confusion(confusion, weights=None):
    """Cohen's kappa from confusion matrices [..., k, k] (rows rater 1, columns rater 2).
    weights: None (unweighted), 'linear' or 'quadratic' disagreement weights. NaN if undefined."""
    require(weights in (None, 'linear', 'quadratic'), f'Unknown weights: {weights}')
    c = np.asarray(confusion, float)
    k = c.shape[-1]
    i, j = np.indices((k, k))
    w = dict(quadratic=(i - j) ** 2 / (k - 1) ** 2, linear=np.abs(i - j) / (k - 1)).get(weights, (i != j).astype(float))
    p = c / c.sum(axis=(-2, -1), keepdims=True)
    expected = p.sum(-1)[..., :, None] * p.sum(-2)[..., None, :]
    observed_disagreement = (w * p).sum(axis=(-2, -1))
    expected_disagreement = (w * expected).sum(axis=(-2, -1))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(expected_disagreement > 0, 1 - observed_disagreement / expected_disagreement, np.nan)


def confusion(a, b, k):
    """Counts of category pairs; a and b are integer categories 0..k-1."""
    return np.bincount(np.asarray(a) * k + np.asarray(b), minlength=k * k).reshape(k, k)


def cohen_kappa(a, b, categories, weights=None):
    index = {c: n for n, c in enumerate(categories)}
    return float(kappa_from_confusion(confusion([index[x] for x in a], [index[x] for x in b], len(categories)), weights))


def icc_2_1(x):
    """ICC(2,1) of Shrout & Fleiss (1979): two-way random effects, absolute agreement, single
    rater (McGraw & Wong's ICC(A,1)). x: [targets, raters]."""
    x = np.asarray(x, float)
    n, k = x.shape
    m = x.mean()
    ss_rows = k * ((x.mean(1) - m) ** 2).sum()
    ss_cols = n * ((x.mean(0) - m) ** 2).sum()
    ss_error = ((x - m) ** 2).sum() - ss_rows - ss_cols
    msr, msc, mse = ss_rows / (n - 1), ss_cols / (k - 1), ss_error / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def moments(a, b):
    """Sufficient statistics of two raters' scores: n, sum a, sum b, sum a^2, sum b^2, sum ab."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.array([len(a), a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), (a * b).sum()])


def icc_from_moments(m):
    """ICC(2,1) for two raters from summed moments [..., 6]; equals icc_2_1 on the same scores."""
    n, sa, sb, saa, sbb, sab = np.moveaxis(np.asarray(m, float), -1, 0)
    grand = (sa + sb) / (2 * n)
    ss_total = saa + sbb - 2 * n * grand ** 2
    ss_rows = (saa + sbb + 2 * sab) / 2 - 2 * n * grand ** 2
    ss_cols = n * ((sa / n - grand) ** 2 + (sb / n - grand) ** 2)
    msr, msc, mse = ss_rows / (n - 1), ss_cols, (ss_total - ss_rows - ss_cols) / (n - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (msr - mse) / (msr + mse + 2 * (msc - mse) / n)


def label_kappa(value):
    """Landis & Koch (1977) descriptive label."""
    if not np.isfinite(value):
        return 'undefined'
    if value < 0:
        return 'poor'
    for limit, name in ((.2, 'slight'), (.4, 'fair'), (.6, 'moderate'), (.8, 'substantial')):
        if value <= limit:
            return name
    return 'almost perfect'


def label_icc(value):
    """Koo & Li (2016) descriptive label."""
    if not np.isfinite(value):
        return 'undefined'
    for limit, name in ((.5, 'poor'), (.75, 'moderate'), (.9, 'good')):
        if value < limit:
            return name
    return 'excellent'


# ---------- analysis ----------

def image_table(key, ratings):
    """One row per image with both raters' answers, in reference, context and frame order."""
    table = key[['code', 'reference', 'family', 'context', 'frame']].copy()
    for rater in RATERS:
        r = ratings[rater].set_index('code')
        for column in ('realism', 'artifact', 'comment', 'position'):
            table[f'{column}_{rater}'] = table['code'].map(r[column])
    table['realism_mean'] = (table['realism_A'] + table['realism_B']) / 2
    table['realism_difference_A_minus_B'] = table['realism_A'] - table['realism_B']
    table['major_either'] = (table['artifact_A'] == 'major') | (table['artifact_B'] == 'major')
    order = dict(context={c: i for i, c in enumerate(CONTEXTS)}, frame={f: i for i, f in enumerate(FRAMES)})
    table = table.sort_values(['reference', 'context', 'frame'], key=lambda col: col.map(order[col.name])
                              if col.name in order else col)
    return table.reset_index(drop=True)


def cohort_sources(table, cohort):
    sources = np.array(sorted(table['reference'].unique()))
    return sources if cohort == 'all' else sources[sources != EXCLUDED_REFERENCE]


def per_source(table, values, mask, sources):
    """Mean of values over the selected images of each source, in the order of sources."""
    frame = pd.DataFrame(dict(reference=table['reference'][mask], value=np.asarray(values, float)[mask]))
    out = frame.groupby('reference')['value'].mean().reindex(sources).to_numpy()
    require(np.isfinite(out).all(), 'A source has no images in this selection')
    return out


def estimate_row(values, index):
    low, high, _ = interval(values[index].mean(axis=1))
    return dict(estimate=float(values.mean()), ci_low=low, ci_high=high)


def rated(table, rater, item):
    """Per-image values for one rater ('A', 'B') or both ('both': mean of the two)."""
    raters = RATERS if rater == 'both' else (rater,)
    if item == 'realism':
        values = [table[f'realism_{r}'] for r in raters]
    else:
        threshold = ARTIFACT.index('minor' if item == 'any_artifact' else 'major')
        values = [table[f'artifact_{r}'].map(ARTIFACT.index) >= threshold for r in raters]
    return np.mean([np.asarray(v, float) for v in values], axis=0)


def context_summary(table):
    """Mean realism and artifact rates per context; rates are shares of ratings (0-1)."""
    rows = []
    for cohort in COHORTS:
        sources = cohort_sources(table, cohort)
        index = bootstrap_index(len(sources))
        for context in CONTEXTS:
            mask = (table['context'] == context) & table['reference'].isin(sources)
            for statistic, item in (('realism_mean', 'realism'), ('any_artifact_rate', 'any_artifact'),
                                    ('major_artifact_rate', 'major_artifact')):
                for rater in ('both',) + RATERS:
                    values = per_source(table, rated(table, rater, item), mask, sources)
                    rows.append(dict(cohort=cohort, context=context, statistic=statistic, rater=rater,
                                     n_images=int(mask.sum()), n_sources=len(sources), **estimate_row(values, index)))
    return pd.DataFrame(rows)


def frame_differences(table):
    """Movement minus static within each source and context (the movement frame is the edited one)."""
    rows = []
    for cohort in COHORTS:
        sources = cohort_sources(table, cohort)
        index = bootstrap_index(len(sources))
        for context in CONTEXTS:
            masks = {fr: (table['context'] == context) & (table['frame'] == fr) & table['reference'].isin(sources)
                     for fr in FRAMES}
            for measure, item in (('realism', 'realism'), ('any_artifact_rate', 'any_artifact'),
                                  ('major_artifact_rate', 'major_artifact')):
                for rater in ('both',) + RATERS:
                    values = rated(table, rater, item)
                    d = per_source(table, values, masks['movement'], sources) - per_source(table, values, masks['static'],
                                                                                         sources)
                    rows.append(dict(cohort=cohort, context=context, measure=measure, rater=rater,
                                     **estimate_row(d, index), n_sources=len(sources), n_movement_higher=int((d > 0).sum()),
                                     n_movement_lower=int((d < 0).sum()), n_equal=int((d == 0).sum())))
    return pd.DataFrame(rows)


AGREEMENT = ('realism_weighted_kappa', 'realism_icc_2_1', 'realism_exact_agreement_percent',
             'realism_within_one_percent', 'realism_mean_difference_A_minus_B', 'artifact_kappa',
             'artifact_percent_agreement')


def agreement_statistics(realism_confusion, artifact_confusion, realism_moments):
    """All agreement statistics from summed confusion matrices and moments (vectorised)."""
    k = realism_confusion.shape[-1]
    i, j = np.indices((k, k))
    n = realism_confusion.sum(axis=(-2, -1))
    return dict(
        realism_weighted_kappa=kappa_from_confusion(realism_confusion, 'quadratic'),
        realism_icc_2_1=icc_from_moments(realism_moments),
        realism_exact_agreement_percent=100 * np.trace(realism_confusion, axis1=-2, axis2=-1) / n,
        realism_within_one_percent=100 * (realism_confusion * (np.abs(i - j) <= 1)).sum(axis=(-2, -1)) / n,
        realism_mean_difference_A_minus_B=(realism_moments[..., 1] - realism_moments[..., 2]) / realism_moments[..., 0],
        artifact_kappa=kappa_from_confusion(artifact_confusion),
        artifact_percent_agreement=100 * np.trace(artifact_confusion, axis1=-2, axis2=-1) / artifact_confusion.sum(
            axis=(-2, -1)))


def agreement(table):
    """Agreement between the raters over all images and within each context, with intervals from
    resampling sources (all images of a resampled source enter together)."""
    rows = []
    for cohort in COHORTS:
        sources = cohort_sources(table, cohort)
        index = bootstrap_index(len(sources))
        for subset in ('all',) + CONTEXTS:
            mask = table['reference'].isin(sources)
            if subset != 'all':
                mask &= table['context'] == subset
            sub = table[mask]
            where = np.searchsorted(sources, sub['reference'].to_numpy())
            rc = np.zeros((len(sources), len(REALISM), len(REALISM)))
            np.add.at(rc, (where, sub['realism_A'].to_numpy() - 1, sub['realism_B'].to_numpy() - 1), 1)
            ac = np.zeros((len(sources), len(ARTIFACT), len(ARTIFACT)))
            np.add.at(ac, (where, sub['artifact_A'].map(ARTIFACT.index).to_numpy(),
                           sub['artifact_B'].map(ARTIFACT.index).to_numpy()), 1)
            rm = np.stack([moments(g['realism_A'], g['realism_B']) for _, g in sub.groupby('reference')])
            point = agreement_statistics(rc.sum(0), ac.sum(0), rm.sum(0))
            boot = agreement_statistics(rc[index].sum(1), ac[index].sum(1), rm[index].sum(1))
            for statistic in AGREEMENT:
                low, high, undefined = interval(boot[statistic])
                value = float(point[statistic])
                label = (label_kappa(value) if statistic.endswith('kappa') else
                         label_icc(value) if statistic.endswith('icc_2_1') else '')
                rows.append(dict(cohort=cohort, subset=subset, statistic=statistic, estimate=value, ci_low=low,
                                 ci_high=high, undefined_resamples=undefined, label=label, n_images=int(mask.sum()),
                                 n_sources=len(sources)))
    return pd.DataFrame(rows)


def distributions(table):
    rows = []
    for rater in RATERS:
        for context in CONTEXTS:
            sub = table[table['context'] == context]
            for item, categories in (('realism', REALISM), ('artifact', ARTIFACT)):
                counts = Counter(sub[f'{item}_{rater}'])
                rows += [dict(rater=rater, context=context, item=item, category=c, count=counts[c]) for c in categories]
    return pd.DataFrame(rows)


def flagged(table):
    out = table[table['major_either']].copy()
    out['flagged_by'] = np.where((out['artifact_A'] == 'major') & (out['artifact_B'] == 'major'), 'both',
                                 np.where(out['artifact_A'] == 'major', 'A', 'B'))
    return out[['code', 'reference', 'family', 'context', 'frame', 'flagged_by', 'artifact_A', 'artifact_B', 'realism_A',
                'realism_B', 'comment_A', 'comment_B']]


def analyse(rating_paths, directory=OUTPUT, output=None):
    output = Path(output or directory)
    manifest, key = load_sample(directory)
    ratings = {}
    for path in rating_paths:
        frame = read_ratings(path, key, manifest['orders'])
        rater = frame['rater'].iloc[0]
        require(rater not in ratings, f'Both rating files are from order {rater}; one file per order (A, B) is needed')
        ratings[rater] = frame.assign(file=str(path))
    require(set(ratings) == set(RATERS), 'One rating file per rater (A and B) is needed')
    table = image_table(key, ratings)
    results = dict(images=table, contexts=context_summary(table), agreement=agreement(table),
                   frames=frame_differences(table), flagged=flagged(table), distributions=distributions(table))
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, frame in results.items():
        frame.to_csv(output / f'{name}.csv', index=False, float_format='%.6g')
        files[f'{name}.csv'] = s.sha(output / f'{name}.csv')
    main = results['agreement'].query("cohort == 'all'")
    contexts = results['contexts'].query("cohort == 'all' and rater == 'both'")
    frames = results['frames'].query("cohort == 'all' and rater == 'both'")
    summary = dict(
        protocol='docs/rater-protocol.md', code_sha256=s.sha(__file__), sample_manifest_sha256=s.sha(layout(directory)['manifest']),
        key_sha256=s.sha(layout(directory)['key']), packet_id=manifest['packet_id'],
        ratings={r: dict(file=str(Path(ratings[r]['file'].iloc[0]).resolve()),
                         sha256=s.sha(ratings[r]['file'].iloc[0]), rows=len(ratings[r])) for r in RATERS},
        bootstrap=dict(unit='source', samples=SAMPLES, seed=SEED, interval=INTERVAL), cohorts=list(COHORTS),
        agreement={f'{r.subset}/{r.statistic}': dict(estimate=r.estimate, ci=[r.ci_low, r.ci_high], label=r.label)
                   for r in main.itertuples()},
        contexts={f'{r.context}/{r.statistic}': dict(estimate=r.estimate, ci=[r.ci_low, r.ci_high])
                  for r in contexts.itertuples()},
        movement_minus_static={f'{r.context}/{r.measure}': dict(estimate=r.estimate, ci=[r.ci_low, r.ci_high],
                                                                movement_higher=r.n_movement_higher,
                                                                movement_lower=r.n_movement_lower, equal=r.n_equal)
                               for r in frames.itertuples()},
        flagged_major_either=int(table['major_either'].sum()),
        use='Report only: ratings do not exclude images. Any exclusion requires an analysis-spec amendment '
            'approved by the project owner.',
        files=files)
    s.write(output / 'summary.json', clean(summary))
    return summary


def clean(value):
    """JSON-ready copy: numpy scalars to Python numbers, NaN to null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


# ---------- rating form ----------

def rate_html(orders, packet_id):
    """Self-contained rating form: inline CSS and JavaScript, images loaded from images/."""
    return (FORM.replace('__ORDERS__', json.dumps(orders, separators=(',', ':')))
            .replace('__PACKET__', packet_id).replace('__COUNT__', str(IMAGES)))


FORM = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image rating</title>
<style>
:root { color-scheme: dark; --bg: #262626; --panel: #303030; --field: #3a3a3a; --line: #505050;
        --text: #e8e8e8; --muted: #a6a6a6; --accent: #9dbbd8; --warn: #e6c27a; }
* { box-sizing: border-box; }
html, body { margin: 0; min-height: 100%; background: var(--bg); color: var(--text);
             font: 16px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
[hidden] { display: none !important; }
h1 { font-size: 1.6rem; margin: 0 0 1rem; font-weight: 600; }
.page { max-width: 760px; margin: 0 auto; padding: 40px 20px; }
.page p { margin: 0 0 0.9rem; }
fieldset { border: 1px solid var(--line); border-radius: 8px; margin: 0 0 1rem; padding: 0.8rem 1rem 1rem; }
legend { padding: 0 0.4rem; font-weight: 600; }
button { font: inherit; color: var(--text); background: var(--field); border: 1px solid var(--line);
         border-radius: 6px; padding: 0.55rem 1.1rem; cursor: pointer; }
button.primary { background: var(--accent); border-color: var(--accent); color: #141414; font-weight: 600; }
button:disabled { opacity: 0.4; cursor: default; }
.inline label { margin-right: 1.4rem; cursor: pointer; }
.warning { color: var(--warn); }
.note { color: var(--muted); font-size: 0.92rem; }
#rate { display: flex; height: 100vh; }
#stage { flex: 1; min-width: 0; height: 100vh; display: flex; align-items: center; justify-content: center;
         padding: 12px; overflow: hidden; cursor: zoom-in; position: relative; }
#stage img { display: block; max-width: 100%; max-height: calc(100vh - 24px); }
#stage.zoomed { display: block; overflow: auto; cursor: zoom-out; }
#stage.zoomed img { max-width: none; max-height: none; width: 1024px; height: 1024px; margin: 0 auto; }
#imageError { position: absolute; max-width: 420px; text-align: center; color: var(--warn); }
#panel { width: 380px; flex: none; background: var(--panel); border-left: 1px solid var(--line);
         padding: 18px 18px 14px; overflow-y: auto; display: flex; flex-direction: column; }
#panel fieldset { margin-bottom: 0.9rem; }
#panel legend { font-weight: 600; line-height: 1.35; }
.progress { color: var(--muted); font-size: 0.9rem; margin-bottom: 0.9rem; }
.bar { height: 4px; background: var(--field); border-radius: 2px; margin-top: 6px; }
.bar div { height: 100%; width: 0; background: var(--accent); border-radius: 2px; }
.option { position: relative; display: block; }
.option input { position: absolute; opacity: 0; width: 1px; height: 1px; margin: 0; }
.option span { display: block; border: 1px solid var(--line); border-radius: 6px; background: var(--field);
               cursor: pointer; user-select: none; }
.option input:checked + span { background: var(--accent); border-color: var(--accent); color: #141414; }
.option input:focus-visible + span { outline: 2px solid var(--text); outline-offset: 1px; }
.scale { display: grid; grid-template-columns: repeat(7, 1fr); gap: 5px; margin-top: 0.4rem; }
.scale span { text-align: center; padding: 0.5rem 0; font-weight: 600; }
.anchors { display: flex; justify-content: space-between; gap: 1rem; color: var(--muted); font-size: 0.85rem;
           margin-top: 0.35rem; }
.anchors span:last-child { text-align: right; }
.choices { display: grid; gap: 6px; margin-top: 0.4rem; }
.choices span { padding: 0.45rem 0.7rem; }
.choices small { display: block; font-size: 0.82rem; opacity: 0.85; }
textarea { width: 100%; font: inherit; color: var(--text); background: var(--field); border: 1px solid var(--line);
           border-radius: 6px; padding: 0.45rem; resize: vertical; margin-top: 0.3rem; }
.nav { display: flex; justify-content: space-between; margin-top: 0.9rem; }
.hint { color: var(--muted); font-size: 0.85rem; margin: 0.8rem 0 0; }
#csvText { width: 100%; height: 12rem; font: 12px/1.3 ui-monospace, Menlo, Consolas, monospace; }
@media (max-width: 900px) {
  #rate { flex-direction: column; height: auto; }
  #stage { height: 70vh; flex: none; }
  #stage img { max-height: calc(70vh - 24px); }
  #panel { width: auto; border-left: 0; border-top: 1px solid var(--line); }
}
</style>
</head>
<body>
<section id="start" class="page">
  <h1 data-t="title"></h1>
  <form id="startForm" autocomplete="off">
    <fieldset class="inline">
      <legend data-t="raterLegend"></legend>
      <label><input type="radio" name="rater" value="A"> A</label>
      <label><input type="radio" name="rater" value="B"> B</label>
    </fieldset>
    <fieldset class="inline">
      <legend data-t="langLegend"></legend>
      <label><input type="radio" name="lang" value="en" checked> English</label>
      <label><input type="radio" name="lang" value="fr"> Français</label>
    </fieldset>
    <p id="startMessage" class="warning" hidden></p>
    <p id="resumeNote" hidden></p>
    <p id="storageNote" class="warning" data-t="storageNote" hidden></p>
    <button type="submit" id="startButton" class="primary"></button>
    <button type="button" id="restartButton" data-t="restart" hidden></button>
  </form>
</section>

<section id="intro" class="page" hidden>
  <h1 data-t="introTitle"></h1>
  <div id="instructions"></div>
  <button type="button" id="beginButton" class="primary"></button>
</section>

<section id="rate" hidden>
  <div id="stage"><img id="photo" alt=""><p id="imageError" data-t="imageError" hidden></p></div>
  <div id="panel">
    <div class="progress"><span id="progressText"></span><div class="bar"><div id="progressBar"></div></div></div>
    <fieldset>
      <legend data-t="q1"></legend>
      <div class="scale">
        <label class="option"><input type="radio" name="realism" value="1"><span>1</span></label>
        <label class="option"><input type="radio" name="realism" value="2"><span>2</span></label>
        <label class="option"><input type="radio" name="realism" value="3"><span>3</span></label>
        <label class="option"><input type="radio" name="realism" value="4"><span>4</span></label>
        <label class="option"><input type="radio" name="realism" value="5"><span>5</span></label>
        <label class="option"><input type="radio" name="realism" value="6"><span>6</span></label>
        <label class="option"><input type="radio" name="realism" value="7"><span>7</span></label>
      </div>
      <div class="anchors"><span data-t="anchorLow"></span><span data-t="anchorHigh"></span></div>
    </fieldset>
    <fieldset>
      <legend data-t="q2"></legend>
      <div class="choices">
        <label class="option"><input type="radio" name="artifact" value="none"><span><b data-t="none"></b><small data-t="noneHint"></small></span></label>
        <label class="option"><input type="radio" name="artifact" value="minor"><span><b data-t="minor"></b><small data-t="minorHint"></small></span></label>
        <label class="option"><input type="radio" name="artifact" value="major"><span><b data-t="major"></b><small data-t="majorHint"></small></span></label>
      </div>
    </fieldset>
    <label for="comment" data-t="commentLabel"></label>
    <textarea id="comment" rows="3" maxlength="500"></textarea>
    <div class="nav">
      <button type="button" id="backButton" data-t="back"></button>
      <button type="button" id="nextButton" class="primary" disabled></button>
    </div>
    <p class="hint" data-t="zoomHint"></p>
  </div>
</section>

<section id="end" class="page" hidden>
  <h1 data-t="endTitle"></h1>
  <p id="endText"></p>
  <p><button type="button" id="downloadButton" class="primary" data-t="download"></button></p>
  <p id="downloadNote" hidden></p>
  <p><button type="button" id="showDataButton" data-t="showData"></button></p>
  <div id="dataBox" hidden>
    <p class="note" data-t="showDataHelp"></p>
    <textarea id="csvText" readonly></textarea>
  </div>
  <p><button type="button" id="reviewButton" data-t="review"></button></p>
</section>

<script>
(function () {
  'use strict';
  var ORDERS = __ORDERS__;
  var PACKET = '__PACKET__';
  var TOTAL = __COUNT__;
  var COLUMNS = ['code', 'realism', 'artifact', 'comment', 'rater', 'position', 'timestamp'];
  var TEXT = {
    en: {
      title: 'Image rating',
      raterLegend: 'Rater letter (given to you by the experimenter)',
      langLegend: 'Language',
      start: 'Continue',
      resume: 'Resume',
      restart: 'Start again',
      chooseRater: 'Please choose your rater letter.',
      resumeNote: 'Saved progress found in this browser: {n} of {total} pictures rated.',
      restartConfirm: 'Delete the ratings saved in this browser and start again from the first picture?',
      storageNote: 'This browser cannot save your progress. Do not close or reload this page until you have downloaded your ratings at the end.',
      introTitle: 'Instructions',
      instructions: [
        'You will see {total} pictures, one at a time. Some show a whole scene; others show only part of it, and the rest of the picture is black. The black areas and the cropping are intentional and are not errors in themselves: judge how the visible part looks.',
        'For each picture, answer two questions. First: How realistic does this photograph look? Answer from 1 to 7, where 1 means clearly artificial and 7 means indistinguishable from a real photograph. All seven values may be used.',
        'Second: Do you see any visible error or artifact (e.g. distorted fingers, smeared skin, wrong number of fingers, unnatural edges)? Answer none, minor or major. Minor means a small flaw that you notice only when looking closely. Major means an error that is obvious or that makes the picture look wrong.',
        'You may add a comment, for example to say where an error is. There are no right or wrong answers: we want your own impression. Many pictures look alike; rate each one on its own.',
        'You can click a picture to enlarge it, and go back to change an answer. Work at a steady pace; about 10 to 15 seconds per picture is typical, and you may take a short break at any time. At the end, download your ratings and give the file to the experimenter.'
      ],
      begin: 'Start rating',
      continueRating: 'Continue rating',
      q1: 'How realistic does this photograph look?',
      anchorLow: '1 = clearly artificial',
      anchorHigh: '7 = indistinguishable from a real photograph',
      q2: 'Do you see any visible error or artifact (e.g. distorted fingers, smeared skin, wrong number of fingers, unnatural edges)?',
      none: 'None', noneHint: 'No visible error',
      minor: 'Minor', minorHint: 'A small flaw, noticed only when looking closely',
      major: 'Major', majorHint: 'An obvious error, or one that makes the picture look wrong',
      commentLabel: 'Comment (optional), for example where the error is',
      back: 'Back', next: 'Next', finish: 'Finish',
      progress: 'Picture {i} of {total}',
      zoomHint: 'Click the picture to enlarge it; click again to fit it to the screen.',
      imageError: 'This picture could not be loaded. Check that the folder "images" is next to rate.html, then reload the page.',
      endTitle: 'Thank you',
      endText: 'You have rated all {total} pictures. Click the button below to save your ratings as a file, then give the file to the experimenter.',
      download: 'Download ratings (CSV)',
      downloaded: 'Saved as {file} (usually in your Downloads folder).',
      showData: 'Show the data as text',
      showDataHelp: 'If the download does not work: select all the text below, copy it and send it to the experimenter.',
      review: 'Back to the last picture',
      leave: 'Your ratings have not been downloaded yet.'
    },
    fr: {
      title: 'Évaluation d’images',
      raterLegend: 'Lettre d’évaluateur (donnée par l’expérimentateur)',
      langLegend: 'Langue',
      start: 'Continuer',
      resume: 'Reprendre',
      restart: 'Recommencer',
      chooseRater: 'Veuillez choisir votre lettre d’évaluateur.',
      resumeNote: 'Progression enregistrée dans ce navigateur\u00a0: {n} images sur {total} déjà évaluées.',
      restartConfirm: 'Effacer les réponses enregistrées dans ce navigateur et recommencer à la première image\u00a0?',
      storageNote: 'Ce navigateur ne peut pas enregistrer votre progression. Ne fermez pas et ne rechargez pas cette page avant d’avoir téléchargé vos réponses à la fin.',
      introTitle: 'Consignes',
      instructions: [
        'Vous allez voir {total} images, une à la fois. Certaines montrent une scène entière\u00a0; d’autres n’en montrent qu’une partie, et le reste de l’image est noir. Les zones noires et le recadrage sont voulus et ne sont pas des erreurs en eux-mêmes\u00a0: jugez l’aspect de la partie visible.',
        'Pour chaque image, répondez à deux questions. D’abord\u00a0: À quel point cette photographie paraît-elle réaliste\u00a0? Répondez de 1 à 7, où 1 signifie clairement artificielle et 7 impossible à distinguer d’une vraie photographie. Les sept valeurs peuvent toutes être utilisées.',
        'Ensuite\u00a0: Voyez-vous une erreur ou un artefact visible (par ex. doigts déformés, peau qui paraît étalée ou floue, mauvais nombre de doigts, contours non naturels)\u00a0? Répondez aucun, mineur ou majeur. Mineur signifie un petit défaut que l’on ne remarque qu’en regardant attentivement. Majeur signifie une erreur évidente, ou qui donne à l’image un aspect faux.',
        'Vous pouvez ajouter un commentaire, par exemple pour indiquer où se trouve une erreur. Il n’y a pas de bonne ou de mauvaise réponse\u00a0: nous voulons votre impression personnelle. Beaucoup d’images se ressemblent\u00a0; évaluez chacune pour elle-même.',
        'Vous pouvez cliquer sur une image pour l’agrandir, et revenir en arrière pour modifier une réponse. Travaillez à un rythme régulier (comptez en général 10 à 15 secondes par image)\u00a0; vous pouvez faire une courte pause à tout moment. À la fin, téléchargez vos réponses et remettez le fichier à l’expérimentateur.'
      ],
      begin: 'Commencer',
      continueRating: 'Reprendre l’évaluation',
      q1: 'À quel point cette photographie paraît-elle réaliste\u00a0?',
      anchorLow: '1 = clairement artificielle',
      anchorHigh: '7 = impossible à distinguer d’une vraie photographie',
      q2: 'Voyez-vous une erreur ou un artefact visible (par ex. doigts déformés, peau qui paraît étalée ou floue, mauvais nombre de doigts, contours non naturels)\u00a0?',
      none: 'Aucun', noneHint: 'Aucune erreur visible',
      minor: 'Mineur', minorHint: 'Un petit défaut, remarqué seulement en regardant attentivement',
      major: 'Majeur', majorHint: 'Une erreur évidente, ou qui donne à l’image un aspect faux',
      commentLabel: 'Commentaire (facultatif), par exemple où se trouve l’erreur',
      back: 'Précédent', next: 'Suivant', finish: 'Terminer',
      progress: 'Image {i} sur {total}',
      zoomHint: 'Cliquez sur l’image pour l’agrandir\u00a0; cliquez à nouveau pour l’ajuster à l’écran.',
      imageError: 'Cette image n’a pas pu être chargée. Vérifiez que le dossier «\u00a0images\u00a0» se trouve à côté de rate.html, puis rechargez la page.',
      endTitle: 'Merci',
      endText: 'Vous avez évalué les {total} images. Cliquez sur le bouton ci-dessous pour enregistrer vos réponses dans un fichier, puis remettez ce fichier à l’expérimentateur.',
      download: 'Télécharger les réponses (CSV)',
      downloaded: 'Enregistré sous le nom {file} (en général dans le dossier Téléchargements).',
      showData: 'Afficher les données sous forme de texte',
      showDataHelp: 'Si le téléchargement ne fonctionne pas\u00a0: sélectionnez tout le texte ci-dessous, copiez-le et envoyez-le à l’expérimentateur.',
      review: 'Revenir à la dernière image',
      leave: 'Vos réponses n’ont pas encore été téléchargées.'
    }
  };

  var state = { rater: null, lang: 'en', index: 0, answers: {}, downloaded: false };
  var storageOk = (function () {
    try { localStorage.setItem('rating-test', '1'); localStorage.removeItem('rating-test'); return true; }
    catch (e) { return false; }
  })();

  function $(id) { return document.getElementById(id); }
  function fill(text, values) {
    return text.replace(/\{(\w+)\}/g, function (match, name) { return name in values ? values[name] : match; });
  }
  function t(name, values) { return fill(TEXT[state.lang][name], values || { total: TOTAL }); }
  function checkedValue(name) {
    var input = document.querySelector('input[name="' + name + '"]:checked');
    return input ? input.value : null;
  }
  function setChecked(name, value) {
    var inputs = document.querySelectorAll('input[name="' + name + '"]');
    for (var i = 0; i < inputs.length; i++) { inputs[i].checked = inputs[i].value === String(value); }
  }
  function storageKey(rater) { return 'image-rating-' + PACKET + '-' + rater; }
  function saved(rater) {
    if (!storageOk) { return null; }
    try { return JSON.parse(localStorage.getItem(storageKey(rater))); } catch (e) { return null; }
  }
  function save() {
    if (!storageOk || !state.rater) { return; }
    try {
      localStorage.setItem(storageKey(state.rater), JSON.stringify(
        { lang: state.lang, index: state.index, answers: state.answers, downloaded: state.downloaded }));
    } catch (e) { storageOk = false; }
  }
  function order() { return ORDERS[state.rater]; }
  function answeredCount() {
    return order().filter(function (code) { return state.answers[code]; }).length;
  }
  function firstUnanswered() {
    var codes = order();
    for (var i = 0; i < codes.length; i++) { if (!state.answers[codes[i]]) { return i; } }
    return codes.length;
  }
  function show(id) {
    ['start', 'intro', 'rate', 'end'].forEach(function (name) { $(name).hidden = name !== id; });
    window.scrollTo(0, 0);
  }

  function applyTexts() {
    document.documentElement.lang = state.lang;
    document.title = t('title');
    var nodes = document.querySelectorAll('[data-t]');
    for (var i = 0; i < nodes.length; i++) { nodes[i].textContent = t(nodes[i].getAttribute('data-t')); }
    updateStart();
  }

  function updateStart() {
    var rater = checkedValue('rater');
    var previous = rater ? saved(rater) : null;
    var count = previous && previous.answers ? Object.keys(previous.answers).length : 0;
    $('resumeNote').hidden = count === 0;
    $('resumeNote').textContent = count ? t('resumeNote', { n: count, total: TOTAL }) : '';
    $('restartButton').hidden = count === 0;
    $('startButton').textContent = count ? t('resume') : t('start');
    $('storageNote').hidden = storageOk;
  }

  function startSession(event) {
    event.preventDefault();
    var rater = checkedValue('rater');
    if (!rater) { $('startMessage').textContent = t('chooseRater'); $('startMessage').hidden = false; return; }
    state.rater = rater;
    var previous = saved(rater);
    state.answers = previous && previous.answers ? previous.answers : {};
    state.downloaded = Boolean(previous && previous.downloaded);
    state.index = firstUnanswered();
    save();
    var box = $('instructions');
    box.textContent = '';
    TEXT[state.lang].instructions.forEach(function (paragraph) {
      var p = document.createElement('p');
      p.textContent = fill(paragraph, { total: TOTAL });
      box.appendChild(p);
    });
    $('beginButton').textContent = answeredCount() ? t('continueRating') : t('begin');
    show('intro');
  }

  function restart() {
    var rater = checkedValue('rater');
    if (!rater || !window.confirm(t('restartConfirm'))) { return; }
    try { localStorage.removeItem(storageKey(rater)); } catch (e) { /* nothing saved */ }
    updateStart();
  }

  function begin() {
    if (state.index >= TOTAL) { finish(); return; }
    show('rate');
    renderItem();
  }

  function renderItem() {
    var code = order()[state.index];
    var stage = $('stage');
    stage.classList.remove('zoomed');
    stage.scrollTop = 0;
    stage.scrollLeft = 0;
    $('imageError').hidden = true;
    $('photo').hidden = false;
    $('photo').src = 'images/' + code + '.png';
    if (state.index + 1 < TOTAL) { new Image().src = 'images/' + order()[state.index + 1] + '.png'; }
    var answer = state.answers[code];
    setChecked('realism', answer ? answer.realism : null);
    setChecked('artifact', answer ? answer.artifact : null);
    $('comment').value = answer ? answer.comment : '';
    $('progressText').textContent = t('progress', { i: state.index + 1, total: TOTAL });
    $('progressBar').style.width = (100 * state.index / TOTAL) + '%';
    $('backButton').disabled = state.index === 0;
    $('nextButton').textContent = state.index + 1 === TOTAL ? t('finish') : t('next');
    updateNext();
  }

  function updateNext() { $('nextButton').disabled = !(checkedValue('realism') && checkedValue('artifact')); }

  function commit() {
    var realism = checkedValue('realism');
    var artifact = checkedValue('artifact');
    if (!realism || !artifact) { return false; }
    var code = order()[state.index];
    var answer = { realism: Number(realism), artifact: artifact, comment: $('comment').value.trim() };
    var old = state.answers[code];
    if (!old || old.realism !== answer.realism || old.artifact !== answer.artifact || old.comment !== answer.comment) {
      answer.timestamp = new Date().toISOString();
      state.answers[code] = answer;
      state.downloaded = false;
    }
    save();
    return true;
  }

  function next() {
    if (!commit()) { return; }
    state.index += 1;
    save();
    if (state.index >= TOTAL) { finish(); } else { renderItem(); }
  }

  function back() {
    commit();
    if (state.index > 0) { state.index -= 1; save(); renderItem(); }
  }

  function csvField(value) {
    var text = value === undefined || value === null ? '' : String(value);
    return /[",\r\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }

  function buildCsv() {
    var lines = [COLUMNS.join(',')];
    order().forEach(function (code, i) {
      var a = state.answers[code] || {};
      lines.push([code, a.realism, a.artifact, a.comment, state.rater, i + 1, a.timestamp].map(csvField).join(','));
    });
    return '\ufeff' + lines.join('\r\n') + '\r\n';
  }

  function fileName() {
    var d = new Date();
    var pad = function (x) { return (x < 10 ? '0' : '') + x; };
    return 'ratings_' + state.rater + '_' + d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + '-' +
      pad(d.getHours()) + pad(d.getMinutes()) + '.csv';
  }

  function finish() {
    state.index = TOTAL;
    save();
    $('endText').textContent = t('endText');
    $('downloadNote').hidden = true;
    $('dataBox').hidden = true;
    show('end');
  }

  function download() {
    var name = fileName();
    var url = URL.createObjectURL(new Blob([buildCsv()], { type: 'text/csv;charset=utf-8' }));
    var link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(function () { URL.revokeObjectURL(url); }, 10000);
    state.downloaded = true;
    save();
    $('downloadNote').textContent = t('downloaded', { file: name });
    $('downloadNote').hidden = false;
  }

  function showData() {
    $('csvText').value = buildCsv().replace(/^\ufeff/, '');
    $('dataBox').hidden = false;
    $('csvText').focus();
    $('csvText').select();
  }

  function review() {
    state.index = TOTAL - 1;
    save();
    show('rate');
    renderItem();
  }

  document.querySelectorAll('input[name="lang"]').forEach(function (input) {
    input.addEventListener('change', function () { state.lang = input.value; applyTexts(); });
  });
  document.querySelectorAll('input[name="rater"]').forEach(function (input) {
    input.addEventListener('change', function () { $('startMessage').hidden = true; updateStart(); });
  });
  document.querySelectorAll('input[name="realism"], input[name="artifact"]').forEach(function (input) {
    input.addEventListener('change', updateNext);
  });
  $('startForm').addEventListener('submit', startSession);
  $('restartButton').addEventListener('click', restart);
  $('beginButton').addEventListener('click', begin);
  $('nextButton').addEventListener('click', next);
  $('backButton').addEventListener('click', back);
  $('downloadButton').addEventListener('click', download);
  $('showDataButton').addEventListener('click', showData);
  $('reviewButton').addEventListener('click', review);
  $('stage').addEventListener('click', function () { $('stage').classList.toggle('zoomed'); });
  $('photo').addEventListener('error', function () { $('photo').hidden = true; $('imageError').hidden = false; });
  window.addEventListener('beforeunload', function (event) {
    if (state.rater && !state.downloaded && answeredCount() > 0) {
      event.preventDefault();
      event.returnValue = t('leave');
      return t('leave');
    }
  });
  var preset = checkedValue('lang');
  state.lang = preset === 'fr' ? 'fr' : 'en';
  applyTexts();
})();
</script>
</body>
</html>
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('sample', 'check', 'analyse'))
    parser.add_argument('--dir', type=Path, default=OUTPUT, help='Where the sample, key and manifest live')
    parser.add_argument('--force', action='store_true', help='sample: rebuild an existing packet')
    parser.add_argument('--ratings', nargs=2, type=Path, metavar=('A.csv', 'B.csv'), help='analyse: the two rating files')
    parser.add_argument('--output', type=Path, help='analyse: result directory (default: --dir)')
    args = parser.parse_args()
    if args.command == 'sample':
        result = sample(args.dir, args.force)
        size = sum(p.stat().st_size for p in layout(args.dir)['packet'].rglob('*') if p.is_file())
        print(f'Wrote {result["images"]} images and rate.html to {layout(args.dir)["packet"]} ({size / 1e6:.1f} MB); '
              f'key and manifest in {args.dir}')
        for row in result['sample']:
            print(f'  r{row["reference"]:02d}  {row["family"]}')
    elif args.command == 'check':
        check(args.dir)
        print('Packet, key and manifest agree; no file or word in the packet unblinds it.')
    else:
        if not args.ratings:
            parser.error('analyse needs --ratings A.csv B.csv')
        result = analyse(args.ratings, args.dir, args.output)
        print(json.dumps({k: result[k] for k in ('agreement', 'flagged_major_either')}, indent=2))
