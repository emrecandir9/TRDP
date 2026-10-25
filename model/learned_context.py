"""Learned context pathways (docs/model-method.md §7.1): H2-learned and H4.

Both give each display a match m in [0, 1], fitted per outer fold on its 19 training sources only and
used, like H2's gate, as g = c + (1 - c) m on the trained H2 networks at evaluation. No condition
label, display name, MEP analogue or human datum enters.

H2-learned, attachment learned by forearm prediction: a ridge regression learns, from full-body
images, the luminance of the forearm band just below the wrist (0 to 1 P, +-0.5 P across, 16 x 16)
from the hand window (0 to 1.5 P towards the fingers, +-0.75 P across, 24 x 24); P = palm width,
wrist and hand axis from the source's landmarks. A display's error e is the mean squared difference
between the predicted and displayed band (static frame).

H4, own-body familiarity: principal components (90% of variance) of the training sources' standardised
full-body static encoder features; a display's distance d is the norm of its whitened scores.

Calibration, both: E = the leave-one-training-source-out errors (distances) of full-body static
frames, everything refitted without the left-out source; m = min(1, 2 P(E >= e)).
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as nd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'model'))
import features as fx  # noqa: E402
import trials  # noqa: E402

OUTPUT = ROOT / 'output/learned_context'
DISPLAYS = tuple(c for c in fx.CONDITIONS if c != 'hand_only')
WINDOWS = dict(hand=dict(along=(0., 1.5), across=(-.75, .75), size=24), band=dict(along=(-1., 0.), across=(-.5, .5), size=16))
RIDGE = tuple(10. ** k for k in range(-2, 5))
VARIANCE = .9
require = trials.require


def hand_geometry(case):
    """Wrist point (x, y), unit axis towards the fingers, unit axis across, palm width P (2048-px
    image coordinates), from the static landmarks, as scripts/controls.py."""
    points = np.array(case['static_points'], float)
    wrist = points[0]
    axis = (points[9] - wrist) / np.linalg.norm(points[9] - wrist)
    return wrist, axis, np.array([-axis[1], axis[0]]), float(np.linalg.norm(points[5] - points[17]))


def sample(luma, geometry, window):
    """Luminance on a size x size grid aligned with the hand axis; the image is blurred first with a
    Gaussian of half the grid spacing, so the samples are not aliased."""
    wrist, axis, across, palm = geometry
    w = WINDOWS[window]
    along = np.linspace(*w['along'], w['size']) * palm
    side = np.linspace(*w['across'], w['size']) * palm
    a, c = np.meshgrid(along, side, indexing='ij')
    x = wrist[0] + a * axis[0] + c * across[0]
    y = wrist[1] + a * axis[1] + c * across[1]
    spacing = (w['along'][1] - w['along'][0]) * palm / (w['size'] - 1)
    blurred = nd.gaussian_filter(luma, spacing / 2)
    return nd.map_coordinates(blurred, [y.ravel(), x.ravel()], order=1, mode='constant', cval=0.)


_renderer = {}


def family_windows(family):
    """Hand window and forearm band of every display (static frame) and of the full-body movement
    frame of one family."""
    if 'r' not in _renderer:
        _renderer['r'] = fx.Renderer()
    case = fx.s.read(fx.s.DATA / 'inputs' / family['source']['id'] / 'case.json')
    geometry = hand_geometry(case)
    images = _renderer['r'].render(family)
    out = {}
    for name in DISPLAYS:
        luma = fx.luminance(images[name][0])
        out[name] = (sample(luma, geometry, 'hand'), sample(luma, geometry, 'band'))
    luma = fx.luminance(images['full_body'][1])
    out['full_body_movement'] = (sample(luma, geometry, 'hand'), sample(luma, geometry, 'band'))
    return family['id'], family['reference'], out


def extract(workers=8):
    """Windows of all 480 families; written once to OUTPUT/windows.npz."""
    families = fx.families()
    with Pool(workers) as pool:
        results = pool.map(family_windows, families)
    ids = [r[0] for r in results]
    sources = np.array([r[1] for r in results])
    keys = DISPLAYS + ('full_body_movement',)
    hand = np.stack([[r[2][k][0] for k in keys] for r in results])     # [families, keys, 576]
    band = np.stack([[r[2][k][1] for k in keys] for r in results])     # [families, keys, 256]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / 'windows.npz', ids=np.array(ids), sources=sources, keys=np.array(keys), hand=hand, band=band)


def ridge_fit(x, y, penalty):
    mean_x, mean_y = x.mean(0), y.mean(0)
    xc = x - mean_x
    weights = np.linalg.solve(xc.T @ xc + penalty * np.eye(x.shape[1]), xc.T @ (y - mean_y))
    return lambda z: (z - mean_x) @ weights + mean_y


def match(error, calibration):
    """m = min(1, 2 P(E >= e)) for each e, against the calibration errors E."""
    calibration = np.sort(calibration)
    above = 1 - np.searchsorted(calibration, error, side='left') / len(calibration)
    return np.minimum(1., 2 * above)


def forearm_fold(windows, held_out):
    """H2-learned for one outer fold: match m of every family and display."""
    keys = list(windows['keys'])
    fb, mv = keys.index('full_body'), keys.index('full_body_movement')
    train = windows['sources'] != held_out
    hand, band, sources = windows['hand'], windows['band'], windows['sources']
    rows = lambda mask: (np.concatenate([hand[mask, fb], hand[mask, mv]]), np.concatenate([band[mask, fb], band[mask, mv]]))
    training_sources = sorted(set(sources[train]))

    def loso(penalty):
        errors = []
        for s in training_sources:
            fit = ridge_fit(*rows(train & (sources != s)), penalty)
            left = train & (sources == s)
            errors.append(((fit(hand[left, fb]) - band[left, fb]) ** 2).mean(1))
        return np.concatenate(errors)

    scores = {p: loso(p) for p in RIDGE}
    penalty = min(RIDGE, key=lambda p: scores[p].mean())
    fit = ridge_fit(*rows(train), penalty)
    errors = ((fit(hand.reshape(-1, hand.shape[-1])) - band.reshape(-1, band.shape[-1])) ** 2).mean(1).reshape(hand.shape[:2])
    return match(errors, scores[penalty]), dict(penalty=penalty, calibration_median=float(np.median(scores[penalty])))


def familiarity_fold(x, sources, conditions, families, held_out):
    """H4 for one outer fold: x [images, D] encoder features of static frames; returns m per image."""
    train_fb = (sources != held_out) & (conditions == 'full_body')

    def model(mask):
        mean, sd = x[mask].mean(0), x[mask].std(0)
        sd = np.where(sd > 0, sd, 1.)
        z = (x[mask] - mean) / sd
        u, singular, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
        share = np.cumsum(singular ** 2) / np.sum(singular ** 2)
        k = int(np.searchsorted(share, VARIANCE) + 1)
        centre, scale = z.mean(0), singular[:k] / np.sqrt(mask.sum() - 1)
        return lambda y: np.linalg.norm((((y - mean) / sd - centre) @ vt[:k].T) / scale, axis=1), k

    calibration = []
    for s in sorted(set(sources[train_fb])):
        distance, _ = model(train_fb & (sources != s))
        calibration.append(distance(x[train_fb & (sources == s)]))
    distance, k = model(train_fb)
    return match(distance(x), np.concatenate(calibration)), dict(components=k)


def build(representations=('openclip', 'dinov2_cls', 'dinov2_patchmean')):
    """Gate tables of both pathways for every outer fold; written to OUTPUT/gates.csv and summary.json."""
    import pandas as pd
    import bank as bk
    with np.load(OUTPUT / 'windows.npz') as z:
        windows = {k: z[k] for k in z.files}
    keys = list(windows['keys'])
    rows, summary = [], dict(forearm={}, familiarity={r: {} for r in representations})
    for held_out in range(1, 21):
        m, info = forearm_fold(windows, held_out)
        summary['forearm'][held_out] = info
        for i, family in enumerate(windows['ids']):
            for j, display in enumerate(DISPLAYS):
                rows.append(dict(fold=held_out, family=family, source=int(windows['sources'][i]), display=display,
                                 pathway='forearm', m=float(m[i, keys.index(display)])))
    for representation in representations:
        bank = bk.Bank(representation)
        index = [(f, c) for (f, c, frame) in bank.rows if frame == 'static' and c in DISPLAYS]
        x = bank.matrix([(f, c, 'static') for f, c in index])
        sources = np.array([int(f[1:3]) for f, _ in index])
        conditions = np.array([c for _, c in index])
        for held_out in range(1, 21):
            m, info = familiarity_fold(x, sources, conditions, None, held_out)
            summary['familiarity'][representation][held_out] = info
            rows += [dict(fold=held_out, family=f, source=int(s), display=c, pathway=f'familiarity_{representation}', m=float(v))
                     for (f, c), s, v in zip(index, sources, m)]
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / 'gates.csv', index=False)
    # held-out source only: the values the evaluation uses
    held = table[table.source == table.fold]
    means = held.groupby(['pathway', 'display']).m.mean().unstack()
    summary['held_out_mean_m'] = {p: {d: round(float(v), 3) for d, v in row.items()} for p, row in means.iterrows()}
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return means


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=('extract', 'build'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.step == 'extract':
        extract(args.workers)
    else:
        print(build().round(3).to_string())
