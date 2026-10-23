"""Gate 1 visual baselines (analysis spec §6.1, §8) and latent-space analyses; method in docs/baselines.md.

Sources, not images, are the independent units: every per-family value is first averaged over
the 24 appearance variants of its source, and every interval is a percentile bootstrap over
the 20 sources (10,000 resamples, seed 20261022). No human data enter any computation; the
human intervals appear only as reference lines. Decoders are fitted on training sources only,
with the regularisation chosen by grouped inner folds (spec §7.1).

Input: output/features/ from analysis/features.py. Output: output/baselines/ (tables,
summary.json) and docs/figures/baselines-*.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
import features as fx  # noqa: E402

FEATURES = ROOT / 'output/features'
OUTPUT = ROOT / 'output/baselines'
FIGURES = ROOT / 'docs/figures'
SEED, SAMPLES, INTERVAL = 20261022, 10000, .95
# name -> (array in features.npz, per-dimension scaling). Low-level statistics have different
# units, so each is divided by its own between-source spread; embeddings use the total spread.
REPRESENTATIONS = dict(lowlevel=('lowlevel', True), pixels64=('pixels64', False), dinov2_cls=('dinov2_cls', False),
                       dinov2_patchmean=('dinov2_patchmean', False), openclip=('openclip', False),
                       dinov2_448_cls=('dinov2_448_cls', False), dinov2_448_patchmean=('dinov2_448_patchmean', False))
PRIMARY = ('lowlevel', 'pixels64', 'dinov2_cls', 'dinov2_patchmean', 'openclip')
LOW_LEVEL = ('lowlevel', 'pixels64')
LABEL = dict(lowlevel='Low-level statistics', pixels64='Pixels (64 x 64 luminance)', dinov2_cls='DINOv2 summary token',
             dinov2_patchmean='DINOv2 mean patch token', openclip='OpenCLIP bigG embedding',
             dinov2_448_cls='DINOv2 summary token, 448 px', dinov2_448_patchmean='DINOv2 mean patch token, 448 px')
SHORT = dict(full_body='FB', upper_limb='UL', detached_hand_silhouette='DH', detached_hand='DH rect',
             hand_stub='stub', hand_gap_stub='gap stub', window_hand='window hand', window_arm='window arm',
             scrambled='scrambled', occluded_wrist='occluded wrist', hand_only='hand only')
CONNECTED = ('full_body', 'upper_limb', 'hand_stub', 'window_arm')
DISCONNECTED = ('detached_hand_silhouette', 'detached_hand', 'hand_gap_stub', 'window_hand', 'scrambled')
CONTEXTS = fx.MAIN + fx.controls.CONTROLS   # everything except hand_only, which has no context
HUMAN = dict(ratio_DH_to_connected=(-.11, -.67, .38), ratio_UL_to_FB=(.77, .34, 1.78))   # spec §4.5, reference only
ATTACHMENT_TASKS = dict(stub_pair=(('hand_stub',), ('hand_gap_stub',)), window_pair=(('window_arm',), ('window_hand',)),
                        main=(('full_body', 'upper_limb'), ('detached_hand_silhouette',)))
require = fx.require


# ---------- data ----------

class Data:
    """Features as arrays [family, condition, frame, dim] in the order of families()."""

    def __init__(self, directory=FEATURES, arrays=None, index=None):
        if arrays is None:
            index = pd.read_csv(directory / 'index.csv')
            with np.load(directory / 'features.npz') as z:
                arrays = {k: z[k] for k in z.files}
        order = list(dict.fromkeys(index.family))
        expected = [(f, c, r) for f in order for c in fx.CONDITIONS for r in fx.FRAMES]
        require(list(zip(index.family, index.condition, index.frame)) == expected, 'Unexpected feature order')
        by_id = {f['id']: f for f in fx.families()}
        self.families = [by_id[f] for f in order]
        self.source = np.array([f['reference'] for f in self.families])
        self.variant = np.array([f['variant'] for f in self.families])
        self.sources = np.unique(self.source)
        shape = (len(order), len(fx.CONDITIONS), len(fx.FRAMES))
        self.x = {k: v.reshape(*shape, -1).astype(np.float64) for k, v in arrays.items()}
        self.directory = directory


def column(condition):
    return fx.CONDITIONS.index(condition)


# ---------- movement signal (spec §6.1) ----------

def spread(static, variant, per_dimension=False):
    """Between-source spread: sqrt(E||x_s - x_t||^2 / 2) over pairs of different sources that
    share an appearance variant (same desk, seed and white balance); per dimension if asked.
    For one dimension this is the between-source standard deviation."""
    squared = []
    for v in np.unique(variant):
        x = static[variant == v]
        i, j = np.triu_indices(len(x), 1)
        squared.append((x[i] - x[j]) ** 2)
    mean = np.concatenate(squared).mean(axis=0)
    return np.sqrt(mean / 2) if per_dimension else float(np.sqrt(mean.sum() / 2))


def movement_signal(x, variant, per_dimension=False, measure='normalised'):
    """Per family and condition: static -> movement distance. x: [family, condition, frame, dim]."""
    a, b = x[:, :, 0], x[:, :, 1]
    out = np.empty(a.shape[:2])
    for c in range(a.shape[1]):
        if measure == 'normalised':
            scale = spread(a[:, c], variant, per_dimension)
            out[:, c] = np.linalg.norm((b[:, c] - a[:, c]) / scale, axis=1)
        elif measure == 'raw':
            out[:, c] = np.linalg.norm(b[:, c] - a[:, c], axis=1)
        elif measure == 'cosine':
            out[:, c] = 1 - (a[:, c] * b[:, c]).sum(1) / (np.linalg.norm(a[:, c], axis=1) * np.linalg.norm(b[:, c], axis=1))
        else:
            raise ValueError(measure)
    return out


def source_means(values, source):
    """Average families within each source: [source, ...]."""
    return np.stack([values[source == s].mean(axis=0) for s in np.unique(source)])


def bootstrap_index(n, samples=SAMPLES, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, (samples, n))


def interval(values):
    tail = (1 - INTERVAL) / 2
    return float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))


def contrasts(dh='detached_hand_silhouette'):
    """Spec §4.3 applied to one movement signal per condition (no muscles): (name, f(cells))."""
    c = lambda name: (lambda m: m[..., column(name)])
    connected = lambda m: (c('full_body')(m) + c('upper_limb')(m)) / 2
    out = [('C', lambda m: connected(m) - c(dh)(m))]
    out += [(f'delta_{SHORT[n]}', c(n)) for n in fx.CONDITIONS]
    out += [('FB_minus_UL', lambda m: c('full_body')(m) - c('upper_limb')(m)),
            ('stub_minus_gap_stub', lambda m: c('hand_stub')(m) - c('hand_gap_stub')(m)),
            ('window_arm_minus_window_hand', lambda m: c('window_arm')(m) - c('window_hand')(m)),
            ('UL_minus_scrambled', lambda m: c('upper_limb')(m) - c('scrambled')(m)),
            ('UL_minus_occluded_wrist', lambda m: c('upper_limb')(m) - c('occluded_wrist')(m)),
            ('stub_minus_DH', lambda m: c('hand_stub')(m) - c(dh)(m))]
    ratios = [('ratio_DH_to_connected', lambda m: c(dh)(m) / connected(m)),
              ('ratio_UL_to_FB', lambda m: c('upper_limb')(m) / c('full_body')(m)),
              ('ratio_DH_to_FB', lambda m: c(dh)(m) / c('full_body')(m))]
    return out, ratios


def summarise_signal(cells, boot, dh='detached_hand_silhouette'):
    """cells: [source, condition]; boot: bootstrap source indices. Contrasts and ratios with intervals."""
    means = cells[boot].mean(axis=1)
    rows = []
    differences, ratios = contrasts(dh)
    for name, f in differences:
        per_source = f(cells)
        low, high = interval(f(means))
        rows.append(dict(statistic=name, kind='contrast', estimate=float(per_source.mean()), ci_low=low, ci_high=high,
                         n_positive=int((per_source > 0).sum()), n=len(per_source)))
    for name, f in ratios:
        low, high = interval(f(means))
        rows.append(dict(statistic=name, kind='ratio', estimate=float(f(cells.mean(axis=0))), ci_low=low, ci_high=high,
                         n_positive=np.nan, n=len(cells)))
    return rows


def signal_tables(data, representations=REPRESENTATIONS):
    """All movement-signal summaries: representation x measure x cohort x detached-hand version."""
    rows, families = [], []
    cohorts = dict(all=np.ones(len(data.source), bool), without_17=data.source != 17)
    for name, (key, per_dimension) in representations.items():
        if key not in data.x:
            continue
        for measure in ('normalised', 'raw', 'cosine'):
            if per_dimension and measure != 'normalised':
                continue   # raw and cosine distances are undefined across mixed units
            signal = movement_signal(data.x[key], data.variant, per_dimension, measure)
            if measure == 'normalised':
                for i, f in enumerate(data.families):
                    families.extend(dict(representation=name, family=f['id'], source=f['reference'], desk=f['desk'],
                                         seed=f['seed'], balance=f['balance'], condition=c, signal=signal[i, j])
                                    for j, c in enumerate(fx.CONDITIONS))
            for cohort, keep in cohorts.items():
                cells = source_means(signal[keep], data.source[keep])
                boot = bootstrap_index(len(cells))
                for dh in ('detached_hand_silhouette', 'detached_hand'):
                    for r in summarise_signal(cells, boot, dh):
                        rows.append(dict(representation=name, measure=measure, cohort=cohort, detached_hand=dh, **r))
    return pd.DataFrame(rows), pd.DataFrame(families)


def gate1(table):
    """P1 for each baseline (C > 0, interval excluding 0) and S1/S2 compatibility, primary analysis."""
    primary = table.query("measure == 'normalised' and cohort == 'all' and detached_hand == 'detached_hand_silhouette'")
    rows = []
    for name, group in primary.groupby('representation', sort=False):
        g = group.set_index('statistic')
        c, r1, r2 = g.loc['C'], g.loc['ratio_DH_to_connected'], g.loc['ratio_UL_to_FB']
        rows.append(dict(representation=name, C=c.estimate, C_low=c.ci_low, C_high=c.ci_high,
                         sources_positive=int(c.n_positive), meets_P1=bool(c.ci_low > 0),
                         reverse_pattern=bool(c.ci_high < 0),
                         ratio_DH_to_connected=r1.estimate, S1_compatible=bool(HUMAN['ratio_DH_to_connected'][1] <= r1.estimate
                                                                               <= HUMAN['ratio_DH_to_connected'][2]),
                         ratio_UL_to_FB=r2.estimate, S2_compatible=bool(HUMAN['ratio_UL_to_FB'][1] <= r2.estimate
                                                                       <= HUMAN['ratio_UL_to_FB'][2])))
    return pd.DataFrame(rows)


def variance_shares(families, representation, conditions=('full_body', 'upper_limb', 'detached_hand_silhouette')):
    """Share of between-family variance of the signal due to each factor (balanced design)."""
    rows = []
    for condition in conditions:
        d = families.query('representation == @representation and condition == @condition')
        total = ((d.signal - d.signal.mean()) ** 2).sum()
        shares = {f: float(d.groupby(f).signal.transform('mean').sub(d.signal.mean()).pow(2).sum() / total)
                  for f in ('source', 'desk', 'seed', 'balance')}
        rows.append(dict(representation=representation, condition=condition, **shares,
                         residual=1 - sum(shares.values())))
    return pd.DataFrame(rows)


# ---------- linear decoding ----------

def decoder():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced'))
    return GridSearchCV(model, {'logisticregression__C': np.logspace(-4, 2, 7)}, cv=GroupKFold(5), scoring='accuracy')


def paired_accuracy(positive, negative):
    """Fraction of matched pairs whose positive item scores higher (ties count half)."""
    return float(((positive > negative) + .5 * (positive == negative)).mean())


def fit_fold(x_train, y_train, groups):
    model = decoder()
    model.fit(x_train, y_train, groups=groups)
    return model


def movement_decoding_fold(x, source, held_out, train_condition):
    """Train static vs movement on one condition of 19 sources; test every condition of the 20th."""
    train, test = source != held_out, source == held_out
    c = column(train_condition)
    xt = np.concatenate([x[train, c, 0], x[train, c, 1]])
    y = np.r_[np.zeros(train.sum()), np.ones(train.sum())]
    model = fit_fold(xt, y, np.r_[source[train], source[train]])
    out = []
    for j, condition in enumerate(fx.CONDITIONS):
        a, b = model.decision_function(x[test, j, 0]), model.decision_function(x[test, j, 1])
        out.append(dict(source=int(held_out), train=train_condition, test=condition, paired=paired_accuracy(b, a),
                        single=float(((a < 0).mean() + (b > 0).mean()) / 2),
                        C=float(model.best_params_['logisticregression__C'])))
    return out


def attachment_fold(x, source, held_out, task):
    """Train connected vs disconnected (static and movement frames) on 19 sources; test all tasks."""
    train, test = source != held_out, source == held_out
    positive, negative = ATTACHMENT_TASKS[task]
    xs = [x[train, column(c)].reshape(-1, x.shape[-1]) for c in positive + negative]
    labels = [np.full(train.sum() * 2, int(c in positive)) for c in positive + negative]
    groups = np.concatenate([np.repeat(source[train], 2) for _ in positive + negative])
    model = fit_fold(np.concatenate(xs), np.concatenate(labels), groups)
    out = []
    for test_task, (pos, neg) in ATTACHMENT_TASKS.items():
        score = {c: model.decision_function(x[test, column(c)].reshape(-1, x.shape[-1])) for c in pos + neg}
        accuracy = np.mean([paired_accuracy(score[p], score[n]) for p in pos for n in neg])
        out.append(dict(source=int(held_out), train=task, test=test_task, paired=float(accuracy),
                        C=float(model.best_params_['logisticregression__C'])))
    return out


def decoding_tables(data, representations=PRIMARY, jobs=-1):
    from joblib import Parallel, delayed
    movement, attachment = [], []
    for name in representations:
        x = data.x[REPRESENTATIONS[name][0]]
        runs = Parallel(n_jobs=jobs)(delayed(movement_decoding_fold)(x, data.source, s, c)
                                     for c in fx.CONDITIONS for s in data.sources)
        movement += [dict(representation=name, **r) for run in runs for r in run]
        runs = Parallel(n_jobs=jobs)(delayed(attachment_fold)(x, data.source, s, t)
                                     for t in ATTACHMENT_TASKS for s in data.sources)
        attachment += [dict(representation=name, **r) for run in runs for r in run]
        print(f'Decoded {name}', flush=True)
    return pd.DataFrame(movement), pd.DataFrame(attachment)


def summarise_folds(folds, keys, value='paired'):
    """Mean over held-out sources with a source bootstrap interval."""
    rows = []
    for key, group in folds.groupby(list(keys), sort=False):
        v = group.sort_values('source')[value].to_numpy()
        boot = v[bootstrap_index(len(v))].mean(axis=1)
        low, high = interval(boot)
        rows.append(dict(zip(keys, key), mean=float(v.mean()), ci_low=low, ci_high=high,
                         sources_above_chance=int((v > .5).sum()), n=len(v)))
    return pd.DataFrame(rows)


# ---------- latent-space geometry ----------

def directions(x, variant, per_dimension):
    """Unit static -> movement vectors [family, condition, dim]."""
    d = x[:, :, 1] - x[:, :, 0]
    if per_dimension:
        d = d / np.stack([spread(x[:, c, 0], variant, True) for c in range(x.shape[1])], 1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def alignment_tables(data, representations=PRIMARY):
    """Cross-context alignment cos(u_c, u_FB) within a family and cross-source consistency within a context."""
    rows = []
    for name in representations:
        key, per_dimension = REPRESENTATIONS[name]
        u = directions(data.x[key], data.variant, per_dimension)
        to_fb = (u * u[:, [column('full_body')]]).sum(-1)
        cells = source_means(to_fb, data.source)
        consistency = np.zeros((len(data.sources), len(fx.CONDITIONS)))
        counts = np.zeros(len(data.sources))
        for v in np.unique(data.variant):
            members = np.flatnonzero(data.variant == v)
            g = np.einsum('icd,jcd->ijc', u[members], u[members])
            g[np.arange(len(members)), np.arange(len(members))] = np.nan
            for k, i in enumerate(members):
                consistency[np.searchsorted(data.sources, data.source[i])] += np.nanmean(g[k], axis=0)
                counts[np.searchsorted(data.sources, data.source[i])] += 1
        consistency /= counts[:, None]
        boot = bootstrap_index(len(cells))
        for measure, values in (('alignment_with_FB', cells), ('cross_source_consistency', consistency)):
            for j, condition in enumerate(fx.CONDITIONS):
                low, high = interval(values[boot, j].mean(axis=1))
                rows.append(dict(representation=name, measure=measure, condition=condition,
                                 mean=float(values[:, j].mean()), ci_low=low, ci_high=high))
    return pd.DataFrame(rows)


def rdm(points):
    """Euclidean distances between condition patterns, divided by sqrt(dimension)."""
    d = points[:, None] - points[None]
    return np.sqrt((d ** 2).sum(-1) / points.shape[-1])


def rank_residual(y, covariates):
    y = stats.rankdata(y)
    z = np.column_stack([np.ones(len(y))] + [stats.rankdata(c) for c in covariates])
    return y - z @ np.linalg.lstsq(z, y, rcond=None)[0]


def partial_spearman(y, x, covariates):
    return float(np.corrcoef(rank_residual(y, covariates), rank_residual(x, covariates))[0, 1])


def rsa_tables(data, representations=PRIMARY):
    """Per-source RDMs over the context conditions (static frames, standardised dimensions),
    compared with an attachment model RDM, partialling out visible area and luminance RDMs."""
    conditions = [c for c in CONTEXTS if c != 'occluded_wrist']   # no declared attachment label
    cols = [column(c) for c in conditions]
    attached = np.array([c in CONNECTED for c in conditions], float)
    i, j = np.triu_indices(len(conditions), 1)
    model = (attached[:, None] != attached[None]).astype(float)[i, j]
    low = data.x['lowlevel'][:, cols, 0]
    rows, matrices = [], {}
    for name in representations:
        x = data.x[REPRESENTATIONS[name][0]][:, cols, 0]
        flat = x.reshape(-1, x.shape[-1])
        z = (x - flat.mean(0)) / np.where(flat.std(0) > 0, flat.std(0), 1)
        per_source = []
        for s in data.sources:
            members = data.source == s
            r = np.mean([rdm(z[k]) for k in np.flatnonzero(members)], axis=0)
            area = np.log(low[members, :, 0].mean(0))
            luminance = low[members, :, 1].mean(0)
            covariates = [np.abs(area[:, None] - area[None])[i, j], np.abs(luminance[:, None] - luminance[None])[i, j]]
            per_source.append(dict(source=int(s), spearman=float(stats.spearmanr(r[i, j], model)[0]),
                                   partial=partial_spearman(r[i, j], model, covariates), rdm=r))
        matrices[name] = np.mean([p['rdm'] for p in per_source], axis=0)
        boot = bootstrap_index(len(per_source))
        for measure in ('spearman', 'partial'):
            v = np.array([p[measure] for p in per_source])
            lo, hi = interval(v[boot].mean(axis=1))
            rows.append(dict(representation=name, measure=measure, mean=float(v.mean()), ci_low=lo, ci_high=hi,
                             sources_positive=int((v > 0).sum()), n=len(v)))
    return pd.DataFrame(rows), matrices, conditions


def hand_patches(source, grid=16, dilation=6):
    """Patches (of the 2048 image on a grid x grid lattice) touching the hand or its movement."""
    hand = (fx.s.load(source, 'static') > 0) | (fx.s.load(source, 'movement') > 0)
    hand = fx.nd.binary_dilation(hand, iterations=dilation)
    size = hand.shape[0] // grid
    return hand.reshape(grid, size, grid, size).any(axis=(1, 3))


def patch_tables(data, directory=FEATURES):
    """Mean |patch(movement) - patch(static)| maps (DINOv2, 224 px) and the share outside the hand."""
    path = directory / 'dinov2_patches.npy'
    if not path.is_file():
        return None, None
    tokens = np.load(path, mmap_mode='r')
    n, c = len(data.families), len(fx.CONDITIONS)
    maps = np.zeros((c, 256))
    outside = np.zeros((len(data.sources), c))
    first = {f['reference']: f['source'] for f in data.families}
    masks = {reference: hand_patches(source).ravel() for reference, source in first.items()}
    for i, family in enumerate(data.families):
        block = np.asarray(tokens[i * c * 2:(i + 1) * c * 2], np.float64).reshape(c, 2, 256, -1)
        change = np.linalg.norm(block[:, 1] - block[:, 0], axis=-1)   # [condition, patch]
        maps += change / n
        energy = change ** 2
        k = np.searchsorted(data.sources, family['reference'])
        outside[k] += energy[:, ~masks[family['reference']]].sum(1) / energy.sum(1) / (data.source == family['reference']).sum()
    boot = bootstrap_index(len(outside))
    rows = []
    for j, condition in enumerate(fx.CONDITIONS):
        if condition == 'hand_only':
            continue   # different geometry: the hand mask does not apply
        lo, hi = interval(outside[boot, j].mean(axis=1))
        rows.append(dict(condition=condition, share_outside_hand=float(outside[:, j].mean()), ci_low=lo, ci_high=hi))
    for a, b in (('hand_stub', 'hand_gap_stub'), ('window_arm', 'window_hand'), ('upper_limb', 'scrambled')):
        d = outside[:, column(a)] - outside[:, column(b)]
        lo, hi = interval(d[boot].mean(axis=1))
        rows.append(dict(condition=f'{a} - {b}', share_outside_hand=float(d.mean()), ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows), maps.reshape(c, 16, 16)


# ---------- run ----------

def run(directory=FEATURES, output=OUTPUT, figures=FIGURES, jobs=-1, decode=True):
    data = Data(directory)
    output.mkdir(parents=True, exist_ok=True)
    primary = tuple(r for r in PRIMARY if REPRESENTATIONS[r][0] in data.x)   # all of them after a GPU run
    signal, family_signal = signal_tables(data)
    tables = dict(movement_signal=signal, family_signal=family_signal, gate1=gate1(signal),
                  variance=pd.concat([variance_shares(family_signal, r) for r in primary]),
                  alignment=alignment_tables(data, primary))
    tables['rsa'], matrices, rsa_conditions = rsa_tables(data, primary)
    patches, maps = patch_tables(data, directory)
    if patches is not None:
        tables['patch_outside_hand'] = patches
    differences = pd.read_csv(directory / 'differences.csv')
    tables['differences'] = differences.groupby('condition', sort=False).mean(numeric_only=True).reset_index()
    if decode:
        movement, attachment = decoding_tables(data, primary, jobs=jobs)
        tables['decoding_folds'], tables['attachment_folds'] = movement, attachment
        tables['decoding'] = summarise_folds(movement, ('representation', 'train', 'test'))
        tables['attachment'] = summarise_folds(attachment, ('representation', 'train', 'test'))
    for name, frame in tables.items():
        frame.to_csv(output / f'{name}.csv', index=False)
    manifest = fx.s.read(directory / 'manifest.json')
    summary = dict(code_sha256=fx.s.sha(__file__), features_manifest_sha256=fx.s.sha(directory / 'manifest.json'),
                   features_files=manifest['files'], families=len(data.families), sources=len(data.sources),
                   bootstrap=dict(samples=SAMPLES, seed=SEED, unit='source', interval='percentile'),
                   gate1=tables['gate1'].to_dict('records'),
                   files={f'{name}.csv': fx.s.sha(output / f'{name}.csv') for name in tables})
    fx.s.write(output / 'summary.json', summary)
    import baseline_figures
    baseline_figures.all_figures(tables, matrices, rsa_conditions, maps, data, figures, primary)
    return tables, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features', type=Path, default=FEATURES)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--figures', type=Path, default=FIGURES)
    parser.add_argument('--jobs', type=int, default=-1)
    parser.add_argument('--skip-decoding', action='store_true')
    args = parser.parse_args()
    tables, summary = run(args.features, args.output, args.figures, args.jobs, not args.skip_decoding)
    pd.set_option('display.width', 220)
    print(tables['gate1'].round(3).to_string(index=False))
