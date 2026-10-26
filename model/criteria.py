"""Model criteria (spec §5.2) with the readout range rule (spec §5.1; docs/model-method.md §6.17, §6.19).

Per instance, each population's readouts on a display and trial type are over-driven when that
population's mean pre-probe rate there exceeds its own readout range (the rate at its excitability
curve's peak). Two treatments: 'remove' drops those readouts before trimming and normalisation (as the
paper removed trials with background EMG before normalising), 'maximum' sets them to the population's
readout maximum (the response at the curve's peak). Contrasts are computed per instance, so a removed
cell leaves exactly the contrasts that need it missing (complete cases per contrast); a ratio uses the
instances complete for all its terms. Estimates are means over instances; intervals come from one set
of hierarchical-bootstrap draws (source groups, then initialisations) shared by every quantity of an
analysis, and paired comparisons are differenced per matched instance before resampling.

A criterion is met if it is met under both treatments, not met if it fails under both, and otherwise
undetermined, which never counts in the hypothesis's favour.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
sys.path.insert(0, str(ROOT / 'analysis'))
import human_target as ht  # noqa: E402
import trials  # noqa: E402

TREATMENTS = ('remove', 'maximum')
POPULATIONS = ('fdi', 'adm')
KINDS = ('static', 'movement')
P3_SHARE = .8
BOOTSTRAP = dict(samples=10000, seed=20261022)
PAIRS = (('hand_stub', 'hand_gap_stub'), ('window_arm', 'window_hand'))      # S3(a) matched control pairs
require = trials.require


def human_intervals(path=ROOT / 'output/human_target/ratios.csv'):
    """S1 and S2 intervals: the human 95% intervals of the ratios on the primary scale (spec §4.5)."""
    r = pd.read_csv(path).set_index(['scale', 'ratio'])
    get = lambda name: tuple(r.loc[('participant_z', name), ['ci_low', 'ci_high']].astype(float))
    return dict(S1=get('ratio_DH_to_connected_FDI'), S2=get('ratio_UL_to_FB_FDI'))


def readout_ranges(report):
    """{population: (readout range in Hz, readout maximum)} of one network (health report)."""
    return dict(fdi=(report['readout_range_hz'], report['readout_peak']),
                adm=(report['adm_readout_range_hz'], report['adm_readout_peak']))


def treated_cells(table, displays, ranges, treatment=None, trim=True):
    """One instance's cells [len(displays), 2 trial types, 2 populations] under a treatment (None: as
    simulated), NaN where removed, and the over-driven flags of the same shape."""
    require(treatment in (None,) + TREATMENTS, f'Unknown treatment: {treatment}')
    task = table[table.kind.isin(KINDS) & table.display.isin(displays)]
    long = pd.concat([pd.DataFrame(dict(display=task.display, kind=task.kind, population=p, value=task[p],
                                        pre=task[f'pre_{p}_hz'])) for p in POPULATIONS], ignore_index=True)
    index = pd.MultiIndex.from_product([displays, KINDS, POPULATIONS])
    limit = long.population.map({p: ranges[p][0] for p in POPULATIONS})
    over_cell = long.assign(limit=limit).groupby(['display', 'kind', 'population']).apply(
        lambda g: g.pre.mean() > g.limit.iloc[0], include_groups=False)
    flagged = long.set_index(['display', 'kind', 'population']).index.map(over_cell).to_numpy(bool)
    if treatment == 'remove':
        long = long[~flagged]
    elif treatment == 'maximum':
        long = long.assign(value=np.where(flagged, long.population.map({p: ranges[p][1] for p in POPULATIONS}), long.value))
    if trim and len(long):
        group = long.groupby(['display', 'kind', 'population']).value
        sd = group.transform(lambda v: v.std(ddof=1))
        long = long[sd.isna() | ((long.value - group.transform('mean')).abs() <= 2 * sd)]
    cells = np.full(len(index), np.nan)
    sd = long.value.std(ddof=1) if len(long) > 1 else np.nan
    if np.isfinite(sd) and sd > 0:
        means = long.assign(z=(long.value - long.value.mean()) / sd).groupby(['display', 'kind', 'population']).z.mean()
        cells = means.reindex(index).to_numpy()
    flags = over_cell.reindex(index).fillna(False).to_numpy(bool)
    return cells.reshape(len(displays), 2, 2), flags.reshape(len(displays), 2, 2)


def deltas(cells, display_index):
    """Movement effect of the FDI population per instance: cells [..., displays, 2, 2]."""
    return cells[..., display_index, 1, 0] - cells[..., display_index, 0, 0]


def draws(groups, samples=BOOTSTRAP['samples'], seed=BOOTSTRAP['seed']):
    """Hierarchical-bootstrap index sets: source groups with replacement, then instances within each
    drawn group with replacement (spec §5.1)."""
    groups = np.asarray(groups)
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(samples):
        picked = rng.integers(0, len(members), len(members))
        out.append(np.concatenate([members[i][rng.integers(0, len(members[i]), len(members[i]))] for i in picked]))
    return out


def resampled_means(values, index_sets):
    """Mean over instances, ignoring missing values, of each column of values [n, q], per index set."""
    values = np.asarray(values, float)
    out = np.empty((len(index_sets), values.shape[1]))
    for s, idx in enumerate(index_sets):
        v = values[idx]
        with np.errstate(invalid='ignore'):
            out[s] = np.nanmean(v, 0) if v.size else np.nan
    return out


def interval(boot, level=.95):
    tail = (1 - level) / 2
    return float(np.nanquantile(boot, tail)), float(np.nanquantile(boot, 1 - tail))


def score(cells, groups, initialisations, index_sets, intervals, level=.95):
    """Criteria P1, P2, P3, S1, S2 of one model under one treatment. cells [instances, 3, 2, 2] in the
    human order (full body, upper limb, detached hand), NaN where removed."""
    cells = np.asarray(cells, float)
    fb, ul, dh = (deltas(cells, i) for i in range(3))
    c_fdi = (fb + ul) / 2 - dh
    complete1 = np.isfinite(fb) & np.isfinite(ul) & np.isfinite(dh)
    complete2 = np.isfinite(fb) & np.isfinite(ul)
    mask = lambda x, m: np.where(m, x, np.nan)
    columns = np.stack([c_fdi, fb, ul, mask(dh, complete1), mask((fb + ul) / 2, complete1), mask(ul, complete2), mask(fb, complete2)], 1)
    with np.errstate(invalid='ignore'):
        est = np.nanmean(columns, 0)
    boot = resampled_means(columns, index_sets)
    ci = {name: interval(boot[:, j], level) for j, name in enumerate(('C_FDI', 'delta_FB', 'delta_UL'))}
    s1 = est[3] / est[4] if est[4] > 0 else np.nan
    s2 = est[5] / est[6] if est[6] > 0 else np.nan
    per_init = {}
    for i in np.unique(initialisations):
        values = c_fdi[np.asarray(initialisations) == i]
        if np.isfinite(values).any():
            per_init[int(i)] = float(np.nanmean(values))
    positive = sum(v > 0 for v in per_init.values())
    result = dict(
        C_FDI=dict(estimate=float(est[0]), ci=ci['C_FDI']), delta_FB=dict(estimate=float(est[1]), ci=ci['delta_FB']),
        delta_UL=dict(estimate=float(est[2]), ci=ci['delta_UL']), S1_ratio=float(s1), S2_ratio=float(s2),
        P3_initialisations=per_init, instances=dict(C_FDI=int(np.isfinite(c_fdi).sum()), S1=int(complete1.sum()),
                                                     S2=int(complete2.sum()), total=len(cells)))
    result['criteria'] = dict(
        P1=bool(est[0] > 0 and ci['C_FDI'][0] > 0),
        P2=bool(ci['delta_FB'][0] > 0 and ci['delta_UL'][0] > 0),
        P3=bool(len(per_init) > 0 and positive >= P3_SHARE * len(per_init)),
        S1=bool(np.isfinite(s1) and intervals['S1'][0] <= s1 <= intervals['S1'][1]),
        S2=bool(np.isfinite(s2) and intervals['S2'][0] <= s2 <= intervals['S2'][1]))
    return result


def paired_difference(a, b, index_sets, level=.95):
    """Mean of a - b over matched instances (both present) with its bootstrap interval; 'exceeds' if
    the interval lies above 0."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    boot = resampled_means(diff[:, None], index_sets)[:, 0]
    with np.errstate(invalid='ignore'):
        est = float(np.nanmean(diff))
    low, high = interval(boot, level)
    return dict(estimate=est, ci=(low, high), instances=int(np.isfinite(diff).sum()), exceeds=bool(low > 0))


def pair_effects(control_cells, displays):
    """S3(a): per instance, the FDI movement-effect difference of each matched control pair."""
    index = {d: i for i, d in enumerate(displays)}
    return {f'{x}-{y}': deltas(control_cells, index[x]) - deltas(control_cells, index[y]) for x, y in PAIRS}


def verdict(per_treatment):
    """Combine one criterion's per-treatment results: met / not met / undetermined."""
    values = [bool(per_treatment[t]) for t in TREATMENTS]
    return 'met' if all(values) else 'not met' if not any(values) else 'undetermined'


def decision(primary, robustness_p1):
    """Spec §5.2 decision rule from combined verdicts: P1, P2, P3, S1, S2 and S3 with the primary
    encoder, and P1 with each robustness encoder. Met if all are met, not met if any is not met,
    otherwise undetermined."""
    parts = [primary[k] for k in ('P1', 'P2', 'P3', 'S1', 'S2', 'S3')] + list(robustness_p1.values())
    return 'met' if all(p == 'met' for p in parts) else 'not met' if 'not met' in parts else 'undetermined'


def align(keys_a, keys_b):
    """Positions of the instance keys (source, initialisation) common to two models, in one order."""
    common = sorted(set(keys_a) & set(keys_b))
    pos_a, pos_b = {k: i for i, k in enumerate(keys_a)}, {k: i for i, k in enumerate(keys_b)}
    return common, [pos_a[k] for k in common], [pos_b[k] for k in common]
