"""Participant-level human MEP target from the authors' deposit (Matamala-Gomez et al., 2025).

Participants, not trials, are the independent units: each participant contributes one
mean per body x movement x muscle cell, and every contrast and bootstrap interval is
computed over those 32 paired values. The paper's own trial-level mixed model is refit
separately, only to check that the deposit matches the published analysis.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/analysis.json'
BODIES = ('full_body', 'upper_limb', 'detached_hand')
SHORT = dict(full_body='FB', upper_limb='UL', detached_hand='DH')
MOVEMENTS = ('static', 'movement')
MUSCLES = ('FDI', 'ADM')
DEPOSIT_BODY = {'Full Body': 'full_body', 'Upper Limb': 'upper_limb', 'Hand Detached': 'detached_hand'}
SCALES = ('participant_z', 'raw_mv', 'log', 'participant_muscle_z', 'participant_z_min50')
ITEMS = ('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7')
QUESTIONNAIRE_BODY = dict(full_body='FullBody', upper_limb='UpperLimb', detached_hand='HandDetached')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path=CONFIG):
    return json.loads(Path(path).read_text())


def verified(path, expected):
    path = Path(path)
    require(path.is_file(), f'Missing {path}. Copy the unchanged OSF files into data/human/.')
    require(sha(path) == expected, f'Changed human data file: {path}')
    return path


def load_meps(path, type_map):
    """Deposited MEPs as one row per trial: id, body, movement, muscle, uv."""
    d = pd.read_csv(path)
    required = {'ID', 'Type', 'Muscle', 'Condition', 'Amplitude (microvolt)', 'Amplitude (millivolt)', 'Filter 1'}
    require(required <= set(d.columns), f'Missing MEP columns: {required - set(d.columns)}')
    require(not d[sorted(required)].isna().any().any(), 'Missing MEP values')
    require(np.allclose(d['Amplitude (millivolt)'] * 1000, d['Amplitude (microvolt)'], atol=1e-6),
            'mV and microvolt columns disagree')
    # Filter 1 separates the muscles exactly; it is not an exclusion flag.
    require((d['Filter 1'] == d.Muscle.map({'FDI': 0, 'ADM': 1})).all(), 'Filter 1 is no longer the muscle indicator')
    out = pd.DataFrame(dict(id=d.ID, body=d.Condition.map(DEPOSIT_BODY), movement=d.Type.astype(str).map(type_map),
                            muscle=d.Muscle, uv=d['Amplitude (microvolt)'].astype(float)))
    require(not out.isna().any().any() and set(out.muscle) == set(MUSCLES), 'Unknown body, movement or muscle label')
    require((out.uv > 0).all() and np.isfinite(out.uv).all(), 'Non-positive or nonfinite amplitude')
    return out


def audit(meps, trials_per_cell):
    per_participant = meps.groupby('id').size()
    designed = meps.id.nunique() * len(BODIES) * len(MOVEMENTS) * trials_per_cell * len(MUSCLES)
    return dict(records=len(meps), participants=int(meps.id.nunique()),
                designed_records=designed, missing_records=designed - len(meps),
                missing_percent=100 * (designed - len(meps)) / designed,
                records_per_participant_min=int(per_participant.min()),
                records_per_participant_max=int(per_participant.max()),
                below_50_microvolt=int((meps.uv < 50).sum()),
                min_microvolt=float(meps.uv.min()),
                cell_counts={f"{b}/{m}/{u}": int(n) for (b, m, u), n in
                             meps.groupby(['body', 'movement', 'muscle']).size().items()})


def standardise(values, keys):
    group = values.groupby(keys)
    sd = group.transform(lambda x: x.std(ddof=1))
    require((sd > 0).all(), 'Undefined participant standard deviation')
    return (values - group.transform('mean')) / sd


def scaled(meps, scale):
    """Response column on one declared scale; exclusions always precede standardisation."""
    d = meps.copy()
    if scale == 'participant_z':
        d['response'] = standardise(d.uv, d.id)
    elif scale == 'raw_mv':
        d['response'] = d.uv / 1000
    elif scale == 'log':
        d['response'] = np.log(d.uv)
    elif scale == 'participant_muscle_z':
        d['response'] = standardise(d.uv, [d.id, d.muscle])
    elif scale == 'participant_z_min50':
        d = d[d.uv >= 50].copy()
        d['response'] = standardise(d.uv, d.id)
    else:
        raise ValueError(f'Unknown scale: {scale}')
    return d


def participant_cells(d):
    """Array [participant, body, movement, muscle] of participant cell means."""
    means = d.groupby(['id', 'body', 'movement', 'muscle']).response.mean()
    index = pd.MultiIndex.from_product([sorted(d.id.unique()), BODIES, MOVEMENTS, MUSCLES])
    means = means.reindex(index)
    require(not means.isna().any(), 'A participant lacks a cell; pairing would be broken')
    return means.index.levels[0].tolist(), means.to_numpy().reshape(-1, 3, 2, 2)


def delta(x, body, muscle):
    b, u = BODIES.index(body), MUSCLES.index(muscle)
    return x[..., b, 1, u] - x[..., b, 0, u]


def body_mean(x, body, muscle, movement=None):
    b, u = BODIES.index(body), MUSCLES.index(muscle)
    return x[..., b, :, u].mean(-1) if movement is None else x[..., b, MOVEMENTS.index(movement), u]


def contrasts():
    """(name, family, function of cell array). Primary family has exactly one test."""
    out = [('C_FDI', 'primary',
            lambda x: (delta(x, 'full_body', 'FDI') + delta(x, 'upper_limb', 'FDI')) / 2 - delta(x, 'detached_hand', 'FDI'))]
    for muscle in MUSCLES:
        for body in BODIES:
            out.append((f'delta_{SHORT[body]}_{muscle}', 'secondary', lambda x, b=body, u=muscle: delta(x, b, u)))
    for body in ('full_body', 'upper_limb'):
        out.append((f'threeway_{SHORT[body]}_vs_DH', 'secondary', lambda x, b=body:
                    (delta(x, b, 'FDI') - delta(x, b, 'ADM')) -
                    (delta(x, 'detached_hand', 'FDI') - delta(x, 'detached_hand', 'ADM'))))
    out.append(('delta_FDI_FB_minus_UL', 'secondary',
                lambda x: delta(x, 'full_body', 'FDI') - delta(x, 'upper_limb', 'FDI')))
    out.append(('C_ADM', 'exploratory',
                lambda x: delta(x, 'upper_limb', 'ADM') - (delta(x, 'full_body', 'ADM') + delta(x, 'detached_hand', 'ADM')) / 2))
    for movement in (None, 'static', 'movement'):
        label = movement or 'average'
        for a, b in (('full_body', 'detached_hand'), ('upper_limb', 'detached_hand'), ('full_body', 'upper_limb')):
            out.append((f'FDI_{label}_{SHORT[a]}_minus_{SHORT[b]}', 'exploratory', lambda x, a=a, b=b, m=movement:
                        body_mean(x, a, 'FDI', m) - body_mean(x, b, 'FDI', m)))
    return out


def ratios():
    """Scale-free summaries for later comparison with model units (ratios of group means)."""
    connected = lambda x: (delta(x, 'full_body', 'FDI') + delta(x, 'upper_limb', 'FDI')) / 2
    return [('ratio_DH_to_connected_FDI', lambda x: delta(x, 'detached_hand', 'FDI') / connected(x)),
            ('ratio_UL_to_FB_FDI', lambda x: delta(x, 'upper_limb', 'FDI') / delta(x, 'full_body', 'FDI')),
            ('ratio_ADM_to_FDI_FB', lambda x: delta(x, 'full_body', 'ADM') / delta(x, 'full_body', 'FDI')),
            ('ratio_ADM_to_FDI_UL', lambda x: delta(x, 'upper_limb', 'ADM') / delta(x, 'upper_limb', 'FDI'))]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1)
    out = np.empty_like(p)
    out[order] = adjusted
    return out


def bootstrap_means(x, samples, seed):
    """Participant resampling, paired across all 12 cells."""
    indices = np.random.default_rng(seed).integers(0, len(x), (samples, len(x)))
    return x[indices].mean(axis=1)


def summarise(x, boot, interval=.95):
    tail = (1 - interval) / 2
    cells = []
    for bi, body in enumerate(BODIES):
        for mi, movement in enumerate(MOVEMENTS):
            for ui, muscle in enumerate(MUSCLES):
                values = boot[:, bi, mi, ui]
                cells.append(dict(body=body, movement=movement, muscle=muscle, mean=x[:, bi, mi, ui].mean(),
                                  ci_low=np.quantile(values, tail), ci_high=np.quantile(values, 1 - tail)))
    rows = []
    for name, family, function in contrasts():
        v, b = function(x), function(boot)
        sd = v.std(ddof=1)
        t = stats.ttest_1samp(v, 0)
        rows.append(dict(contrast=name, family=family, estimate=v.mean(), ci_low=np.quantile(b, tail),
                         ci_high=np.quantile(b, 1 - tail), sd=sd, dz=v.mean() / sd, t=t.statistic, df=len(v) - 1,
                         p=t.pvalue, p_wilcoxon=stats.wilcoxon(v).pvalue, n_positive=int((v > 0).sum()), n=len(v)))
    table = pd.DataFrame(rows)
    table['p_holm'] = table.p
    for family in ('secondary', 'exploratory'):
        rows = table.family == family
        table.loc[rows, 'p_holm'] = holm(table.loc[rows, 'p'])
    mean = x.mean(axis=0)
    ratio_rows = [dict(ratio=name, estimate=function(mean), ci_low=np.quantile(function(boot), tail),
                       ci_high=np.quantile(function(boot), 1 - tail)) for name, function in ratios()]
    return pd.DataFrame(cells), table, pd.DataFrame(ratio_rows)


def paper_model(meps):
    """Refit the published model: raw mV, participant random intercept, ML as in Stata's mixed."""
    import statsmodels.formula.api as smf
    d = meps.assign(y=meps.uv / 1000,
                    body=pd.Categorical(meps.body, BODIES), movement=pd.Categorical(meps.movement, MOVEMENTS),
                    muscle=pd.Categorical(meps.muscle, ('ADM', 'FDI')))
    fit = smf.mixedlm('y ~ body * movement * muscle', d, groups=d.id).fit(reml=False)
    rename = {'body[T.upper_limb]': 'upper_limb', 'body[T.detached_hand]': 'detached_hand',
              'movement[T.movement]': 'movement', 'muscle[T.FDI]': 'fdi'}
    def term(name):
        return ':'.join(rename[part] for part in name.split(':'))
    fixed = pd.DataFrame(dict(term=[term(n) for n in fit.fe_params.index if n != 'Intercept'],
                              coef=fit.fe_params.drop('Intercept').values,
                              z=fit.tvalues[fit.fe_params.index].drop('Intercept').values))
    # The published post-hoc z values are reproduced by separate random-intercept models on
    # each subset (one muscle x body for movement effects, one muscle x movement for body
    # differences), not by contrasts of the three-way model above.
    def wald(subset, factor, plus, minus):
        fit = smf.mixedlm(f'y ~ 0 + {factor}', subset.assign(**{factor: subset[factor].astype(str)}),
                          groups=subset.id).fit(reml=False)
        names = list(fit.fe_params.index)
        weights = np.zeros(len(names))
        weights[names.index(f'{factor}[{plus}]')], weights[names.index(f'{factor}[{minus}]')] = 1, -1
        covariance = fit.cov_params().loc[names, names].to_numpy()
        return float(weights @ fit.fe_params.to_numpy() / np.sqrt(weights @ covariance @ weights))
    posthoc = {}
    for muscle in MUSCLES:
        for body in BODIES:
            subset = d[(d.muscle == muscle) & (d.body == body)]
            posthoc[f'{muscle} {body} movement - static'] = wald(subset, 'movement', 'movement', 'static')
        for movement in MOVEMENTS:
            subset = d[(d.muscle == muscle) & (d.movement == movement)]
            for a, b in (('full_body', 'detached_hand'), ('upper_limb', 'detached_hand'), ('upper_limb', 'full_body')):
                posthoc[f'{muscle} {movement} {a} - {b}'] = wald(subset, 'body', a, b)
    return fixed, posthoc, dict(participant_variance=float(fit.cov_re.iloc[0, 0]), residual_variance=float(fit.scale))


def compare_published(fixed, posthoc, cfg):
    rows = []
    for term, published in cfg['published_table2']['coefficients'].items():
        ours = fixed.set_index('term').loc[term]
        rows.append(dict(source='Table 2', effect=term, published_coef=published.get('coef'), refit_coef=ours.coef,
                         published_abs_z=abs(published['z']), refit_abs_z=abs(ours.z)))
    for effect, published in cfg['published_posthoc_abs_z']['contrasts'].items():
        rows.append(dict(source='post hoc', effect=effect, published_abs_z=published,
                         refit_abs_z=abs(posthoc[effect])))
    return pd.DataFrame(rows)


def questionnaire(path):
    q = pd.read_csv(path)
    for body, column in QUESTIONNAIRE_BODY.items():
        index = {'full_body': 'INDEX_DISOWNERSHIP_FullBody', 'upper_limb': 'INDEX_DISOWNERSHIP_Upper_Limb',
                 'detached_hand': 'INDEX_DISOWNERSHIP_HandDetached'}[body]
        require(np.allclose(q[index], q[[f'Q{i}_{column}' for i in (3, 5, 7)]].mean(axis=1)),
                'Disownership index is not the mean of Q3, Q5 and Q7')
    rows = []
    for item in ITEMS + ('disownership_index',):
        values = {b: (q[f'{item}_{c}'] if item != 'disownership_index' else q[[f'Q{i}_{c}' for i in (3, 5, 7)]].mean(axis=1))
                  for b, c in QUESTIONNAIRE_BODY.items()}
        pairs = (('full_body', 'upper_limb'), ('full_body', 'detached_hand'), ('upper_limb', 'detached_hand'))
        p = [stats.wilcoxon(values[a], values[b]).pvalue for a, b in pairs]
        for (a, b), raw, adjusted in zip(pairs, p, holm(p)):
            rows.append(dict(item=item, comparison=f'{SHORT[a]} - {SHORT[b]}', mean_a=values[a].mean(),
                             mean_b=values[b].mean(), median_difference=(values[a] - values[b]).median(),
                             p_wilcoxon=raw, p_holm_within_item=adjusted, n=len(q)))
    return pd.DataFrame(rows)


def run(output, cfg, with_paper_model=True):
    human = cfg['human_data']
    directory = ROOT / human['directory']
    files = {name: verified(directory / name, digest) for name, digest in human['files'].items()}
    meps = load_meps(files['MEPs_amplitude.csv'], human['type_map'])
    target = cfg['human_target']
    output.mkdir(parents=True, exist_ok=True)
    cell_tables, contrast_tables, ratio_tables, participant_tables = [], [], [], []
    for scale in SCALES:
        ids, x = participant_cells(scaled(meps, scale))
        cells, table, ratio = summarise(x, bootstrap_means(x, target['bootstrap_samples'], target['bootstrap_seed']),
                                        target['interval'])
        for frame in (cells, table, ratio):
            frame.insert(0, 'scale', scale)
        cell_tables.append(cells), contrast_tables.append(table), ratio_tables.append(ratio)
        long = pd.DataFrame(x.reshape(len(ids), -1), index=pd.Index(ids, name='participant'),
                            columns=[f'{b}|{m}|{u}' for b in BODIES for m in MOVEMENTS for u in MUSCLES])
        participant_tables.append(long.assign(scale=scale).reset_index())
    tables = dict(cells=pd.concat(cell_tables), contrasts=pd.concat(contrast_tables),
                  ratios=pd.concat(ratio_tables), participant_cells=pd.concat(participant_tables),
                  questionnaire=questionnaire(files['Embodiment_Quesionnaire.csv']))
    summary = dict(config_sha256=sha(CONFIG), code_sha256=sha(__file__),
                   human_files={name: sha(path) for name, path in files.items()},
                   audit=audit(meps, human['design_trials_per_cell']), primary_scale=target['primary_scale'],
                   bootstrap=dict(samples=target['bootstrap_samples'], seed=target['bootstrap_seed'],
                                  unit='participant, paired across all 12 cells', interval='percentile'))
    if with_paper_model:
        fixed, posthoc, variances = paper_model(meps)
        tables['paper_model'] = fixed
        tables['paper_comparison'] = compare_published(fixed, posthoc, cfg)
        summary['paper_model'] = dict(variances, scale='raw mV', estimation='maximum likelihood',
                                      random_effects='participant intercept only')
        comparison = tables['paper_comparison']
        summary['published_statistics_reproduced'] = bool(
            ((comparison.published_abs_z - comparison.refit_abs_z).abs() <= .01).all())
    for name, frame in tables.items():
        frame.to_csv(output / f'{name}.csv', index=False)
        summary[f'{name}_csv_sha256'] = sha(output / f'{name}.csv')
    primary = tables['contrasts'].query("contrast == 'C_FDI'")
    summary['primary_C_FDI_positive_on_every_scale'] = bool((primary.ci_low > 0).all())
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return tables, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/human_target')
    parser.add_argument('--skip-paper-model', action='store_true', help='Omit the statsmodels refit of Table 2')
    args = parser.parse_args()
    tables, summary = run(args.output, read_config(), not args.skip_paper_model)
    pd.set_option('display.width', 200)
    shown = tables['contrasts'].query("family != 'exploratory' and scale == 'participant_z'")
    print(shown[['contrast', 'estimate', 'ci_low', 'ci_high', 'dz', 't', 'p', 'p_holm', 'n_positive']]
          .round(4).to_string(index=False))
    print('\nC_FDI across scales:')
    print(tables['contrasts'].query("contrast == 'C_FDI'")[['scale', 'estimate', 'ci_low', 'ci_high', 't', 'p']]
          .round(4).to_string(index=False))
    if 'paper_comparison' in tables:
        print('\nPublished vs refit (raw mV mixed model):')
        print(tables['paper_comparison'].round(3).to_string(index=False))
    print(f"\nWrote {args.output}; primary C_FDI interval excludes 0 on every scale: "
          f"{summary['primary_C_FDI_positive_on_every_scale']}")
