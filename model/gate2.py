"""Gate 2 checks on training sources (spec §8; docs/model-method.md §6.19): control predictions and a
probe-strength sweep. No held-out source and no paper display other than the full body is used.

Control predictions: every H0, H1 and H2 spiking network runs the six control displays (spec §7.2) on
its inner training source, which it never trained on and which is not its held-out source; H2 with
its fitted c under each treatment of over-driven readouts. Per network the prediction is 8 numbers:
each control's FDI movement effect and the two matched-pair differences of S3(a). Two hypotheses make
different control predictions if, for at least one of the 8, the paired difference between their
networks of the same fold and initialisation has a two-sided hierarchical-bootstrap p < .05 after
Holm correction over the 8, under both treatments. The S3(a) visible-area model is not run on the
controls before the evaluation, so S3(a) itself is not previewed.

Probe-strength sweep: every network, full-body trials of its inner source, probe at 1.0-1.5 x the
resting threshold (common random numbers across strengths). A network's probe response varies
smoothly with strength if the mean FDI readout on static and on movement trials both rise with it
(Spearman >= 0.9, the health check's dose-response threshold).

    python model/gate2.py predictions --shard 0 --shards 3
    python model/gate2.py sweep --shard 0 --shards 3
    python model/gate2.py summary
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
sys.path.insert(0, str(ROOT / 'analysis'))
import bank as bk  # noqa: E402
import criteria as cr  # noqa: E402
import evaluate as ev  # noqa: E402
import human_target as ht  # noqa: E402
import network as nw  # noqa: E402
import train as tr  # noqa: E402
import trials  # noqa: E402

OUTPUT = ROOT / 'output/model/gate2'
FIT = ROOT / 'output/model/gate_fit'
CONTROLS = trials.CONTROL_DISPLAYS
QUANTITIES = tuple(f'delta_{d}' for d in CONTROLS) + tuple(f'{x}-{y}' for x, y in cr.PAIRS)
STRENGTHS = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
KIND_CODES = dict(predictions=4, sweep=5)
HYPOTHESES = ('h0', 'h1', 'h2')
ENCODERS = ('openclip', 'dinov2_cls', 'dinov2_patchmean')
require = trials.require


def prediction_vector(cells):
    """The 8 control quantities of one instance: cells [6 controls, 2, 2] in CONTROLS order."""
    effects = cr.pair_effects(cells, CONTROLS)
    return [float(cr.deltas(cells, i)) for i in range(len(CONTROLS))] + [float(effects[f'{x}-{y}']) for x, y in cr.PAIRS]


def predictions_network(path, device='cpu'):
    net, _, scaler, mean_salience, report = tr.checkpoint(path, device)
    hypothesis = report.get('hypothesis', 'h2')
    bank = bk.Bank(report['representation'])
    source = report['inner_source']
    require(source != report['held_out_source_not_used'], 'The inner source must not be the held-out source')
    if hypothesis == 'h2':
        fit = json.loads((FIT / Path(path).parent.name / (Path(path).stem + '.json')).read_text())
        c = {t: fit['fits']['continuity'][t]['c'] for t in cr.TREATMENTS}
        models = {t: ev.model_for(report, c=c[t]) for t in cr.TREATMENTS if c[t] is not None}
    else:
        model = ev.model_for(report, mean_salience)
        models = {t: model for t in cr.TREATMENTS}
        c = None
    amplitude = ev.PROBE_FACTOR * nw.threshold(net, gate=next(iter(models.values())).gate(trials.BLACK), device=device) if models else None
    session = trials.session(ev.families_of(bank, source), source, source - 1, report['seed'], CONTROLS, KIND_CODES['predictions'])
    order = list(models)
    tables = ev.simulate_grid(net, [models[t] for t in order], bank, scaler, session, [amplitude] * len(order), device) if order else []
    out = dict(run=Path(path).stem, configuration=Path(path).parent.name, hypothesis=hypothesis, fold=report['held_out_source_not_used'],
               initialisation=report['seed'], representation=report['representation'], source=source, c=c, predictions={})
    for t, table in zip(order, tables):
        cells, flags = cr.treated_cells(table, CONTROLS, cr.readout_ranges(report), t)
        out['predictions'][t] = dict(values=prediction_vector(cells), over_driven=flags.any(axis=(1, 2)).tolist())
    return out


def sweep_network(path, device='cpu'):
    net, _, scaler, mean_salience, report = tr.checkpoint(path, device)
    bank = bk.Bank(report['representation'])
    source = report['inner_source']
    model = ev.model_for(report, mean_salience, ungated=True)       # H2: g = 1 on the full body (a = 1)
    threshold = nw.threshold(net, gate=model.gate(trials.BLACK), device=device)
    session = trials.session(ev.families_of(bank, source), source, source - 1, report['seed'], ('full_body',), KIND_CODES['sweep'])
    tables = ev.simulate_grid(net, [model] * len(STRENGTHS), bank, scaler, session, [k * threshold for k in STRENGTHS], device,
                              task_only=True)
    means = {kind: [float(t.loc[t.kind == kind, 'fdi'].mean()) for t in tables] for kind in ('static', 'movement')}
    rho = {kind: float(stats.spearmanr(STRENGTHS, v)[0]) for kind, v in means.items()}
    return dict(run=Path(path).stem, configuration=Path(path).parent.name, hypothesis=report.get('hypothesis', 'h2'),
                representation=report['representation'], spiking=report['spiking'], source=source, strengths=STRENGTHS,
                threshold=threshold, mean_fdi=means, spearman=rho,
                smooth=bool(min(rho.values()) >= tr.HEALTH['dose_spearman']),
                facilitation=[m - s for m, s in zip(means['movement'], means['static'])])


def bootstrap_p(diff, index_sets):
    """Two-sided hierarchical-bootstrap p of a mean paired difference: 2 x the smaller share of
    resampled means on either side of 0 (at least 1 / samples)."""
    boot = cr.resampled_means(np.asarray(diff, float)[:, None], index_sets)[:, 0]
    boot = boot[np.isfinite(boot)]
    share = min((boot <= 0).mean(), (boot >= 0).mean())
    return float(min(1., max(2 * share, 1 / len(boot))))


def distinct(a, b, groups, index_sets):
    """Whether two hypotheses' matched prediction vectors [instances, 8] differ (module doc)."""
    p = [bootstrap_p(a[:, j] - b[:, j], index_sets) for j in range(a.shape[1])]
    adjusted = ht.holm(p)
    return dict(p=p, p_holm=adjusted.tolist(), distinct=bool((adjusted < .05).any()))


def summary():
    """Distinctness of H0, H1 and H2 per encoder; probe sweep per configuration. Writes summary.json."""
    out = dict(predictions={}, sweep={})
    for encoder in ENCODERS:
        loaded = {}
        for h in HYPOTHESES:
            for p in sorted(glob.glob(str(OUTPUT / 'predictions' / f'{h}_{encoder}_snn' / '*.json'))):
                r = json.loads(Path(p).read_text())
                loaded.setdefault(h, {})[(r['fold'], r['initialisation'])] = r
        if set(loaded) != set(HYPOTHESES):
            continue
        result = dict(pairs={}, means={})
        for t in cr.TREATMENTS:
            for h in HYPOTHESES:
                values = np.array([r['predictions'][t]['values'] for r in loaded[h].values() if t in r['predictions']])
                result['means'].setdefault(h, {})[t] = dict(zip(QUANTITIES, np.nanmean(values, 0).round(4).tolist()))
        for x, y in (('h0', 'h1'), ('h0', 'h2'), ('h1', 'h2')):
            per_t = {}
            for t in cr.TREATMENTS:
                keys = sorted(k for k in set(loaded[x]) & set(loaded[y]) if t in loaded[x][k]['predictions'] and t in loaded[y][k]['predictions'])
                a = np.array([loaded[x][k]['predictions'][t]['values'] for k in keys])
                b = np.array([loaded[y][k]['predictions'][t]['values'] for k in keys])
                groups = [loaded[x][k]['source'] for k in keys]
                per_t[t] = dict(distinct(a, b, groups, cr.draws(groups)), instances=len(keys))
            result['pairs'][f'{x}-{y}'] = dict(per_t, verdict=cr.verdict({t: per_t[t]['distinct'] for t in cr.TREATMENTS}))
        result['gate2'] = 'met' if all(v['verdict'] == 'met' for v in result['pairs'].values()) else 'not met'
        out['predictions'][encoder] = result
    for d in sorted(glob.glob(str(OUTPUT / 'sweep' / '*'))):
        rs = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(d + '/*.json'))]
        fac = np.array([r['facilitation'] for r in rs])
        out['sweep'][Path(d).name] = dict(networks=len(rs), smooth=sum(r['smooth'] for r in rs),
                                          not_smooth=[r['run'] for r in rs if not r['smooth']],
                                          facilitation_median=np.median(fac, 0).round(3).tolist(),
                                          facilitation_positive_share=(fac > 0).mean(0).round(3).tolist())
    (OUTPUT / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('step', choices=('predictions', 'sweep', 'summary'))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if args.step == 'summary':
        print(json.dumps(summary(), indent=1)[:6000])
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.step == 'predictions':
        configs = [f'{h}_{e}_snn' for h in HYPOTHESES for e in ENCODERS]
        run = predictions_network
    else:
        configs = sorted(Path(p).name for p in glob.glob(str(ROOT / 'output/model/campaign/*')))
        run = sweep_network
    paths = sorted(p for c in configs for p in glob.glob(str(ROOT / 'output/model/campaign' / c / 'health_*.pt')))
    for path in paths[args.shard::args.shards]:
        target = OUTPUT / args.step / Path(path).parent.name / (Path(path).stem + '.json')
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        result = run(path, device)
        result['seconds'] = time.time() - start
        tr.atomic(target, lambda t: t.write_text(json.dumps(result)))
    print(f'{args.step} shard {args.shard}/{args.shards}: done')


if __name__ == '__main__':
    main()
