"""Fit of the gate parameter c on training sources (docs/model-method.md §6.6, §6.19).

Per trained H2 network (outer fold f, initialisation, encoder, spiking or twin) and gated model (H2's
forearm continuity, the S3(a) visible-area model, H2-learned, H4): full-body and silhouette
detached-hand sessions of the fold's 19 training sources (30 static and 30 movement trials per display,
every variant used as in the evaluation, task probe at 1.2 x threshold) are simulated once for every c
on the grid {0, 0.05, ..., 1}, all values in one batch with common random numbers. Within each source
the readouts are trimmed and z-normalised as in the evaluation; c is the grid value whose ratio of
source-mean Delta(DH, FDI) to Delta(FB, FDI) is closest to -0.10, ties to the smaller c, among the values
with source-mean Delta(FB, FDI) > 0 (otherwise undefined). The fit is made as declared (raw readouts) and
under each treatment of over-driven readouts (criteria.py), whose c the evaluation uses in that
treatment. The held-out source never enters.

    python model/fit_gate.py --shard 0 --shards 3
    python model/fit_gate.py --summary
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
import bank as bk  # noqa: E402
import criteria as cr  # noqa: E402
import evaluate as ev  # noqa: E402
import network as nw  # noqa: E402
import train as tr  # noqa: E402
import trials  # noqa: E402

GRID = tuple(round(k * .05, 2) for k in range(21))
TARGET = -.10                       # human ratio Delta(DH, FDI) / Delta(FB, FDI), rounded as declared (spec §6.2)
DISPLAYS = ('full_body', 'detached_hand_silhouette')
KIND_CODE = 3                       # random streams of fit sessions, apart from evaluation sessions (0-2)
OUTPUT = ROOT / 'output/model/gate_fit'
CONFIGURATIONS = ('h2_openclip_snn', 'h2_dinov2_cls_snn', 'h2_dinov2_patchmean_snn',
                  'h2_openclip_rate', 'h2_dinov2_cls_rate', 'h2_dinov2_patchmean_rate')
require = trials.require


def choose(curve):
    """c from a fit curve [(c, ratio, mean delta FB, ...)]: closest ratio to TARGET among the values
    with mean delta FB > 0, ties to the smaller c; None if no value qualifies."""
    usable = [(abs(row[1] - TARGET), row[0]) for row in curve if np.isfinite(row[1]) and row[2] > 0]
    if not usable:
        return None
    best = min(d for d, _ in usable)
    return min(c for d, c in usable if d <= best + 1e-12)


def curve(per_source):
    """Fit curve from per-source deltas {c: [(delta FB, delta DH, over FB, over DH), ...]}: rows
    (c, ratio of source means, mean delta FB, mean delta DH, sources used, over-driven FB share,
    over-driven DH share), sources used = those with both deltas present."""
    rows = []
    for c in sorted(per_source):
        v = np.asarray(per_source[c], float)
        both = np.isfinite(v[:, 0]) & np.isfinite(v[:, 1])
        fb, dh = (v[both, 0].mean(), v[both, 1].mean()) if both.any() else (np.nan, np.nan)
        ratio = dh / fb if both.any() and fb > 0 else np.nan
        rows.append((c, float(ratio), float(fb), float(dh), int(both.sum()), float(v[:, 2].mean()), float(v[:, 3].mean())))
    return rows


def fit(net, scaler, bank, report, attachments, ranges, sources, device='cpu'):
    """Fit curves and c of every gate in `attachments` ({gate: a(family, display)}) for one network."""
    closed, open_ = nw.threshold(net, gate=0., device=device), nw.threshold(net, gate=1., device=device)
    require(closed == open_, 'The resting threshold depends on the fixation gate')
    amplitude = ev.PROBE_FACTOR * closed
    models = [(gate, c, ev.Model('gated', c=c, attachment=a)) for gate, a in attachments.items() for c in GRID]
    collected = {(gate, t): {c: [] for c in GRID} for gate in attachments for t in (None,) + cr.TREATMENTS}
    for source in sources:
        session = trials.session(ev.families_of(bank, source), source, source - 1, report['seed'], DISPLAYS, KIND_CODE)
        tables = ev.simulate_grid(net, [m for _, _, m in models], bank, scaler, session, [amplitude] * len(models), device,
                                  task_only=True)
        for (gate, c, _), table in zip(models, tables):
            for t in (None,) + cr.TREATMENTS:
                cells, flags = cr.treated_cells(table, DISPLAYS, ranges, t)
                collected[(gate, t)][c].append((cr.deltas(cells, 0), cr.deltas(cells, 1), flags[0].any(), flags[1].any()))
    fits = {}
    for gate in attachments:
        fits[gate] = {}
        for t in (None,) + cr.TREATMENTS:
            rows = curve(collected[(gate, t)])
            fits[gate][t or 'raw'] = dict(c=choose(rows), curve=rows)
    return amplitude, fits


def fit_network(path, device='cpu'):
    net, _, scaler, _, report = tr.checkpoint(path, device)
    require(report.get('hypothesis', 'h2') == 'h2', 'c is fitted on H2 networks')
    fold = report['held_out_source_not_used']
    bank = bk.Bank(report['representation'])
    attachments = {gate: ev.attachment(gate, fold, report['representation']) for gate in ev.GATES}
    sources = [s for s in range(1, 21) if s != fold]
    amplitude, fits = fit(net, scaler, bank, report, attachments, cr.readout_ranges(report), sources, device)
    return dict(run=Path(path).stem, configuration=Path(path).parent.name, fold=fold, initialisation=report['seed'],
                representation=report['representation'], spiking=report['spiking'], amplitude=amplitude, sources=sources,
                target=TARGET, fits=fits)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--configurations', nargs='+', default=list(CONFIGURATIONS))
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if args.summary:
        return summary(args.configurations)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    paths = sorted(p for c in args.configurations for p in glob.glob(str(ROOT / 'output/model/campaign' / c / 'health_*.pt')))
    for path in paths[args.shard::args.shards]:
        target = OUTPUT / Path(path).parent.name / (Path(path).stem + '.json')
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        result = fit_network(path, device)
        result['seconds'] = time.time() - start
        tr.atomic(target, lambda t: t.write_text(json.dumps(result)))
    print(f'shard {args.shard}/{args.shards}: done')


def summary(configurations):
    """Distribution of fitted c per configuration, gate and treatment; agreement of the treatments."""
    for configuration in configurations:
        results = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(OUTPUT / configuration / '*.json')))]
        if not results:
            continue
        print(configuration, len(results), 'networks')
        for gate in ev.GATES:
            line = []
            for t in ('raw',) + cr.TREATMENTS:
                cs = [r['fits'][gate][t]['c'] for r in results]
                defined = [c for c in cs if c is not None]
                line.append(f"{t}: median c {np.median(defined) if defined else float('nan'):.2f} "
                            f"[{min(defined, default=float('nan')):.2f}, {max(defined, default=float('nan')):.2f}], undefined {len(cs) - len(defined)}")
            same = sum(r['fits'][gate]['remove']['c'] == r['fits'][gate]['maximum']['c'] for r in results)
            print(f'  {gate}: ' + '; '.join(line) + f'; remove = maximum in {same}/{len(results)}')


if __name__ == '__main__':
    main()
