"""Completeness audit of the training-source outputs before the freeze (final-phase plan, task B2).

Counts, per configuration, the networks that have every output the evaluation needs, and compares them with
the declared numbers (model-method §6.16, §6.20, §6.21, §6.23):

- a finished health check (`campaign/<configuration>/health_*.pt` with its report);
- FDI and ADM readout ranges measured inside the excitability sweep (`criteria.readout_peak_found`,
  `adm_readout_peak_found`, finite `readout_range_hz` and `adm_readout_range_hz`);
- for H2 networks, a fit of c (`gate_fit/<configuration>/<run>.json`) for all four gates under the three
  analyses (raw, remove, maximum), and how many of those fits are undefined;
- the Gate 2 probe-strength sweep for the 2,400 campaign networks and the control predictions for the 1,800
  H0/H1/H2 spiking networks (`gate2/sweep`, `gate2/predictions`);
- optionally, the rehearsal record (`rehearsal/<configuration>/<run>.json`, task A3).

    python scripts/audit_outputs.py [--root output/model] [--json audit.json]

Exit status 0 only if every count equals its declared number. Reads files only; changes nothing.
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = ('openclip', 'dinov2_cls', 'dinov2_patchmean')
VARIATIONS = ('lr3e-3', 'lr3e-4', 'nv128', 'nv512', 'tm10', 'tm40', 'ts10', 'ts2', 'u6000')
DECLARED = {**{f'{h}_{e}_snn': 200 for h in ('h0', 'h1', 'h2') for e in ENCODERS},
            **{f'h2_{e}_rate': 200 for e in ENCODERS},
            'h2_openclip_snn_rec': 200, **{f'h2_openclip_snn_{v}': 60 for v in VARIATIONS},
            'h2_openclip_snn_allctx': 60, 'h2_openclip_snn_dt0.5': 20}
CAMPAIGN = tuple(c for c in DECLARED if c.endswith(('_snn', '_rate')))       # the 2,400 of §6.16
PREDICTED = tuple(c for c in CAMPAIGN if c.endswith('_snn'))                  # the 1,800 spiking networks
GATES = ('continuity', 'area', 'forearm', 'familiarity')
ANALYSES = ('raw', 'remove', 'maximum')


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def networks(root, configuration):
    """Finished networks of a configuration: health_*.pt with a report (as train.finished_networks)."""
    paths = sorted(glob.glob(str(Path(root) / 'campaign' / configuration / 'health_*.pt')))
    return [Path(p) for p in paths if not p.endswith(('.trained.pt', '.partial.pt')) and Path(p).with_suffix('.json').is_file()]


def audit_configuration(root, configuration, rehearsal=False):
    root = Path(root)
    nets = networks(root, configuration)
    out = dict(declared=DECLARED.get(configuration), networks=len(nets), ranges=0, healthy=0, missing_ranges=[])
    h2 = configuration.startswith('h2_')
    if h2:
        out.update(fit=0, fit_undefined=0, missing_fit=[])
    if configuration in CAMPAIGN:
        out.update(sweep=0, missing_sweep=[])
    if configuration in PREDICTED:
        out.update(predictions=0, missing_predictions=[])
    if rehearsal:
        out.update(rehearsal=0, rehearsal_failed=0)
    for path in nets:
        run = path.stem
        report = json.loads(path.with_suffix('.json').read_text())
        if (report.get('criteria', {}).get('readout_peak_found') and report.get('adm_readout_peak_found')
                and finite(report.get('readout_range_hz')) and finite(report.get('adm_readout_range_hz'))):
            out['ranges'] += 1
        else:
            out['missing_ranges'].append(run)
        out['healthy'] += bool(report.get('criteria')) and all(report['criteria'].values())
        if h2:
            fit = root / 'gate_fit' / configuration / f'{run}.json'
            fits = json.loads(fit.read_text()).get('fits', {}) if fit.is_file() else {}
            if all(g in fits and all(a in fits[g] and 'c' in fits[g][a] for a in ANALYSES) for g in GATES):
                out['fit'] += 1
                out['fit_undefined'] += sum(fits[g][a]['c'] is None for g in GATES for a in ANALYSES)
            else:
                out['missing_fit'].append(run)
        for step, key in (('sweep', 'sweep'), ('predictions', 'predictions')):
            if key in out:
                if (root / 'gate2' / step / configuration / f'{run}.json').is_file():
                    out[key] += 1
                else:
                    out[f'missing_{key}'].append(run)
        if rehearsal:
            out['rehearsal'] += (root / 'rehearsal' / configuration / f'{run}.json').is_file()
            out['rehearsal_failed'] += (root / 'rehearsal' / configuration / f'{run}.failed.txt').is_file()
    return out


def complete(entry):
    """Every count equals the declared number of networks."""
    n = entry['declared']
    counts = [entry[k] for k in ('networks', 'ranges', 'fit', 'sweep', 'predictions', 'rehearsal') if k in entry]
    return n is not None and all(c == n for c in counts) and entry.get('rehearsal_failed', 0) == 0


def audit(root=ROOT / 'output/model', rehearsal=False):
    configurations = sorted(set(DECLARED) | {Path(p).name for p in glob.glob(str(Path(root) / 'campaign' / '*'))
                                              if Path(p).name in DECLARED or Path(p).name.startswith(('h0_', 'h1_', 'h2_'))})
    result = {c: audit_configuration(root, c, rehearsal) for c in configurations}
    totals = {k: sum(e.get(k, 0) for e in result.values()) for k in ('networks', 'ranges', 'healthy', 'fit', 'fit_undefined', 'sweep', 'predictions', 'rehearsal', 'rehearsal_failed')}
    declared = dict(networks=sum(DECLARED.values()), fit=sum(n for c, n in DECLARED.items() if c.startswith('h2_')),
                    sweep=sum(DECLARED.get(c, 0) for c in CAMPAIGN), predictions=sum(DECLARED.get(c, 0) for c in PREDICTED))
    return dict(configurations=result, totals=totals, declared=declared,
                complete=all(complete(e) for c, e in result.items() if c in DECLARED) and set(result) >= set(DECLARED))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=str(ROOT / 'output/model'))
    parser.add_argument('--rehearsal', action='store_true', help='Also require a rehearsal record per network')
    parser.add_argument('--json')
    args = parser.parse_args()
    result = audit(args.root, args.rehearsal)
    columns = ('declared', 'networks', 'ranges', 'healthy', 'fit', 'fit_undefined', 'sweep', 'predictions', 'rehearsal', 'rehearsal_failed')
    print(f"{'configuration':26s} " + ' '.join(f'{c[:11]:>11s}' for c in columns))
    for name, e in result['configurations'].items():
        flag = '' if complete(e) else '  <- incomplete'
        print(f'{name:26s} ' + ' '.join(f"{'' if e.get(c) is None else e[c]:>11}" for c in columns) + flag)
        for key in ('missing_ranges', 'missing_fit', 'missing_sweep', 'missing_predictions'):
            if e.get(key):
                print(f'    {key}: {", ".join(e[key][:10])}' + (' …' if len(e[key]) > 10 else ''))
    print('totals:', json.dumps(result['totals']))
    print('declared:', json.dumps(result['declared']))
    print('complete:', result['complete'])
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + '\n')
    sys.exit(0 if result['complete'] else 1)


if __name__ == '__main__':
    main()
