"""Full rehearsal of the evaluation on training-source data (final-phase plan, task A3).

Every trained network runs through the evaluation's own code (`run_evaluation.simulate_network`) on its
inner source instead of its held-out source, with only the displays already simulated on training sources:
full body and silhouette detached hand (primary session) and the six controls. No upper limb, no rectangle
detached hand, no held-out source; `simulate_network` refuses both itself. The rehearsal measures what the
evaluation will cost (seconds, GPU memory, record size per network), lists every network that fails to run,
and scores what these displays allow:

- the gate's effect on the detached hand, as the paired difference of (delta_FB - delta_DH) between H2 and
  the ungated network (the necessity contrast without its upper-limb term), and the same at c+;
- the S3(a) control pairs (stub - gap stub, arm window - hand window) for H2, ungated and area, and H2 - area;
- the share of instances with an over-driven cell per display.

The block orders and z-scoring use two displays here instead of three, so these numbers preview the
pipeline, not the evaluation's values. Nothing here enters the evaluation or changes a setting.

    python scripts/rehearse.py simulate --claim [--shard 0 --shards 2] [--configurations ...]
    python scripts/rehearse.py score [--samples 2000]
    python scripts/rehearse.py summary
"""
import argparse
import glob
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
import criteria as cr  # noqa: E402
import run_evaluation as rv  # noqa: E402
import trials  # noqa: E402

OUTPUT = ROOT / 'output/model/rehearsal'
PRIMARY = ('full_body', 'detached_hand_silhouette')
SESSIONS = dict(primary=(PRIMARY, 0), controls=(trials.CONTROL_DISPLAYS, 2))
GATED_MODELS = ('h2', 'h2_cplus', 'area', 'h2_learned', 'h4')


def target(path, directory=OUTPUT):
    return Path(directory) / Path(path).parent.name / (Path(path).stem + '.json')


def rehearse_network(path, device='cpu', fit_directory=None):
    """One network on its inner source; the record of simulate_network plus cost measures."""
    report = json.loads(Path(path).with_suffix('.json').read_text())
    options = {} if fit_directory is None else dict(fit_directory=fit_directory)
    if device == 'cuda':
        import torch
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    result = rv.simulate_network(path, device, source=report['inner_source'], sessions=SESSIONS, **options)
    result['seconds'] = time.time() - start
    if device == 'cuda':
        import torch
        result['gpu_peak_mb'] = torch.cuda.max_memory_allocated() / 2 ** 20
    result['rehearsal'] = True
    return result


def simulate(configurations=None, shard=0, shards=1, claim=False, directory=OUTPUT):
    import torch
    import train as tr
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configurations = configurations or sorted(Path(p).name for p in glob.glob(str(rv.CAMPAIGN / '*')))
    paths = tr.finished_networks(sorted(p for c in configurations for p in glob.glob(str(rv.CAMPAIGN / c / 'health_*.pt'))))
    done = failed = 0
    for path in paths[shard::shards]:
        out = target(path, directory)
        name = f'{Path(path).parent.name}/{Path(path).stem}'
        if out.exists() or (claim and not tr.claim('rehearsal', name.replace('/', '__'))):
            continue
        try:
            result = rehearse_network(path, device)
        except Exception as error:                          # recorded, not hidden: task A3 lists failures
            failed += 1
            failure = out.with_suffix('.failed.txt')
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(f'{type(error).__name__}: {error}\n\n{traceback.format_exc()}')
            print(f'FAILED {name}: {type(error).__name__}: {error}', flush=True)
            continue
        result['provenance'] = tr.provenance()
        text = json.dumps(result)
        result_size = len(text.encode())
        rv.tr_atomic(out, text)
        done += 1
        print(f'{name} {result["seconds"]:.1f} s {result.get("gpu_peak_mb", 0):.0f} MB {result_size / 1e6:.2f} MB', flush=True)
    print(f'rehearsal shard {shard}/{shards}: {done} written, {failed} failed', flush=True)


# ---------- scoring ----------

def cells(record, session, name, analysis, trim=True):
    displays = SESSIONS[session][0]
    key = record['labels'].get(f'{name}|{analysis}')
    if key is None:
        return np.full((len(displays), 2, 2), np.nan)
    table = pd.DataFrame(record['tables'][session][key])
    return cr.treated_cells(table, displays, record['ranges'], None if analysis == 'raw' else analysis, trim)[0]


def over_driven(records, session, name):
    displays, flags = SESSIONS[session][0], []
    for r in records:
        key = r['labels'].get(f'{name}|remove')
        if key is not None:
            flags.append(cr.treated_cells(pd.DataFrame(r['tables'][session][key]), displays, r['ranges'])[1].any(axis=(1, 2)))
    return dict(zip(displays, np.mean(flags, 0).round(4).tolist())) if flags else {}


def detached_contrast(c):
    """delta_FB - delta_DH per instance, cells [instances, 2 (full body, silhouette), 2, 2]."""
    return cr.deltas(c, 0) - cr.deltas(c, 1)


def score_configuration(records, samples=2000):
    records = sorted(records, key=lambda r: (r['source'], r['initialisation']))
    index_sets = cr.draws([r['source'] for r in records], samples)
    names = sorted({k.split('|')[0] for r in records for k in r['labels']})
    out = dict(instances=len(records), models=names, analyses={},
               over_driven={n: {s: over_driven(records, s, n) for s in SESSIONS} for n in names if n in ('h2', 'ungated', 'h0', 'h1')})
    for analysis in rv.ANALYSES:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            primary = {n: np.array([cells(r, 'primary', n, analysis) for r in records]) for n in names}
            entry = dict(effects={n: {'delta_FB': float(np.nanmean(cr.deltas(c, 0))), 'delta_DH': float(np.nanmean(cr.deltas(c, 1))),
                                      'instances_DH': int(np.isfinite(cr.deltas(c, 1)).sum())} for n, c in primary.items()})
            if 'ungated' in primary:
                entry['gate_on_detached_hand'] = {n: cr.paired_difference(detached_contrast(primary[n]), detached_contrast(primary['ungated']), index_sets)
                                                  for n in GATED_MODELS if n in primary}
                control = {n: np.array([cells(r, 'controls', n, analysis) for r in records]) for n in ('h2', 'ungated', 'area') if n in names}
                pairs = {n: cr.pair_effects(c, SESSIONS['controls'][0]) for n, c in control.items()}
                entry['control_pairs'] = {n: {p: float(np.nanmean(v)) for p, v in e.items()} for n, e in pairs.items()}
                if 'area' in pairs:
                    entry['control_pairs_h2_minus_area'] = {p: cr.paired_difference(pairs['h2'][p], pairs['area'][p], index_sets) for p in pairs['h2']}
        out['analyses'][analysis] = entry
    return out


def load(directory=OUTPUT):
    out = {}
    for folder in sorted(Path(directory).glob('*/')):
        records = [json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))]
        if records:
            out[folder.name] = records
    return out


def costs(data, directory=OUTPUT):
    seconds = np.array([r['seconds'] for rs in data.values() for r in rs])
    memory = np.array([r.get('gpu_peak_mb', np.nan) for rs in data.values() for r in rs])
    sizes = np.array([p.stat().st_size for p in Path(directory).glob('*/*.json')])
    failures = sorted(str(p.relative_to(directory)) for p in Path(directory).glob('*/*.failed.txt'))
    pct = lambda a, q: float(np.nanpercentile(a, q)) if len(a) else None
    return dict(networks=int(len(seconds)), failures=failures,
                seconds=dict(total_hours=float(seconds.sum() / 3600), median=pct(seconds, 50), p95=pct(seconds, 95), maximum=pct(seconds, 100)),
                gpu_peak_mb=dict(median=pct(memory, 50), maximum=pct(memory, 100)),
                record_mb=dict(median=pct(sizes, 50) / 1e6 if len(sizes) else None, total=float(sizes.sum() / 1e6)),
                per_configuration={c: len(rs) for c, rs in data.items()})


def score(directory=OUTPUT, samples=2000):
    data = load(directory)
    result = dict(note='Rehearsal on inner (training) sources: full body, silhouette and controls only. '
                       'Previews the pipeline, not the evaluation (two-display sessions and z-scoring).',
                  costs=costs(data, directory),
                  configurations={c: score_configuration(rs, samples) for c, rs in data.items()})
    rv.tr_atomic(Path(directory) / 'rehearsal.json', json.dumps(result, indent=1) + '\n')
    return result


def summary(directory=OUTPUT):
    result = json.loads((Path(directory) / 'rehearsal.json').read_text())
    print(json.dumps(result['costs'], indent=1))
    for c, r in result['configurations'].items():
        line = [f'{c:28s} n={r["instances"]:4d}']
        for analysis in ('raw', 'remove'):
            g = r['analyses'][analysis].get('gate_on_detached_hand', {}).get('h2')
            if g:
                line.append(f'{analysis}: gate {g["estimate"]:+.3f} [{g["ci"][0]:+.3f}, {g["ci"][1]:+.3f}]')
        od = r['over_driven'].get('h2') or next(iter(r['over_driven'].values()), {})
        if od:
            line.append('over-driven ' + ' '.join(f'{d}={v:.2f}' for d, v in od['primary'].items()))
        print('  '.join(line))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=('simulate', 'score', 'summary'))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--configurations', nargs='+')
    parser.add_argument('--claim', action='store_true')
    parser.add_argument('--samples', type=int, default=2000)
    args = parser.parse_args()
    if args.step == 'simulate':
        simulate(args.configurations, args.shard, args.shards, args.claim)
    elif args.step == 'score':
        r = score(samples=args.samples)
        print(json.dumps(r['costs'], indent=1))
    else:
        summary()


if __name__ == '__main__':
    main()
