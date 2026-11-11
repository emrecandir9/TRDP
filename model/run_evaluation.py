"""The held-out evaluation (spec §5, §8; docs/model-method.md §6.18-6.19a, §6.22).

Two steps. `simulate` runs every trained network on its own held-out source: the primary session (full
body, upper limb, silhouette detached hand), the paired rectangle session and the control session, every
model of the network in one batch with common random numbers. `score` turns the saved trials into the
criteria of spec §5.2 for every model, under both treatments of over-driven readouts and on the readouts
as simulated, with the necessity of the gate (H2 - ungated) and the declared sensitivity analyses.

The lock. `simulate` is the only code that shows a network its held-out source. It refuses to run unless
config/freeze.json exists and the files it lists still have their recorded hashes; `freeze` writes that
record once, on the freeze date (spec §8). Until then nothing here can open the evaluation.

    python model/run_evaluation.py freeze                      # 11 December: record configurations and code
    python model/run_evaluation.py simulate --shard 0 --shards 3 --claim
    python model/run_evaluation.py score
"""
import argparse
import glob
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
import criteria as cr  # noqa: E402
import evaluate as ev  # noqa: E402
import fit_gate as fg  # noqa: E402
import trials  # noqa: E402

OUTPUT = ROOT / 'output/model/evaluation'
CAMPAIGN = ROOT / 'output/model/campaign'
FREEZE = ROOT / 'config/freeze.json'
FROZEN = ('model/*.py', 'analysis/human_target.py', 'config/analysis.json', 'docs/analysis-spec.md',
          'data/context_continuity.json', 'output/human_target/ratios.csv', 'output/features/manifest.json',
          'output/learned_context/gates.csv', 'output/learned_context/gates_crossfit.csv')
# session name: (displays, random-stream code of trials.session); the rectangle session shares the
# primary session's stream, so the two are paired (docs/model-method.md §2)
SESSIONS = dict(primary=(trials.BODIES, 0), rectangle=(trials.SENSITIVITY_BODIES, 0), controls=(trials.CONTROL_DISPLAYS, 2))
ANALYSES = ('raw',) + cr.TREATMENTS                    # readouts as simulated, then the two treatments
GATED = dict(continuity='h2', area='area', forearm='h2_learned', familiarity='h4')
H2_MODELS = ('h2', 'ungated', 'area', 'h2_learned', 'h4', 'h2_cplus')
PRIMARY_ENCODER, ROBUSTNESS_ENCODERS = 'openclip', ('dinov2_cls', 'dinov2_patchmean')
CRITERIA = ('P1', 'P2', 'P3', 'S1', 'S2')
require = trials.require


# ---------- the lock ----------

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_files(root=ROOT, patterns=FROZEN):
    return sorted(str(Path(p).relative_to(root)) for pattern in patterns for p in glob.glob(str(root / pattern)))


def freeze(path=FREEZE, root=ROOT, patterns=FROZEN, date=None):
    """Record the hashes of the code, configuration and inputs the evaluation uses. Written once."""
    require(not Path(path).exists(), f'{path} exists: the configurations are already frozen')
    files = frozen_files(root, patterns)
    missing = [p for p in patterns if not glob.glob(str(root / p))]
    require(not missing, f'Nothing matches {missing}')
    record = dict(date=date or time.strftime('%Y-%m-%d'), files={f: sha(root / f) for f in files},
                  note='The held-out evaluation runs only while every file listed here has this hash.')
    Path(path).write_text(json.dumps(record, indent=2) + '\n')
    return record


def opened(path=FREEZE, root=ROOT):
    """The freeze record, if the evaluation may run: the record exists and no listed file has changed."""
    require(Path(path).is_file(), f'The evaluation is not open: {path} does not exist (written at the freeze, spec §8)')
    record = json.loads(Path(path).read_text())
    changed = [f for f, digest in record['files'].items() if not (root / f).is_file() or sha(root / f) != digest]
    require(not changed, f'Changed since the freeze: {changed}')
    return record


# ---------- simulation ----------

def gated_runs(fits, interval):
    """Runs and labels of an H2 network from its fit of c. runs {key: (gate, c)}: each distinct gate and c
    once. labels {'model|analysis': key}: per analysis the c fitted under it (docs/model-method.md §6.19);
    'h2_cplus' is H2 at the sensitivity value c+ (§6.19a); the ungated network is the same run in every
    analysis. A model whose fit is undefined under an analysis has no label there."""
    runs, labels = {'ungated': None}, {}
    for analysis in ANALYSES:
        labels[f'ungated|{analysis}'] = 'ungated'
        values = {name: fits[gate][analysis]['c'] for gate, name in GATED.items() if gate in fits}
        values['h2_cplus'] = fg.sensitivity_c(fits['continuity'][analysis]['curve'], interval)
        for name, c in values.items():
            if c is None:
                continue
            gate = 'continuity' if name == 'h2_cplus' else next(g for g, n in GATED.items() if n == name)
            key = f'{gate}@{c:.2f}'
            runs[key] = (gate, float(c))
            labels[f'{name}|{analysis}'] = key
    return runs, labels


def run_instance(net, scaler, bank, report, models, labels, amplitude, source, sessions=SESSIONS, device='cpu'):
    """One network's sessions on one source: every model in `models` {key: evaluate.Model} in one batch per
    session, sharing every noise draw. Returns the record that `score` reads."""
    import bank as bk  # noqa: F401  (the bank's families)
    keys = list(models)
    tables = {}
    for name, (displays, kind_code) in sessions.items():
        session = trials.session(ev.families_of(bank, source), source, source - 1, report['seed'], displays, kind_code)
        out = ev.simulate_grid(net, [models[k] for k in keys], bank, scaler, session, [amplitude] * len(keys), device)
        tables[name] = {k: t[['display', 'kind', 'family', 'fdi', 'adm', 'pre_fdi_hz', 'pre_adm_hz']].to_dict('list')
                        for k, t in zip(keys, out)}
    return dict(hypothesis=report.get('hypothesis', 'h2'), representation=report['representation'],
                spiking=report['spiking'], fold=report['held_out_source_not_used'], source=source,
                initialisation=report['seed'], amplitude=amplitude, ranges=cr.readout_ranges(report),
                healthy=bool(all(report['criteria'].values())), labels=labels, tables=tables)


def simulate_network(path, device='cpu', fit_directory=fg.OUTPUT, source=None, sessions=SESSIONS):
    """A campaign network on its held-out source (source=None), which needs the evaluation to be open
    (the freeze record is checked here, not only in main). A rehearsal may name another source, which
    must not be the held-out one, with sessions of displays already simulated on training sources."""
    import bank as bk
    import network as nw
    import train as tr
    net, _, scaler, mean_salience, report = tr.checkpoint(path, device)
    hypothesis, fold = report.get('hypothesis', 'h2'), report['held_out_source_not_used']
    if source is None:
        opened()
        source = fold
    else:
        require(source != fold, f'Source {source} is held out for this network; a rehearsal cannot use it')
        require(not any('upper_limb' in displays for displays, _ in sessions.values()),
                'The upper limb is simulated only in the evaluation')
    bank = bk.Bank(report['representation'])
    if hypothesis == 'h2':
        fit = json.loads((Path(fit_directory) / Path(path).parent.name / (Path(path).stem + '.json')).read_text())
        runs, labels = gated_runs(fit['fits'], cr.human_intervals()['DH_FB'])
        models = {k: ev.Model('ungated') if v is None else
                  ev.Model('gated', c=v[1], attachment=ev.attachment(v[0], fold, report['representation'])) for k, v in runs.items()}
        closed, open_ = nw.threshold(net, gate=0., device=device), nw.threshold(net, gate=1., device=device)
        require(closed == open_, 'The resting threshold depends on the fixation gate')
        amplitude = ev.PROBE_FACTOR * closed
    else:
        models = {hypothesis: ev.model_for(report, mean_salience)}
        labels = {f'{hypothesis}|{a}': hypothesis for a in ANALYSES}
        amplitude = ev.PROBE_FACTOR * nw.threshold(net, device=device)
    require(np.isfinite(amplitude), 'No resting threshold')
    result = run_instance(net, scaler, bank, report, models, labels, amplitude, source, sessions, device)
    return dict(result, run=Path(path).stem, configuration=Path(path).parent.name)


# ---------- scoring ----------

def cells_of(result, session, name, analysis, trim=True):
    """One instance's cells [displays, 2, 2] for a model under an analysis; NaN if the model has no run."""
    displays = SESSIONS[session][0]
    key = result['labels'].get(f'{name}|{analysis}')
    if key is None:
        return np.full((len(displays), 2, 2), np.nan)
    table = pd.DataFrame(result['tables'][session][key])
    cells, _ = cr.treated_cells(table, displays, result['ranges'], None if analysis == 'raw' else analysis, trim)
    return cells


def over_driven(results, session, name):
    """Share of instances with an over-driven FDI or ADM cell, per display (spec §5.1)."""
    displays, flags = SESSIONS[session][0], []
    for r in results:
        key = r['labels'].get(f'{name}|remove')
        if key is not None:
            flags.append(cr.treated_cells(pd.DataFrame(r['tables'][session][key]), displays, r['ranges'])[1].any(axis=(1, 2)))
    return dict(zip(displays, np.mean(flags, 0).round(4).tolist())) if flags else {}


def ordered(results, exclude=(), healthy_only=False):
    kept = [r for r in results if r['source'] not in exclude and (r['healthy'] or not healthy_only)]
    return sorted(kept, key=lambda r: (r['source'], r['initialisation']))


def cell_summary(cells, index_sets, displays, level=.95):
    """Mean of each of the 12 cells over instances with its hierarchical-bootstrap interval, for the report's
    human-vs-model figure (descriptive; no criterion uses it). Nested [display][trial type][population]."""
    cells = np.asarray(cells, float)
    flat = cells.reshape(len(cells), -1)
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(flat, 0) if len(flat) else np.full(flat.shape[1], np.nan)
    bounds = np.array([cr.interval(column, level) for column in cr.resampled_means(flat, index_sets).T])
    shape = cells.shape[1:]
    return dict(order=[list(displays), list(cr.KINDS), list(cr.POPULATIONS)], mean=mean.reshape(shape).tolist(),
                ci_low=bounds[:, 0].reshape(shape).tolist(), ci_high=bounds[:, 1].reshape(shape).tolist())


def score_analysis(h2, h0, h1, analysis, intervals, session='primary', trim=True, samples=cr.BOOTSTRAP['samples']):
    """Every model's criteria under one analysis, with S3 and the necessity of the gate for H2."""
    stack = lambda results, name, s=session: np.array([cells_of(r, s, name, analysis, trim) for r in results])
    key = lambda results: [(r['source'], r['initialisation']) for r in results]
    sets = {}

    def draws(results):
        groups = tuple(r['source'] for r in results)
        if groups not in sets:
            sets[groups] = cr.draws(groups, samples)
        return sets[groups]

    out = dict(models={}, instances=len(h2))
    cells = {}
    for name, results in [(n, h2) for n in H2_MODELS] + [('h0', h0), ('h1', h1)]:
        if not results:
            continue
        cells[name] = stack(results, name)
        out['models'][name] = cr.score(cells[name], [r['source'] for r in results], [r['initialisation'] for r in results],
                                       draws(results), intervals)
        out['models'][name]['cells'] = cell_summary(cells[name], draws(results), SESSIONS[session][0])
    if 'h2' in cells:
        out['necessity'] = cr.necessity(cells['h2'], cells['ungated'], draws(h2))
        out['necessity_cplus'] = cr.necessity(cells['h2_cplus'], cells['ungated'], draws(h2))
    if 'h2' in cells and 'h0' in cells:                      # S3(b): matched by source and initialisation
        common, a, b = cr.align(key(h2), key(h0))
        matched = [h2[i] for i in a]
        difference = cr.paired_difference(cr.c_fdi(cells['h2'][a]), cr.c_fdi(cells['h0'][b]), draws(matched))
        out['S3b'] = dict(difference, h0_meets_P1=out['models']['h0']['criteria']['P1'],
                          met=bool(difference['exceeds'] and not out['models']['h0']['criteria']['P1']))
    if 'h2' in cells and session == 'primary' and all('controls' in r['tables'] for r in h2):   # S3(a)
        controls = SESSIONS['controls'][0]
        effects = {n: cr.pair_effects(stack(h2, n, 'controls'), controls) for n in ('h2', 'area')}
        pairs = {pair: cr.paired_difference(effects['h2'][pair], effects['area'][pair], draws(h2)) for pair in effects['h2']}
        out['S3a'] = dict(pairs=pairs, met=bool(all(p['exceeds'] for p in pairs.values())))
        out['controls'] = {n: {d: float(np.nanmean(cr.deltas(stack(h2, n, 'controls'), i))) for i, d in enumerate(controls)}
                           for n in ('h2', 'ungated', 'area')}
    return out


def score_encoder(h2, h0, h1, intervals, exclude=(), healthy_only=False, session='primary', trim=True,
                  samples=cr.BOOTSTRAP['samples']):
    """All three analyses and the combined verdicts (met / not met / undetermined across the two
    treatments; the readouts as simulated are reported beside them, spec §5.2)."""
    h2, h0, h1 = (ordered(r, exclude, healthy_only) for r in (h2, h0, h1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)       # means of instances that all lack a cell
        analyses = {a: score_analysis(h2, h0, h1, a, intervals, session, trim, samples) for a in ANALYSES}
    verdicts = {}
    for name in analyses['raw']['models']:
        verdicts[name] = {k: cr.verdict({t: analyses[t]['models'][name]['criteria'][k] for t in cr.TREATMENTS}) for k in CRITERIA}
    if 'S3b' in analyses['raw']:
        for part in ('S3a', 'S3b'):
            if part in analyses['raw']:
                verdicts['h2'][part] = cr.verdict({t: analyses[t][part]['met'] for t in cr.TREATMENTS})
        if 'S3a' in analyses['raw']:
            verdicts['h2']['S3'] = cr.verdict({t: analyses[t]['S3a']['met'] and analyses[t]['S3b']['met'] for t in cr.TREATMENTS})
    necessary = {}
    if 'ungated' in verdicts:                                # the gate is necessary for a criterion only if the
        for k in CRITERIA:                                   # ungated network does not meet it while H2 does
            necessary[k] = bool(verdicts['h2'][k] == 'met' and verdicts['ungated'][k] == 'not met')
        necessary['adds_to_C_FDI'] = cr.verdict({t: analyses[t]['necessity']['exceeds'] for t in cr.TREATMENTS})
    return dict(analyses=analyses, verdicts=verdicts, gate_necessary=necessary,
                over_driven={n: over_driven(h2, session, n) for n in ('h2', 'ungated')} if h2 else {},
                instances=dict(h2=len(h2), h0=len(h0), h1=len(h1)))


def decide(by_encoder, primary=PRIMARY_ENCODER, robustness=ROBUSTNESS_ENCODERS):
    """Spec §5.2 decision rule for H2 from the encoders' combined verdicts."""
    return cr.decision(by_encoder[primary]['verdicts']['h2'], {e: by_encoder[e]['verdicts']['h2']['P1'] for e in robustness})


def load(configuration, directory=OUTPUT):
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(Path(directory) / configuration / '*.json')))]


def report(directory=OUTPUT, samples=cr.BOOTSTRAP['samples'], variant='snn'):
    """The evaluation's result: the main analysis per encoder with the decision rule, then the declared
    repetitions (without reference 17; healthy networks only; untrimmed; rectangle detached hand)."""
    intervals = cr.human_intervals()
    data = {e: tuple(load(f'{h}_{e}_{variant}', directory) for h in ('h2', 'h0', 'h1'))
            for e in (PRIMARY_ENCODER,) + ROBUSTNESS_ENCODERS}
    run = lambda **options: {e: score_encoder(*d, intervals, samples=samples, **options) for e, d in data.items() if d[0]}
    out = dict(main=run())
    if set(out['main']) == set(data) and 'S3' in out['main'][PRIMARY_ENCODER]['verdicts']['h2']:
        out['decision'] = decide(out['main'])
    out.update(without_reference_17=run(exclude=(17,)), healthy_only=run(healthy_only=True), untrimmed=run(trim=False),
               rectangle=run(session='rectangle'))
    # Comparison configurations of the primary encoder, each scored as H2 and as the ungated network: the
    # non-spiking twin, the recurrent variant, the robustness variations and the all-context ablation
    # (docs/model-method.md §6.14, §6.20, §6.21, §6.23). They have no H0 or H1 networks, so no S3.
    prefix = f'h2_{PRIMARY_ENCODER}_'
    names = sorted(Path(p).name[len(prefix):] for p in glob.glob(str(Path(directory) / (prefix + '*'))))
    out['configurations'] = {name: score_encoder(load(prefix + name, directory), [], [], intervals, samples=samples)
                             for name in names if name != variant}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=('freeze', 'simulate', 'score'))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--configurations', nargs='+')
    parser.add_argument('--claim', action='store_true', help='Claim each network (several machines)')
    args = parser.parse_args()
    if args.step == 'freeze':
        record = freeze()
        print(f"Frozen {len(record['files'])} files on {record['date']}: {FREEZE}")
        return
    if args.step == 'score':
        result = report()
        tr_atomic(OUTPUT / 'result.json', json.dumps(result, indent=1) + '\n')
        print('decision:', result.get('decision', 'incomplete'))
        for encoder, r in result['main'].items():
            print(encoder, json.dumps(r['verdicts']), json.dumps(r['gate_necessary']))
        return
    record = opened()                                         # refuses unless frozen and unchanged
    import torch
    import train as tr
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configurations = args.configurations or sorted(Path(p).name for p in glob.glob(str(CAMPAIGN / '*')))
    paths = tr.finished_networks(sorted(p for c in configurations for p in glob.glob(str(CAMPAIGN / c / 'health_*.pt'))))
    for path in paths[args.shard::args.shards]:
        target = OUTPUT / Path(path).parent.name / (Path(path).stem + '.json')
        if target.exists() or (args.claim and not tr.claim('evaluation', Path(path).stem)):
            continue
        start = time.time()
        result = dict(simulate_network(path, device), freeze_date=record['date'], seconds=None)
        result['seconds'] = time.time() - start
        result['provenance'] = tr.provenance()
        target.parent.mkdir(parents=True, exist_ok=True)
        tr_atomic(target, json.dumps(result))
    print(f'simulate shard {args.shard}/{args.shards}: done')


def tr_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


if __name__ == '__main__':
    main()
