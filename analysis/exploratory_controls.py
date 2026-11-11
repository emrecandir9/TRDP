"""EXPLORATORY (not declared before the freeze): every model's predictions on the matched control images.

The declared evaluation scored the control pairs only for H2, the ungated network and the visible-area gate
(S3(a), `result.json` → `main.<encoder>.analyses.<A>.controls` and `S3a`). This script reads the same stored
evaluation records (`output/model/evaluation/<configuration>/*.json`, written once by the frozen code) and
summarises the controls for every model the records contain: H2, ungated, visible area, H2-learned, H4, H2 at
c⁺, H0 and H1. Nothing is simulated or re-simulated, no criterion or verdict is computed, and nothing here
changes the declared result. Final-phase plan block D: analyses not declared before the freeze go into
`result_exploratory.json` and are labelled exploratory.

For each model, encoder and analysis (as simulated, over-driven removed, maximum):
- the FDI movement effect Δ on each of the six control displays (mean over instances);
- each matched pair's difference (stub − gap stub; arm window − hand window), with its hierarchical-bootstrap
  interval (same draws, samples and seed as the evaluation);
- for models built on H2's networks, the paired difference from H2 on each pair.

Check: for H2, ungated and visible area the per-display means and the H2 − visible-area pair differences
must equal the declared `result.json` values exactly (`--check`).

    python analysis/exploratory_controls.py [--records output/model/evaluation] [--check docs/evaluation/result.json]
    python analysis/exploratory_controls.py --figure [--out docs/evaluation]
"""
import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))

import criteria as cr  # noqa: E402
import run_evaluation as rv  # noqa: E402

OUTPUT = ROOT / 'output/model/evaluation/result_exploratory.json'
CONFIGURATIONS = {e: (f'h2_{e}_snn', f'h0_{e}_snn', f'h1_{e}_snn') for e in ('openclip', 'dinov2_cls', 'dinov2_patchmean')}
CONFIGURATIONS['openclip_allctx'] = ('h2_openclip_snn_allctx', None, None)
LABELS = dict(h2='H2 (attachment gate)', ungated='Ungated network', area='Visible-area gate', h2_learned='H2-learned',
              h4='H4 (familiarity)', h2_cplus='H2 at c+', h0='H0 (hand only)', h1='H1 (salience)')


def summarise(results, name, analysis, index_sets, h2_effects=None):
    displays = rv.SESSIONS['controls'][0]
    cells = np.array([rv.cells_of(r, 'controls', name, analysis) for r in results])
    out = dict(deltas={d: float(np.nanmean(cr.deltas(cells, i))) for i, d in enumerate(displays)})
    effects = cr.pair_effects(cells, displays)
    out['pairs'] = {p: cr.paired_difference(v, np.zeros_like(v), index_sets) for p, v in effects.items()}
    if h2_effects is not None and name != 'h2':
        out['minus_h2'] = {p: cr.paired_difference(v, h2_effects[p], index_sets) for p, v in effects.items()}
    return out, effects


def run(records=rv.OUTPUT, samples=cr.BOOTSTRAP['samples']):
    out = dict(exploratory=True,
               note=('EXPLORATORY: not declared before the freeze. Summaries of the stored evaluation records on the '
                     'control displays for every model; no criterion, verdict or decision. Same bootstrap as the evaluation.'),
               freeze_date=json.loads((ROOT / 'config/freeze.json').read_text())['date'], encoders={})
    for label, (h2c, h0c, h1c) in CONFIGURATIONS.items():
        h2 = rv.ordered(rv.load(h2c, records))
        if not h2:
            continue
        entry = {}
        for analysis in rv.ANALYSES:
            models = {}
            sets = cr.draws([r['source'] for r in h2], samples)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                models['h2'], h2_effects = summarise(h2, 'h2', analysis, sets)
                for name in rv.H2_MODELS[1:]:
                    models[name], _ = summarise(h2, name, analysis, sets, h2_effects)
                for name, configuration in (('h0', h0c), ('h1', h1c)):
                    if configuration:
                        results = rv.ordered(rv.load(configuration, records))
                        if results:
                            models[name], _ = summarise(results, name, analysis, cr.draws([r['source'] for r in results], samples))
            entry[analysis] = dict(instances=len(h2), models=models)
        out['encoders'][label] = entry
    return out


def check(result, declared):
    """H2, ungated and visible area must reproduce the declared controls and S3(a) differences exactly."""
    problems = []
    for encoder in ('openclip', 'dinov2_cls', 'dinov2_patchmean'):
        for analysis in rv.ANALYSES:
            d = declared['main'][encoder]['analyses'][analysis]
            mine = result['encoders'][encoder][analysis]['models']
            for name in ('h2', 'ungated', 'area'):
                for display, value in d['controls'][name].items():
                    if not np.isclose(mine[name]['deltas'][display], value, rtol=0, atol=1e-12):
                        problems.append((encoder, analysis, name, display))
            for pair, value in d['S3a']['pairs'].items():
                theirs = mine['area']['minus_h2'][pair]          # area − H2: the negative of the declared H2 − area
                if not (np.isclose(-theirs['estimate'], value['estimate'], atol=1e-12)
                        and np.allclose(sorted(-np.array(theirs['ci'])), value['ci'], atol=1e-12)):
                    problems.append((encoder, analysis, 'S3a', pair))
    return problems


def figure(result, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / 'analysis'))
    import report_figures as rf
    models = result['encoders']['openclip']['raw']['models']
    names = [n for n in ('h2', 'h2_learned', 'h4', 'area', 'ungated', 'h1', 'h0') if n in models]
    pairs = (('hand_stub-hand_gap_stub', 'Forearm stub − same stub with a gap'),
             ('window_arm-window_hand', 'Window on the arm − window on the hand'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (pair, title) in zip(axes, pairs):
        for i, n in enumerate(names):
            p = models[n]['pairs'][pair]
            colour, marker, _ = rf.MODEL_STYLE.get(n, ('0.35', 'o', n))
            ax.errorbar(p['estimate'], i, xerr=[[p['estimate'] - p['ci'][0]], [p['ci'][1] - p['estimate']]],
                        fmt=marker, color=colour, capsize=0, lw=2, ms=6)
        ax.axvline(0, color='0.3', lw=0.8)
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
        ax.set_xlabel('Difference in FDI movement effect (z), 95% interval')
        ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
        ax.invert_yaxis()
        ax.grid(axis='x', color='0.9')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.suptitle('EXPLORATORY: model predictions on the matched controls (OpenCLIP, held-out sources, as simulated)',
                 x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.text(0.01, -0.02, 'Same visible area within each pair; only the connection to the arm differs. No human data exist '
             'for these images.', fontsize=9, color='0.35')
    fig.tight_layout()
    path = Path(out_dir) / 'controls_all_models_exploratory.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return path


def table(result):
    rows = ['| Encoder | Model | Stub − gap stub | Arm window − hand window | Minus H2 (stub; window) |', '| --- | --- | --- | --- | --- |']
    fmt = lambda p: f"{p['estimate']:+.2f} [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}]"
    for encoder, entry in result['encoders'].items():
        for name, m in entry['raw']['models'].items():
            d = m.get('minus_h2')
            rows.append(f"| {encoder} | {LABELS[name]} | {fmt(m['pairs']['hand_stub-hand_gap_stub'])} | "
                        f"{fmt(m['pairs']['window_arm-window_hand'])} | "
                        + (f"{fmt(d['hand_stub-hand_gap_stub'])}; {fmt(d['window_arm-window_hand'])}" if d else '—') + ' |')
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--records', default=str(rv.OUTPUT))
    parser.add_argument('--json', default=str(OUTPUT))
    parser.add_argument('--check', help='Declared result.json to reproduce for H2, ungated and visible area')
    parser.add_argument('--figure', action='store_true', help='Figure and table from an existing --json')
    parser.add_argument('--out', default=str(ROOT / 'docs/evaluation'))
    args = parser.parse_args()
    if args.figure:
        result = json.loads(Path(args.json).read_text())
        print(figure(result, args.out))
        (Path(args.out) / 'controls_all_models_exploratory.md').write_text(
            '# EXPLORATORY: every model on the matched controls\n\nGenerated by `analysis/exploratory_controls.py` from '
            '`result_exploratory.json`; not declared before the freeze. As simulated; 95% hierarchical-bootstrap intervals.\n\n'
            + table(result) + '\n')
        return
    result = run(args.records)
    if args.check:
        declared_path = Path(args.check)
        result['declared_result_sha256'] = hashlib.sha256(declared_path.read_bytes()).hexdigest()
        problems = check(result, json.loads(declared_path.read_text()))
        result['check'] = dict(reproduces_declared=not problems, problems=problems)
        print('reproduces declared controls and S3(a):', not problems, problems[:5])
    Path(args.json).write_text(json.dumps(result, indent=1) + '\n')
    print(table(result))


if __name__ == '__main__':
    main()
