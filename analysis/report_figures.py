"""Figures and tables of the two reports, from the evaluation's result (final-phase plan, task A1).

Reads `output/model/evaluation/result.json` (written once by `model/run_evaluation.py score`), the human
target (`output/human_target/`) and the criterion-status table (`docs/criteria-status.json`), and writes
static PNG figures and Markdown tables. Nothing here computes a criterion: every number shown is read from
the result file, so the reports quote exactly what the evaluation produced.

    python analysis/report_figures.py [--result output/model/evaluation/result.json] [--out output/report]

Colour carries one job per figure (style of analysis/baseline_figures.py): models keep one colour each
across figures, the human data are drawn in ink, and marker shape repeats the colour, so identity never
rests on colour alone.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'output/model/evaluation/result.json'
HUMAN = ROOT / 'output/human_target'
STATUS = ROOT / 'docs/criteria-status.json'
OUT = ROOT / 'output/report'

SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
MODEL_STYLE = dict(h2=('#2a78d6', 'o', 'H2 (attachment gate)'), ungated=('#eb6834', 's', 'Ungated network'),
                   area=('#1baf7a', 'D', 'Visible-area gate'), h2_learned=('#8a5cd1', 'v', 'H2-learned'),
                   h4=('#c2477e', '^', 'H4 (familiarity)'), h2_cplus=('#86b6ef', 'P', 'H2 at c+'),
                   h0=('#7a7974', 'X', 'H0 (hand only)'), h1=('#b8860b', 'h', 'H1 (salience)'))
ANALYSIS_LABEL = dict(raw='as simulated', remove='over-driven removed', maximum='over-driven at maximum')
DISPLAY_LABEL = dict(full_body='Full body', upper_limb='Upper limb', detached_hand_silhouette='Detached hand',
                     detached_hand='Detached hand (rectangle)', hand_stub='Hand + stub', hand_gap_stub='Hand + gap + stub',
                     window_hand='Window: hand', window_arm='Window: arm', scrambled='Scrambled',
                     occluded_wrist='Occluded wrist')
HUMAN_BODIES = ('full_body', 'upper_limb', 'detached_hand')
CRITERIA = ('P1', 'P2', 'P3', 'S1', 'S2', 'S3')
STATUS_LABEL = dict(built_in='built in', calibrated='calibrated', previewed='previewed', open='open', not_scored='not scored')
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold',
                     'axes.titlelocation': 'left', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.dpi': 160, 'savefig.bbox': 'tight'})


# ---------- inputs ----------

def load_result(path=RESULT):
    return json.loads(Path(path).read_text())


def human_cells(directory=HUMAN, scale='participant_z'):
    """Human cells as arrays [3 displays, 2 trial types, 2 muscles] (mean, low, high), paper order."""
    t = pd.read_csv(Path(directory) / 'cells.csv')
    t = t[t.scale == scale].set_index(['body', 'movement', 'muscle'])
    index = [(b, k, m) for b in HUMAN_BODIES for k in ('static', 'movement') for m in ('FDI', 'ADM')]
    return tuple(t.loc[index, c].to_numpy(float).reshape(3, 2, 2) for c in ('mean', 'ci_low', 'ci_high'))


def human_contrasts(directory=HUMAN, scale='participant_z'):
    t = pd.read_csv(Path(directory) / 'contrasts.csv')
    t = t[t.scale == scale].set_index('contrast')
    names = dict(C_FDI='C_FDI', delta_FB='delta_FB_FDI', delta_UL='delta_UL_FDI', delta_DH='delta_DH_FDI')
    return {k: (float(t.loc[v, 'estimate']), float(t.loc[v, 'ci_low']), float(t.loc[v, 'ci_high'])) for k, v in names.items()}


def load_status(path=STATUS):
    return json.loads(Path(path).read_text())


def scored(result, encoder='openclip', block='main'):
    """One encoder's (or comparison configuration's) score_encoder output."""
    return result['configurations'][encoder] if block == 'configurations' else result[block][encoder]


# ---------- figures ----------

def _interval(ax, x, mean, low, high, color, marker, label=None, size=6.5):
    ax.plot([x, x], [low, high], color=color, lw=2, solid_capstyle='round')
    ax.plot(x, mean, marker=marker, color=color, markersize=size, markeredgecolor=SURFACE, markeredgewidth=1.2,
            linestyle='', label=label)


def cell_plot(result, path, encoder='openclip', analysis='remove', models=('h2', 'ungated', 'h0', 'h1'), human=None):
    """Human vs model cells: one panel per display and muscle, static and movement, z units."""
    human = human or human_cells()
    analyses = scored(result, encoder)['analyses'][analysis]['models']
    models = [m for m in models if m in analyses]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2), sharey='row')
    for d in range(3):
        for p, muscle in enumerate(('FDI', 'ADM')):
            ax = axes[p, d]
            for k, kind in enumerate(('static', 'movement')):
                base = k * (len(models) + 2)
                _interval(ax, base, human[0][d, k, p], human[1][d, k, p], human[2][d, k, p], INK, 'o',
                          'Human (n = 32)' if (d, p, k) == (0, 0, 0) else None)
                for j, m in enumerate(models):
                    c = analyses[m]['cells']
                    color, marker, label = MODEL_STYLE[m]
                    _interval(ax, base + 1 + j, c['mean'][d][k][p], c['ci_low'][d][k][p], c['ci_high'][d][k][p], color, marker,
                              label if (d, p, k) == (0, 0, 0) else None)
            ax.set_xticks([(len(models) + 1) / 2, len(models) + 2 + (len(models) + 1) / 2], ['static', 'movement'])
            ax.grid(axis='x', visible=False)
            ax.axhline(0, color=MUTED, lw=.6)
            name = analyses[models[0]]['cells']['order'][0][d] if models else HUMAN_BODIES[d]
            ax.set_title(f'{DISPLAY_LABEL.get(name, name)}, {muscle}')
    axes[0, 0].set_ylabel('FDI MEP (z within participant / instance)')
    axes[1, 0].set_ylabel('ADM MEP (z)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=len(models) + 1, bbox_to_anchor=(.5, -.04))
    fig.suptitle(f'Cells, human and model ({encoder}, {ANALYSIS_LABEL[analysis]})', x=.01, ha='left', fontweight='bold')
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def effects_plot(result, path, encoder='openclip', models=None, human=None):
    """C_FDI and the two movement effects per model with 95% intervals, under each analysis; human in ink."""
    human = human or human_contrasts()
    block = scored(result, encoder)['analyses']
    has_data = lambda m: any(np.isfinite(block[a]['models'][m]['C_FDI']['estimate']) for a in block)
    models = models or [m for m in MODEL_STYLE if m in block['raw']['models'] and has_data(m)]
    quantities = ('C_FDI', 'delta_FB', 'delta_UL')
    titles = dict(C_FDI='C_FDI (primary contrast)', delta_FB='Movement effect, full body', delta_UL='Movement effect, upper limb')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=False)
    offsets = dict(raw=-.22, remove=0., maximum=.22)
    for ax, q in zip(axes, quantities):
        _interval(ax, -1, *human[q], INK, 'o', 'Human')
        for i, m in enumerate(models):
            color, marker, label = MODEL_STYLE[m]
            for a, off in offsets.items():
                e = block[a]['models'][m][q]
                _interval(ax, i + off, e['estimate'], e['ci'][0], e['ci'][1], color, marker, size=5 if a != 'remove' else 6.5,
                          label=label if (q == 'C_FDI' and a == 'remove') else None)
        ax.axhline(0, color=MUTED, lw=.6)
        ax.set_xticks([-1] + list(range(len(models))), ['Human'] + [MODEL_STYLE[m][2] for m in models], rotation=40, ha='right')
        ax.grid(axis='x', visible=False)
        ax.set_title(titles[q])
    axes[0].set_ylabel('z units, 95% interval')
    fig.suptitle(f'Contrasts per model ({encoder}); three marks per model, left to right: readouts as simulated, '
                 'over-driven removed, over-driven at maximum', x=.01, ha='left', fontweight='bold', fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def diets_plot(result, path, analysis='remove', ablation='snn_allctx'):
    """The two training diets side by side: H2 and the ungated network trained on the full body only
    (primary) and on every display (all-context ablation)."""
    sets = [('Full-body training', scored(result, 'openclip')['analyses'][analysis]['models'])]
    if ablation in result.get('configurations', {}):
        sets.append(('All-context training', scored(result, ablation, 'configurations')['analyses'][analysis]['models']))
    fig, axes = plt.subplots(1, len(sets), figsize=(4.2 * len(sets), 3.8), sharey=True, squeeze=False)
    quantities = ('delta_FB', 'delta_UL', 'C_FDI')
    for ax, (title, models) in zip(axes[0], sets):
        for j, m in enumerate(('h2', 'ungated')):
            color, marker, label = MODEL_STYLE[m]
            for i, q in enumerate(quantities):
                e = models[m][q]
                _interval(ax, i + (j - .5) * .3, e['estimate'], e['ci'][0], e['ci'][1], color, marker, label if i == 0 else None)
            ax.text(0, -.2 - .08 * j, f"{label}: S1 ratio {models[m]['S1_ratio']:.2f}, S2 ratio {models[m]['S2_ratio']:.2f}",
                    transform=ax.transAxes, fontsize=7, color=color)
        ax.axhline(0, color=MUTED, lw=.6)
        ax.set_xticks(range(3), ['Δ FB', 'Δ UL', 'C_FDI'])
        ax.grid(axis='x', visible=False)
        ax.set_title(title)
    axes[0, 0].set_ylabel('z units, 95% interval')
    axes[0, 0].legend(loc='lower left', fontsize=7)
    fig.suptitle(f'Necessity of the gate under the two training diets ({ANALYSIS_LABEL[analysis]})', x=.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def controls_plot(result, path, encoder='openclip', analysis='remove'):
    """FDI movement effect on each control display for H2, the ungated network and the visible-area gate."""
    block = scored(result, encoder)['analyses'][analysis]
    if 'controls' not in block:
        return False
    controls = block['controls']
    displays = list(next(iter(controls.values())))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for j, (m, values) in enumerate(controls.items()):
        color, marker, label = MODEL_STYLE[m]
        ax.plot(np.arange(len(displays)) + (j - 1) * .2, [values[d] for d in displays], marker=marker, color=color,
                linestyle='', markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    ax.axhline(0, color=MUTED, lw=.6)
    ax.set_xticks(range(len(displays)), [DISPLAY_LABEL.get(d, d) for d in displays], rotation=30, ha='right')
    ax.grid(axis='x', visible=False)
    ax.set_ylabel('FDI movement effect (z), mean of instances')
    ax.legend(loc='best')
    pairs = block.get('S3a', {}).get('pairs', {})
    note = '; '.join(f"{p}: H2 − area {v['estimate']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]" for p, v in pairs.items())
    ax.set_title(f'Control displays ({encoder}, {ANALYSIS_LABEL[analysis]})')
    if note:
        ax.text(0, -.42, note, transform=ax.transAxes, fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


# ---------- tables ----------

def _status(status, model, criterion):
    row = status['models'].get(model, {})
    if criterion == 'P2':
        a, b = (STATUS_LABEL[row[c]['status']] for c in ('P2_FB', 'P2_UL')) if row else ('', '')
        return a if a == b else f'FB {a}, UL {b}'
    if criterion == 'S3':
        return STATUS_LABEL[row['S3a']['status']] + ' / ' + STATUS_LABEL[row['S3b']['status']] if row else ''
    return STATUS_LABEL[row[criterion]['status']] if row else ''


def criterion_table(result, status=None, encoder='openclip', models=('h2', 'ungated', 'area', 'h2_learned', 'h4', 'h0', 'h1')):
    """Verdict (met / not met / undetermined) of every criterion per model, with what the criterion can
    still show for that model (built in, calibrated, previewed, open)."""
    status = status or load_status()
    verdicts = scored(result, encoder)['verdicts']
    rows = []
    for m in models:
        if m not in verdicts:
            continue
        for c in CRITERIA:
            if c in verdicts[m]:
                rows.append(dict(model=MODEL_STYLE[m][2], criterion=c, verdict=verdicts[m][c], status=_status(status, m, c)))
    return pd.DataFrame(rows)


def effects_table(result, encoder='openclip', block='main'):
    """C_FDI, the movement effects and the two ratios per model and analysis, with intervals."""
    rows = []
    for a, entry in scored(result, encoder, block)['analyses'].items():
        for m, s in entry['models'].items():
            fmt = lambda e: f"{e['estimate']:+.2f} [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]"
            rows.append(dict(model=MODEL_STYLE.get(m, (0, 0, m))[2], analysis=ANALYSIS_LABEL[a], C_FDI=fmt(s['C_FDI']),
                             delta_FB=fmt(s['delta_FB']), delta_UL=fmt(s['delta_UL']), S1_ratio=round(s['S1_ratio'], 2),
                             S2_ratio=round(s['S2_ratio'], 2), instances=s['instances']['total']))
    return pd.DataFrame(rows)


def robustness_table(result):
    """H2's verdicts, the ungated network's P1 and whether the gate adds to C_FDI, for every encoder,
    every declared repetition and every comparison configuration."""
    rows = []

    def add(label, s):
        h2, ungated = s['verdicts'].get('h2', {}), s['verdicts'].get('ungated', {})
        c = s['analyses']['remove']['models'].get('h2', {}).get('C_FDI', {})
        rows.append(dict(scored=label, **{f'H2 {k}': h2.get(k, '') for k in CRITERIA}, **{'ungated P1': ungated.get('P1', ''),
                         'gate adds to C_FDI': s['gate_necessary'].get('adds_to_C_FDI', ''),
                         'H2 C_FDI (removed)': f"{c['estimate']:+.2f}" if c else '', 'H2 instances': s['instances']['h2']}))
    for e, s in result['main'].items():
        add(f'main, {e}', s)
    for block in ('without_reference_17', 'healthy_only', 'untrimmed', 'rectangle'):
        for e, s in result.get(block, {}).items():
            add(f'{block.replace("_", " ")}, {e}', s)
    for name, s in result.get('configurations', {}).items():
        add(f'configuration {name}', s)
    return pd.DataFrame(rows)


def over_driven_table(result, encoder='openclip'):
    s = scored(result, encoder)['over_driven']
    return pd.DataFrame({MODEL_STYLE[m][2]: {DISPLAY_LABEL.get(d, d): v for d, v in shares.items()} for m, shares in s.items()})


def markdown(table):
    if table.empty:
        return '(none)\n'
    columns = [str(c) for c in table.columns]
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in table.itertuples(index=False)]
    return '\n'.join(lines) + '\n'


def write_all(result, directory=OUT, status=None, human_directory=HUMAN):
    """Every figure and table of the reports. Returns the written paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    human, contrasts = human_cells(human_directory), human_contrasts(human_directory)
    written = []
    cell_plot(result, directory / 'cells.png', human=human)
    effects_plot(result, directory / 'effects.png', human=contrasts)
    diets_plot(result, directory / 'diets.png')
    written += [directory / 'cells.png', directory / 'effects.png', directory / 'diets.png']
    if controls_plot(result, directory / 'controls.png'):
        written.append(directory / 'controls.png')
    tables = dict(criteria=criterion_table(result, status), effects=effects_table(result), robustness=robustness_table(result),
                  over_driven=over_driven_table(result).reset_index(names='display'))
    text = '# Evaluation tables\n\nGenerated by `analysis/report_figures.py` from `result.json`; do not edit by hand.\n\n'
    if 'decision' in result:
        text += f"**Decision rule (spec §5.2): {result['decision']}.**\n\n"
    for name, table in tables.items():
        text += f'## {name.replace("_", " ").capitalize()}\n\n' + markdown(table) + '\n'
        table.to_csv(directory / f'{name}.csv', index=False)
        written.append(directory / f'{name}.csv')
    (directory / 'tables.md').write_text(text)
    written.append(directory / 'tables.md')
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--result', default=str(RESULT))
    parser.add_argument('--out', default=str(OUT))
    args = parser.parse_args()
    for path in write_all(load_result(args.result), args.out):
        print(path)


if __name__ == '__main__':
    main()
