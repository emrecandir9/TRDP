"""Figures for docs/baselines.md (static PNG, light surface).

Colour carries one job per figure. Display role: blue = connected (full body, upper limb,
stub, arm window), orange = disconnected (detached hands, gap stub, hand window, scrambled),
aqua = no declared role (occluded wrist, hand only); marker shape repeats the role, and
condition names label every axis, so identity never rests on colour alone. Magnitudes use one
blue ramp, light to dark. Palette validated with the dataviz skill's validator (all pairs).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import baselines as b  # noqa: E402
import features as fx  # noqa: E402

SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
ROLE = dict(connected=('#2a78d6', 'o'), disconnected=('#eb6834', 's'), other=('#1baf7a', 'D'))
RAMP = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
HUMAN_BAND = '#f0efec'
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold',
                     'axes.titlelocation': 'left', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.dpi': 160, 'savefig.bbox': 'tight'})


def role(condition):
    return 'connected' if condition in b.CONNECTED else 'disconnected' if condition in b.DISCONNECTED else 'other'


def legend(ax, **kw):
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle='', markersize=6, label=r) for r, (c, m) in ROLE.items()]
    ax.legend(handles=handles, **kw)


def dots(ax, labels, mean, low, high, colors, markers, horizontal=False):
    """Point estimate with 95% interval; 2 px whiskers, 6 pt markers ringed with the surface."""
    pos = np.arange(len(labels))
    for p, m, lo, hi, c, k in zip(pos, mean, low, high, colors, markers):
        if horizontal:
            ax.plot([lo, hi], [p, p], color=c, lw=2, solid_capstyle='round')
            ax.plot(m, p, marker=k, color=c, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2)
        else:
            ax.plot([p, p], [lo, hi], color=c, lw=2, solid_capstyle='round')
            ax.plot(p, m, marker=k, color=c, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    if horizontal:
        ax.set_yticks(pos, labels)
        ax.invert_yaxis()
        ax.grid(axis='y', visible=False)
    else:
        ax.set_xticks(pos, labels, rotation=40, ha='right')
        ax.grid(axis='x', visible=False)


def inputs(data, path):
    """One family (reference 1) at encoder resolution: static frame and |movement - static|."""
    family = next(f for f in data.families if f['reference'] == 1)
    images = fx.Renderer().render(family)
    fig, axes = plt.subplots(2, len(fx.CONDITIONS), figsize=(16, 3.4))
    for k, condition in enumerate(fx.CONDITIONS):
        a, m = (fx.resized(image, 224) for image in images[condition])
        axes[0, k].imshow(a)
        change = np.abs(fx.luminance(m) - fx.luminance(a))
        axes[1, k].imshow(change, cmap=RAMP, vmin=0, vmax=max(change.max(), 1e-6))
        axes[0, k].set_title(b.SHORT[condition], fontsize=8, loc='center')
        for ax in axes[:, k]:
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
    axes[0, 0].set_ylabel('static, 224 px', fontsize=8)
    axes[1, 0].set_ylabel('|movement - static|', fontsize=8)
    fig.suptitle('What the encoders see (reference 1, beech desk, seed 9, neutral balance)', x=.01, ha='left',
                 fontweight='bold')
    fig.savefig(path)
    plt.close(fig)


def movement_signal(table, path, representations=b.PRIMARY):
    t = table.query("measure == 'normalised' and cohort == 'all' and detached_hand == 'detached_hand_silhouette'")
    fig, axes = plt.subplots(1, len(representations), figsize=(3.3 * len(representations), 3.6))
    for ax, name in zip(axes, representations):
        g = t[t.representation == name].set_index('statistic')
        rows = [g.loc[f'delta_{b.SHORT[c]}'] for c in fx.CONDITIONS]
        dots(ax, [b.SHORT[c] for c in fx.CONDITIONS], [r.estimate for r in rows], [r.ci_low for r in rows],
             [r.ci_high for r in rows], [ROLE[role(c)][0] for c in fx.CONDITIONS], [ROLE[role(c)][1] for c in fx.CONDITIONS])
        ax.set_title(b.LABEL[name])
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Movement signal (distance / between-source spread)')
    legend(axes[-1], loc='upper right')
    fig.suptitle('Static -> movement distance per display; mean over 20 sources, 95% source-bootstrap interval',
                 x=.01, ha='left', fontweight='bold', y=1.03)
    fig.savefig(path)
    plt.close(fig)


def gate(table, path):
    t = table.query("measure == 'normalised' and cohort == 'all' and detached_hand == 'detached_hand_silhouette'")
    names = list(dict.fromkeys(t.representation))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), sharey=True)
    for ax, (statistic, title) in zip(axes, (('C', 'C = (FB + UL)/2 - DH'),
                                             ('ratio_DH_to_connected', 'DH / connected (S1)'),
                                             ('ratio_UL_to_FB', 'UL / FB (S2)'))):
        g = t[t.statistic == statistic].set_index('representation').loc[names]
        colors = [ROLE['other'][0] if n in b.LOW_LEVEL else ROLE['connected'][0] for n in names]
        markers = ['D' if n in b.LOW_LEVEL else 'o' for n in names]
        if statistic in ('ratio_DH_to_connected', 'ratio_UL_to_FB'):
            human = b.HUMAN[statistic]
            ax.axvspan(human[1], human[2], color=HUMAN_BAND, zorder=0)
            ax.axvline(human[0], color=MUTED, lw=1, ls='--', zorder=1)
            ax.text(human[0], -.9, 'human 95% interval', color=MUTED, fontsize=7.5, ha='center')
        else:
            ax.axvline(0, color=MUTED, lw=1, zorder=1)
        dots(ax, [b.LABEL[n] for n in names], g.estimate, g.ci_low, g.ci_high, colors, markers, horizontal=True)
        ax.set_title(title)
    axes[0].set_xlabel('normalised movement signal')
    axes[1].set_xlabel('ratio of mean signals')
    axes[2].set_xlabel('ratio of mean signals')
    fig.suptitle('Gate 1: context contrast and scale-free ratios of each baseline (aqua diamonds: low-level)',
                 x=.01, ha='left', fontweight='bold', y=1.04)
    fig.savefig(path)
    plt.close(fig)


def heatmap(ax, values, rows, columns, vmin, vmax, fmt='{:.2f}'):
    image = ax.imshow(values, cmap=RAMP, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(columns)), columns, rotation=40, ha='right')
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                shade = (values[i, j] - vmin) / (vmax - vmin)
                ax.text(j, i, fmt.format(values[i, j]), ha='center', va='center', fontsize=7,
                        color='white' if shade > .55 else INK)
    return image


def decoding(table, path, representations=b.PRIMARY):
    within = table[table.train == table.test]
    transfer = table[table.train == 'full_body']
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.2))
    for ax, t, title in ((axes[0], within, 'Trained and tested on the same display'),
                         (axes[1], transfer, 'Trained on full body, tested on each display')):
        values = np.array([[t[(t.representation == r) & (t.test == c)]['mean'].iloc[0] for c in fx.CONDITIONS]
                           for r in representations])
        image = heatmap(ax, values, [b.LABEL[r] for r in representations], [b.SHORT[c] for c in fx.CONDITIONS], .5, 1)
        ax.set_title(title)
    fig.colorbar(image, ax=axes, shrink=.8, label='paired accuracy (chance 0.5)')
    fig.suptitle('Static vs movement: linear decoding, leave one source out (mean over 20 held-out sources)',
                 x=.01, ha='left', fontweight='bold', y=1.05)
    fig.savefig(path)
    plt.close(fig)


def attachment(table, path, representations=b.PRIMARY):
    order = [('stub_pair', 'stub_pair'), ('window_pair', 'window_pair'), ('main', 'main'),
             ('stub_pair', 'window_pair'), ('window_pair', 'stub_pair'), ('main', 'stub_pair'), ('main', 'window_pair')]
    names = dict(stub_pair='stub vs gap stub', window_pair='arm vs hand window', main='FB+UL vs DH')
    labels = [f'{names[a]}' if a == c else f'{names[a]} -> {names[c]}' for a, c in order]
    fig, axes = plt.subplots(1, len(representations), figsize=(3.2 * len(representations), 3.2), sharey=True)
    for ax, r in zip(axes, representations):
        t = table[table.representation == r].set_index(['train', 'test'])
        rows = [t.loc[k] for k in order]
        colors = [ROLE['connected'][0] if a == c else ROLE['disconnected'][0] for a, c in order]
        markers = ['o' if a == c else 's' for a, c in order]
        ax.axvline(.5, color=MUTED, lw=1)
        dots(ax, labels, [x['mean'] for x in rows], [x.ci_low for x in rows], [x.ci_high for x in rows], colors,
             markers, horizontal=True)
        ax.set_xlim(0, 1.02)
        ax.set_title(b.LABEL[r])
        ax.set_xlabel('paired accuracy')
    handles = [plt.Line2D([], [], color=ROLE['connected'][0], marker='o', ls='', label='within task'),
               plt.Line2D([], [], color=ROLE['disconnected'][0], marker='s', ls='', label='transfer to another task')]
    axes[-1].legend(handles=handles, loc='lower right')
    fig.suptitle('Connected vs disconnected displays: linear decoding of attachment, leave one source out',
                 x=.01, ha='left', fontweight='bold', y=1.04)
    fig.savefig(path)
    plt.close(fig)


def rsa(table, matrices, conditions, path, shown=('pixels64', 'dinov2_cls', 'openclip')):
    fig, axes = plt.subplots(1, len(shown) + 2, figsize=(4 * (len(shown) + 2), 3.6),
                             gridspec_kw=dict(width_ratios=[1] * (len(shown) + 1) + [1.2]))
    labels = [b.SHORT[c] for c in conditions]
    attached = np.array([c in b.CONNECTED for c in conditions], float)
    model = (attached[:, None] != attached[None]).astype(float)
    for ax, (name, m) in zip(axes, [(n, matrices[n]) for n in shown] + [('attachment model', model)]):
        heatmap(ax, m / np.nanmax(m), labels, labels, 0, 1, fmt='')
        ax.set_title(b.LABEL.get(name, 'Attachment model RDM'))
        ax.tick_params(labelsize=7)
    ax = axes[-1]
    t = table.set_index(['representation', 'measure'])
    names = list(dict.fromkeys(table.representation))
    for k, (measure, color, marker, shift) in enumerate((('spearman', ROLE['connected'][0], 'o', -.15),
                                                          ('partial', ROLE['disconnected'][0], 's', .15))):
        for i, n in enumerate(names):
            row = t.loc[(n, measure)]
            ax.plot([row.ci_low, row.ci_high], [i + shift] * 2, color=color, lw=2)
            ax.plot(row['mean'], i + shift, marker=marker, color=color, markersize=6.5, markeredgecolor=SURFACE,
                    ls='', label=('Spearman' if measure == 'spearman' else 'partial (area, luminance removed)')
                    if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(names)), [b.LABEL[n] for n in names])
    ax.invert_yaxis()
    ax.set_title('Correlation with attachment RDM')
    ax.legend(loc='lower right', fontsize=7)
    fig.suptitle('Representational geometry of the static displays (RDMs scaled to their maximum; mean over sources)',
                 x=.01, ha='left', fontweight='bold', y=1.04)
    fig.savefig(path)
    plt.close(fig)


def patch_maps(maps, path):
    shown = [c for c in fx.CONDITIONS if c != 'hand_only']
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    vmax = np.quantile(maps[[fx.CONDITIONS.index(c) for c in shown]], .995)
    for ax, condition in zip(axes.ravel(), shown):
        image = ax.imshow(maps[fx.CONDITIONS.index(condition)], cmap=RAMP, vmin=0, vmax=vmax)
        ax.set_title(b.SHORT[condition], loc='center')
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    fig.colorbar(image, ax=axes, shrink=.7, label='mean |patch token change| (DINOv2, 16 x 16 patches)')
    fig.suptitle('Where the representation changes when the index finger moves (mean over 480 families)',
                 x=.01, ha='left', fontweight='bold')
    fig.savefig(path)
    plt.close(fig)


def alignment(table, path, representations=b.PRIMARY):
    fig, axes = plt.subplots(2, len(representations), figsize=(3.2 * len(representations), 6.4), sharey='row')
    for k, (measure, ylabel) in enumerate((('alignment_with_FB', 'cos(movement direction, FB direction)'),
                                           ('cross_source_consistency', 'mean cos between sources'))):
        for ax, r in zip(axes[k], representations):
            g = table[(table.representation == r) & (table.measure == measure)].set_index('condition').loc[list(fx.CONDITIONS)]
            dots(ax, [b.SHORT[c] for c in fx.CONDITIONS], g['mean'], g.ci_low, g.ci_high,
                 [ROLE[role(c)][0] for c in fx.CONDITIONS], [ROLE[role(c)][1] for c in fx.CONDITIONS])
            ax.axhline(0, color=MUTED, lw=1)
            if k == 0:
                ax.set_title(b.LABEL[r])
        axes[k, 0].set_ylabel(ylabel)
    legend(axes[0, -1], loc='lower left')
    fig.suptitle('Direction of the movement in latent space: shared with full body? shared across sources?',
                 x=.01, ha='left', fontweight='bold', y=1.01)
    fig.savefig(path)
    plt.close(fig)


def all_figures(tables, matrices, rsa_conditions, maps, data, directory, representations=b.PRIMARY):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inputs(data, directory / 'baselines-inputs.png')
    movement_signal(tables['movement_signal'], directory / 'baselines-movement-signal.png', representations)
    gate(tables['movement_signal'], directory / 'baselines-gate1.png')
    alignment(tables['alignment'], directory / 'baselines-alignment.png', representations)
    shown = tuple(r for r in ('pixels64', 'dinov2_cls', 'openclip') if r in representations)
    rsa(tables['rsa'], matrices, rsa_conditions, directory / 'baselines-rsa.png', shown)
    if maps is not None:
        patch_maps(maps, directory / 'baselines-patch-maps.png')
    if 'decoding' in tables:
        decoding(tables['decoding'], directory / 'baselines-decoding.png', representations)
        attachment(tables['attachment'], directory / 'baselines-attachment.png', representations)
