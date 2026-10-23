"""Hand-centred views of every display: docs/baselines.md, Addendum A (exploratory, added after
the Gate 1 results were known).

The Gate 1 encoders saw the whole 2048-px scene resized to 224 px. Here every display of a
source is seen through one square window centred on the source's used_hand_box (side = s x the
box's longer side, s in SCALES; pixels outside the image are black), the same window for every
condition and both frames. The movement signal is then analysed exactly as the declared Gate 1
analysis (analysis/baselines.py; docs/baselines.md §4-§5), separately for each scale.

  extract   CUDA machine. Render (features.Renderer), crop, low-level codes on the crop and the
            encoders of features.Encoders at 224 px -> output/handview/. The same job encodes one
            all-black 2048-px image (the fixation-trial display) -> output/features_extra/.
  analyse   CPU. Tables and summary -> output/handview/; figure docs/figures/baselines-handview.png.
  figure    CPU. Redraw the figure from output/handview/movement_signal.csv.
"""
import argparse
import datetime
import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
import baselines as b  # noqa: E402
import features as fx  # noqa: E402

SCALES = (2, 4)
CONDITIONS = b.CONTEXTS   # the ten displays with a context; hand_only has none
SIZE = fx.SIZES['dinov2']
OUTPUT = ROOT / 'output/handview'
EXTRA = ROOT / 'output/features_extra'
FIGURE = ROOT / 'docs/figures/baselines-handview.png'
GATE1_MANIFEST = fx.OUTPUT / 'manifest.json'
REPRESENTATIONS = {name: b.REPRESENTATIONS[name] for name in b.PRIMARY}
ENCODED = ('dinov2_cls', 'dinov2_patchmean', 'openclip')
require = fx.require
require(fx.SIZES['dinov2'] == fx.SIZES['openclip'] == 224, 'Both encoders are expected at 224 px')


# ---------- views ----------

def window(box, scale):
    """(left, top, side) of the square centred on box = [x, y, right, bottom], side = scale x
    the longer side of the box. A centre between two pixels is rounded down (off by <= 0.5 px)."""
    x, y, right, bottom = (int(v) for v in box)
    side = int(scale * max(right - x, bottom - y))
    return (x + right - side) // 2, (y + bottom - side) // 2, side


def crop(image, win):
    """The window of an image; pixels outside the image are black."""
    left, top, side = win
    out = np.zeros((side, side) + image.shape[2:], image.dtype)
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + side, image.shape[1]), min(top + side, image.shape[0])
    if x1 > x0 and y1 > y0:
        out[y0 - top:y1 - top, x0 - left:x1 - left] = image[y0:y1, x0:x1]
    return out


def iter_views(images, source, scales=SCALES, conditions=CONDITIONS):
    """(scale, condition, (static, movement)) with one window per source and scale, the same for
    every condition and both frames."""
    for scale in scales:
        win = window(source['used_hand_box'], scale)
        for condition in conditions:
            yield scale, condition, tuple(crop(image, win) for image in images[condition])


def views(images, source, scales=SCALES, conditions=CONDITIONS):
    out = {}
    for scale, condition, pair in iter_views(images, source, scales, conditions):
        out.setdefault(scale, {})[condition] = pair
    return out


def identical(first, second):
    """True if two (static, movement) pairs are pixel-identical in both frames."""
    return all(np.array_equal(a, c) for a, c in zip(first, second))


# ---------- extraction (GPU) ----------

_renderer = None


def prepare(family):
    """CPU work for one family: render from the archive, crop, hash, low-level codes and the
    224-px encoder input of every scale, condition and frame; and the FB = UL check."""
    global _renderer
    _renderer = _renderer or fx.Renderer()
    images = _renderer.render(family)
    rows, checks, kept = [], [], {}
    for scale, condition, (a, m) in iter_views(images, family['source']):
        require(a.shape == m.shape, f'Frame shapes differ: {family["id"]} {condition}')
        for frame, image in zip(fx.FRAMES, (a, m)):
            rows.append(dict(family=family['id'], scale=scale, condition=condition, frame=frame,
                             sha256=fx.array_sha(image), lowlevel=fx.lowlevel(image), pixels=fx.pixels(image),
                             small=fx.resized(image, SIZE)))
        rows[-1]['difference'] = fx.difference(a, m)
        if condition in ('full_body', 'upper_limb'):
            kept[condition] = (a, m)
            if len(kept) == 2:
                checks.append(dict(family=family['id'], source=family['reference'], scale=scale,
                                   fb_equals_ul=identical(kept['full_body'], kept['upper_limb'])))
                kept = {}
    return rows, checks


def encode(encoders, smalls):
    cls, mean, _ = encoders.dino(smalls)
    return dict(dinov2_cls=cls, dinov2_patchmean=mean, openclip=encoders.clip(smalls))


def pinned_encoders():
    """features.Encoders, required to resolve to the revisions and weights of the Gate 1 run.
    The Hub revisions are checked before any weight is loaded, so nothing new is downloaded."""
    expected = fx.s.read(GATE1_MANIFEST)['models'] if GATE1_MANIFEST.is_file() else None
    if expected:
        from huggingface_hub import HfApi
        api = HfApi()
        found = dict(dinov2=api.model_info(fx.DINOV2).sha, openclip=api.model_info(fx.OPENCLIP['repository']).sha)
        pinned = {name: expected[name]['revision'] for name in found}
        require(found == pinned, f'Hub revisions differ from the Gate 1 run: {found} vs {pinned}')
    encoders = fx.Encoders()
    if expected:
        require(encoders.revisions == pinned, 'Encoder revisions differ from the Gate 1 run')
        require(encoders.weights['openclip']['sha256'] == expected['openclip']['sha256'],
                'OpenCLIP weights differ from the Gate 1 run')
    return encoders, bool(expected)


def models(encoders):
    return dict(dinov2=dict(repository=fx.DINOV2, revision=encoders.revisions['dinov2'],
                            tokens='last_hidden_state (final layer norm); CLS and patch tokens'),
                openclip=dict(fx.OPENCLIP, revision=encoders.revisions['openclip'],
                              output='encode_image, before L2 normalisation', **encoders.weights['openclip']))


def black(encoders, directory=EXTRA):
    """One all-black 2048 x 2048 RGB image (the fixation-trial display), with the same
    representations and preprocessing as every Gate 1 and Addendum A image."""
    directory.mkdir(parents=True, exist_ok=True)
    image = np.zeros((2048, 2048, 3), np.uint8)
    small = fx.resized(image, SIZE)
    cls, mean, patches = encoders.dino([small])
    arrays = dict(dinov2_cls=cls[0].astype(np.float32), dinov2_patchmean=mean[0].astype(np.float32),
                  openclip=encoders.clip([small])[0].astype(np.float32), lowlevel=fx.lowlevel(image),
                  pixels64=fx.pixels(image).astype(np.float32), dinov2_patches=patches[0].astype(np.float16))
    require(arrays['dinov2_patches'].shape == (256, 1024), 'Unexpected patch-token shape')
    np.savez(directory / 'black.npz', **arrays)
    import torch
    manifest = dict(description='One all-black 2048 x 2048 RGB image (the fixation-trial display), encoded with '
                                'the encoders and preprocessing of the Gate 1 baselines (docs/baselines.md §3)',
                    created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                    code_sha256={'analysis/handview.py': fx.s.sha(__file__), 'analysis/features.py': fx.s.sha(fx.__file__)},
                    image_sha256=fx.array_sha(image),
                    preprocessing=f'whole image resized to {SIZE} px (PIL bicubic, antialiased; no centre crop), '
                                  'normalised with each encoder\'s mean and SD; encoded alone (batch of 1)',
                    lowlevel=list(fx.LOWLEVEL), pixel_grid=fx.PIXEL_GRID, models=models(encoders),
                    precision='float32, TF32 disabled; dinov2_patches stored as float16',
                    gpu=torch.cuda.get_device_name(0),
                    arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()},
                    files={'black.npz': fx.s.sha(directory / 'black.npz')})
    fx.s.write(directory / 'manifest.json', manifest)
    return manifest


def noise_floor(encoders, rows, encoded):
    """Re-encode the first family's images one at a time in reverse order; largest change."""
    again = [encode(encoders, [r['small']]) for r in reversed(rows)][::-1]
    out = {}
    for name in ENCODED:
        first, repeat = encoded[name], np.concatenate([a[name] for a in again])
        out[name] = dict(max_abs=float(np.abs(first - repeat).max()),
                         max_relative=float((np.linalg.norm(first - repeat, axis=1) / np.linalg.norm(first, axis=1)).max()))
    return out


def write_csv(path, records):
    pd.DataFrame(records).to_csv(path, index=False)


def extract(output=OUTPUT, extra=EXTRA, limit=None, workers=8):
    output.mkdir(parents=True, exist_ok=True)
    selected = fx.families()
    if limit:   # whole appearance variants, so every source and the between-source spread remain defined
        keep = list(dict.fromkeys(f['variant'] for f in selected))[:limit]
        selected = [f for f in selected if f['variant'] in keep]
    started = time.time()
    encoders, pinned = pinned_encoders()
    black_manifest = black(encoders, extra)
    print(f'Encoders loaded and black image encoded, {time.time() - started:.0f} s', flush=True)
    index, low, pix, differences, checks, features, floor = [], [], [], [], [], {}, None
    loop = time.time()
    with Pool(workers) as pool:
        for number, (rows, family_checks) in enumerate(pool.imap(prepare, selected), 1):
            encoded = {}
            for scale in SCALES:
                part = encode(encoders, [r['small'] for r in rows if r['scale'] == scale])
                for name, value in part.items():
                    encoded.setdefault(name, []).append(value.astype(np.float32))
            encoded = {name: np.concatenate(v) for name, v in encoded.items()}
            if floor is None:
                floor = noise_floor(encoders, rows, encoded)
            for name, value in encoded.items():
                features.setdefault(name, []).append(value)
            for r in rows:
                index.append(dict(family=r['family'], scale=r['scale'], condition=r['condition'], frame=r['frame'],
                                  sha256=r['sha256']))
                low.append(r['lowlevel'])
                pix.append(r['pixels'])
                if 'difference' in r:
                    differences.append(dict(family=r['family'], scale=r['scale'], condition=r['condition'],
                                            **r['difference']))
            checks += family_checks
            if number % 24 == 0 or number == len(selected):
                elapsed = time.time() - loop
                print(f'{number}/{len(selected)} families, {elapsed:.0f} s, '
                      f'about {elapsed / number * (len(selected) - number):.0f} s left', flush=True)
    seconds = time.time() - started
    arrays = dict(lowlevel=np.array(low), pixels64=np.array(pix, np.float32))
    arrays.update({name: np.concatenate(v) for name, v in features.items()})
    np.savez(output / 'features.npz', **arrays)
    write_csv(output / 'index.csv', index)
    write_csv(output / 'differences.csv', differences)
    write_csv(output / 'identity.csv', checks)
    windows = [dict(reference=src['reference'], used_hand_box=src['used_hand_box'],
                    upper_limb_cutoff=src['upper_limb_cutoff'],
                    windows={str(scale): list(window(src['used_hand_box'], scale)) for scale in SCALES})
               for src in fx.s.sources()]
    import torch
    manifest = dict(method='docs/baselines.md, Addendum A', code_sha256=fx.s.sha(__file__),
                    features_code_sha256=fx.s.sha(fx.__file__), baselines_code_sha256=fx.s.sha(b.__file__),
                    images=len(index), families=len(selected), scales=list(SCALES), conditions=list(CONDITIONS),
                    frames=list(fx.FRAMES), lowlevel=list(fx.LOWLEVEL), pixel_grid=fx.PIXEL_GRID,
                    window='square centred on the used_hand_box centre (corner rounded down), side = scale x '
                           'max(box width, box height); outside the 2048 image black; same for every condition and frame',
                    resize=f'crop to {SIZE} px, PIL bicubic, antialiased (features.resized); low-level codes on the '
                           'full-resolution crop', windows=windows, models=models(encoders),
                    pinned_to_gate1_manifest=pinned, precision='float32, TF32 disabled',
                    gpu=torch.cuda.get_device_name(0), noise_floor=floor, seconds=round(seconds),
                    accepted_manifest_sha256=fx.s.sha(fx.ACCEPTED), controls_sha256=fx.s.sha(fx.controls.MANIFEST),
                    silhouettes_sha256=fx.s.sha(fx.s.MANIFEST), python=platform.python_version(),
                    packages=fx.versions(), black_image=dict(directory=str(extra.relative_to(ROOT)) if extra.is_relative_to(ROOT)
                                                             else str(extra), files=black_manifest['files']),
                    files={name: fx.s.sha(output / name)
                           for name in ('features.npz', 'index.csv', 'differences.csv', 'identity.csv')})
    fx.s.write(output / 'manifest.json', manifest)
    return manifest


# ---------- analysis (CPU) ----------

def load(directory=OUTPUT):
    """Families and features as arrays [family, scale, condition, frame, dim]."""
    index = pd.read_csv(directory / 'index.csv')
    with np.load(directory / 'features.npz') as z:
        arrays = {k: z[k] for k in z.files}
    order = list(dict.fromkeys(index.family))
    expected = [(f, s, c, r) for f in order for s in SCALES for c in CONDITIONS for r in fx.FRAMES]
    require(list(zip(index.family, index.scale, index.condition, index.frame)) == expected, 'Unexpected feature order')
    by_id = {f['id']: f for f in fx.families()}
    shape = (len(order), len(SCALES), len(CONDITIONS), len(fx.FRAMES))
    return [by_id[f] for f in order], {k: v.reshape(*shape, -1).astype(np.float64) for k, v in arrays.items()}


def scale_data(families, x, scale):
    """One scale in the layout of baselines.Data: [family, fx.CONDITIONS, frame, dim]. The
    hand_only column (not part of this addendum) is NaN and dropped from every table."""
    k = SCALES.index(scale)
    columns = [fx.CONDITIONS.index(c) for c in CONDITIONS]
    arrays = {}
    for name, v in x.items():
        full = np.full((v.shape[0], len(fx.CONDITIONS)) + v.shape[3:], np.nan)
        full[:, columns] = v[:, k]
        arrays[name] = full
    return SimpleNamespace(x=arrays, families=families, source=np.array([f['reference'] for f in families]),
                           variant=np.array([f['variant'] for f in families]))


def tables(families, x):
    """The declared movement-signal summaries (baselines.signal_tables) and Gate 1 rule, per scale."""
    signal, family = [], []
    hand_only = f'delta_{b.SHORT["hand_only"]}'
    for scale in SCALES:
        table, per_family = b.signal_tables(scale_data(families, x, scale), REPRESENTATIONS)
        signal.append(table[table.statistic != hand_only].assign(scale=scale))
        family.append(per_family[per_family.condition != 'hand_only'].assign(scale=scale))
    signal = pd.concat(signal, ignore_index=True)
    gate = pd.concat([b.gate1(signal[signal.scale == scale]).assign(scale=scale) for scale in SCALES], ignore_index=True)
    return signal, pd.concat(family, ignore_index=True), gate


def reading(gate):
    """The interpretation rule declared in Addendum A, per representation (primary analysis)."""
    out = {}
    for name, g in gate.groupby('representation', sort=False):
        state = {int(r.scale): 'reverse' if r.C_high < 0 else 'positive' if r.C_low > 0 else 'interval includes 0'
                 for r in g.itertuples()}
        if all(v == 'reverse' for v in state.values()):
            verdict = 'reverse at both scales: the reversal does not depend on the hand being small'
        elif any(v == 'positive' for v in state.values()):
            verdict = 'positive at some scale: the Gate 1 conclusion depends on viewing geometry'
        else:
            verdict = 'other: reported per scale, no general claim'
        out[name] = dict(by_scale=state, reading=verdict)
    return out


def identity_table(directory=OUTPUT):
    checks = pd.read_csv(directory / 'identity.csv')
    return (checks.groupby(['scale', 'source']).fb_equals_ul.agg(families_identical='sum', families='count')
            .reset_index())


def analyse(directory=OUTPUT, output=OUTPUT, figure_path=FIGURE):
    families, x = load(directory)
    output.mkdir(parents=True, exist_ok=True)
    signal, family, gate = tables(families, x)
    identity = identity_table(directory)
    differences = pd.read_csv(directory / 'differences.csv')
    results = dict(movement_signal=signal, family_signal=family, gate=gate, fb_ul_identity=identity,
                   differences_summary=differences.groupby(['scale', 'condition'], sort=False)
                   .mean(numeric_only=True).reset_index())
    for name, frame in results.items():
        frame.to_csv(output / f'{name}.csv', index=False)
    by_scale = identity.groupby('scale')[['families_identical', 'families']].sum()
    summary = dict(method='docs/baselines.md, Addendum A (exploratory)', code_sha256=fx.s.sha(__file__),
                   features_manifest_sha256=fx.s.sha(directory / 'manifest.json'),
                   features_files=fx.s.read(directory / 'manifest.json')['files'], families=len(families),
                   sources=len({f['reference'] for f in families}), scales=list(SCALES),
                   bootstrap=dict(samples=b.SAMPLES, seed=b.SEED, unit='source', interval='percentile'),
                   fb_equals_ul={str(s): dict(families_identical=int(r.families_identical), families=int(r.families),
                                              sources_all_identical=int((identity[identity.scale == s].families_identical
                                                                         == identity[identity.scale == s].families).sum()))
                                 for s, r in by_scale.iterrows()},
                   gate=gate.to_dict('records'), reading=reading(gate),
                   files={f'{name}.csv': fx.s.sha(output / f'{name}.csv') for name in results})
    fx.s.write(output / 'summary.json', summary)
    figure(signal, families, figure_path)
    return results, summary


# ---------- figure ----------

def figure(table, families=None, path=FIGURE, gate1=b.OUTPUT / 'gate1.csv'):
    """(a) one family's static frames as the encoders receive them: whole scene (Gate 1, with the
    two windows outlined) and both windows; (b) movement signal per condition and scale; (c) C
    per representation and view."""
    import baseline_figures as bf   # shared style and palette
    plt = bf.plt
    family = next(f for f in families or fx.families() if f['reference'] == 1)
    images = fx.Renderer().render(family)
    shown = views(images, family['source'], SCALES, fx.MAIN)
    order = sorted(SCALES, reverse=True)   # field of view shrinks downwards, so the hand grows
    fig = plt.figure(figsize=(16.5, 12.6))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.2], hspace=.02)
    left, right = top.subfigures(1, 2, width_ratios=[1.15, 1], wspace=.02)

    axes = left.subplots(1 + len(order), len(fx.MAIN), gridspec_kw=dict(wspace=.04, hspace=.06))
    scale_to_224 = SIZE / 2048
    for i, scale in enumerate([None] + order):
        for j, condition in enumerate(fx.MAIN):
            ax = axes[i, j]
            image = images[condition][0] if scale is None else shown[scale][condition][0]
            ax.imshow(fx.resized(image, SIZE))
            if scale is None:
                for s, style in zip(SCALES, ('-', '--')):
                    l, t, side = window(family['source']['used_hand_box'], s)
                    ax.add_patch(plt.Rectangle((l * scale_to_224, t * scale_to_224), side * scale_to_224,
                                               side * scale_to_224, fill=False, ec='white', lw=1, ls=style))
                ax.set_xlim(-.5, SIZE - .5), ax.set_ylim(SIZE - .5, -.5)
            if i == 0:
                ax.set_title(b.SHORT[condition], loc='center', fontsize=8.5)
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        label = 'whole scene (Gate 1)' if scale is None else f'window s = {scale}'
        axes[i, 0].set_ylabel(label, fontsize=8.5)
    left.suptitle(f'(a) Static frame as the encoders receive it (224 px); reference 1, {family["desk"]} desk, '
                  f'seed {family["seed"]}, {family["balance"]} balance.\n     Top row: window s = 2 solid, '
                  's = 4 dashed (larger than the image); black outside the image',
                  x=.01, ha='left', fontweight='bold', fontsize=9)

    ax = right.subplots()
    primary = table.query("measure == 'normalised' and cohort == 'all' and "
                          "detached_hand == 'detached_hand_silhouette' and statistic == 'C'")
    names = list(REPRESENTATIONS)
    series = [(f'window s = {s}', primary[primary.scale == s].set_index('representation'), mk, col)
              for s, mk, col in zip(order, ('s', 'D'), (bf.MUTED, bf.INK))]
    if Path(gate1).is_file():
        reference = pd.read_csv(gate1).set_index('representation').rename(columns=dict(C='estimate', C_low='ci_low',
                                                                                       C_high='ci_high'))
        series.insert(0, ('whole scene (Gate 1)', reference, 'o', '#a3a29e'))
    offsets = np.linspace(-.24, .24, len(series))
    for (label, g, marker, color), offset in zip(series, offsets):
        for i, name in enumerate(names):
            r = g.loc[name]
            ax.plot([r.ci_low, r.ci_high], [i + offset] * 2, color=color, lw=2, solid_capstyle='round')
            ax.plot(r.estimate, i + offset, marker=marker, color=color, markersize=6.5, markeredgecolor=bf.SURFACE,
                    markeredgewidth=1.2, ls='', label=label if i == 0 else None)
    ax.axvline(0, color=bf.MUTED, lw=1, zorder=1)
    ax.set_yticks(range(len(names)), [b.LABEL[n] for n in names])
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('C = (FB + UL)/2 - DH, normalised movement signal (95% source-bootstrap interval)')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=len(series), fontsize=8, handletextpad=.3)
    right.suptitle('(c) Context contrast C per view (primary: silhouette DH, all 20 sources)', x=.02, ha='left',
                   fontweight='bold', fontsize=9)

    t = table.query("measure == 'normalised' and cohort == 'all' and detached_hand == 'detached_hand_silhouette'")
    axes = bottom.subplots(len(order), len(names), sharey='col', gridspec_kw=dict(hspace=.12, wspace=.3))
    ceiling = {}
    for k, scale in enumerate(order):
        for ax, name in zip(axes[k], names):
            g = t[(t.representation == name) & (t.scale == scale)].set_index('statistic')
            rows = [g.loc[f'delta_{b.SHORT[c]}'] for c in CONDITIONS]
            bf.dots(ax, [b.SHORT[c] for c in CONDITIONS], [r.estimate for r in rows], [r.ci_low for r in rows],
                    [r.ci_high for r in rows], [bf.ROLE[bf.role(c)][0] for c in CONDITIONS],
                    [bf.ROLE[bf.role(c)][1] for c in CONDITIONS])
            ceiling[name] = max(ceiling.get(name, 0), max(r.ci_high for r in rows))
            ax.set_ylim(0, 1.06 * ceiling[name])   # shared within a column: both scales on one axis
            if k == 0:
                ax.set_title(b.LABEL[name])
                ax.tick_params(labelbottom=False)
        axes[k, 0].set_ylabel(f'window s = {scale}\ndistance / between-source spread')
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle='', markersize=6, label=r) for r, (c, m) in bf.ROLE.items()]
    bottom.legend(handles=handles, loc='upper right', bbox_to_anchor=(.99, 1), ncol=len(handles), fontsize=8)
    bottom.suptitle('(b) Static -> movement distance per display in the hand-centred windows; mean over 20 sources, '
                    '95% source-bootstrap interval', x=.01, ha='left', fontweight='bold', fontsize=9)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('extract', help='GPU: render, crop and encode')
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--extra', type=Path, default=EXTRA, help='Where the black fixation image is written')
    p.add_argument('--limit', type=int, help='First N appearance variants only, all 20 sources each (smoke test)')
    p.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    for name, text in (('analyse', 'CPU: tables, summary and figure'), ('figure', 'CPU: redraw the figure only')):
        p = commands.add_parser(name, help=text)
        p.add_argument('--features', type=Path, default=OUTPUT)
        p.add_argument('--output', type=Path, default=OUTPUT)
        p.add_argument('--figure', type=Path, default=FIGURE)
    args = parser.parse_args()
    if args.command == 'extract':
        result = extract(args.output, args.extra, args.limit, args.workers)
        print(json.dumps({k: v for k, v in result.items() if k in ('images', 'seconds', 'models', 'noise_floor', 'files')},
                         indent=2))
    elif args.command == 'analyse':
        results, summary = analyse(args.features, args.output, args.figure)
        pd.set_option('display.width', 220)
        print(results['gate'].round(3).to_string(index=False))
        print(json.dumps(dict(fb_equals_ul=summary['fb_equals_ul'], reading=summary['reading']), indent=2))
    else:
        figure(pd.read_csv(args.output / 'movement_signal.csv'), path=args.figure)
