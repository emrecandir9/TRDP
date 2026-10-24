"""H2 context pathway: visible continuity of the forearm at the wrist (docs/model-method.md §7).

The attachment estimate a of a display is the length of forearm that stays visible without a
break from the wrist line downwards, in palm widths P and capped at 1 P:

    a = min(L, P) / P,   L = number of 1-px cross-sections of the forearm, counted from the
                             wrist line, before the first one that is less than half visible.

"Visible" means the displayed pixel equals the full-body pixel and is not black. The forearm is
the source's SAM-HQ arm mask (data/inputs/*/sam_arm.png) and the wrist line, the hand axis and P
are those of the controls (scripts/controls.py geometry). Only the displayed image and the hand's
geometry enter; no condition label does. The cap at 1 P is the forearm length of `hand_stub`,
which spec §7.2 already treats as an attached arm. The forearm does not move between frames, so a
is computed on the static frame; the build checks that it is identical across the 24 appearance
variants of a source.

The alternative of the amended criterion S3 replaces a by the global visible-area fraction of
the display (Gate 1 `lowlevel` visible_area; 1 for the full body), with the same gate.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
import features as fx  # noqa: E402

OUTPUT = ROOT / 'data/context_continuity.json'
PARAMETERS = dict(cross_section_visible=.5, cap_palms=1.0, step_px=1)
DISPLAYS = tuple(c for c in fx.CONDITIONS if c != 'hand_only')


def continuity(visible, arm, along, palm, parameters=PARAMETERS):
    """Contiguous visible forearm below the wrist line, in palm widths, capped (see module doc).
    visible, arm: boolean images; along: signed distance from the wrist line (positive towards
    the fingers)."""
    limit = int(np.ceil(parameters['cap_palms'] * palm))
    length = 0
    for k in range(0, limit, parameters['step_px']):
        section = arm & (along < -k) & (along >= -k - parameters['step_px'])
        if not section.any() or visible[section].mean() < parameters['cross_section_visible']:
            break
        length = k + parameters['step_px']
    return min(length, parameters['cap_palms'] * palm) / palm


_cache = {}


def family_continuity(family):
    """a for every display of one family (static frame)."""
    source = family['source']
    if source['id'] not in _cache:
        case = fx.s.read(fx.s.DATA / 'inputs' / source['id'] / 'case.json')
        _cache[source['id']] = (fx.controls.geometry(case), fx.controls.arm_mask(source), fx.Renderer())
    g, arm, renderer = _cache[source['id']]
    images = renderer.render(family)
    full = images['full_body'][0]
    lit = full.max(axis=2) > 0
    out = {}
    for name in DISPLAYS:
        visible = np.all(images[name][0] == full, axis=2) & lit
        out[name] = continuity(visible, arm, g['along'], g['palm'])
    return family['id'], family['reference'], out


def build(output=OUTPUT, workers=8):
    families = fx.families()
    with Pool(workers) as pool:
        results = pool.map(family_continuity, families)
    by_source = {}
    for family_id, reference, values in results:
        by_source.setdefault(reference, []).append(values)
    sources = {}
    for reference, rows in sorted(by_source.items()):
        for name in DISPLAYS:
            values = {round(r[name], 6) for r in rows}
            fx.require(len(values) == 1, f'Continuity differs across variants: reference {reference}, {name}: {values}')
        sources[str(reference)] = {name: rows[0][name] for name in DISPLAYS}
    record = dict(method=__doc__.strip().splitlines()[0], parameters=PARAMETERS, displays=list(DISPLAYS),
                  code_sha256=fx.s.sha(__file__), controls_sha256=fx.s.sha(fx.controls.MANIFEST),
                  arm_segmentation_sha256=fx.s.sha(fx.controls.ARMS), accepted_manifest_sha256=fx.s.sha(fx.ACCEPTED),
                  families_checked=len(results), variants_identical=True, sources=sources,
                  mean={name: float(np.mean([s[name] for s in sources.values()])) for name in DISPLAYS})
    fx.s.write(output, record)
    return record


def load(path=OUTPUT):
    """{reference: {display: a}}; the black fixation display has no forearm, a = 0."""
    record = fx.s.read(path)
    fx.require(record['code_sha256'] == fx.s.sha(__file__), 'Continuity table was built by other code')
    return {int(k): v for k, v in record['sources'].items()}


def gate(a, c):
    """H2 gate g = c + (1 - c) a: c is the residual gain of a fully disconnected hand, the one
    parameter fitted to the human DH/FB ratio on training sources."""
    return c + (1 - c) * np.asarray(a, float)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=8)
    record = build(workers=parser.parse_args().workers)
    print(json.dumps(record['mean'], indent=2))
