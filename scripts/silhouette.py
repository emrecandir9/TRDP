"""Paper-matched detached-hand silhouettes: the hand on black, separated at the wrist.

Matamala-Gomez et al. (2025) show the detached hand as an isolated hand on a black
background. This module builds one soft hand matte per frame for each frozen V7 source:

- Segmentation: the pinned SAM-HQ model segments the final static and movement frames
  themselves (scripts/segment_hands.py, data/hand_segmentation.json). The earlier reviewed
  matte was drawn on a pre-repair image and can clip fingertips of the final frames.
- Prompts for the movement frame come from a geometric reconstruction (movement_mask): the
  index shaft moved rigidly by the rotation that built the movement, re-fitted by texture
  registration, with a bending root and a limited colour model at the edges.
- The movement matte equals the static matte outside the changed region (dilated by 6 px),
  so stationary fingers keep identical outlines in both frames.
- Edges are anti-aliased (Gaussian, sigma 0.7 px). Both frames are cut at the wrist: a line
  through landmark 0, perpendicular to the wrist -> middle-knuckle axis, with a 2 px ramp.

Condition pixels are round(full_body * alpha / 255), so every visible pixel comes from the
full-body image and everything else is black.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
MANIFEST = DATA / 'silhouettes.json'
SEGMENTATION = DATA / 'hand_segmentation.json'
ARCHIVE_MANIFEST = ROOT / 'docs/accepted-silhouette.json'
CONTEXT = 'detached_hand_silhouette'
FRAMES = ('static', 'movement')
PARAMETERS = dict(shaft_start_px=40, shaft_width_section_px=[80, 140], root_bend_step_px=2,
                  colour_band_px=3, prior_pull_px=4, web_reach_px=15, histogram_bins=24,
                  desk_sample_reach_px=120, colour_confidence=1.0, boundary_smoothing_sigma_px=1.5,
                  root_closing_px=3, ghost_zone_px=8, ghost_opening_px=2,
                  movement_zone_px=6, edge_spread_px=3, seam_closing_px=2, edge_sigma_px=0.7, wrist_ramp_px=2)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sources():
    return read(DATA / 'frozen_sources.json')['sources']


def rgb(path):
    with Image.open(path) as image:
        require(image.mode == 'RGB' and image.size == (2048, 2048), f'Expected 2048-square RGB: {path}')
        return np.array(image)


def apply(full, alpha):
    """Condition pixels: the full-body image weighted by the matte, black elsewhere."""
    return np.rint(full.astype(np.float64) * alpha[..., None] / 255).astype(np.uint8)


def rotation(degrees):
    r = np.deg2rad(degrees)
    return np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])


def place(mask, pivot, degrees, dx, dy):
    """Rotate about pivot (x, y), then translate; nearest-grid resampling of a float mask."""
    inverse = rotation(degrees).T
    matrix = np.array([[inverse[1, 1], inverse[1, 0]], [inverse[0, 1], inverse[0, 0]]])
    centre = np.array([pivot[1], pivot[0]])
    return nd.affine_transform(mask.astype(float), matrix, offset=centre - matrix @ (centre + [dy, dx]), order=1)


def lab(image):
    x = image.astype(float) / 255
    x = np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
    xyz = x @ np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]]).T
    f = xyz / np.array([.9505, 1, 1.089])
    f = np.where(f > .008856, np.cbrt(f), 7.787 * f + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def skin_logit(image, hand, changed):
    """Log-likelihood ratio skin vs desk from smoothed Lab histograms of unchanged pixels."""
    bins, low, high = PARAMETERS['histogram_bins'], np.array([0, -20, -10]), np.array([100, 50, 60])
    colours = lab(image)
    index = tuple(np.moveaxis(np.clip(((colours - low) / (high - low) * bins).astype(int), 0, bins - 1), -1, 0))
    def density(selected):
        counts = np.zeros((bins,) * 3)
        np.add.at(counts, tuple(i[selected] for i in index), 1)
        counts = nd.gaussian_filter(counts, 1) + 1e-3
        return counts / counts.sum()
    train = ~nd.binary_dilation(changed, iterations=4)
    near = nd.distance_transform_edt(hand < 5) < PARAMETERS['desk_sample_reach_px']
    skin, desk = density(train & (hand > 250)), density(train & (hand < 5) & near)
    return nd.gaussian_filter(np.log(skin[index]) - np.log(desk[index]), 1.)


def signed_distance(mask):
    return nd.distance_transform_edt(mask) - nd.distance_transform_edt(~mask)


def decide(logit, prior, region):
    """Confident colour evidence decides inside region, pulled weakly toward the geometric prior."""
    band, pull = PARAMETERS['colour_band_px'], PARAMETERS['prior_pull_px']
    region = region & (np.abs(logit) >= PARAMETERS['colour_confidence'])
    mask = np.where(region, logit + np.clip(signed_distance(prior), -band, band) / pull > 0, prior)
    mask = nd.binary_opening(mask, iterations=1)
    labels, count = nd.label(mask)
    require(count > 0, 'Empty hand mask')
    mask = labels == 1 + np.argmax(nd.sum(mask, labels, range(1, count + 1)))
    return nd.binary_fill_holes(mask)


def register(static, movement, shaft, pivot, degrees):
    """Rigid fit of the static shaft texture to the movement frame: (mean abs RGB error, angle, dx, dy)."""
    ys, xs = np.nonzero(nd.binary_erosion(shaft, iterations=3))
    reference, points = static[ys, xs].astype(float), np.stack([xs, ys], 1) - pivot
    channels = [movement[..., c].astype(float) for c in range(3)]
    def cost(angle, dx, dy):
        q = points @ rotation(angle).T + pivot + [dx, dy]
        sampled = np.stack([nd.map_coordinates(c, [q[:, 1], q[:, 0]], order=1, mode='nearest') for c in channels], 1)
        return float(np.abs(sampled - reference).mean())
    best = min((cost(t, dx, dy), t, dx, dy) for t in np.arange(degrees - 2, degrees + 2.01, .25)
               for dx in range(-8, 9, 2) for dy in range(-8, 9, 2))
    # Pattern search: move to any better neighbour; halve the steps when none improves.
    step_angle, step_shift = .25, 2.
    while step_angle >= .005:
        _, t0, x0, y0 = best
        candidate = min((cost(t0 + i * step_angle, x0 + j * step_shift, y0 + k * step_shift),
                         t0 + i * step_angle, x0 + j * step_shift, y0 + k * step_shift)
                        for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1))
        if candidate[0] < best[0] - 1e-9:
            best = candidate
        else:
            step_angle, step_shift = step_angle / 2, step_shift / 2
    return tuple(float(v) for v in best)


def movement_mask(static, movement, hand, case, crop):
    """Binary movement-frame hand mask in crop coordinates, plus diagnostics."""
    x, y = crop[:2]
    start, (low, high) = PARAMETERS['shaft_start_px'], PARAMETERS['shaft_width_section_px']
    sp, tp = np.array(case['static_points']) - [x, y], np.array(case['target_points']) - [x, y]
    recorded = case['image_plane_abduction_degrees']
    degrees = next((d for d in (recorded, -recorded)
                    if np.abs((sp[6:9] - sp[5]) @ rotation(d).T + sp[5] - tp[6:9]).max() < 1), None)
    require(degrees is not None, 'Case target landmarks are not a rotation about landmark 5')
    changed = np.any(static != movement, axis=2)
    yy, xx = np.indices(hand.shape)
    axis = (sp[8] - sp[5]) / np.linalg.norm(sp[8] - sp[5])
    along = (xx - sp[5][0]) * axis[0] + (yy - sp[5][1]) * axis[1]
    across = -(xx - sp[5][0]) * axis[1] + (yy - sp[5][1]) * axis[0]
    static_mask = hand > 127
    labels, count = nd.label(static_mask & (along > start) & nd.binary_dilation(changed, iterations=2))
    require(count > 0, 'No index shaft inside the changed region')
    shaft = labels == 1 + np.argmax(nd.sum(np.ones_like(hand), labels, range(1, count + 1)))
    section = shaft & (along > low) & (along < high)
    require(section.sum() > 100, 'Index shaft section too small')
    centre = np.median(across[section])
    half = np.percentile(np.abs(across[section] - centre), 99) + 2
    shaft &= np.abs(across - centre) <= half
    require(shaft.sum() < .2 * static_mask.sum(), 'Index shaft implausibly large')
    fit = register(static, movement, shaft, sp[5], degrees)
    moved = place(shaft, sp[5], *fit[1:]) > .5
    root = changed & static_mask & (along > 0) & (along <= start) & (np.abs(across - centre) <= half)
    bent = np.zeros_like(moved)
    step = PARAMETERS['root_bend_step_px']
    for offset in range(0, start, step):
        piece = root & (along > offset) & (along <= offset + step)
        if piece.any():
            share = (offset + step / 2) / start
            bent |= place(piece, sp[5], fit[1] * share, fit[2] * share, fit[3] * share) > .5
    old = (changed & (along > start) & nd.binary_dilation(shaft, iterations=3)) | root | \
          (changed & static_mask & (along > start) & (np.abs(across - centre) <= half + 1))
    prior = np.where(changed, (static_mask & ~old) | moved | bent, static_mask)
    near_root = (xx - sp[5][0]) ** 2 + (yy - sp[5][1]) ** 2 < 60 ** 2
    prior |= changed & near_root & nd.binary_closing(moved | bent, iterations=1)
    edge = changed & (np.abs(signed_distance(prior)) <= PARAMETERS['colour_band_px'])
    web = changed & prior & ~moved & nd.binary_dilation(old, iterations=PARAMETERS['web_reach_px'])
    mask = decide(skin_logit(movement, hand, changed), prior, edge | web)
    # Regularise the decided boundary: close small notches at the root, then smooth.
    closed = nd.binary_closing(mask, iterations=PARAMETERS['root_closing_px'])
    mask = np.where(near_root, closed, mask)
    mask = nd.gaussian_filter(mask.astype(float), PARAMETERS['boundary_smoothing_sigma_px']) > .5
    mask = np.where(changed, mask, static_mask)
    # Edge pixels of the old index that the movement left unchanged are a ghost outline in the
    # movement frame, not hand: remove thin structures inside the old-finger zone only.
    zone = nd.binary_dilation(old, iterations=PARAMETERS['ghost_zone_px'])
    opened = nd.binary_opening(mask, iterations=PARAMETERS['ghost_opening_px'])
    mask = np.where(zone, opened, mask)
    labels, count = nd.label(mask)
    mask = nd.binary_fill_holes(labels == 1 + np.argmax(nd.sum(mask, labels, range(1, count + 1))))
    edit = changed | (zone & ~mask & static_mask)
    # Validation on the static frame: shrink its hand by 3 px inside the changed region and
    # recover it with the same colour rule; report errors against the reviewed matte.
    shrunk = np.where(changed, nd.binary_erosion(static_mask, iterations=3), static_mask)
    band = changed & (np.abs(signed_distance(shrunk)) <= PARAMETERS['colour_band_px'])
    recovered = np.where(changed, decide(skin_logit(static, hand, changed), shrunk, band), static_mask)
    diagnostics = dict(recorded_degrees=float(degrees), fitted_degrees=fit[1], fitted_shift_px=[fit[2], fit[3]],
                       registration_mean_abs_rgb=fit[0], changed_pixels=int(changed.sum()),
                       colour_decided_pixels=int((edge | web).sum()),
                       static_recovery_errors_before=int((shrunk != static_mask)[changed].sum()),
                       static_recovery_errors_after=int((recovered != static_mask)[changed].sum()))
    diagnostics['ghost_pixels_removed_outside_changed'] = int((edit & ~changed).sum())
    return mask, edit, diagnostics


def wrist_weight(shape, case):
    """1 on the hand side of the wrist line, 0 beyond, linear over the declared ramp."""
    wrist, knuckle = np.array(case['static_points'][0]), np.array(case['static_points'][9])
    axis = (knuckle - wrist) / np.linalg.norm(knuckle - wrist)
    yy, xx = np.indices(shape)
    distance = (xx - wrist[0]) * axis[0] + (yy - wrist[1]) * axis[1]
    ramp = PARAMETERS['wrist_ramp_px']
    return np.clip(distance / ramp + .5, 0, 1), dict(point=wrist.tolist(), normal=axis.tolist(), ramp_px=ramp)


def segmentation(source, frame):
    """Hash-checked SAM-HQ mask of one frame, cleaned to one hand without holes."""
    entry = read(SEGMENTATION)['sources'][source['id']]
    path = DATA / 'inputs' / source['id'] / f'sam_{frame}.png'
    require(path.is_file() and sha(path) == entry['files'][f'sam_{frame}.png'], f'Missing or changed mask: {path}')
    with Image.open(path) as image:
        return clean(np.array(image) > 127)


def disk(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def clean(mask):
    """One hand without holes; hairline seams between touching fingers are closed."""
    mask = nd.binary_closing(mask, disk(PARAMETERS['seam_closing_px']))
    labels, count = nd.label(mask)
    require(count > 0, 'Empty hand mask')
    return nd.binary_fill_holes(labels == 1 + np.argmax(nd.sum(mask, labels, range(1, count + 1))))


def mattes(source, inputs):
    """(static alpha, movement alpha, record) as 2048-square uint8 arrays."""
    case = read(inputs / 'case.json')
    x, y, right, bottom = case['crop']
    directory = DATA / 'sources' / source['source_directory'] / 'full_body'
    static, movement = rgb(directory / 'static.png'), rgb(directory / 'movement.png')
    changed = np.any(static != movement, axis=2)
    zone = nd.binary_dilation(changed, iterations=PARAMETERS['movement_zone_px'])
    masks = [segmentation(source, 'static')]
    masks.append(clean(np.where(zone, segmentation(source, 'movement'), masks[0])))
    soft = [nd.gaussian_filter(m.astype(float), PARAMETERS['edge_sigma_px']) * 255 for m in masks]
    edited = nd.binary_dilation(zone, iterations=PARAMETERS['edge_spread_px'])
    soft[1] = np.where(edited, soft[1], soft[0])
    cut, line = wrist_weight((2048, 2048), case)
    alphas = [np.rint(a * cut).astype(np.uint8) for a in soft]
    require(np.array_equal(alphas[0][~edited], alphas[1][~edited]),
            'Movement matte differs from the static matte outside the changed region')
    window = (slice(y, bottom), slice(x, right))
    outside = np.ones((2048, 2048), bool)
    outside[window] = False
    require(not any(a[outside].any() for a in alphas), 'Segmentation extends outside the hand crop')
    with Image.open(inputs / 'hand_alpha.png') as matte:
        reviewed = np.zeros((2048, 2048), bool)
        reviewed[window] = np.array(matte.convert('L').resize((right - x, bottom - y), Image.Resampling.BOX)) > 127
    kept = cut > .5
    def iou(a, b):
        return float((a & b & kept).sum() / ((a | b) & kept).sum())
    seg = read(SEGMENTATION)['sources'][source['id']]
    areas = [int((a > 127).sum()) for a in alphas]
    record = dict(segmentation=dict(predicted_iou={f: seg[f]['predicted_iou'] for f in FRAMES},
                                    static_iou_with_reviewed_matte=iou(masks[0], reviewed),
                                    movement_iou_with_prompt_mask=seg['movement']['prompt_mask_iou']),
                  prompt_mask_diagnostics=seg['prompt_mask_diagnostics'], changed_pixels=int(changed.sum()),
                  wrist_cut=line, crop=case['crop'], area_px=dict(zip(FRAMES, areas)),
                  movement_to_static_area=areas[1] / areas[0],
                  visible_fraction=dict(zip(FRAMES, [float((a > 0).mean()) for a in alphas])))
    return alphas[0], alphas[1], record


def import_inputs(origin):
    """Copy the hash-bound case.json and hand_alpha.png of each source from the historical folder."""
    for source in sources():
        destination = DATA / 'inputs' / source['id']
        for name, key in (('case.json', 'frozen_case_sha256'), ('hand_alpha.png', 'static_hand_alpha_sha256')):
            path = Path(origin) / source['id'] / name
            require(path.is_file() and sha(path) == source[key], f'Missing or changed historical input: {path}')
            shutil.copy2(path, destination / name)


def review_sheet(records, path):
    tiles = []
    for source in sources():
        inputs = DATA / 'inputs' / source['id']
        directory = DATA / 'sources' / source['source_directory'] / 'full_body'
        x, y, right, bottom = records[source['id']]['crop']
        row = []
        for frame in FRAMES:
            with Image.open(inputs / f'silhouette_{frame}.png') as matte:
                alpha = np.array(matte)
            row.append(apply(rgb(directory / f'{frame}.png'), alpha)[y - 40:bottom + 40, x - 40:right + 40])
        tile = Image.fromarray(np.hstack(row)).resize((560, 280))
        ImageDraw.Draw(tile).text((6, 4), f"r{source['reference']:02d}", fill=(255, 80, 80))
        tiles.append(np.array(tile))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])).save(path, quality=92)
    # Close-ups of the moved index on black, where any matte error would be visible.
    closeups = []
    for source in sources():
        knuckle = np.array(read(DATA / 'inputs' / source['id'] / 'case.json')['static_points'][5]).astype(int)
        full = rgb(DATA / 'sources' / source['source_directory'] / 'full_body/movement.png')
        with Image.open(DATA / 'inputs' / source['id'] / 'silhouette_movement.png') as matte:
            window = (slice(knuckle[1] - 230, knuckle[1] + 50), slice(knuckle[0] - 150, knuckle[0] + 150))
            tile = Image.fromarray(np.hstack([full[window], apply(full, np.array(matte))[window]]))
        ImageDraw.Draw(tile).text((6, 4), f"r{source['reference']:02d}", fill=(255, 80, 80))
        closeups.append(np.array(tile))
    Image.fromarray(np.vstack([np.hstack(closeups[i:i + 4]) for i in range(0, len(closeups), 4)])).save(
        path.with_name('movement_closeups.jpg'), quality=92)


def build(origin, sheet):
    if origin:
        import_inputs(origin)
    records = {}
    for source in sources():
        inputs = DATA / 'inputs' / source['id']
        require(sha(inputs / 'case.json') == source['frozen_case_sha256'] and
                sha(inputs / 'hand_alpha.png') == source['static_hand_alpha_sha256'],
                f"Import historical inputs first: {source['id']}")
        alphas = mattes(source, inputs)
        files = {}
        for frame, alpha in zip(FRAMES, alphas[:2]):
            Image.fromarray(alpha).save(inputs / f'silhouette_{frame}.png')
            files[f'silhouette_{frame}.png'] = sha(inputs / f'silhouette_{frame}.png')
        records[source['id']] = dict(alphas[2], files=files)
        print(source['id'], 'movement/static area %.4f' % alphas[2]['movement_to_static_area'],
              'IoU with reviewed matte %.4f' % alphas[2]['segmentation']['static_iou_with_reviewed_matte'], flush=True)
    write(MANIFEST, dict(method=__doc__.strip().splitlines()[0], parameters=PARAMETERS,
                         code_sha256=sha(__file__), frozen_sources_sha256=sha(DATA / 'frozen_sources.json'),
                         hand_segmentation_sha256=sha(SEGMENTATION),
                         visual_review='pending', sources=records))
    recipe = read(ROOT / 'config/recipe.json')
    recipe['silhouettes_sha256'] = sha(MANIFEST)
    write(ROOT / 'config/recipe.json', recipe)
    review_sheet(records, sheet)
    return records


def load(source, frame):
    """Hash-checked matte for one source and frame."""
    entry = read(MANIFEST)['sources'][source['id']]
    path = DATA / 'inputs' / source['id'] / f'silhouette_{frame}.png'
    require(path.is_file() and sha(path) == entry['files'][f'silhouette_{frame}.png'], f'Missing or changed matte: {path}')
    with Image.open(path) as image:
        require(image.mode == 'L' and image.size == (2048, 2048), f'Invalid matte: {path}')
        return np.array(image)


def check_source(source, static, movement):
    """Invariants of the frozen mattes against the frozen source pixels."""
    entry = read(MANIFEST)['sources'][source['id']]
    alphas = [load(source, frame) for frame in FRAMES]
    manifest = read(MANIFEST)
    require(manifest['hand_segmentation_sha256'] == sha(SEGMENTATION), 'Mattes were built from other segmentations')
    reach = manifest['parameters']['movement_zone_px'] + manifest['parameters']['edge_spread_px']
    edited = nd.binary_dilation(np.any(static != movement, axis=2), iterations=reach)
    require(np.array_equal(alphas[0][~edited], alphas[1][~edited]),
            'Mattes differ where the frames are identical')
    x, y, right, bottom = entry['crop']
    outside = np.ones((2048, 2048), bool)
    outside[y:bottom, x:right] = False
    cut = entry['wrist_cut']
    yy, xx = np.indices((2048, 2048))
    beyond = (xx - cut['point'][0]) * cut['normal'][0] + (yy - cut['point'][1]) * cut['normal'][1] < -cut['ramp_px']
    for alpha in alphas:
        require(not alpha[outside | beyond].any(), 'Matte extends outside its crop or past the wrist')
        require((alpha > 127).sum() > 50000, 'Matte implausibly small')
    return alphas


def export_archive(archive, destination):
    """Silhouette images for every accepted family, from its own full-body pixels."""
    accepted = read(ROOT / 'docs/accepted-dataset.json')
    by_id = {s['id']: s for s in sources()}
    families = {}
    for index, family in enumerate(accepted['families'], 1):
        source = by_id[family['source_id']]
        alphas = [load(source, frame) for frame in FRAMES]
        folder = Path(destination) / family['id'] / CONTEXT
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for frame, alpha in zip(FRAMES, alphas):
            full = Path(archive) / family['id'] / 'full_body' / f'{frame}.png'
            require(sha(full) == family['condition_png_sha256'][f'full_body/{frame}.png'], f'Changed archive image: {full}')
            Image.fromarray(apply(rgb(full), alpha)).save(folder / f'{frame}.png')
            files[f'{CONTEXT}/{frame}.png'] = sha(folder / f'{frame}.png')
        families[family['id']] = dict(source_id=source['id'], files=files)
        if index % 48 == 0:
            print(f'Exported {index}/{len(accepted["families"])} families', flush=True)
    write(ARCHIVE_MANIFEST, dict(context=CONTEXT, rule='round(full_body * alpha / 255) per channel',
                                 accepted_manifest_sha256=sha(ROOT / 'docs/accepted-dataset.json'),
                                 silhouettes_sha256=sha(MANIFEST), families_count=len(families),
                                 condition_pngs=2 * len(families), visual_review='pending', families=families))


def verify_archive(archive, destination):
    manifest = read(ARCHIVE_MANIFEST)
    require(manifest['silhouettes_sha256'] == sha(MANIFEST), 'Archive silhouettes were built from other mattes')
    require(manifest['accepted_manifest_sha256'] == sha(ROOT / 'docs/accepted-dataset.json'), 'Accepted manifest changed')
    by_id = {s['id']: s for s in sources()}
    for family_id, family in manifest['families'].items():
        source = by_id[family['source_id']]
        for frame in FRAMES:
            path = Path(destination) / family_id / CONTEXT / f'{frame}.png'
            require(sha(path) == family['files'][f'{CONTEXT}/{frame}.png'], f'Changed silhouette image: {path}')
            expected = apply(rgb(Path(archive) / family_id / 'full_body' / f'{frame}.png'), load(source, frame))
            require(np.array_equal(rgb(path), expected), f'Silhouette pixels disagree with full body: {path}')
    return len(manifest['families'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('build', 'check', 'export-archive', 'verify-archive'))
    parser.add_argument('--from', dest='origin', type=Path,
                        help='Historical frozen_v4_inputs folder; copies case.json and hand_alpha.png (build only)')
    parser.add_argument('--archive', type=Path, default=DATA / 'accepted')
    parser.add_argument('--destination', type=Path, default=DATA / 'accepted_silhouette')
    parser.add_argument('--sheet', type=Path, default=ROOT / 'output/silhouette_review/sources.jpg')
    args = parser.parse_args()
    if args.command == 'build':
        build(args.origin, args.sheet)
        print(f'Wrote 40 mattes, {MANIFEST.relative_to(ROOT)} and review sheet {args.sheet}')
    elif args.command == 'check':
        for source in sources():
            directory = DATA / 'sources' / source['source_directory'] / 'full_body'
            check_source(source, rgb(directory / 'static.png'), rgb(directory / 'movement.png'))
        print('Verified 40 silhouette mattes against hashes, frozen frames, crops and wrist cuts.')
    elif args.command == 'export-archive':
        export_archive(args.archive, args.destination)
        print(f'Wrote 960 silhouette PNGs and {ARCHIVE_MANIFEST.relative_to(ROOT)}')
    else:
        print(f'Verified {verify_archive(args.archive, args.destination)} families of silhouette images.')
