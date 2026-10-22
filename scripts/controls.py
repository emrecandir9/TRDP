"""Control stimuli of analysis spec §7.2: masks over frozen full-body pixels.

Each control varies one property of the visible context while the hand, and the index movement,
stay pixel-identical to the other conditions. Lengths scale with each source's palm width P
(index to little-finger knuckle, landmarks 5 and 17). The wrist line is the silhouette's: through
landmark 0, perpendicular to the wrist -> middle-knuckle axis; s is the signed distance from it
(positive towards the fingers).

- hand_stub: the detached-hand silhouette plus 1 P of forearm on black: the hand attached to
  a short arm segment, with no body and no desk.
- hand_gap_stub: the silhouette plus a forearm segment left in place but starting 0.5 P below
  the wrist, on black, so black background shows through the missing wrist (missing-wrist
  condition of Tieri et al., 2015). The segment length is set so its visible area equals
  hand_stub's; the pair differs only in connection.
- window_hand / window_arm: two rectangles aligned with the hand axis, both containing the whole
  hand, its movement and the surrounding desk, with equal visible area. window_arm extends 1 P
  down the arm; window_hand instead extends sideways, over desk on the little-finger side.
- scrambled: the upper-limb view divided into 32 px blocks; blocks touching the hand or the
  movement (6 px guard) stay in place, all other blocks above the desk edge are permuted, so the
  image keeps exactly the upper-limb pixels while the arm is scattered.
- occluded_wrist (exploratory): the upper-limb view with an opaque black band over 0.5 P of the
  wrist. An occluder lets the arm be completed behind it (amodal completion), so this is an
  occlusion control, not a discontinuity control.

Condition pixels are round(full_body * alpha / 255) for the mask controls and a fixed block
permutation of the upper-limb view for the scrambled control. The same masks and permutation
apply to every appearance variant of a source and to both frames.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd

import silhouette as s

MANIFEST = s.DATA / 'controls.json'
ARMS = s.DATA / 'arm_segmentation.json'
CONTROLS = ('hand_stub', 'hand_gap_stub', 'window_hand', 'window_arm', 'scrambled', 'occluded_wrist')
PARAMETERS = dict(gap_length_palms=0.5, stub_length_palms=1.0, window_extension_palms=1.0,
                  window_margin_px=20, window_area_tolerance=0.005, gap_arm_dilation_px=3,
                  block_px=32, hand_guard_px=6, edge_sigma_px=0.7, cut_ramp_px=2)
FILES = dict(occluded_wrist=('control_occluded_wrist.png',),
             hand_stub=('control_hand_stub_static.png', 'control_hand_stub_movement.png'),
             hand_gap_stub=('control_hand_gap_stub_static.png', 'control_hand_gap_stub_movement.png'),
             window_hand=('control_window_hand.png',), window_arm=('control_window_arm.png',),
             scrambled=('control_scrambled.npy',))
require = s.require


def geometry(case):
    """Wrist point, unit axis towards the fingers, perpendicular axis, palm width and grids s, v."""
    points = np.array(case['static_points'], float)
    wrist = points[0]
    axis = (points[9] - wrist) / np.linalg.norm(points[9] - wrist)
    across = np.array([-axis[1], axis[0]])
    yy, xx = np.indices((2048, 2048))
    along_grid = (xx - wrist[0]) * axis[0] + (yy - wrist[1]) * axis[1]
    across_grid = (xx - wrist[0]) * across[0] + (yy - wrist[1]) * across[1]
    little_side = np.sign((points[17] - wrist) @ across - (points[5] - wrist) @ across)
    return dict(palm=float(np.linalg.norm(points[5] - points[17])), along=along_grid, across=across_grid,
                little_side=float(little_side))


def ramp(distance):
    return np.clip(distance / PARAMETERS['cut_ramp_px'] + .5, 0, 1)


def soft(mask):
    return nd.gaussian_filter(mask.astype(float), PARAMETERS['edge_sigma_px'])


def arm_mask(source):
    entry = s.read(ARMS)['sources'][source['id']]
    path = s.DATA / 'inputs' / source['id'] / 'sam_arm.png'
    require(path.is_file() and s.sha(path) == entry['files']['sam_arm.png'], f'Missing or changed arm mask: {path}')
    with Image.open(path) as image:
        return s.clean(np.array(image) > 127)


def rectangle(g, along_range, across_range):
    return ((g['along'] >= along_range[0]) & (g['along'] <= along_range[1]) &
            (g['across'] >= across_range[0]) & (g['across'] <= across_range[1]))


def windows(g, hand_visible):
    """Equal-area rectangles: hand + 1 P of arm, or hand + the same area of desk sideways.

    hand_visible covers the hand in both frames and every pixel the movement changes.
    """
    margin = PARAMETERS['window_margin_px']
    top = g['along'][hand_visible].max() + margin
    lo, hi = g['across'][hand_visible].min() - margin, g['across'][hand_visible].max() + margin
    extension = PARAMETERS['window_extension_palms'] * g['palm']
    arm = soft(rectangle(g, (-extension, top), (lo, hi))) * 255
    target = arm.sum()
    def hand_window(width):
        side = (lo - width, hi) if g['little_side'] < 0 else (lo, hi + width)
        return soft(rectangle(g, (-PARAMETERS['cut_ramp_px'], top), side)) * 255
    low, high = 0., 4 * extension * (hi - lo) / top
    for _ in range(40):   # bisection on the sideways extension until the areas match
        middle = (low + high) / 2
        low, high = (middle, high) if hand_window(middle).sum() < target else (low, middle)
    hand = hand_window((low + high) / 2)
    return np.rint(hand).astype(np.uint8), np.rint(arm).astype(np.uint8), float((low + high) / 2)


def block_permutation(guard, cutoff, seed):
    """Indices of movable blocks and their shuffled order; blocks touching the guard stay fixed."""
    size = PARAMETERS['block_px']
    rows, cols = cutoff // size, 2048 // size
    touched = guard[:rows * size].reshape(rows, size, cols, size).any(axis=(1, 3))
    movable = np.flatnonzero(~touched.ravel())
    order = np.random.default_rng(seed).permutation(movable)
    return np.stack([movable, order], 1).astype(np.int32), rows, cols


def scramble(image, table, cutoff):
    size = PARAMETERS['block_px']
    cols = 2048 // size
    out = np.zeros_like(image)
    out[:cutoff] = image[:cutoff]
    blocks = lambda index: (slice(index // cols * size, index // cols * size + size),
                            slice(index % cols * size, index % cols * size + size))
    for destination, origin in table:
        out[blocks(destination)] = image[blocks(origin)]
    return out


def build_source(source):
    """Write one source's control files and return its record (checked separately)."""
    inputs = s.DATA / 'inputs' / source['id']
    case = s.read(inputs / 'case.json')
    g = geometry(case)
    directory = s.DATA / 'sources' / source['source_directory'] / 'full_body'
    static, movement = s.rgb(directory / 'static.png'), s.rgb(directory / 'movement.png')
    action = np.any(static != movement, axis=2)
    hands = {f: s.load(source, f) for f in s.FRAMES}
    hand_visible = (hands['static'] > 0) | (hands['movement'] > 0)
    arm = arm_mask(source)
    cutoff = source['upper_limb_cutoff']
    upper_limb = np.zeros((2048, 2048))
    upper_limb[:cutoff] = 1
    palm = g['palm']
    gap = PARAMETERS['gap_length_palms'] * palm
    # The gap starts where the silhouette's wrist ramp ends, so its soft edge never reaches the hand.
    top = -PARAMETERS['cut_ramp_px']
    band = (g['along'] <= top) & (g['along'] >= top - gap) & nd.binary_dilation(arm, iterations=PARAMETERS['gap_arm_dilation_px'])
    occluded_wrist = np.rint(255 * upper_limb * (1 - soft(band))).astype(np.uint8)
    stub_length = PARAMETERS['stub_length_palms'] * palm
    arm_soft = soft(arm)
    segment = lambda start, length: arm_soft * ramp(g['along'] - start + length) * (1 - ramp(g['along'] - start))
    stub = segment(0, stub_length)
    # Same visible area as the stub, but the segment begins one gap below the wrist.
    low, high = 0., 3 * stub_length
    for _ in range(40):
        middle = (low + high) / 2
        low, high = (middle, high) if segment(-gap, middle).sum() < stub.sum() else (low, middle)
    gap_segment_length = (low + high) / 2
    gap_stub = segment(-gap, gap_segment_length)
    hand_stub = {f: np.rint(np.clip(hands[f] + 255 * stub, 0, 255)).astype(np.uint8) for f in s.FRAMES}
    hand_gap_stub = {f: np.rint(np.clip(hands[f] + 255 * gap_stub, 0, 255)).astype(np.uint8) for f in s.FRAMES}
    window_hand, window_arm, sideways = windows(g, hand_visible | action)
    guard = nd.binary_dilation(hand_visible | action, iterations=PARAMETERS['hand_guard_px'])
    table, rows, cols = block_permutation(guard, cutoff, source["reference"])
    arrays = dict(occluded_wrist=[occluded_wrist], hand_stub=[hand_stub['static'], hand_stub['movement']],
                  hand_gap_stub=[hand_gap_stub['static'], hand_gap_stub['movement']],
                  window_hand=[window_hand], window_arm=[window_arm])
    files = {}
    for name, images in arrays.items():
        for filename, image in zip(FILES[name], images):
            Image.fromarray(image).save(inputs / filename)
            files[filename] = s.sha(inputs / filename)
    np.save(inputs / 'control_scrambled.npy', table)
    files['control_scrambled.npy'] = s.sha(inputs / 'control_scrambled.npy')
    areas = {name: float(a.astype(float).sum() / 255) for name, a in (('window_hand', window_hand), ('window_arm', window_arm))}
    record = dict(palm_px=palm, gap_px=gap, stub_px=stub_length, window_extension_px=PARAMETERS['window_extension_palms'] * palm,
                  window_sideways_px=sideways, window_areas_px=areas,
                  window_area_difference=abs(areas['window_hand'] - areas['window_arm']) / areas['window_arm'],
                  gap_segment_px=gap_segment_length, occluded_band_px=int(band.sum()),
                  stub_area_px=float(stub.sum()), gap_stub_area_px=float(gap_stub.sum()),
                  stub_area_difference=abs(float(gap_stub.sum() - stub.sum())) / float(stub.sum()),
                  scrambled_blocks=dict(movable=int(len(table)), fixed=int(rows * cols - len(table)), grid=[rows, cols]),
                  seed=source['reference'], files=files)
    return record


def load(source, name, frame='static'):
    """Hash-checked control matte (uint8) or block table (scrambled)."""
    files = s.read(MANIFEST)['sources'][source['id']]['files']
    filename = FILES[name][0] if len(FILES[name]) == 1 else FILES[name][s.FRAMES.index(frame)]
    path = s.DATA / 'inputs' / source['id'] / filename
    require(path.is_file() and s.sha(path) == files[filename], f'Missing or changed control file: {path}')
    if filename.endswith('.npy'):
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        return np.array(image)


def render(full, source, name, frame):
    """Control image from a (white-balanced) full-body image of any family of this source."""
    require(name in CONTROLS, f'Unknown control: {name}')
    if name == 'scrambled':
        return scramble(full, load(source, name), source['upper_limb_cutoff'])
    return s.apply(full, load(source, name, frame))


def check(source, record, static, movement):
    """Invariants: the hand and its movement are untouched, and each control differs as declared."""
    action = np.any(static != movement, axis=2)
    hands = {f: s.load(source, f) for f in s.FRAMES}
    for name in CONTROLS:
        a, b = (render(image, source, name, f) for image, f in ((static, 'static'), (movement, 'movement')))
        if name in ('hand_stub', 'hand_gap_stub'):
            changed = np.any(a != b, axis=2)
            require(not changed[(hands['static'] == hands['movement']) & ~action].any(), 'Stub frames differ off the hand')
            continue
        require(np.array_equal(np.any(a != b, axis=2), action), f'{name}: static/movement difference is not the action')
        require(np.array_equal(a[action], static[action]) and np.array_equal(b[action], movement[action]),
                f'{name}: action pixels altered')
        if name != 'scrambled':
            require(np.array_equal(a[hands['static'] == 255], static[hands['static'] == 255]),
                    f'{name}: opaque hand pixels altered')
    table = load(source, 'scrambled')
    require(np.array_equal(np.sort(table[:, 0]), np.sort(table[:, 1])), 'Scramble is not a permutation')
    require(record['window_area_difference'] <= PARAMETERS['window_area_tolerance'], 'Window areas differ')
    require(record['stub_area_difference'] <= PARAMETERS['window_area_tolerance'], 'Stub areas differ')


def review_sheet(path):
    tiles = []
    labels = ('upper limb', 'silhouette', 'hand stub', 'hand gap stub', 'window hand', 'window arm', 'scrambled',
              'occluded wrist')
    for source in s.sources():
        directory = s.DATA / 'sources' / source['source_directory'] / 'full_body'
        full = s.rgb(directory / 'static.png')
        upper = full.copy()
        upper[source['upper_limb_cutoff']:] = 0
        images = [upper, s.apply(full, s.load(source, 'static'))] + [
            render(full, source, name, 'static') for name in CONTROLS]
        row = []
        for label, image in zip(labels, images):
            tile = Image.fromarray(image[:1400, 150:1250]).resize((240, 305))
            ImageDraw.Draw(tile).text((5, 5), f"r{source['reference']:02d} {label}", fill=(255, 80, 80))
            row.append(np.array(tile))
        tiles.append(np.hstack(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    for part in range(0, len(tiles), 5):
        Image.fromarray(np.vstack(tiles[part:part + 5])).save(path.with_name(f'{path.stem}_{part // 5 + 1}.jpg'), quality=88)


def manifest(records):
    return dict(method=__doc__.strip().splitlines()[0], parameters=PARAMETERS, controls=list(CONTROLS),
                code_sha256=s.sha(__file__), silhouettes_sha256=s.sha(s.MANIFEST), arm_segmentation_sha256=s.sha(ARMS),
                frozen_sources_sha256=s.sha(s.DATA / 'frozen_sources.json'), visual_review='pending', sources=records)


def frames(source):
    directory = s.DATA / 'sources' / source['source_directory'] / 'full_body'
    return s.rgb(directory / 'static.png'), s.rgb(directory / 'movement.png')


def build(sheet):
    records = {}
    for source in s.sources():
        records[source['id']] = build_source(source)
        s.write(MANIFEST, manifest(records))   # load() inside check() reads the hashes just written
        check(source, records[source['id']], *frames(source))
        r = records[source['id']]
        print(source['id'], 'palm %.0f px, window area difference %.3f%%, movable blocks %d' % (
            r['palm_px'], 100 * r['window_area_difference'], r['scrambled_blocks']['movable']), flush=True)
    review_sheet(sheet)
    return records


def check_all():
    data = s.read(MANIFEST)
    require(data['silhouettes_sha256'] == s.sha(s.MANIFEST) and data['arm_segmentation_sha256'] == s.sha(ARMS),
            'Controls were built from other silhouettes or arm masks')
    for source in s.sources():
        check(source, data['sources'][source['id']], *frames(source))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('build', 'check', 'review'))
    parser.add_argument('--sheet', type=Path, default=s.ROOT / 'output/control_review/controls.jpg')
    args = parser.parse_args()
    if args.command == 'build':
        build(args.sheet)
        print(f'Wrote control files for 20 sources, {MANIFEST.relative_to(s.ROOT)} and review sheets')
    elif args.command == 'check':
        check_all()
        print('Verified 6 controls x 20 sources: hand and movement untouched, areas matched, permutations valid.')
    else:
        review_sheet(args.sheet)
