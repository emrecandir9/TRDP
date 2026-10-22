"""Segment the hand in each frozen V7 frame with the pinned SAM-HQ model (CPU).

The reviewed hand matte was drawn on an earlier image version (before the V5 hand repair),
so its outline can clip fingertips of the final frames. Here SAM-HQ segments the final static
and movement frames themselves. Prompts come from the earlier masks: the 21 hand landmarks
(static or target positions) as foreground points, background points sampled in the gaps
between fingers, the crop corners, and a box around the hand. For the movement frame the
prompt mask is the geometric reconstruction of silhouette.movement_mask.

Environment: pip install -r requirements-segmentation.txt (torch CPU, timm, SAM-HQ at a pinned
commit). The checkpoint is downloaded at a pinned revision and checked against its SHA-256.
"""
import argparse
import importlib.metadata
import platform

import numpy as np
from PIL import Image
from scipy import ndimage as nd

import silhouette as s

CHECKPOINT = dict(repository='lkeab/hq-sam', filename='sam_hq_vit_b.pth',
                  revision='09b02a333b37772133eff3997bdba997867374b7',
                  sha256='14a9d662cd6f5a9c2dba6d40ab0058d88d287e4a18fd6fdc6ad5fb1a3fdeaa57')
CODE = 'git+https://github.com/SysCV/sam-hq.git@e696978d60352dc9a26b12631cd91781502c6546'
MANIFEST = s.DATA / 'hand_segmentation.json'
PROMPTS = dict(gap_points=24, gap_min_distance_px=4, gap_hull_closing_px=25, corner_inset_px=10,
               box_margin_px=15, landmark_erosion_px=2, multimask_output=False, hq_token_only=False)


def predictor(threads):
    import torch
    from huggingface_hub import hf_hub_download
    from segment_anything import SamPredictor, sam_model_registry
    torch.set_num_threads(threads)
    path = hf_hub_download(CHECKPOINT['repository'], filename=CHECKPOINT['filename'], revision=CHECKPOINT['revision'])
    s.require(s.sha(path) == CHECKPOINT['sha256'], 'SAM-HQ checkpoint hash mismatch')
    model = sam_model_registry['vit_b'](checkpoint=None)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return SamPredictor(model.eval())


def prompts(prior, landmarks, rng):
    """Foreground landmarks inside the prior, background gap and corner points, and a box."""
    inside = nd.binary_erosion(prior, iterations=PROMPTS['landmark_erosion_px'])
    xy = np.rint(landmarks).astype(int)
    valid = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < prior.shape[1]) & (xy[:, 1] < prior.shape[0])
    positive = landmarks[valid][inside[xy[valid, 1], xy[valid, 0]]]
    hull = nd.binary_fill_holes(nd.binary_closing(prior, iterations=PROMPTS['gap_hull_closing_px']))
    gaps = hull & ~prior & (nd.distance_transform_edt(~prior) >= PROMPTS['gap_min_distance_px'])
    ys, xs = np.nonzero(gaps)
    pick = rng.choice(len(ys), size=min(PROMPTS['gap_points'], len(ys)), replace=False) if len(ys) else []
    h, w = prior.shape
    inset = PROMPTS['corner_inset_px']
    corners = np.array([[inset, inset], [w - inset, inset], [inset, h - inset], [w - inset, h - inset]], float)
    corners = corners[~prior[corners[:, 1].astype(int), corners[:, 0].astype(int)]]
    negative = np.vstack([np.stack([xs[pick], ys[pick]], 1).astype(float), corners]) if len(pick) else corners
    ys, xs = np.nonzero(prior)
    margin = PROMPTS['box_margin_px']
    box = np.array([max(xs.min() - margin, 0), max(ys.min() - margin, 0), min(xs.max() + margin, w - 1),
                    min(ys.max() + margin, h - 1)], float)
    return positive, negative, box


def segment(model, image, positive, negative, box):
    model.set_image(image)
    points = np.vstack([positive, negative])
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    masks, scores, _ = model.predict(point_coords=points, point_labels=labels, box=box,
                                     multimask_output=PROMPTS['multimask_output'],
                                     hq_token_only=PROMPTS['hq_token_only'])
    return masks[0], float(scores[0])


def run(threads):
    model = predictor(threads)
    records = {}
    for source in s.sources():
        inputs = s.DATA / 'inputs' / source['id']
        case = s.read(inputs / 'case.json')
        x, y, right, bottom = case['crop']
        window = (slice(y, bottom), slice(x, right))
        directory = s.DATA / 'sources' / source['source_directory'] / 'full_body'
        static, movement = s.rgb(directory / 'static.png')[window], s.rgb(directory / 'movement.png')[window]
        with Image.open(inputs / 'hand_alpha.png') as matte:
            hand = np.array(matte.convert('L').resize((right - x, bottom - y), Image.Resampling.BOX))
        geometric, _, diagnostics = s.movement_mask(static, movement, hand, case, (x, y))
        rng = np.random.default_rng(source['reference'])
        record = dict(prompt_mask_diagnostics=diagnostics, files={})
        for frame, image, prior, key in (('static', static, hand > 127, 'static_points'),
                                          ('movement', movement, geometric, 'target_points')):
            positive, negative, box = prompts(prior, np.array(case[key], float) - [x, y], rng)
            mask, score = segment(model, image, positive, negative, box)
            full = np.zeros((2048, 2048), np.uint8)
            full[window] = mask.astype(np.uint8) * 255
            Image.fromarray(full).save(inputs / f'sam_{frame}.png')
            record['files'][f'sam_{frame}.png'] = s.sha(inputs / f'sam_{frame}.png')
            record[frame] = dict(predicted_iou=score, positive_points=positive.round(2).tolist(),
                                 negative_points=negative.round(2).tolist(), box=box.tolist(),
                                 prompt_mask_iou=float((mask & prior).sum() / (mask | prior).sum()))
        records[source['id']] = record
        print(source['id'], 'predicted IoU %.3f / %.3f' % (record['static']['predicted_iou'],
                                                          record['movement']['predicted_iou']), flush=True)
    import torch
    s.write(MANIFEST, dict(model=dict(CHECKPOINT, code=CODE, variant='vit_b'), prompts=PROMPTS,
                           runtime=dict(python=platform.python_version(), platform=platform.platform(), threads=threads,
                                        **{p: importlib.metadata.version(p) for p in ('torch', 'timm', 'numpy', 'scipy')}),
                           code_sha256=s.sha(__file__), silhouette_code_sha256=s.sha(s.__file__),
                           crop_coordinates='mask stored in full 2048 frame; values only inside the hand crop',
                           sources=records))
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    run(args.threads)
    print(f'Wrote 40 SAM-HQ masks and {MANIFEST.relative_to(s.ROOT)}')
