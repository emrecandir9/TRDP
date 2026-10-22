"""Segment the hand and forearm in each frozen V7 static frame with the pinned SAM-HQ model (CPU).

The control stimuli (scripts/controls.py) need the forearm below the wrist, which lies outside
the hand crop used by segment_hands.py. The forearm is identical in the static and movement
frames, so only the static frame is segmented. Prompts: six hand landmarks (wrist, thumb base,
index, middle, ring and little knuckles) as foreground; desk points beside the forearm and in
the top corners as background; and a box from the top of the hand mask to the desk edge.
Results go to data/arm_segmentation.json, separate from the approved hand segmentation.

Environment: pip install -r requirements-segmentation.txt.
"""
import argparse
import importlib.metadata
import platform

import numpy as np
from PIL import Image

import segment_hands
import silhouette as s

MANIFEST = s.DATA / 'arm_segmentation.json'
PROMPTS = dict(landmarks=[0, 1, 5, 9, 13, 17], box_left_margin_px=300, box_right_margin_px=60,
               box_top_margin_px=10, box_bottom_inset_px=5, desk_point_inset_px=15,
               desk_point_below_wrist_px=150, multimask_output=False, hq_token_only=False)


def window(source, hand):
    ys, xs = np.nonzero(hand)
    x0 = max(xs.min() - PROMPTS['box_left_margin_px'], 0)
    x1 = min(xs.max() + PROMPTS['box_right_margin_px'], 2047)
    y0 = max(ys.min() - PROMPTS['box_top_margin_px'], 0)
    y1 = source['upper_limb_cutoff'] - PROMPTS['box_bottom_inset_px']
    return x0, y0, x1, y1


def prompts(case, box):
    x0, y0, x1, y1 = box
    points = np.array(case['static_points'], float)
    positive = points[PROMPTS['landmarks']] - [x0, y0]
    inset, below = PROMPTS['desk_point_inset_px'], PROMPTS['desk_point_below_wrist_px']
    w, h = x1 - x0, y1 - y0
    negative = np.array([[w - inset, h - 20], [w - inset, points[0][1] - y0 + below],
                         [inset, inset], [w - inset, inset]], float)
    return positive, negative, np.array([0, 0, w - 1, h - 1], float)


def run(threads):
    model = segment_hands.predictor(threads)
    records = {}
    for source in s.sources():
        inputs = s.DATA / 'inputs' / source['id']
        case = s.read(inputs / 'case.json')
        hand = s.segmentation(source, 'static')
        box = window(source, hand)
        x0, y0, x1, y1 = box
        image = s.rgb(s.DATA / 'sources' / source['source_directory'] / 'full_body/static.png')[y0:y1, x0:x1]
        positive, negative, crop_box = prompts(case, box)
        mask, score = segment_hands.segment(model, image, positive, negative, crop_box)
        full = np.zeros((2048, 2048), np.uint8)
        full[y0:y1, x0:x1] = mask.astype(np.uint8) * 255
        Image.fromarray(full).save(inputs / 'sam_arm.png')
        records[source['id']] = dict(window=list(map(int, box)), predicted_iou=score,
                                     positive_points=positive.round(2).tolist(),
                                     negative_points=negative.round(2).tolist(),
                                     files={'sam_arm.png': s.sha(inputs / 'sam_arm.png')})
        print(source['id'], 'predicted IoU %.3f' % score, flush=True)
    s.write(MANIFEST, dict(model=dict(segment_hands.CHECKPOINT, code=segment_hands.CODE, variant='vit_b'),
                           prompts=PROMPTS, frame='static (forearm identical in the movement frame)',
                           runtime=dict(python=platform.python_version(), platform=platform.platform(), threads=threads,
                                        **{p: importlib.metadata.version(p) for p in ('torch', 'timm', 'numpy')}),
                           code_sha256=s.sha(__file__), hand_segmentation_sha256=s.sha(s.SEGMENTATION),
                           sources=records))
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--threads', type=int, default=4)
    run(parser.parse_args().threads)
    print(f'Wrote 20 SAM-HQ arm masks and {MANIFEST.relative_to(s.ROOT)}')
