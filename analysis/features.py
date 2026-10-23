"""Image features for the Gate 1 baselines of analysis spec §6.1 (method: docs/baselines.md).

Every condition image is rendered from the frozen, hash-checked archive: the three archive
contexts, the paper-matched silhouette, the six controls of spec §7.2 and the hand-only input
(silhouette cropped to the source's used_hand_box and padded to a square). For each image this
writes low-level statistics, a 64 x 64 luminance vector, DINOv2 ViT-L/14 summary and patch
tokens (224 px; summary and mean patch token also at 448 px) and the OpenCLIP ViT-bigG-14
image embedding. Encoders run in float32 with TF32 disabled, so differences between the two
frames of a family are not swamped by reduced precision; the noise floor is measured at the end.

Rendering and low-level statistics need only the CPU (`--lowlevel-only`); the encoders need a
CUDA GPU and the packages of requirements-features.txt. Output: output/features/.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as nd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import controls  # noqa: E402
import silhouette as s  # noqa: E402

ACCEPTED = ROOT / 'docs/accepted-dataset.json'
OUTPUT = ROOT / 'output/features'
MAIN = ('full_body', 'upper_limb', 'detached_hand_silhouette', 'detached_hand')
CONDITIONS = MAIN + controls.CONTROLS + ('hand_only',)
FRAMES = s.FRAMES
LUMA = np.array([.2126, .7152, .0722])   # Rec. 709 weights on the stored (gamma-encoded) values
LOWLEVEL = ('visible_area', 'mean_luminance', 'rms_contrast', 'edge_energy')
PIXEL_GRID = 64
SIZES = dict(dinov2=224, dinov2_448=448, openclip=224)
DINOV2 = 'facebook/dinov2-large'
OPENCLIP = dict(architecture='ViT-bigG-14', pretrained='laion2b_s39b_b160k',
                repository='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
IMAGENET = (np.array([.485, .456, .406]), np.array([.229, .224, .225]))
CLIP_NORM = (np.array([.48145466, .4578275, .40821073]), np.array([.26862954, .26130258, .27577711]))
require = s.require


def families():
    """Accepted families with source and appearance variant (desk, seed, white balance)."""
    by_id = {x['id']: x for x in s.sources()}
    out = []
    for family in s.read(ACCEPTED)['families']:
        reference, desk, seed, balance = family['id'].split('_')
        require(int(reference[1:]) == family['reference'], f'Unexpected family id: {family["id"]}')
        out.append(dict(id=family['id'], source_id=family['source_id'], reference=family['reference'],
                        desk=desk, seed=int(seed[1:]), balance=balance, variant=f'{desk}_{seed}_{balance}',
                        hashes=family['condition_png_sha256'], source=by_id[family['source_id']]))
    require(len(out) == 480 and len({f['variant'] for f in out}) == 24, 'Unexpected cohort layout')
    return out


def archive_image(family, context, frame):
    path = ROOT / 'data/accepted' / family['id'] / context / f'{frame}.png'
    require(s.sha(path) == family['hashes'][f'{context}/{frame}.png'], f'Changed archive image: {path}')
    return s.rgb(path)


def square_crop(image, box):
    """Crop [x, y, right, bottom] and pad with black to a centred square (no rescaling here)."""
    x, y, right, bottom = box
    crop = image[y:bottom, x:right]
    side = max(crop.shape[:2])
    out = np.zeros((side, side, 3), np.uint8)
    top, left = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    out[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    return out


class Renderer:
    """All condition images of a family; masks are loaded and hash-checked once per source."""

    def __init__(self):
        self.mattes, self.silhouettes = {}, s.read(ROOT / 'docs/accepted-silhouette.json')['families']

    def matte(self, source, name, frame):
        key = (source['id'], name, frame)
        if key not in self.mattes:
            self.mattes[key] = (s.load(source, frame) if name == 'silhouette' else controls.load(source, name, frame))
        return self.mattes[key]

    def control(self, full, source, name, frame):
        if name == 'scrambled':
            return controls.scramble(full, self.matte(source, name, 'static'), source['upper_limb_cutoff'])
        return s.apply(full, self.matte(source, name, frame))

    def render(self, family):
        source, out = family['source'], {}
        full = {f: archive_image(family, 'full_body', f) for f in FRAMES}
        for context in ('full_body', 'upper_limb', 'detached_hand'):
            out[context] = tuple(full[f] if context == 'full_body' else archive_image(family, context, f) for f in FRAMES)
        silhouette = []
        for frame in FRAMES:
            path = ROOT / 'data/accepted_silhouette' / family['id'] / s.CONTEXT / f'{frame}.png'
            require(s.sha(path) == self.silhouettes[family['id']]['files'][f'{s.CONTEXT}/{frame}.png'],
                    f'Changed silhouette image: {path}')
            silhouette.append(s.rgb(path))
        out['detached_hand_silhouette'] = tuple(silhouette)
        for name in controls.CONTROLS:
            out[name] = tuple(self.control(full[f], source, name, f) for f in FRAMES)
        out['hand_only'] = tuple(square_crop(image, source['used_hand_box']) for image in silhouette)
        return {name: out[name] for name in CONDITIONS}


def luminance(image):
    return image.astype(np.float64) @ LUMA / 255


def lowlevel(image):
    """Visible area, mean luminance, RMS contrast and edge energy over the whole image."""
    y = luminance(image)
    gradient = nd.sobel(y, axis=0) ** 2 + nd.sobel(y, axis=1) ** 2
    return np.array([(image.max(axis=2) > 0).mean(), y.mean(), y.std(), gradient.mean()])


def pixels(image, grid=PIXEL_GRID):
    """Luminance averaged over equal blocks: a fixed, non-learned pixel representation."""
    y = luminance(image)
    require(y.shape[0] == y.shape[1], 'Square image expected')
    return np.array(Image.fromarray(y.astype(np.float32), mode='F').resize((grid, grid), Image.BOX)).ravel()


def difference(a, b):
    """Statistics of the static -> movement change itself."""
    changed = np.any(a != b, axis=2)
    visible = (a.max(axis=2) > 0) | (b.max(axis=2) > 0)
    dy = luminance(b) - luminance(a)
    return dict(changed_fraction=float(changed.mean()), changed_per_visible=float(changed.sum() / visible.sum()),
                rms_luminance_change=float(np.sqrt((dy ** 2).mean())),
                mean_abs_change_on_changed=float(np.abs(dy[changed]).mean()) if changed.any() else 0.)


def resized(image, size):
    """Whole image to size x size, bicubic with antialiasing (PIL); no crop, so nothing is cut off."""
    return np.array(Image.fromarray(image).resize((size, size), Image.BICUBIC))


def array_sha(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


_renderer = None


def prepare(family):
    """CPU work for one family: render, hash, low-level features and encoder-sized copies."""
    global _renderer
    _renderer = _renderer or Renderer()
    images = _renderer.render(family)
    rows = []
    for condition in CONDITIONS:
        a, b = images[condition]
        require(a.shape == b.shape, f'Frame shapes differ: {family["id"]} {condition}')
        for frame, image in zip(FRAMES, (a, b)):
            rows.append(dict(family=family['id'], condition=condition, frame=frame, sha256=array_sha(image),
                             lowlevel=lowlevel(image), pixels=pixels(image),
                             small={size: resized(image, size) for size in sorted(set(SIZES.values()))}))
        rows[-1]['difference'] = difference(a, b)
    return rows


class Encoders:
    """Frozen pretrained encoders, pinned to the revisions resolved at load time."""

    def __init__(self, device='cuda'):
        import torch
        from huggingface_hub import HfApi, hf_hub_download
        from transformers import AutoModel
        import open_clip
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_grad_enabled(False)
        self.torch, self.device = torch, device
        api = HfApi()
        self.revisions = dict(dinov2=api.model_info(DINOV2).sha, openclip=api.model_info(OPENCLIP['repository']).sha)
        self.dinov2 = AutoModel.from_pretrained(DINOV2, revision=self.revisions['dinov2']).to(device).eval().float()
        names = {f.rfilename for f in api.model_info(OPENCLIP['repository'], files_metadata=False).siblings}
        weights = next(n for n in ('open_clip_model.safetensors', 'open_clip_pytorch_model.bin') if n in names)
        path = hf_hub_download(OPENCLIP['repository'], weights, revision=self.revisions['openclip'])
        self.weights = dict(openclip=dict(file=weights, sha256=s.sha(path)))
        self.openclip = open_clip.create_model(OPENCLIP['architecture'], pretrained=path).to(device).eval().float()

    def tensor(self, images, norm):
        x = np.stack(images).astype(np.float32) / 255
        x = (x - norm[0]) / norm[1]
        return self.torch.from_numpy(x.transpose(0, 3, 1, 2).astype(np.float32)).to(self.device)

    def dino(self, images):
        tokens = self.dinov2(pixel_values=self.tensor(images, IMAGENET)).last_hidden_state
        return tokens[:, 0].cpu().numpy(), tokens[:, 1:].mean(1).cpu().numpy(), tokens[:, 1:].cpu().numpy()

    def clip(self, images):
        return self.openclip.encode_image(self.tensor(images, CLIP_NORM)).cpu().numpy()

    def __call__(self, rows):
        cls, mean, patches = self.dino([r['small'][SIZES['dinov2']] for r in rows])
        cls448, mean448, _ = self.dino([r['small'][SIZES['dinov2_448']] for r in rows])
        clip = self.clip([r['small'][SIZES['openclip']] for r in rows])
        return dict(dinov2_cls=cls, dinov2_patchmean=mean, dinov2_448_cls=cls448, dinov2_448_patchmean=mean448,
                    openclip=clip), patches


def versions():
    import importlib.metadata as m
    names = ('numpy', 'pillow', 'scipy', 'torch', 'torchvision', 'transformers', 'open_clip_torch', 'huggingface-hub',
             'safetensors')
    out = {}
    for name in names:
        try:
            out[name] = m.version(name)
        except m.PackageNotFoundError:
            pass
    return out


def run(output, limit=None, workers=8, encode=True):
    output.mkdir(parents=True, exist_ok=True)
    selected = families()
    if limit:   # whole appearance variants, so every source and the between-source spread remain defined
        keep = list(dict.fromkeys(f['variant'] for f in selected))[:limit]
        selected = [f for f in selected if f['variant'] in keep]
    count = len(selected) * len(CONDITIONS) * len(FRAMES)
    encoders = Encoders() if encode else None
    patches = (np.lib.format.open_memmap(output / 'dinov2_patches.npy', 'w+', np.float16, (count, 256, 1024))
               if encode else None)
    index, low, pix, differences, features = [], [], [], [], {}
    started = time.time()
    with Pool(workers) as pool:
        for number, rows in enumerate(pool.imap(prepare, selected), 1):
            start = len(index)
            if encode:
                encoded, tokens = encoders(rows)
                patches[start:start + len(rows)] = tokens.astype(np.float16)
                for name, value in encoded.items():
                    features.setdefault(name, []).append(value.astype(np.float32))
            for r in rows:
                index.append(dict(family=r['family'], condition=r['condition'], frame=r['frame'], sha256=r['sha256']))
                low.append(r['lowlevel'])
                pix.append(r['pixels'])
                if 'difference' in r:
                    differences.append(dict(family=r['family'], condition=r['condition'], **r['difference']))
            if number % 24 == 0 or number == len(selected):
                print(f'{number}/{len(selected)} families, {time.time() - started:.0f} s', flush=True)
    arrays = dict(lowlevel=np.array(low), pixels64=np.array(pix, np.float32))
    arrays.update({name: np.concatenate(v) for name, v in features.items()})
    np.savez(output / 'features.npz', **arrays)
    lines = ['family,condition,frame,sha256'] + [f"{r['family']},{r['condition']},{r['frame']},{r['sha256']}" for r in index]
    (output / 'index.csv').write_text('\n'.join(lines) + '\n')
    keys = list(differences[0])
    (output / 'differences.csv').write_text('\n'.join([','.join(keys)] + [
        ','.join(str(d[k]) for k in keys) for d in differences]) + '\n')
    manifest = dict(method='docs/baselines.md', code_sha256=s.sha(__file__), images=len(index),
                    conditions=list(CONDITIONS), frames=list(FRAMES), lowlevel=list(LOWLEVEL), pixel_grid=PIXEL_GRID,
                    resize='whole 2048 image (hand_only: padded square crop) to the encoder size; PIL bicubic, antialiased',
                    sizes=SIZES, accepted_manifest_sha256=s.sha(ACCEPTED), controls_sha256=s.sha(controls.MANIFEST),
                    silhouettes_sha256=s.sha(s.MANIFEST), python=platform.python_version(), packages=versions(),
                    files={name: None for name in ('features.npz', 'index.csv', 'differences.csv')})
    if encode:
        patches.flush()
        manifest['files']['dinov2_patches.npy'] = None
        import torch
        manifest.update(models=dict(dinov2=dict(repository=DINOV2, revision=encoders.revisions['dinov2'],
                                                tokens='last_hidden_state (final layer norm); CLS and patch tokens'),
                                    openclip=dict(OPENCLIP, revision=encoders.revisions['openclip'],
                                                  output='encode_image, before L2 normalisation',
                                                  **encoders.weights['openclip'])),
                        precision='float32, TF32 disabled', gpu=torch.cuda.get_device_name(0),
                        noise_floor=noise_floor(encoders, selected[:2]))
    for name in manifest['files']:
        manifest['files'][name] = s.sha(output / name)
    s.write(output / 'manifest.json', manifest)
    return manifest


def noise_floor(encoders, selected):
    """Re-encode two families one image at a time in reverse order; report the largest change."""
    rows = [r for family in selected for r in prepare(family)]
    first = encoders(rows)[0]
    again = [encoders([r])[0] for r in reversed(rows)][::-1]
    out = {}
    for name, value in first.items():
        repeat = np.concatenate([a[name] for a in again])
        out[name] = dict(max_abs=float(np.abs(value - repeat).max()),
                         max_relative=float((np.linalg.norm(value - repeat, axis=1) / np.linalg.norm(value, axis=1)).max()))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--limit', type=int, help='First N appearance variants only, all 20 sources each (smoke test)')
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--lowlevel-only', action='store_true', help='CPU only: skip the encoders')
    args = parser.parse_args()
    result = run(args.output, args.limit, args.workers, not args.lowlevel_only)
    print(json.dumps({k: v for k, v in result.items() if k in ('images', 'models', 'noise_floor', 'files')}, indent=2))
