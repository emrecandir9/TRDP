"""CUDA worker: immutable checkpoints, protected pixels and a visual canary gate."""
import argparse
import ast
import fcntl
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
from dataset import (ROOT, DATA, RECIPE, read, write, sha, rgb, require, plan, check_sources,
                     identity, frozen_files, shared_desk, lighting, export_family, verify_family, require_canary)


class ResourceUnavailable(Exception):
    pass


def gpu_state():
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                    '--format=csv,noheader'], text=True, timeout=10).strip()
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.free',
                                       '--format=csv,noheader,nounits'], text=True, timeout=10).strip().splitlines()
    require(len(devices) == 1, 'Use a machine with one visible NVIDIA GPU')
    name, free = devices[0].rsplit(',', 1)
    others = [line for line in apps.splitlines() if int(line.split(',')[0]) != os.getpid()]
    return dict(name=name.strip(), free_mib=int(free), other_processes=others)


def available(cfg):
    state = gpu_state()
    if state['other_processes'] or state['free_mib'] < cfg['minimum_gpu_vram_mib']:
        raise ResourceUnavailable(str(state))
    return state


def check_checkpoints(output):
    for group, record in (('parents', 'result.json'), ('candidates', 'audit.json')):
        for path in (output / group).glob('*'):
            require(path.is_dir() and (path / record).is_file() and not (path / 'failure.json').exists(),
                    f'Incomplete checkpoint retained: {path}. Inspect it; use a new output directory for a revised run.')


def preflight(cfg):
    from models import model_paths
    paths = model_paths()
    from transformers import CLIPTokenizer
    tokens = [CLIPTokenizer.from_pretrained(paths['model'] / name, local_files_only=True)
              for name in ('tokenizer', 'tokenizer_2')]
    for desk in cfg['desks'].values():
        for prompt in (cfg['prompt'].format(desk=desk), cfg['negative_prompt']):
            require(all(len(tokenizer(prompt).input_ids) <= 77 for tokenizer in tokens), 'Prompt exceeds 77 tokens')
    import diffusers
    api = Path(diffusers.__file__).parent / 'pipelines/controlnet/pipeline_controlnet_inpaint_sd_xl.py'
    tree = ast.parse(api.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StableDiffusionXLControlNetInpaintPipeline')
    call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
    parameters = {n.arg for n in call.args.args + call.args.kwonlyargs}
    require({'prompt_2', 'negative_prompt_2', 'mask_image', 'control_image', 'callback_on_step_end',
             'controlnet_conditioning_scale', 'control_guidance_end'} <= parameters, 'Diffusers API changed')
    import torch
    require(not torch.cuda.is_initialized(), 'Preflight must not initialize CUDA')
    runtime = {p: importlib.metadata.version(p) for p in
               ('torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'safetensors',
                'huggingface-hub', 'numpy', 'Pillow')}
    runtime.update(python=platform.python_version(), compiled_cuda=torch.version.cuda)
    require(torch.version.cuda is not None, 'Install the CUDA environment with ./setup.sh cuda')
    return paths, dict(identity=identity(), runtime=runtime, gpu=gpu_state()['name'], models=cfg['models'])


def load_pipeline(paths, cfg):
    import torch
    import diffusers as df
    torch.set_num_threads(cfg['render']['cpu_threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    control = df.ControlNetModel.from_pretrained(paths['control'], torch_dtype=torch.float16,
                                               use_safetensors=True, local_files_only=True)
    vae = df.AutoencoderKL.from_pretrained(paths['vae'], torch_dtype=torch.float16,
                                          use_safetensors=True, local_files_only=True)
    pipe = df.StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        paths['model'], controlnet=control, vae=vae, torch_dtype=torch.float16, variant='fp16',
        use_safetensors=True, local_files_only=True, add_watermarker=False)
    pipe.scheduler = df.DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type='dpmsolver++', use_karras_sigmas=True)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def render_parent(pipe, cache, job, source, cfg, output, a, b, weight):
    import torch
    cache.mkdir(parents=True, exist_ok=False)
    steps = []
    def callback(pipeline, index, timestep, kwargs):
        steps.append(float(timestep))
        return kwargs
    inputs = DATA / 'inputs' / source['id']
    r = cfg['render']
    tick = time.monotonic()
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        prompt = cfg['prompt'].format(desk=cfg['desks'][job['desk_id']])
        raw = pipe(prompt=prompt, prompt_2=prompt, negative_prompt=cfg['negative_prompt'],
                   negative_prompt_2=cfg['negative_prompt'],
                   image=Image.open(inputs / 'image_1024.png').convert('RGB'),
                   mask_image=Image.open(inputs / 'desk_mask.png').resize((r['size'], r['size']), Image.Resampling.NEAREST),
                   control_image=Image.open(inputs / 'control_1024.png').convert('RGB'),
                   width=r['size'], height=r['size'], strength=r['strength'], num_inference_steps=r['steps'],
                   guidance_scale=r['cfg'], controlnet_conditioning_scale=r['control_scale'],
                   control_guidance_end=r['control_end'], generator=torch.Generator('cpu').manual_seed(job['seed']),
                   callback_on_step_end=callback).images[0]
        torch.cuda.synchronize()
        require(len(steps) == int(r['steps'] * r['strength']), 'Unexpected number of denoising steps')
        raw.save(cache / 'raw_1024.png')
        proposal = np.array(raw.resize((r['export_size'], r['export_size']), Image.Resampling.LANCZOS))
        new_a, _, composite = shared_desk(a, b, proposal, weight, cfg['colour_offset_limit_rgb'])
        Image.fromarray(new_a).save(cache / 'static_with_shared_desk.png')
        scheduler = dict(pipe.scheduler.config)
        if '_use_default_values' in scheduler:
            scheduler['_use_default_values'] = sorted(scheduler['_use_default_values'])
        record = dict(base_id=job['base_id'], source_id=source['id'], seed=job['seed'],
                      prompt=prompt, negative_prompt=cfg['negative_prompt'],
                      run_sha256=sha(output / 'provenance.json'), input_sha256=source['input_sha256'],
                      seconds=time.monotonic() - tick, effective_steps=len(steps), timesteps=steps,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      scheduler=scheduler, unet_in_channels=pipe.unet.config.in_channels, composition=composite,
                      files={name: sha(cache / name) for name in ('raw_1024.png', 'static_with_shared_desk.png')})
        write(cache / 'result.json', record)
    except Exception:
        write(cache / 'failure.json', dict(error=traceback.format_exc()))
        raise


def run(args):
    cfg, sources, jobs = plan()
    check_sources(sources)
    check_checkpoints(args.output)
    if args.scope == 'full':
        require_canary(args.output)
    selected = [j for j in jobs if args.scope == 'full' or j['canary']]
    pending = []
    for job in selected:
        if (args.output / 'candidates' / job['id']).exists():
            verify_family(args.output, job, sources[job['source_id']], cfg)
        else:
            pending.append(job)
    if not pending and not args.preflight:
        print('All selected families already exist and pass saved-image checks; no models loaded.')
        return
    require(platform.system() == 'Linux', 'Generation requires Linux and NVIDIA CUDA; use dataset.py check on Mac')
    require(socket.gethostname().split('.')[0] not in {'jaguar', 'puma', 'leopard', 'sshproxy'}, 'Do not compute on a gateway')
    args.output.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(args.output).free >= cfg['minimum_free_workspace_gib'] * 2**30, 'Need 16 GiB of disk headroom')
    if not args.preflight:
        available(cfg)
    paths, provenance = preflight(cfg)
    if args.preflight:
        print('Preflight passed without loading weights or initializing CUDA.')
        return
    provenance_path = args.output / 'provenance.json'
    if provenance_path.exists():
        require(read(provenance_path) == provenance, 'Runtime, GPU, code or recipe changed; use a new output directory')
    else:
        write(provenance_path, provenance)
    pipe = None
    for job in pending:
        available(cfg)
        source = sources[job['source_id']]
        directory = DATA / 'sources' / source['source_directory']
        a, b = rgb(directory / 'full_body/static.png'), rgb(directory / 'full_body/movement.png')
        weight = np.array(Image.open(DATA / 'inputs' / source['id'] / 'desk_weight.png'))
        cache = args.output / 'parents' / job['base_id']
        if not (cache / 'result.json').exists():
            if pipe is None:
                pipe = load_pipeline(paths, cfg)
            render_parent(pipe, cache, job, source, cfg, args.output, a, b, weight)
        record = read(cache / 'result.json')
        require(record['run_sha256'] == sha(provenance_path) and record['base_id'] == job['base_id'] and
                record['input_sha256'] == source['input_sha256'], 'Parent provenance changed')
        frozen_files(cache, record['files'])
        new_a = rgb(cache / 'static_with_shared_desk.png')
        new_b = b.copy()
        new_b[weight > 0] = new_a[weight > 0]
        gains = cfg['lighting'][job['light_id']]
        export_family(args.output / 'candidates' / job['id'], lighting(new_a, gains), lighting(new_b, gains),
                      a, b, source, job, weight, gains, dict(identity=identity(), run_sha256=sha(provenance_path),
                                                           parent_sha256=sha(cache / 'result.json')))
        print('FAMILY', job['id'], 'visual_review_pending', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope', choices=('canary', 'full'))
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'output')
    args = parser.parse_args()
    try:
        with open(f'/tmp/trdp-gpu-{os.getuid()}.lock', 'a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ResourceUnavailable('Another TRDP worker holds the GPU lock') from error
            run(args)
    except ResourceUnavailable as error:
        print('RESOURCE_WAIT:', error, flush=True)
        sys.exit(75)
