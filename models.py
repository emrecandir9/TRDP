"""Download only the pinned generation files into the Hugging Face cache."""
import argparse
from pathlib import Path
from dataset import RECIPE, read, require

FILES = {
    'model': ['model_index.json', 'scheduler/scheduler_config.json',
              'tokenizer/*', 'tokenizer_2/*', 'text_encoder/config.json',
              'text_encoder/model.fp16.safetensors', 'text_encoder_2/config.json',
              'text_encoder_2/model.fp16.safetensors', 'unet/config.json',
              'unet/diffusion_pytorch_model.fp16.safetensors'],
    'control': ['config.json', 'diffusion_pytorch_model.safetensors'],
    'vae': ['config.json', 'diffusion_pytorch_model.safetensors'],
}


def model_paths(download=False):
    from huggingface_hub import snapshot_download
    paths = {}
    for name, model in read(RECIPE)['models'].items():
        path = Path(snapshot_download(repo_id=model['id'], revision=model['revision'],
                                     allow_patterns=FILES[name], local_files_only=not download))
        for pattern in FILES[name]:
            require(any(p.is_file() for p in path.glob(pattern)), f'Missing {name}/{pattern}; run models.py --download')
        paths[name] = path
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Fetch missing files; otherwise check the cache')
    args = parser.parse_args()
    for name, path in model_paths(args.download).items():
        print(name, path)
