# TRDP

Controlled visual stimuli for studying body context and observed index-finger movement.
The current generator expands **20 approved V7 image pairs into 480 families / 2,880 PNGs**:
four desks × two seeds × three shared white balances per source.
It preserves the approved movement; it does not generate new hand motion.

## Start locally

Use Python 3.12 or 3.13. Run commands from this folder.

```sh
./setup.sh review             # Mac or Linux: checks and image review only
source .venv/bin/activate
python -m unittest -v test_dataset
python dataset.py check       # requires the local frozen data
```

The dataset and large review PDF stay **local and ignored by Git**. This working folder
already contains them. After cloning elsewhere, copy `data/sources/`, `data/inputs/`,
and optionally `docs/480-family-review.pdf` from your local TRDP copy into the same paths.
`data/frozen_sources.json` verifies all 220 source/input files. Do not replace them with
newly generated approximations. Tests work without the local image bundle.

## Generate on CUDA

Linux, one NVIDIA GPU, at least 12,000 MiB free VRAM, and 16 GiB free output space.
The CUDA environment uses the production package versions and PyTorch CUDA 12.8 wheels.
Allow additional disk space for the model cache (roughly 12 GB).

```sh
./setup.sh cuda
source .venv/bin/activate
python models.py --download   # three pinned models; uses the Hugging Face cache
python generate.py canary --preflight   # no weights loaded or CUDA initialized
python run.py canary
python dataset.py review-template > review-canary.json
# Inspect all six PNGs per family; fill every check and explicit decision in this JSON.
python dataset.py review review-canary.json
python run.py full --cooperative
python dataset.py review-template > review-full.json
# Inspect the remaining families and complete their decisions.
python dataset.py review review-full.json
```

Full generation requires all ten canaries to pass visual review. Numeric checks alone
never approve realism. `output/accepted.json` records actual reviewers and decisions;
the historical approval records in `docs/` do not approve a new run.

Workers run with low CPU priority. Cooperative full runs wait for 60 seconds of continuous
GPU idle, release the worker before waiting, and resume only resource-availability stops.
Sampling errors, altered files and incomplete checkpoints stop the run. Inspect failures;
use a separate output directory for a revised run. Interrupting the launcher stops its worker.

Outputs go to ignored `output/`: `parents/` holds 160 shared desk proposals,
`candidates/<family>/<context>/{static,movement}.png` holds the six 2048-square images,
and JSON records bind pixels, recipe, code, models, runtime and reviews. Resume requires
the same recorded environment and GPU. Use `--output PATH` consistently to choose another run.

```sh
python dataset.py verify
python dataset.py verify --family r06_birch_s9_warm
```

## Where things live

| Path | Purpose |
| --- | --- |
| `recipe.json` | Used parameters, prompts, model revisions and canary IDs |
| `dataset.py` | Frozen data checks, composition, exports, verification and review |
| `generate.py` / `run.py` | CUDA worker / logged cooperative launcher |
| `setup.sh` / `models.py` | Environment setup / pinned model downloads |
| `test_dataset.py` | CPU regression checks, including failure cases |
| `data/` | Tracked hash manifest; source pairs and inputs remain local |
| [Project guide](docs/project-guide.pdf) | Current method, parameters, worked example, trial history and roadmap |
| `docs/480-family-review.pdf` | Original interactive review, local only |
| [Accepted dataset](docs/accepted-dataset.json) / [receipt](docs/approval-receipt.json) | Historical acceptance and image hashes |

## Current position

The original 480-family batch was visually accepted on 11 October 2026. Reference 17
has a known clothing issue, accepted with that flag. The clean code passes CPU regression
and frozen-input checks; a fresh CUDA canary is still required before trusting its new outputs.
The original production folder and generated batch are unchanged and are not copied here.

These are **20 source groups with 24 appearances each**, not 480 independent people or poses.
Before evaluation, audit upstream lineage and keep related sources, all variants and all
six conditions together. Run a sensitivity analysis without reference 17 (456 families).
Next: freeze the analysis cohort, evaluate visual encoders with controlled baselines,
then assess a spiking-network model against the intended motor-response question.
The guide distinguishes earlier experiments from completed scientific validation.

Models retain their own licenses: [RealVisXL](https://huggingface.co/SG161222/RealVisXL_V4.0),
[Xinsir Canny](https://huggingface.co/xinsir/controlnet-canny-sdxl-1.0),
[fixed SDXL VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix).
Setup follows the [PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/)
and [Hugging Face cache/download API](https://huggingface.co/docs/huggingface_hub/guides/download).
