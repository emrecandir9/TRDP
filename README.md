# TRDP

Controlled visual stimuli for studying body context and observed index-finger movement.
The generator expands **20 approved V7 image pairs into 480 families / 2,880 PNGs**:
four desks × two seeds × three shared white balances per source.
It preserves the approved hand movement while varying the desk and white balance.

Read the [illustrated project guide](docs/project-guide.pdf) for the complete workflow:
original references, source generation, hand and clothing repairs, V7 approval, masks,
final expansion, earlier methods and next steps. [Validation](docs/validation.md) records
what was checked and what remains untested.

## Directory map

```text
TRDP/
├── README.md                    # Start here: setup, commands and project status
├── requirements.txt             # Small CPU review environment: NumPy and Pillow
├── .gitignore                   # Keeps data, models, environments and runs local
├── .gitattributes               # Treats PDFs as binary files; no Git LFS
├── config/
│   └── recipe.json              # Parameters, prompts, pinned models and canary IDs
├── scripts/
│   ├── setup.sh                 # Creates .venv for review or Linux CUDA generation
│   ├── models.py                # Downloads / resolves the three pinned models
│   ├── run.py                   # Logs and launches the CUDA worker cooperatively
│   ├── generate.py              # Makes shared desk proposals and exports families
│   ├── dataset.py               # Source checks, composition, verification and review
│   └── validate_dataset.py      # Checks the downloaded, historically accepted archive
├── tests/
│   └── test_dataset.py          # CPU regression checks, including failure cases
├── data/
│   ├── frozen_sources.json      # Tracked metadata and hashes for 220 frozen files
│   ├── sources/                 # LOCAL: 20 approved V7 pairs in all three contexts
│   ├── inputs/                  # LOCAL: source image, Canny control, desk mask/weight
│   └── accepted/                # LOCAL: original 480-family accepted dataset
│       ├── _provenance/          # LOCAL: original masks and hash-bound parent records
│       │   ├── inputs/           # 20 original desk_weight.png files
│       │   └── diffusion_parents/ # 160 original result.json records
│       ├── r06_birch_s9_warm/    # Example: reference 6, birch, seed 9, warm balance
│       │   ├── full_body/       # static.png + movement.png
│       │   ├── upper_limb/      # static.png + movement.png
│       │   ├── detached_hand/   # static.png + movement.png
│       │   └── audit.json       # Original per-family provenance and image hashes
│       └── validation.json      # Local archive verification result
├── docs/
│   ├── project-guide.pdf        # Step-by-step method with actual before/after images
│   ├── accepted-dataset.json    # Historical approval and hashes of all 2,880 PNGs
│   ├── approval-receipt.json    # Acceptance scope and known reference-17 issue
│   ├── validation.md            # Validation scope, results and limitations
│   └── 480-family-review.pdf    # LOCAL: unchanged original interactive review
├── .venv/                       # LOCAL: environment created by setup.sh
└── output/                      # LOCAL: new runs, proposals, audits and review records
```

Only code, configuration, compact metadata and documentation go to GitHub.
The dataset, frozen input images and large review PDF stay local; no Git LFS is used.
All condition images are **RGB PNG, 2048 × 2048**. Contexts keep their original positions;
hidden regions are black. A family is six images, not six independent observations.

## Review locally (Mac or Linux)

Use Python 3.12 or 3.13. Run commands from this repository's root.

```sh
./scripts/setup.sh review
source .venv/bin/activate
python -m unittest discover -s tests -v
python scripts/dataset.py check
python scripts/validate_dataset.py
```

Tests need no dataset or GPU. The last two commands require local data. This working
copy contains the frozen inputs and full accepted archive. After cloning elsewhere,
copy `data/sources/`, `data/inputs/`, `data/accepted/` and optionally
`docs/480-family-review.pdf` from this local copy into the same paths. Do not regenerate
approximations of the frozen inputs. Archive validation checks hashes, image formats,
approval, protected pixels under the original masks, action support and contexts; it writes
`data/accepted/validation.json`. It does not replace human visual review.

## Mask correction (11 October 2026)

The hand matte was a cropped image, mistakenly stretched across the full scene.
The 20 local mask/weight/preview sets now place it at its native crop coordinates;
the existing manifest and recipe hashes have been updated. **Black protects** both
hand poses, a 36 px guard and the forearm/contact region; **white edits the desk**.
The separate blending weight softens the editable boundary over 48 px. Masks are
processing inputs and are never displayed as stimuli. The guide shows before/after photos.

The approved 480 families remain unchanged and were generated with the old masks.
Their acceptance hashes and original production inputs are preserved. Historical
pixel checks use those original weights; validation separately reports deviations
from the corrected guard. A future corrected run needs a fresh output folder and new visual
canary approval. Reuse the corrected frozen inputs; the legacy preparation script
still contains the original scaling mistake.

## Generate on CUDA

Linux, one NVIDIA GPU, at least 12,000 MiB free VRAM and 16 GiB free output space.
Allow roughly 12 GB more for the Hugging Face model cache. CUDA setup uses the recorded
production package versions and PyTorch CUDA 12.8 wheels.

```sh
./scripts/setup.sh cuda
source .venv/bin/activate
python scripts/models.py --download
python scripts/generate.py canary --preflight   # checks only; no model loaded
python scripts/run.py canary
python scripts/dataset.py review-template > review-canary.json
# Inspect all six PNGs per family; complete every check and explicit decision.
python scripts/dataset.py review review-canary.json
python scripts/run.py full --cooperative
python scripts/dataset.py review-template > review-full.json
# Inspect the remaining families and complete their decisions.
python scripts/dataset.py review review-full.json
python scripts/dataset.py verify
```

Full generation requires all ten canaries to pass visual review. Numeric checks alone
never approve realism. `output/accepted.json` records the new reviewers and decisions;
the historical records in `docs/` do not approve a new run.

Workers use low CPU priority. Cooperative full runs wait for 60 seconds of continuous
GPU idle, release the worker before waiting and resume resource-availability stops.
Sampling errors, altered files and incomplete checkpoints stop the run. Inspect failures;
use a separate output directory for a revised run. Interrupting the launcher stops its worker.
Resume requires the same recorded environment and GPU.

New runs use `output/parents/<desk-parent>/` for the 160 raw desk proposals and native
shared composites, and `output/candidates/<family>/<context>/{static,movement}.png`
for the condition images. JSON records bind pixels, recipe, code, models, runtime and
reviews. Use `--output PATH` consistently for another run. To check just one family:
`python scripts/dataset.py verify --family r06_birch_s9_warm`.

## Current position and next steps

The original 480-family batch was visually accepted on 11 October 2026. Reference 17
has a known clothing issue, accepted with that flag. The clean repository covers the
**final expansion from frozen V7 sources**; earlier exploratory runners remain in the
original project folders. The guide explains their role and limitations.

1. Validate a fresh CUDA canary before trusting newly generated outputs.
2. Freeze the analysis cohort and audit upstream lineage. Keep related sources, all
   variants and all six conditions together in evaluation splits. These are 20 source
   groups with 24 appearances each; upstream groups may share earlier origins.
3. Evaluate visual encoders with hand-only and low-level baselines, then assess the
   intended motor-response model. Repeat analyses without reference 17: 456 families.

Image acceptance does not establish physiological or scientific replication.
The guide details the project history and the remaining evaluation work.

Models retain their own licenses: [RealVisXL](https://huggingface.co/SG161222/RealVisXL_V4.0),
[Xinsir Canny](https://huggingface.co/xinsir/controlnet-canny-sdxl-1.0),
[fixed SDXL VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix).
