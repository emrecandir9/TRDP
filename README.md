# TRDP

Controlled visual stimuli for studying body context and observed index-finger movement.
The generator expands **20 approved V7 image pairs into 480 families**:
four desks × two seeds × three shared white balances per source.
It preserves the approved hand movement while varying the desk and white balance.
The accepted archive holds 2,880 PNGs in three contexts; the paper-matched
[detached-hand silhouette](docs/detached-hand-silhouette.md) adds 960 PNGs derived from
the same full-body pixels.

Read the [illustrated project guide](docs/project-guide.pdf) for the complete workflow:
original references, source generation, hand and clothing repairs, V7 approval, masks,
final expansion and earlier methods. The guide dates from 11 October and is partly superseded:
it predates the human target, the [analysis spec](docs/analysis-spec.md), the silhouette and the
[control stimuli](docs/controls.md). Its page 17 shows the rectangular detached hand, now the
sensitivity condition, and its roadmap on page 25 is replaced by the
[next-phase plan](docs/next-phase-plan.md). [Validation](docs/validation.md) records
what was checked and what remains untested.

## Directory map

```text
TRDP/
├── README.md                    # Start here: setup, commands and project status
├── requirements.txt             # CPU environment: NumPy, Pillow, pandas, SciPy, statsmodels,
│                                #   scikit-learn, matplotlib
├── requirements-features.txt    # GPU feature extraction: torch (CUDA 12.8), transformers, open_clip
├── requirements-segmentation.txt # Optional: torch, timm, pinned SAM-HQ for segment_hands.py
├── .gitignore                   # Keeps data, models, environments and runs local
├── .gitattributes               # Treats PDFs as binary files; no Git LFS
├── config/
│   ├── recipe.json              # Parameters, prompts, pinned models and canary IDs
│   └── analysis.json            # Frozen cohort, human-data hashes, target settings
├── analysis/
│   ├── human_target.py          # Participant-level human MEP target from the OSF deposit
│   ├── features.py              # Gate 1 inputs: rendering, low-level statistics, DINOv2/OpenCLIP (GPU)
│   ├── baselines.py             # Gate 1 baselines: movement signal, decoding, latent-space analyses
│   └── baseline_figures.py      # Figures of docs/baselines.md
├── scripts/
│   ├── setup.sh                 # Creates .venv for review or Linux CUDA generation
│   ├── models.py                # Downloads / resolves the three pinned models
│   ├── run.py                   # Logs and launches the CUDA worker cooperatively
│   ├── generate.py              # Makes shared desk proposals and exports families
│   ├── dataset.py               # Source checks, composition, verification and review
│   ├── segment_hands.py         # SAM-HQ hand segmentation of the final frames (CPU)
│   ├── silhouette.py            # Paper-matched detached-hand mattes and archive export
│   ├── segment_arms.py          # SAM-HQ hand and forearm segmentation for the controls
│   ├── controls.py              # Control stimuli of analysis spec §7.2 (masks, checks, rendering)
│   └── validate_dataset.py      # Checks the downloaded, historically accepted archive
├── tests/
│   ├── test_dataset.py          # CPU regression checks, including failure cases
│   ├── test_silhouette.py       # Silhouette geometry, registration, seams, wrist cut, frozen mattes
│   ├── test_controls.py         # Control geometry, equal areas, scramble, frozen controls
│   ├── test_human_target.py     # Human target checks; published-statistics reproduction
│   └── test_baselines.py        # Baseline statistics, leave-one-source-out decoding, rendering
├── data/
│   ├── frozen_sources.json      # Tracked metadata and hashes for 220 frozen files
│   ├── hand_segmentation.json   # Tracked SAM-HQ provenance, prompts and mask hashes
│   ├── silhouettes.json         # Tracked matte hashes, parameters and per-source diagnostics
│   ├── source_metadata.json     # Reference and confirmed apparent sex per source, with provenance
│   ├── arm_segmentation.json    # Tracked SAM-HQ forearm masks: provenance, prompts, hashes
│   ├── controls.json            # Tracked control masks, permutations, parameters, diagnostics
│   ├── sources/                 # LOCAL: 20 approved V7 pairs in all three contexts
│   ├── inputs/                  # LOCAL: source image, Canny control, desk mask/weight,
│   │                            #        hand matte, case geometry, SAM-HQ masks, mattes
│   ├── human/                   # LOCAL: unchanged OSF files of the reference paper
│   ├── accepted/                # LOCAL: original 480-family accepted dataset
│   │   ├── _provenance/          # LOCAL: original masks and hash-bound parent records
│   │   │   ├── inputs/           # 20 original desk_weight.png files
│   │   │   └── diffusion_parents/ # 160 original result.json records
│   │   ├── r12_birch_s9_warm/    # Example: reference 12, birch, seed 9, warm balance
│   │   │   ├── full_body/       # static.png + movement.png
│   │   │   ├── upper_limb/      # static.png + movement.png
│   │   │   ├── detached_hand/   # rectangle: static.png + movement.png
│   │   │   └── audit.json       # Original per-family provenance and image hashes
│   │   └── validation.json      # Local archive verification result
│   └── accepted_silhouette/     # LOCAL: <family>/detached_hand_silhouette/{static,movement}.png
├── docs/
│   ├── project-guide.pdf        # Stimulus method with before/after images (11 Oct; partly superseded)
│   ├── accepted-dataset.json    # Historical approval and hashes of all 2,880 PNGs
│   ├── approval-receipt.json    # Acceptance scope and known reference-17 issue
│   ├── accepted-silhouette.json # Hashes of the 960 archive silhouette PNGs
│   ├── silhouette-approval.json # Hash-bound human approval of the silhouettes
│   ├── detached-hand-silhouette.md # Silhouette method, checks and residual defects
│   ├── controls.md              # Control stimuli: design, parameters, checks, figures
│   ├── controls-approval.json   # Hash-bound human approval of the controls
│   ├── figures/                 # Silhouette and control review sheets
│   ├── validation.md            # Validation scope, results and limitations
│   ├── next-phase-plan.md       # Priorities and schedule to 18 December 2026
│   ├── analysis-spec.md         # Endpoints, splits, models and success criteria
│   ├── baselines.md             # Gate 1 baselines: method fixed before results, then results
│   └── 480-family-review.pdf    # LOCAL: unchanged original interactive review
├── .venv/                       # LOCAL: environment created by setup.sh
└── output/                      # LOCAL: new runs, proposals, audits and review records
```

Only code, configuration, compact metadata and documentation go to GitHub.
The dataset, frozen input images and large review PDF stay local; no Git LFS is used.
All condition images are **RGB PNG, 2048 × 2048**. Contexts keep their original positions;
hidden regions are black. Contexts: `full_body`, `upper_limb`, `detached_hand_silhouette`
(primary detached hand: skin only, cut at the wrist, as in the paper) and `detached_hand`
(the earlier rectangle with a desk patch, kept as a sensitivity condition). New runs export
eight images per family; the historical archive has six plus the derived silhouette pair.
A family's images are paired views, not independent observations.

## Review locally (Mac or Linux)

Use Python 3.12 or 3.13. Run commands from this repository's root.

```sh
./scripts/setup.sh review
source .venv/bin/activate
python -m unittest discover -s tests -v
python scripts/dataset.py check
python scripts/validate_dataset.py
python scripts/silhouette.py check
python scripts/silhouette.py verify-archive
python scripts/controls.py check
python scripts/validate_dataset.py --hand-deviation
python analysis/human_target.py
```

Gate 1 baselines ([method and results](docs/baselines.md)): `python analysis/features.py`
on a CUDA machine (`pip install -r requirements-features.txt`), then `python analysis/baselines.py`.

Tests need no dataset or GPU; the published-statistics test runs only when the OSF files
are present. The last seven commands require local data. This working
copy contains the frozen inputs and full accepted archive. After cloning elsewhere,
copy `data/sources/`, `data/inputs/`, `data/accepted/`, `data/accepted_silhouette/`,
`data/human/` and optionally
`docs/480-family-review.pdf` from this local copy into the same paths. Do not regenerate
approximations of the frozen inputs. Archive validation checks hashes, image formats,
approval, protected pixels under the original masks, action support and contexts; it writes
`data/accepted/validation.json`. It does not replace human visual review.

## Human target

`analysis/human_target.py` reads the unchanged OSF files of Matamala-Gomez et al. (2025),
checks their hashes against `config/analysis.json`, and writes `output/human_target/`:
participant cell means, the 12-cell table, all declared contrasts with paired bootstrap
intervals on five scales, scale-free ratios, the questionnaire comparison and a refit of the
paper's own mixed models. `summary.json` reports whether every published statistic was
reproduced. Endpoints and their use are defined in the [analysis spec](docs/analysis-spec.md).

## Generate on CUDA

Paused until after the December evaluation; see the [next-phase plan](docs/next-phase-plan.md).

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
# Inspect all eight PNGs per family; complete every check and explicit decision.
python scripts/dataset.py review review-canary.json
python scripts/run.py full --cooperative
python scripts/dataset.py review-template > review-full.json
# Inspect the remaining families and complete their decisions.
python scripts/dataset.py review review-full.json
python scripts/dataset.py verify
```

Full generation requires all ten canaries to pass visual review. Numeric checks alone
never approve realism, and only decisions recorded as `reviewer: human_user` are accepted:
automated checks and assistant inspections are advisory. `output/accepted.json` records the decisions;
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
`python scripts/dataset.py verify --family r12_birch_s9_warm`.

## Current position and next steps

The original 480-family batch was visually accepted on 11 October 2026. Reference 17
has a known clothing issue, accepted with that flag. The clean repository covers the
**final expansion from frozen V7 sources**; earlier exploratory runners remain in the
original project folders. The guide explains their role and limitations.

Stimulus work is paused; the [next-phase plan](docs/next-phase-plan.md) sets the
priorities and schedule to 18 December 2026.

Done on 22 October:

- The accepted archive is frozen as the analysis cohort.
- The participant-level human MEP target is built and reproduces the paper's published
  statistics.
- The [analysis spec](docs/analysis-spec.md) is frozen, before any model result.
- The detached-hand silhouette and the six control stimuli are built and visually approved.

Done on 23 October:

- **Gate 1, visual baselines** ([method and results](docs/baselines.md)): low-level, pixel,
  hand-only, DINOv2 and OpenCLIP baselines on all 10,560 condition images. None reproduces
  the human pattern; every one shows the reverse (the largest movement signal for the
  detached hand). Frozen encoders show no attachment variable beyond a pixel cue.

Next:

1. Build and test the temporal motor-response model; Gate 2 on 4 December, single held-out
   evaluation 14–18 December. Every analysis is repeated without reference 17 (456 families).
   A fresh CUDA canary is deferred.

Image acceptance does not establish physiological or scientific replication.

Models retain their own licenses: [RealVisXL](https://huggingface.co/SG161222/RealVisXL_V4.0),
[Xinsir Canny](https://huggingface.co/xinsir/controlnet-canny-sdxl-1.0),
[fixed SDXL VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix).
