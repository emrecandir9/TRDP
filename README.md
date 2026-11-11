# TRDP

Controlled visual stimuli for studying body context and observed index-finger movement.
The generator expands **20 approved V7 image pairs into 480 families**:
four desks × two seeds × three shared white balances per source.
It preserves the approved hand movement while varying the desk and white balance.
The accepted archive holds 2,880 PNGs in three contexts; the paper-matched
[detached-hand silhouette](docs/detached-hand-silhouette.md) adds 960 PNGs derived from
the same full-body pixels.

`docs/TRDP Study Short Version.pdf` and `docs/TRDP Study Progress.pptx` describe the state before the
fit of c (they predate 9 November) and the [independent review](docs/independent-review.md).

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
├── requirements-model.txt       # Model phase: requirements-features.txt plus snnTorch
├── .gitignore                   # Keeps data, models, environments and runs local
├── .gitattributes               # Treats PDFs as binary files; no Git LFS
├── config/
│   ├── recipe.json              # Parameters, prompts, pinned models and canary IDs
│   └── analysis.json            # Frozen cohort, human-data hashes, target settings
├── analysis/
│   ├── human_target.py          # Participant-level human MEP target from the OSF deposit
│   ├── features.py              # Gate 1 inputs: rendering, low-level statistics, DINOv2/OpenCLIP (GPU)
│   ├── baselines.py             # Gate 1 baselines: movement signal, decoding, latent-space analyses
│   ├── baseline_figures.py      # Figures of docs/baselines.md
│   ├── report_figures.py        # Report figures and tables from the evaluation's result.json
│   └── handview.py              # Baselines Addendum A: hand-centred views
├── model/
│   ├── trials.py                # Trial simulator: sessions, timing, block orders (paper's procedure)
│   ├── bank.py                  # Frozen-encoder inputs, hash-checked; training-only standardisation
│   ├── context.py               # H2 context pathway: forearm continuity at the wrist
│   ├── network.py               # Spiking visuomotor network, rate twin, probe readout, threshold
│   ├── train.py                 # Auxiliary training per hypothesis and Gate 2 health check
│   ├── stack.py                 # Trains many networks at once on the GPU (same procedure, ~9x faster)
│   ├── campaign.py              # Trains every fold × initialisation through the health-checked path
│   ├── learned_context.py       # Learned context pathways: forearm prediction (H2-learned), familiarity (H4)
│   ├── extend_curves.py         # Readout excitability curve of every campaign network over the extended sweep
│   ├── evaluate.py              # Evaluation simulator: sessions, probes, trimming, cells, hierarchical bootstrap
│   ├── criteria.py              # Criteria of spec §5.2, treatments of over-driven readouts, verdicts, necessity
│   ├── fit_gate.py              # Fit of c on training sources; sensitivity value c+
│   ├── gate2.py                 # Gate 2 checks: control predictions, probe-strength sweep
│   ├── recurrence.py            # How much a recurrent network uses its recurrent weights
│   └── run_evaluation.py        # The held-out evaluation: simulate, score, decision; locked until the freeze
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
│   ├── rehearse.py              # Rehearsal of the evaluation on inner (training) sources
│   └── validate_dataset.py      # Checks the downloaded, historically accepted archive
├── tests/
│   ├── test_dataset.py          # CPU regression checks, including failure cases
│   ├── test_silhouette.py       # Silhouette geometry, registration, seams, wrist cut, frozen mattes
│   ├── test_controls.py         # Control geometry, equal areas, scramble, frozen controls
│   ├── test_human_target.py     # Human target checks; published-statistics reproduction
│   ├── test_baselines.py        # Baseline statistics, leave-one-source-out decoding, rendering
│   ├── test_handview.py         # Hand-centred windows and analysis
│   ├── test_raters.py           # Rater-validation package
│   ├── test_trials.py           # Trial simulator and feature bank
│   ├── test_context.py          # Forearm-continuity estimate
│   ├── test_network.py          # Network units, probe readout, training, health measures, campaign
│   ├── test_learned_context.py  # Learned context pathways
│   ├── test_evaluate.py         # Evaluation simulator
│   ├── test_criteria.py, test_fit_gate.py, test_gate2.py  # Criteria, fit of c, Gate 2 checks
│   ├── test_run_evaluation.py   # Evaluation driver and its freeze lock
│   ├── test_rehearse.py, test_report_figures.py, test_criteria_status.py  # Rehearsal, report figures, status table
│   └── test_spec.py             # The spec equals its recorded hash
├── report/
│   └── project-report.md        # Project report (skeleton; results filled after the evaluation)
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
│   ├── independent-review.md    # Independent review of 10 Nov 2026 and progress on its findings
│   ├── decision-audit.md        # What each decision rests on
│   ├── model-method.md          # Model methods, health checks, fit of c, Gate 2
│   ├── final-phase-plan.md      # Plan from 11 November 2026 to the hand-in
│   ├── criteria-status.json     # Per criterion and model: built in, calibrated, previewed or open
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

Done on 24 October:

- Hand-centred views (baselines Addendum A): the reversal persists when every display is seen
  through the same window around the hand, with the hand up to twice as large.
- Trial simulator and model inputs ([model method](docs/model-method.md)); rater-validation
  package prepared, not run by the owner's decision.
- Encoder decided (OpenCLIP primary, DINOv2 for robustness); spiking visuomotor network, auxiliary
  training and health check built; H2's context pathway (forearm continuity) decided.
- Pre-training validation ([model method](docs/model-method.md) §6.10, §8): the probe readout did
  not register the trained motor response, because the motor population was driven far above rest.
  Training now keeps the motor populations near rest during observation (as the paper's MEPs were
  taken at rest), and new health criteria check the readout itself. H0 and H1 training paths, an
  ungated comparison model, the paper's ± 2 SD trimming and a campaign driver
  (`model/campaign.py`) were added (spec amendment). The corrected spiking network passes every
  health criterion for H2, H0 and H1; the non-spiking twin does not yet learn the task.

Done on 25–26 October:

- Stacked GPU training (about nine times faster); the non-spiking twin repaired; learned context
  pathways (H2-learned, H4) declared; a spinal motoneuron readout tried and not adopted
  ([model method](docs/model-method.md) §6.13–6.15, §7.1).
- **Training campaign finished** ([model method](docs/model-method.md) §6.16): 2,400 networks
  (H2, H0, H1 spiking and the H2 twin, three encoders, 20 folds × 10 initialisations) on six CREMI
  GPUs; 2,016 pass every health criterion on their inner training source. Healthy per 200:
  OpenCLIP 181–200 in every configuration; DINOv2 120–188, lowest for H2 and H1. Failures follow
  the fold, not the seed: on some unseen training sources the motor population is driven past
  its readout peak, on others it does not respond.
- **Readout ranges re-measured and the rule for failing networks declared** ([model method](docs/model-method.md)
  §6.17, spec amendment): the excitability sweep was too short for the twin; extended for all 2,400
  networks. No network is excluded for a source-dependent health failure; over-driven held-out
  displays count only if a result holds both with them removed and at the readout maximum. Every
  premise was tested on all 2,400 networks.
- **Evaluation simulator** built ([model method](docs/model-method.md) §6.18): its fast trialwise mode
  equals the full session on trained networks.
- **Decision audit** ([decision audit](docs/decision-audit.md)): all 48 decisions (49 since 10 November) classed by what grounds
  them, 29 references verified; the model's over-driven readout falls, unlike a human MEP, and is
  reported as a limitation.

- **Criteria, fit of c and Gate 2 checks declared and built** ([model method](docs/model-method.md) §6.19, spec
  amendment), tested on synthetic data before any run; the learned pathways now score training sources by
  cross-fitting.

Done on 8 November:

- **Fit of c and Gate 2 checks started** on four CREMI GPUs (`jessica`, `daniel`, `vladimir`, `cheonech`; machines share the per-network work through claims, `scripts/cremi_parallel.sh`): ADM readout ranges for all 2,400 networks, then c for
  the 1,200 H2 networks (spiking first, then the twins), the probe-strength sweep on every network and the
  H0/H1/H2 control predictions, all on training sources. By 23:40 the sweep (2,400) was complete, c was fitted
  for 599 of 600 spiking networks and 314 of 600 twins, and 1,799 of 1,800 predictions were done; fits that ran
  out of GPU memory left claims that are now taken over automatically (a dead process's claim on the same
  machine is retaken, `train.claim`; `scripts/cremi_retry.sh`).
- **Recurrent variant declared, built and training** ([model method](docs/model-method.md) §6.20, spec
  amendment): recurrent weights in the visual layer, starting at zero so that each recurrent network starts
  as the feedforward network of the same seed; H2, OpenCLIP, 200 networks. Gate 2 keeps the feedforward
  network as primary unless it fails Gate 2's conditions.
- **Robustness variations declared and built** ([model method](docs/model-method.md) §6.21, spec amendment):
  visual layer, learning rate, training length and time constants, one at a time, 60 networks each; training
  in parallel with the recurrent campaign. The 0.5-ms time step is declared; its implementation is pending.
  From 23:26 the remaining recurrent and robustness stacks run from one shared job list on eight CREMI GPUs
  (adding `moneo`, `wensicia`, `hayt`, `albans`, two stacks per new GPU): `campaign.py --claim` claims each
  stack and each health check, so no network is trained twice (`scripts/cremi_work.sh`,
  `scripts/cremi_handover.sh`).
- The twin's simulation no longer copies its rate table to the GPU every millisecond (same values, about
  20 times faster threshold search).

Done on 9 November:

- **Fit of c and Gate 2 checks finished; Gate 2 met** ([model method](docs/model-method.md) §6.19,
  training sources only). The detached-hand target (Δ(DH)/Δ(FB) = −0.10) is outside what a gate can reach:
  with the gate fully closed the ratio is ≈ 0, so the declared fit gives c = 0 in almost every network
  (reported as found; nothing changed). H0, H1 and H2 make distinct control predictions with every encoder,
  and 2,397 of 2,400 probe responses rise smoothly with strength (all primary-encoder networks; the three
  exceptions are H1 DINOv2 networks with a silent readout). The feedforward spiking network stays primary;
  the recurrent network is the ablation.
- **Recurrent and robustness campaigns finished** (eight CREMI GPUs, 23:26–03:10): 200 recurrent networks
  (180 healthy; the feedforward network has 181) and 540 robustness networks (52–60 of 60 healthy per
  variation). The recurrent networks use their recurrent weights (recurrent input a median 29% of the
  feedforward input; trialwise simulation valid, mode check passed). The fit of c gives c = 0 in every one
  of these 740 networks, so the closed-gate boundary holds for recurrence and every varied setting
  ([model method](docs/model-method.md) §6.20–6.21).

Done on 10 November:

- **Independent review** ([review](docs/independent-review.md)): verdict "sound with major revisions", 18
  ranked findings; its progress table tracks what has been handled. No setting is changed in response to it.
- **Ungated results recorded** (review finding F2; [model method](docs/model-method.md) §6.19, from the saved
  fit curves, training sources only): without any gate the trained network already responds less to the
  detached hand than to the full body (ratio Δ(DH)/Δ(FB) 0.49 with OpenCLIP, 0.04 and 0.00 with DINOv2). The
  gate is therefore probably not necessary for the main contrast, and the "clean target" reading of Gate 1 is
  qualified ([baselines](docs/baselines.md)). The H2 − ungated comparison is the test of necessity.

- **Reporting amendments declared and built** (spec amendment of 10 November; [model method](docs/model-method.md)
  §6.19a): necessity of the gate (H2 − ungated, paired), every criterion also on the readouts as simulated, a
  sensitivity value c⁺ for the unidentified gate parameter, and the statement that for H2 only the upper-limb
  half of P2 and S2 are open tests. No criterion, parameter or decision rule changed. `config/analysis.json`
  now lists all 12 amendments with what existed at each, and a test binds the spec to its recorded hash.
- Document contradictions fixed (counts, links, stale predictions).
- **Evaluation driver built and locked** ([model method](docs/model-method.md) §6.22): `model/run_evaluation.py`
  simulates, scores and applies the decision rule; it cannot show a network its held-out source until
  `config/freeze.json` is written at the 11 December freeze and unchanged. Rehearsed on inner sources: it
  reproduces the stored Gate 2 predictions.
- **All-context training ablation declared and built** ([model method](docs/model-method.md) §6.23): 60 networks
  trained on all three displays, to test whether the weak detached-hand response is a training-distribution effect.
- **0.5-ms time step implemented** ([model method](docs/model-method.md) §6.21): durations and rates are expressed in
  time throughout; the 1-ms results are unchanged (four stored networks reproduce their health values exactly).
- **All-context ablation finished** ([model method](docs/model-method.md) §6.23; `jessica`, 19:32–21:52): 60
  networks, 57 healthy. Trained on all three displays, the ungated network facilitates for the detached hand
  slightly more than for the full body (ratio 1.14 [1.03, 1.25]; primary, same folds and seeds: 0.53). The
  primary network's weak detached-hand response is therefore a training-distribution effect, and with
  experience of every display a gate is needed to obtain the human pattern.
- **0.5-ms subset finished** (§6.21; 21:52–23:18): 20 networks, 18 healthy (matched primary: 19); c = 0 in every
  one; ungated ratio 0.51. The readout ranges shift with the step (FDI 7.95 → 6.37 Hz, ADM 4.79 → 6.47 Hz).

Done on 11 November:

- **External timestamps found** ([push times](docs/push-times.json)): GitHub's record shows every commit pushed
  within 65 minutes of its commit time, so the dates in the amendment log are externally supported.
- **Author's answers to the review's questions recorded** ([review](docs/independent-review.md) §8). Claim adopted
  for the report: the pipeline tests attachment accounts of the human pattern; it does not assert that
  attachment explains it. Two documents are planned from one set of results: the full project report and a
  paper-style methods article (outline in the review).
- **Final phase started** ([final phase plan](docs/final-phase-plan.md), block A). Criterion-status table as data
  ([criteria-status.json](docs/criteria-status.json), A2): for every criterion and model, whether the held-out run
  can still show anything (for H2 only the upper-limb half of P2 and S2 are open). Report figures and tables as code
  (`analysis/report_figures.py`, A1): human vs model cells, contrasts per model, the two training diets, control
  displays, criterion, robustness and over-driven tables, tested end to end on a synthetic `result.json`; the
  scored result now also carries each model's mean cells with intervals (descriptive, no criterion uses them).
  Rehearsal of the evaluation on every network's inner source started on `joe` (`scripts/rehearse.py`, A3; full
  body, silhouette and controls only). Project report skeleton drafted (`report/project-report.md`, A4): every
  section in place, methods and pre-evaluation results written, evaluation sections as stubs naming the
  `result.json` fields they will quote. 189 tests pass.

Next:

1. Follow the [final phase plan](docs/final-phase-plan.md) (11 November to the hand-in): report figures and
   skeletons and a full rehearsal on inner sources by 20 November; no setting changes.
2. Discuss the closed-gate boundary of the fit of c (a gate cannot make the detached hand fall below
   baseline) before the configurations are frozen; no setting is changed in response to it.
3. Gate 2 on 4 December, configurations frozen 11 December, single held-out evaluation 14–18 December. Every
   analysis is repeated without reference 17 (456 families). A fresh CUDA canary is deferred.

Image acceptance does not establish physiological or scientific replication.

Models retain their own licenses: [RealVisXL](https://huggingface.co/SG161222/RealVisXL_V4.0),
[Xinsir Canny](https://huggingface.co/xinsir/controlnet-canny-sdxl-1.0),
[fixed SDXL VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix).
