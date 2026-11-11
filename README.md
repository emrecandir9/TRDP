# TRDP: can an image-driven visuomotor model reproduce display-dependent motor facilitation?

Muhammed Emre Candir · TRDP project, University of Bordeaux, 2026

Matamala-Gomez, Frisco et al. (2025, *Sci. Rep.* 15, 13364) recorded motor-evoked potentials
(MEPs) while participants watched an index finger move in three displays: a **full body**, an
**upper limb** and a **detached hand**. Watching the movement facilitated the finger muscle for the
full body and the upper limb, but not for the detached hand.

This project asks whether a model that sees only images of these displays, with the paper's trial
timing, can reproduce that pattern, and what the model has to contain. It builds a controlled
stimulus set, a participant-level human target from the paper's open data, and a spiking
visuomotor network with competing context accounts. All of this was tested in one pre-specified,
frozen evaluation on held-out images. The project tests attachment accounts of the human pattern.
It does not assert that attachment explains it, and it makes no claim of physiological equivalence
between a model readout and an MEP.

## Study in brief

- **Stimuli.** 480 image families were generated from 20 approved source pairs (4 desks × 2 seeds
  × 3 white balances). Each family has a static and a movement frame in every display: 2,880
  PNGs (2048 × 2048), plus 960 paper-matched detached-hand silhouettes and six matched control
  displays. Every image was checked by hash and approved by a person.
- **Human target.** `analysis/human_target.py` rebuilds participant-level effects from the
  unchanged OSF deposit and reproduces every published statistic.
- **Visual baselines (Gate 1).** Pixel, low-level, DINOv2 and OpenCLIP baselines all show the
  reverse of the human pattern: the largest movement signal is for the detached hand.
- **Models.** A spiking visuomotor network on frozen image features (OpenCLIP primary, DINOv2
  for robustness), compared under several context accounts:
  - **H0:** no context.
  - **H1:** low-level salience.
  - **H2 (primary):** an attachment gate driven by the visible forearm continuity at the wrist.
  - **Comparison models:** the ungated network, learned attachment and familiarity estimates,
    a rate-based twin, a recurrent variant and robustness variations.

  In total, 3,220 networks were trained, 20 source folds × 10 initialisations per primary
  configuration.
- **Procedure.** The analysis spec was frozen before any model result, and each later amendment
  is dated and hash-recorded. Gate 2 checks ran on training sources only. Code and inputs were
  frozen on 11 November 2026 (`config/freeze.json`, tag `freeze-2026-11-11`), and the held-out
  evaluation was then run and scored once.

## Result

The decision rule was met. The one blind prediction holds: on held-out sources the network
facilitates for the upper limb, Δ(UL) = 1.54 [1.34, 1.73], at **0.96 of the full-body effect**
(human 0.77 [0.34, 1.78]). This holds under every analysis and in all 13 comparison
configurations, except DINOv2 mean-patch, where the result is undetermined.

What the result does not show:

- **The gate is not needed for the transfer.** The ungated network transfers the same way.
- **The gate matters only for the detached-hand ratio, and only after full-body training.**
  Without the gate that ratio is 0.66, above the human interval.
- **The paper's three displays cannot separate the accounts.** The visible-area gate, the
  learned attachment estimate and salience meet the same criteria there.
- **Only the matched controls separate them, and they have no human data.** In an exploratory
  analysis after the evaluation, only H2 predicts a difference on both control pairs, so a human
  experiment on these images could test it.

Results, figures and tables are in [docs/evaluation/](docs/evaluation/), with
[tables.md](docs/evaluation/tables.md) as the summary. The method and every intermediate result
are in [docs/model-method.md](docs/model-method.md).

## Repository

Each folder has its own `README.md` with a file-by-file map.

```text
config/     recipe.json (generation), analysis.json (cohort, hashes, amendments), freeze.json
analysis/   human target, Gate 1 features and baselines, report figures, exploratory controls
model/      trial simulator, network, training, campaigns, fit of the gate, Gate 2, evaluation
scripts/    dataset generation, review and validation; silhouettes; controls; rehearsal; audit
tests/      192 unit and regression tests (no dataset or GPU needed)
data/       tracked metadata and hashes; images and OSF files are local only (see below)
docs/       spec, methods, validation, review, approvals, figures and evaluation results
report/     project report, paper-style report, verified references
```

Main documents:

- [Analysis spec](docs/analysis-spec.md): endpoints, splits, models, criteria and the amendment log
- [Model method](docs/model-method.md): network, training, health checks, Gate 2 and evaluation
- [Baselines](docs/baselines.md): the Gate 1 visual baselines
- [Controls](docs/controls.md) and the [detached-hand silhouette](docs/detached-hand-silhouette.md): stimulus derivations
- [Validation](docs/validation.md): what was checked and what remains untested
- [Independent review](docs/independent-review.md) and [decision audit](docs/decision-audit.md)
- [Final-phase plan](docs/final-phase-plan.md): the schedule to the hand-in

## Setup

Python 3.12 or 3.13. Run all commands from the repository root.

```sh
./scripts/setup.sh review              # CPU environment in .venv
source .venv/bin/activate
python -m unittest discover -s tests   # all tests; no data needed
```

GPU work (feature extraction, training, evaluation) uses `requirements-model.txt`, which builds on
`requirements-features.txt` (PyTorch, CUDA 12.8). Regenerating the stimuli needs
`./scripts/setup.sh cuda` and one NVIDIA GPU with at least 12 GB free.

**Local data.** Only code, configuration, metadata and documents are in Git. To run the checks and
analyses, copy these from the project's local copy into the same paths:

- `data/accepted/`, `data/accepted_silhouette/`, `data/sources/`, `data/inputs/` (the images)
- `data/human/` (the OSF deposit)
- `output/features/` (the GPU embeddings)

Do not regenerate approximations of the frozen inputs.

## Reproducing the pipeline

```sh
# Stimuli and target (CPU)
python scripts/validate_dataset.py         # archive hashes, formats, approval, protected pixels
python scripts/silhouette.py verify-archive
python scripts/controls.py check
python analysis/human_target.py            # -> output/human_target/

# Gate 1 baselines
python analysis/features.py                # GPU: image embeddings
python analysis/baselines.py

# Model (GPU; every step can be sharded across machines with --shard/--shards)
python model/campaign.py --seeds 0-9       # train and health-check one configuration
                                           #   (--hypothesis, --representation, --variation)
python model/extend_curves.py              # readout ranges, then --apply
python model/fit_gate.py                   # fit of the gate parameter c on training sources
python model/gate2.py predictions          # Gate 2 checks; also: sweep, summary
python model/run_evaluation.py simulate    # held-out evaluation (locked unless freeze.json matches)
python model/run_evaluation.py score
python analysis/report_figures.py          # figures and tables from result.json
```

`model/run_evaluation.py` refuses to show any network its held-out source unless every file
recorded in `config/freeze.json` is unchanged. `scripts/cremi_evaluate.sh` is the launcher that ran
the evaluation on several lab PCs. Stimulus generation and review are described in the
[project guide](docs/project-guide.pdf) (11 October 2026; it predates the silhouette, the controls
and the model).

## Notes

- Image acceptance does not establish physiological or scientific replication. Reference 17 has
  a known clothing issue, accepted with that flag ([approval receipt](docs/approval-receipt.json)).
- A family's images are paired views, not independent observations.
- Generation models keep their own licenses: [RealVisXL V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0),
  [Xinsir ControlNet Canny SDXL](https://huggingface.co/xinsir/controlnet-canny-sdxl-1.0),
  [SDXL VAE fp16 fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix).
