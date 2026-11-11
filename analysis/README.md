# analysis/

The human target, the Gate 1 visual baselines, and the report figures. Run from the repository root.

| File | What it does | Output |
| --- | --- | --- |
| `human_target.py` | Participant-level MEP target from the unchanged OSF deposit of Matamala-Gomez et al. (2025): cell means, the 12-cell table, contrasts with paired bootstrap intervals, scale-free ratios, questionnaire comparison, refit of the paper's mixed models (CPU) | `output/human_target/` |
| `features.py` | Gate 1 inputs: renders every condition image, low-level statistics, DINOv2 and OpenCLIP embeddings (GPU) | `output/features/` |
| `baselines.py` | Gate 1 baselines: movement signal, leave-one-source-out decoding, latent-space analyses ([docs/baselines.md](../docs/baselines.md)) | `output/baselines/` |
| `baseline_figures.py` | Figures of `docs/baselines.md` | `docs/figures/baselines-*.png` |
| `handview.py` | Baselines Addendum A: the same analyses through a fixed window around the hand (`extract` on GPU, then `analyse`, `figure`) | `output/handview/`, `docs/figures/baselines-handview.png` |
| `report_figures.py` | Report figures and tables from the evaluation's `result.json` | `docs/evaluation/` |
| `exploratory_controls.py` | EXPLORATORY, after the evaluation: every model's predictions on the matched controls | `docs/evaluation/controls_all_models_exploratory.*` |

Typical order: `human_target.py` → `features.py` → `baselines.py` → `baseline_figures.py`; after
the evaluation, `report_figures.py` and `exploratory_controls.py`.
