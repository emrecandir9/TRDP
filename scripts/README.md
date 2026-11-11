# scripts/

Environment setup, stimulus production and validation, derived stimuli, and the integrity checks
before and during the evaluation.

**Setup**

| File | Contents |
| --- | --- |
| `setup.sh` | Creates `.venv`: `review` (CPU) or `cuda` (Linux generation with pinned package versions) |
| `models.py` | Downloads or resolves the three pinned generation models |

**Stimulus generation (CUDA)**

| File | Contents |
| --- | --- |
| `run.py` | Logs and launches the CUDA worker; `--cooperative` waits for an idle GPU |
| `generate.py` | Shared desk proposals and export of the image families (`canary`, `full`; `--preflight` checks only) |
| `dataset.py` | Source checks, composition, verification and human review records (`check`, `review-template`, `review`, `verify`) |

Generation workflow: `models.py --download` → `generate.py canary --preflight` → `run.py canary` →
human review → `run.py full --cooperative` → human review → `dataset.py verify`. Only decisions
recorded as `reviewer: human_user` are accepted.

**Validation and derived stimuli**

| File | Contents |
| --- | --- |
| `validate_dataset.py` | Checks the accepted archive: hashes, formats, approval, protected pixels, contexts; `--hand-deviation` |
| `segment_hands.py` | SAM-HQ hand segmentation of the final frames (CPU) |
| `silhouette.py` | Paper-matched detached-hand silhouettes (`build`, `check`, `export-archive`, `verify-archive`) |
| `segment_arms.py` | SAM-HQ hand and forearm segmentation for the controls |
| `controls.py` | Control stimuli of analysis spec §7.2 (`build`, `check`, `review`) |
| `raters.py` | Rater-validation package (prepared; not run, by the owner's decision) |

**Evaluation support**

| File | Contents |
| --- | --- |
| `rehearse.py` | Rehearsal of the evaluation on every network's inner (training) source |
| `audit_outputs.py` | Completeness audit of all training-source outputs before the freeze |
| `cremi_evaluate.sh` | Launcher that ran the held-out evaluation with claim-based workers on several CREMI lab PCs |
