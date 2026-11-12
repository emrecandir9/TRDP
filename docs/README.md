# docs/

Specification, methods, results, reviews and approval records.

**Specification and methods**

| File | Contents |
| --- | --- |
| `analysis-spec.md` | Endpoints, splits, models, success criteria, decision rule and the dated amendment log (hash-locked) |
| `model-method.md` | Network, training, health checks, fit of c, Gate 2, evaluation and every intermediate result |
| `baselines.md` | Gate 1 visual baselines: method fixed before results, then the results |
| `controls.md` | Control stimuli: design, parameters, checks, figures |
| `detached-hand-silhouette.md` | Silhouette method, checks and residual defects |
| `rater-protocol.md` | Rater-validation protocol (prepared, not run) |
| `project-guide.pdf` | Illustrated stimulus method (11 Oct 2026; predates the silhouette, the controls and the model) |

**Results**

| Path | Contents |
| --- | --- |
| `evaluation/result.json` | Scored held-out evaluation (`model/run_evaluation.py score`) |
| `evaluation/tables.md` | Decision, criteria and effects in readable form (generated, do not edit) |
| `evaluation/*.csv` | `effects`, `criteria`, `robustness`, `over_driven` tables |
| `evaluation/cells.png`, `effects.png`, `diets.png`, `controls.png` | Report figures (`analysis/report_figures.py`) |
| `evaluation/result_exploratory.json`, `controls_all_models_exploratory.*` | EXPLORATORY: every model on the matched controls, after the evaluation |
| `figures/baselines-*.png` | Figures of `baselines.md` |
| `figures/controls-*.jpg`, `silhouette-*.jpg` | Review sheets of the controls and silhouettes |
| `criteria-status.json` | For each criterion and model: built in, calibrated, previewed or open |

**Review, audit and planning**

| File | Contents |
| --- | --- |
| `validation.md` | What was checked, results and what remains untested |
| `independent-review.md` | Independent review of 10 Nov 2026, the author's answers, and progress on its findings |
| `decision-audit.md` | What each design and analysis decision rests on, with verified references |
| `final-phase-plan.md` | Plan from 11 Nov 2026 to the hand-in |
| `push-times.json` | GitHub's record of push times, an external timestamp for the commit history |

**Approval records** (hash-bound)

| File | Contents |
| --- | --- |
| `accepted-dataset.json` | Approval and hashes of all 2,880 archive PNGs |
| `accepted-silhouette.json` | Hashes of the 960 silhouette PNGs |
| `approval-receipt.json` | Acceptance scope and the known reference-17 clothing issue |
| `silhouette-approval.json` | Human approval of the silhouettes |
| `controls-approval.json` | Human approval of the controls |
| `480-family-review.pdf` | Local only: the original interactive review of the 480 families |

**Presentations** (updated 12 November 2026, after the held-out evaluation)

| File | Contents |
| --- | --- |
| `TRDP Study Progress.pptx` | Full deck, 81 slides with speaker notes; section 7 covers Gate 2, the 820 extra networks, the review, the freeze and the result |
| `TRDP Study Short Version.pdf` | Plain-language short version, 24 pages; pages 17–24 cover Gate 2 to the result and next steps |
