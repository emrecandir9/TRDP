# Validation — 11 October 2026

The original accepted archive passes its **historical integrity checks**. The corrected
input masks pass their own checks. These are different mask versions: the accepted
images were preserved unchanged, as requested. A corrected-mask generation run has
not been sampled or visually approved.

## Results

| Scope | Result |
| --- | --- |
| CPU regression suite | 10 tests passed, including protected composition, changed-file rejection, original-mask provenance, immutable checkpoints, canary gating, idle timing and pinned model resolution |
| Frozen local bundle | All 220 hashes match: 120 RGB 2048-square source PNGs and 100 preparation inputs; all formats, dimensions and derived context pixels checked |
| Corrected masks | Binary mask equals weight > 0; no overlap with the original action. All 20 hash-bound historical hand mattes were resized into their native crops and independently confirmed protected |
| Approved archive | All 480 unique families, 2,880 original PNG hashes and 480 original audit records checked |
| Original provenance | All 160 parent JSONs match the family audit hashes; all 20 original desk weights match the parent input hashes and recorded previous versions |
| Historical pixel invariants | Zero protected-pixel errors under the original masks; action differences stay inside original support; retained context pixels match; hidden pixels are black |
| Corrected-guard comparison | All 480 families have deviations: 4,116,516 changed pixel positions summed across the 960 full-body A/B images. This includes the larger protected contact buffer; it is not a count of distinct people, anatomical defects or corrupted files |
| Worked example | Reference 12 (male), birch, seed 9: composition from the original saved proposal and original weight matches the saved native composite exactly; warm export pixels also match |
| Illustrated guide | 25 pages rendered and visually inspected; bookmarks, text extraction, page dimensions and image provenance checked |
| Large local review PDF | 480 pages, 960 form fields, readable and unencrypted; retained unchanged. Its 480 pages were not freshly re-rendered |
| Publication boundary | Dataset, original-mask provenance, frozen input images, environment, new outputs and large review PDF are ignored by Git; no Git LFS |

The detailed local result is `data/accepted/validation.json`, including every family ID,
audit hash and corrected-guard deviation count. The original archive contains six PNGs
plus `audit.json` per family. Its `_provenance/` holds only the 20 original weights and
160 parent records needed to validate it. No replacement images or redundant preview
exports were added.

An additional check of the 20 beech / seed-9 / neutral static exemplars found changed
pixels inside the opaque source-hand matte (alpha >= 250) for references 10, 11, 12,
16, 18 and 20: respectively 136, 107, 352, 6, 162 and 78 pixels. The other 14 exemplars
were exact there. This confirms that the earlier mask issue can affect hand pixels,
in addition to the contact buffer. This exemplar check is not an exhaustive anatomy
assessment; the approved originals remain preserved.

## Project files checked

| File | Check |
| --- | --- |
| `README.md` | Directory map, local links and documented CLI entry points checked |
| `requirements.txt` | Pinned NumPy 2.3.5 / Pillow 12.0.0 installed by the review setup |
| `.gitignore` | Required local-only paths confirmed ignored |
| `.gitattributes` | PDF binary handling; no LFS rules |
| `config/recipe.json` | Valid JSON; 20 sources / 160 parents / 480 jobs / 10 canaries; frozen-manifest hash matches |
| `data/frozen_sources.json` | Valid JSON; every referenced local image rehashed and checked; correction retains previous input hashes |
| `scripts/setup.sh` | Shell syntax and actual Mac review-environment setup passed |
| `scripts/models.py` | Syntax, CLI and pinned/cache-only/download behavior tested with a mock hub |
| `scripts/generate.py` | Syntax, CLI, imports, existing-output and incomplete-checkpoint behavior checked; sampling untested |
| `scripts/run.py` | Syntax, CLI, moved worker path and continuous-idle reset behavior checked |
| `scripts/dataset.py` | Protected composition, white balance, exports, source/context checks and hash-bound review safeguards exercised |
| `scripts/validate_dataset.py` | Full downloaded archive checked; corrupt approval/weight cases covered by regression tests |
| `tests/test_dataset.py` | Syntax checked; all 10 CPU tests passed |
| `docs/project-guide.pdf` | Full visual layout review and saved-image/parameter checks completed |
| `docs/accepted-dataset.json` | 480 approvals, unique IDs, all 2,880 PNG hashes and 24 reference-17 flags checked |
| `docs/approval-receipt.json` | Historical approval scope, counts and known issue checked; original receipt preserved |
| `docs/validation.md` | Results reconciled with the local verification record and command outputs |

## Recheck

Run from the repository root after `./scripts/setup.sh review`:

```sh
source .venv/bin/activate
python -m unittest discover -s tests -v
python scripts/dataset.py check
python scripts/validate_dataset.py
```

Checks need the local bundle described in the README; CPU tests work without it.
The archive validator reports historical success and separately prints the corrected-
guard note. It does not rewrite the original accepted PNGs or audit records.

Current frozen-source manifest SHA-256:
`2495d9098ad6b7b8df014b7e56223c76ebf9cb5295fd88d0aa26c748c8e38faf`.

Local large-review PDF SHA-256:
`bc53cede317c3733c5bd5078b310f7d7d650e41593539ef6fe6bc99f31841447`.

## Human target — 22 October 2026

| Check | Result |
| --- | --- |
| OSF files | `data/human/` hashes equal the 26 September download record |
| Deposit audit | 11,280 MEPs, 32 participants; 240 missing of 11,520 designed (2.08%, as reported); 93 below 50 µV retained |
| Published Table 2 | 8/8 legible coefficients and z values reproduced: raw mV, participant random intercept, maximum likelihood |
| Published post-hoc z values | 10/10 values in the Results text reproduced by separate subset models |
| Primary contrast `C_FDI` | 0.283 participant SD [0.147, 0.419], t(31) = 4.06; interval excludes 0 on all five scales |
| Test suite | 17 CPU tests passed, including 7 for the human target |

The refit verifies that the deposit matches the published analysis; it does not endorse that
analysis. Discrepancies with the paper's Methods are listed in
[analysis-spec.md](analysis-spec.md) §4.2.

## Detached-hand silhouette — 22 October 2026

| Check | Result |
| --- | --- |
| Inputs | 20 hand mattes and 20 case geometries copied into `data/inputs/`; hashes equal `frozen_sources.json` |
| Segmentation | SAM-HQ ViT-B, checkpoint SHA-256 verified, code at a pinned commit, CPU; 40 masks byte-identical across repeated runs; prompts and predicted IoU (0.972–0.989) in `data/hand_segmentation.json` |
| Mattes | 40 mattes, deterministic; hashes in `data/silhouettes.json`, bound by `config/recipe.json`; static IoU with the earlier reviewed matte 0.978–0.993 |
| Invariants | Movement and static mattes identical outside the changed region dilated by 9 px; nothing outside the crop or past the wrist line; area ratio 0.990–1.005 |
| Archive export | 960 PNGs for 480 families; every image recomputed exactly from its full-body PNG by `verify-archive` |
| Code paths | `dataset.py check` verifies the mattes; new runs export 8 PNGs per family; the archive validator still checks the historical 6 (sample of 24 families re-run on real data) |
| Test suite | 23 CPU tests passed, including 6 for the silhouette (segmentation hash check and seam closing among them) |
| Visual | Assistant inspected every fingertip, index web and finger valley of all 40 frames at 2×, suspicious spots at 4×; remaining observations are image content, listed in [detached-hand-silhouette.md](detached-hand-silhouette.md). Approved by the project owner on 22 October ([record](silhouette-approval.json)) |

## Archive hand pixels under the silhouette — 22 October 2026

`python scripts/validate_dataset.py --hand-deviation` compares every archive full-body image
with its frozen V7 source after the same white balance, inside the silhouette mattes.

| Measure | Result |
| --- | --- |
| Affected families | 144 / 480: every variant of references 10, 11, 12, 16, 18 and 20 |
| Deviating visible pixels per image | median 0; maximum 645 (0.65% of the visible hand) |
| Largest difference | 6 of 255 levels; no pixel differs by more than 10 |
| Static vs movement | Deviations identical in both frames, so the movement contrast is unaffected |
| Location | Outer palm edge on the little-finger side near the wrist, away from the moving index |

Cause: the historical desk masks reached into the hand; the diffusion blend there was faint.
**Decision (project owner, 22 October 2026): keep all 480 families, with no exclusion or
sensitivity analysis for these deviations.** Reason: they are below 2.4% of the intensity
range, cover under 1% of the hand, sit away from the action and cancel exactly in every
static/movement comparison. This is consistent with the frozen cohort; no spec amendment is
needed.

## Review rules — 22 October 2026

`dataset.py review` records only `reviewer: human_user`; assistant inspections and numeric
checks are advisory. A regression test rejects an assistant review file.

## Control stimuli — 22 October 2026

| Check | Result |
| --- | --- |
| Forearm segmentation | SAM-HQ (pinned), 20 static frames, predicted IoU 0.971–0.998, byte-identical on repeat |
| Integrity | 6 controls × 20 sources: hand and every changed pixel identical to the full-body image; static/movement renders differ exactly at the movement |
| Matched pairs | `hand_stub` vs `hand_gap_stub` area difference < 10⁻¹¹; `window_hand` vs `window_arm` < 10⁻⁵ |
| Scramble | Valid block permutation per source; fixed blocks unchanged |
| Low-level statistics | Matched pairs agree in visible area, luminance, changed-pixel share and movement distance ([controls.md](controls.md)) |
| Tests | 28 CPU tests passed, including 4 for the controls |
| Visual | Assistant inspected all 20 sources; approved by the project owner on 22 October ([record](controls-approval.json)) |

## Reproduction on CREMI and Gate 1 baselines — 23 October 2026

| Check | Result |
| --- | --- |
| Transfer | Working copy `/net/travail/mucandy/TRDP` on CREMI: all 5,276 files (excluding environments and caches) hash-identical to this copy; git at the same commit as GitHub |
| Linux reproduction (Python 3.13, `daniel`) | 28 tests; `dataset.py check`; `validate_dataset.py` (480 families, rerun identical to the frozen `validation.json` except its timestamp, which was restored); `silhouette.py check` and `verify-archive`; `controls.py check`; hand-pixel deviation CSV byte-identical; human target identical within 4 × 10⁻¹³ (floating-point library differences) |
| Baseline inputs | 10,560 condition images rendered from the hash-checked archive, each hashed in `output/features/index.csv`; rendered controls equal `controls.render` (test) |
| Encoders | DINOv2 and OpenCLIP at pinned revisions, float32 with TF32 off; noise floor ≤ 1.2 × 10⁻⁵ of the feature norm, at least 700 times below the smallest movement |
| Tests | 44 CPU tests, including 16 for the baselines (statistics, leakage-free decoding, rendering) |

## Hand-centred baseline on a second GPU — 24 October 2026

| Check | Result |
| --- | --- |
| Cross-machine reproduction | One Gate 1 family (22 images) re-encoded on `jessica`: features bit-identical to those computed on `daniel`, identical pixel hashes, same pinned revisions and weights |
| Addendum A run | 28,800 hand-centred images; encoders checked against the Gate 1 revisions and weight hash; noise floor ≤ 3 × 10⁻⁶ |
| Geometry | Full-body and upper-limb crops pixel-identical in all 480 families at s = 1.5 and 2, different in all 480 at s = 3, as predicted |
| Tests | 87 CPU tests pass |

## Pre-training validation of the model — 24 October 2026

Review of every step before the training campaign, against the paper's Methods, the frozen spec and
the code ([model method](model-method.md) §6.10–6.11, §8; spec amendment of 24 October). Training
sources and full-body trials only; no held-out source, and no probe or movement contrast on another
display.

| Check | Result |
| --- | --- |
| Probe readout measures the trained response | **Failed, corrected.** FDI fired at 26 Hz before the probe on movement trials and the readout did not differ from static trials (−0.47 spikes [−1.32, 0.38]); the population's probe response falls once it fires above about 9 Hz. After holding the motor populations near rest (the paper excluded trials with background EMG) and making each population a homogeneous pool: facilitation +2.3 [1.5, 3.1], +2.6 [1.9, 3.4] and +3.0 [2.3, 3.7] spikes in three H2 networks (two seeds, two encoders), +1.5 for H0 and +2.8 for H1; all criteria pass for these five. DINOv2 mean patch token misses the task criterion (AUC 0.891 vs 0.9) with positive facilitation (+1.3) |
| Non-spiking twin | **Not usable as built**: its softplus rate unit is far from the spiking neuron's regime and it did not learn the task (AUC 0.0); to be repaired before Gate 2 |
| Paper's preprocessing mirrored | Background-EMG exclusion and ± 2 SD trimming had not been mirrored; added |
| Health criteria implemented as declared | "> 200 Hz" missing, "silent" measured on one batch, accuracy confounded by source offsets; corrected (AUC, all training sources) |
| Gate necessity testable | Ungated comparison model added |
| What is built in for H2 | Stated per criterion ([model method](model-method.md) §7) |
| Citations | Three added references checked against publisher records |
| Tests | 105 CPU tests pass, including 12 for the network, health measures and campaign driver |

## Training campaign — 25–26 October 2026

Health of every campaign network on its inner training source ([model method](model-method.md)
§6.16). No held-out source was used, and no setting was changed in response.

| Check | Result |
| --- | --- |
| Completeness | 12 configurations × 20 folds × 10 initialisations: 2,400 networks trained, 2,400 health reports, none failed to run, no partial stacks left (checked on CREMI, 26 Oct) |
| Health, OpenCLIP | H2 spiking 181, H0 198, H1 190, H2 twin 200 of 200 |
| Health, DINOv2 summary token | H2 spiking 143, H0 188, H1 138, H2 twin 171 of 200 |
| Health, DINOv2 mean patch token | H2 spiking 120, H0 185, H1 126, H2 twin 176 of 200 |
| Failing criteria | Only task, facilitation and background; all other criteria pass in every network |
| Failure pattern | Follows the fold, not the seed: over-driven (pre-probe FDI 8–24 Hz spiking, 15–21 Hz twin, past the readout peak, so facilitation collapses) or no transfer (rest-level rate, task AUC < 0.9). H1's gate on sources 17 and 18 exceeds its training range |
| Readout ranges | Excitability sweep extended to 0.6 for all 2,400 networks (model-method §6.17): every curve peaks inside it; spiking ranges unchanged; twin ranges 14.1 → 15.7 Hz, 20 twins no longer over-driven (the twin counts above include this) |
| Rule for failing networks | Premises tested on all 2,400: network-level criteria pass in all; pre-probe rate rises with excitability in all; readout falls past its peak in all; within-range readouts never exceed the peak (largest ratio 1.005); non-transferring networks read near rest (145 / 145). In 7 of 12 configurations 1–6 folds have no healthy network, which no longer matters: no network is excluded for these failures |
| Evaluation simulator | Trialwise vs continuous session on trained networks (full body, inner source): identical trials without noise (spiking 100 / 100, twin within 9 × 10⁻⁵); no cell differs with noise (20 repeats, Holm) |

## Criteria, fit of c, recurrent variant and robustness variations — 26 October to 8 November 2026

Code checks only; no fit of c, Gate 2 check or recurrent network result exists yet
([model method](model-method.md) §6.19–6.21).

| Check | Result |
| --- | --- |
| Criteria, fit of c, Gate 2 checks (26 Oct) | Tested on synthetic data before any run: treatments of over-driven cells, verdicts, paired bootstrap, choice of c (ties, undefined fits), batched grid bit-identical to the §6.18 trialwise simulation for one model |
| Twin speed (8 Nov) | The rate table was copied to the GPU every millisecond; copied once now; the twin's resting threshold is unchanged (0.0390625 both ways), 62 s → 3.4 s |
| Recurrent variant (8 Nov) | Zero recurrent weights reproduce the feedforward network of the same seed exactly (spiking and twin); self-connections have no effect; recurrence acts with a one-step delay; training changes W_rec from the first update and never its diagonal; stacked training equals one-at-a-time training; checkpoints round-trip |
| Robustness variations (8 Nov) | Each changes exactly one declared setting; names and campaign commands carry it; a stack with a changed time constant equals the networks trained alone; real-feature preparation gives the declared sizes, constants and learning rates |
| Tests | 152 tests pass on the Mac (one GPU-only test skipped); the 143 of 26 Oct pass on `jessica`, GPU tests included |
| Claims across machines (8 Nov) | A dead process's claim on the same machine is taken over by exactly one process, never another machine's; campaign stacks and health checks are claimed; 154 tests pass |

## Fit of c and Gate 2 checks — 8–9 November 2026

Training sources only ([model method](model-method.md) §6.19).

| Check | Result |
| --- | --- |
| Completeness | ADM curves 2,400/2,400 (every peak inside the sweep); c for 1,200/1,200 H2 networks; sweep 2,400/2,400; predictions 1,800/1,800. Fourteen out-of-memory fits rerun |
| Fit of c | Target −0.10 reachable in ≤ 2 of 200 networks per configuration and gate; closed gate gives a ratio ≈ 0, so c = 0 almost everywhere; flat curves in H2 DINOv2 patch-mean make c uninformative there; treatments agree; no undefined fit |
| Distinct predictions | H0–H1, H0–H2, H1–H2 distinct under both treatments for all three encoders (condition: primary encoder) |
| Probe sweep | 2,397/2,400 smooth; all primary-encoder networks; three H1 DINOv2 networks with a silent readout are not |
| Gate 2 | Met; feedforward spiking network stays primary (§6.20) |
| Recurrent mode check | Fold 1, seed 0: noise off 100/100 identical; noise on no cell differs (smallest adjusted p 0.67); trialwise mode used |
| Recurrent and robustness networks | 740/740 trained, ADM curves applied, c fitted (two fits on `vladimir` ran out of memory beside another user's job and were rerun on `jessica`); c = 0 in every network; recurrent input median 29% of feedforward |

## Independent review — 10 November 2026

[Review](independent-review.md) of the whole project at commit `4fd2d50`; training-source outputs read on
CREMI; no held-out source or upper-limb display simulated.

| Check | Result |
| --- | --- |
| Tests | 154 pass on the Mac (one GPU-only test skipped) and 154 on `jessica` with none skipped; 176 after this day's additions |
| Human target | Re-run: 18 of 18 published statistics reproduced; tables byte-identical to the stored ones |
| Integrity scripts | `dataset.py check`, `silhouette.py check` and `verify-archive`, `controls.py check` pass |
| Criteria code | Hand-built cases: bootstrap at source level (two implementations identical), treatments, ratios, verdicts as declared. A model with full-body facilitation only meets P1, P3 and S1 |
| Campaign and fit numbers | Health counts and fitted c match the documents; ungated ratios newly recorded (model-method §6.19) |
| Code on CREMI | `model/*.py` hash-identical to the repository |
| Stacked training | Fold 4, seed 0, alone vs in a stack: different weights, same health measures |
| Provenance (added the same day) | New outputs record the hash of every model source file and the commit; the feature manifest and both learned gate tables are pinned in `config/analysis.json` (identical on the Mac and CREMI) and checked at load |
| 0.5-ms time step (implemented the same day) | Six tests; four stored campaign networks give exactly their stored health values with the refactored code at 1 ms |
| All-context ablation and 0.5-ms subset (run the same evening) | 60 and 20 networks trained, none failed to run; 57 and 18 healthy; ADM curves peak inside the sweep in all 80; c fitted for all 80 (c = 0 throughout, no undefined fit, no error in any log) |
| Verdict | Sound with major revisions; findings and progress in the review |
| Evaluation driver (built the same day) | 9 tests (lock, runs and labels, open gate = ungated, verdicts, repetitions); rehearsal on four real networks' inner sources reproduces the stored Gate 2 control predictions; 176 tests pass at the end of the day |

## Final phase, block A — 11 November 2026

Training-source data only; no held-out source or upper-limb display simulated.

| Check | Result |
| --- | --- |
| Tests | 189 pass on the Mac (one GPU-only test skipped): 13 new for the rehearsal, the report figures and the criterion-status table |
| Report figures (A1) | Run end to end on a synthetic `result.json` made by the evaluation's own scoring from small untrained networks (all three encoders and the all-context configuration); every figure and table written; figures inspected |
| Per-cell summaries | `run_evaluation.cell_summary` added for the cell figure; the existing evaluation tests pass unchanged, and no criterion reads it |
| Criterion status (A2) | `docs/criteria-status.json` covers every model the evaluation scores and every criterion; a test checks that for H2 only P2-UL and S2 are open, as spec §5.2 (10 November) states |
| References (A6) | The 11 references of review answer 40 looked up in Crossref: title, authors, journal, volume, issue, pages and DOI match in all 11 |
| Completeness audit (B2) | `scripts/audit_outputs.py` on CREMI: networks 3,220/3,220; readout ranges inside the sweep 3,220; fits of c 2,020/2,020 (4 gates × 3 analyses, 0 undefined); sweeps 2,400/2,400; Gate 2 predictions 1,800/1,800; healthy 2,767 (campaign 2,016, recurrent 180, variations 52–60 per 60, all-context 57, 0.5 ms 18), as recorded. A first run reported no readout ranges because the script read the FDI peak flag from the wrong field; fixed and rerun |
| Rehearsal (A3, B3) | All 3,220 networks on their inner source through `run_evaluation.simulate_network` (full body, silhouette, controls). First pass: 411 CUDA out-of-memory failures, caused by snnTorch's neuron registry (about 10 MB kept per network); fixed in `network.py`, test added (fails on the old code), three stored records re-simulated identically; rerun with 0 failures and flat memory. 3.1 GPU-process hours, median 4.0 s, 0.27 MB per record |
| S3(a) withholding | The rehearsal ran the visible-area gate on the inner-source controls, contrary to spec §8 ("not run on the controls before the evaluation"); recorded by spec amendment of 11 November 2026 (hash in `config/analysis.json`, `test_spec` passes) |
| Tests | 192 pass at the end of the day (audit and memory tests added; one GPU-only test skipped) |

## Gate 2 record and freeze — 11 November 2026

Training-source data only; no held-out source or upper-limb display simulated before the freeze.

| Check | Result |
| --- | --- |
| Gate 2 record | Spec §8: primary variant feedforward spiking; 10 initialisations for the primary configurations, 3 for all-context and the variations, 1 for 0.5 ms; 24 configurations, 3,220 networks. Freeze and evaluation moved forward by the project owner (§8, §10) |
| Inputs | Every encoder vector the evaluation looks up (3 encoders × 20 sources × families × all displays × 2 frames, 9,600 per encoder) and every gate value (4 gates × 3 encoders × 20 folds, 4,800 each) exists and is finite; lookups only, nothing simulated |
| Integrity scripts | `dataset.py check`, `silhouette.py check`, `verify-archive`, `controls.py check` pass on the Mac and on `joe` |
| Human target | Rerun on the Mac: every CSV byte-identical (only `summary.json`'s config hash moves with the config). On `joe` (Linux): `ratios.csv`, `cells.csv`, `participant_cells.csv` byte-identical; four CSVs differ by at most 9.2 × 10⁻¹⁴ (floating point), text identical; stored copies restored |
| Tests | 192 pass on the Mac (one GPU-only skip) and 192 on `joe`'s GPU (no skip) |
| CREMI copy | At the pushed commit `ddeca21`, `git status` clean; the five frozen inputs (`ratios.csv`, feature manifest, two learned-gate tables, continuity table) have the same SHA-256 on CREMI and the Mac |
| Freeze | `python model/run_evaluation.py freeze` on CREMI: `config/freeze.json`, 23 files, dated 2026-11-11; the lock opens on the Mac with the same record; a copy with one changed line in `model/network.py` is refused; a second `freeze` is refused. Committed, pushed and tagged `freeze-2026-11-11` Since the freeze, the test that needs a closed evaluation (`test_run_evaluation`, refusal without a record) skips by design; the suite reports 2 skips on the Mac |

## Held-out evaluation — 11 November 2026

Run once, after the freeze, with the frozen code (block D of the final-phase plan).

| Check | Result |
| --- | --- |
| Simulation | `scripts/cremi_evaluate.sh 3` on `joe`, `vladimir`, `moneo`, `jamis`, `daniel`, `cheonech` (18 workers), 16:49–17:01 CET: 3,220 records, 0 failures, 3.34 GPU-process hours, median 4.3 s per network |
| Completeness | One record per declared network in every configuration; all 3,220 written at commit `3e689fd` (clean working copy) with model-code and configuration hashes equal to `config/freeze.json` and freeze date 2026-11-11; no record outside the declared configurations; no record re-simulated |
| Scoring | `run_evaluation.py score` once on `joe`, 31 min 45 s, exit 0. `output/model/evaluation/result.json` SHA-256 `d7430f650beaa277d1089dae91fda353cffa7273b4467e9323bf878e45aecbe4` (2,120,220 bytes); the copy on the Mac has the same hash |
| Decision rule | `decision`: **met** |
| Exploratory controls | `analysis/exploratory_controls.py` on the Mac from the stored records (10 configurations copied from CREMI through the gateway; lab PCs were off): reproduces the declared per-display control effects and H2 − visible-area differences for H2, ungated and visible area exactly (all encoders and analyses). Labelled exploratory (model-method §6.27) |
| Result recorded | [model-method](model-method.md) §6.26; copy of `result.json` (same hash) and the figures and tables of `analysis/report_figures.py` in `docs/evaluation/`. Text values checked against `result.json` field by field |

## Remaining limits

**From the independent review (10 November 2026).** For H2, P1, P3, S1 and S3(b) are built in and the
full-body half of P2 was calibrated; full-body facilitation, the detached-hand ratio and H2's control
predictions are already known on unseen inner sources, so only the upper-limb display is new in the
evaluation. H2's gate is supplied from masks, not computed from the image. The gate parameter is not
identified. The readout is not physiological and its over-driven treatments are a sensitivity analysis.
The spec is a dated pre-specification, not an externally registered preregistration; its commit dates are supported by GitHub's push record (every commit pushed within 65 minutes, `docs/push-times.json`, 11 November 2026).
Further limits to carry into the report: the motor limits, the pooled motor populations and the background
criterion were set in five rounds on 24–25 October against full-body facilitation on unseen inner sources, so
full-body facilitation is a calibrated property of the model, not a finding (F4); the movement frames are
edited images accepted by one unblinded reviewer, so a network may respond to traces of the edit, which
cannot bias a contrast between displays (they share the hand pixels) but limits what "movement" means (F11);
the 20 networks of a seed share their initial weights, trial draws, noise and 17 of 18 training sources, so
intervals describe generalisation over these images given nearly the same training set, and initialisations
are crossed with sources although the bootstrap resamples them as nested (F12; Bengio & Grandvalet, 2004).


**Single unblinded review.** All 480 families were accepted by one reviewer (the project owner),
who knew the hypothesis and which frame was edited. An independent two-rater check was prepared
([protocol](rater-protocol.md)) but not run, by the owner's decision of 24 October 2026; in
particular, no independent check shows that movement frames carry no more visible flaws than
static frames. On 11 November 2026 the owner reported that two other people reviewed the stimuli and
agreed with every acceptance decision; this informal check has no records in the repository and did not
follow the protocol (it does not compare movement and static frames), so the limitation is reduced, not removed.

No fresh CUDA inference or actual model download was run during this cleanup. No new
human review of every downloaded image is implied. The accepted reference-17 clothing
issue remains flagged. A future corrected-mask run requires new canary and full-batch
review. Upstream source independence, anatomical ground truth and physiological
replication remain unvalidated.
