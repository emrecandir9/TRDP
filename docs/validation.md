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

## Remaining limits

**Single unblinded review.** All 480 families were accepted by one reviewer (the project owner),
who knew the hypothesis and which frame was edited. An independent two-rater check was prepared
([protocol](rater-protocol.md)) but not run, by the owner's decision of 24 October 2026; in
particular, no independent check shows that movement frames carry no more visible flaws than
static frames.

No fresh CUDA inference or actual model download was run during this cleanup. No new
human review of every downloaded image is implied. The accepted reference-17 clothing
issue remains flagged. A future corrected-mask run requires new canary and full-batch
review. Upstream source independence, anatomical ground truth and physiological
replication remain unvalidated.
