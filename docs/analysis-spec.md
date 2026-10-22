# Analysis specification

**Status: DRAFT — 22 October 2026. Freeze on 30 October 2026, before any model result exists.**
After freezing, change this document only through the amendment log (§10), with a date and
reason. Machine-readable values live in [`config/analysis.json`](../config/analysis.json).

Reference study: Matamala-Gomez, Frisco et al. (2025), *Virtual body continuity during action
observation affects motor cortical excitability*, Sci Rep 15:13364,
[doi:10.1038/s41598-025-97695-9](https://doi.org/10.1038/s41598-025-97695-9); data from the
authors' [OSF deposit](https://osf.io/473wg/?view_only=c89faf2295bd4178a0446fab4027c1a5).

## 1. Question and claim boundaries

**Question.** Can a model that receives only images of the paper's three body displays,
presented with the paper's trial timing, reproduce the human pattern of motor facilitation
during observation of index-finger abduction: facilitation for full body and upper limb,
none for the detached hand?

**What may be claimed.** Whether a specified visual-to-motor model reproduces declared human
contrasts on held-out stimulus groups, and which model components are necessary for that.

**What may not be claimed.** Replication of the human experiment; ownership, agency or
presence (the model has no such variable); physiological equivalence between a model
readout and a peripheral MEP; generalisation beyond the 20 source hands.

## 2. Correspondence with the paper

| Paper (Methods) | This project | Deviation |
| --- | --- | --- |
| Left hand, index abduction, first-person view | 20 frozen left-hand static/movement pairs; image-plane abduction 10.5–24.7° (median 18.3°) | Synthetic photographs, not the authors' 360° stereo images |
| Full body: 360° stereo photo of the lab, body co-located | Full first-person scene: desk, arm, torso, legs | Flat image; no co-location; no surrounding room |
| Upper limb: 180° image of arm on table, black surround | Rows above the fitted desk edge kept, black below | Desk remains visible |
| Detached hand: isolated hand, black surround | **Primary:** skin-only silhouette cut at the wrist, on black ([method](detached-hand-silhouette.md)). Sensitivity: the archive's rectangle with desk patch | Outline from SAM-HQ on the final frames; not hand-annotated ground truth |
| Body blocks counterbalanced across participants | Six block orders cycled across model instances | — |
| 60 s static familiarisation per block | 60 s of the static image at block start; model state carried | — |
| 30 static + 30 movement trials per block, randomised | Same counts, randomised per block and instance | — |
| First frame 2,250–3,250 ms (jittered), second frame 750 ms | Same; jitter drawn uniformly in integer ms | Original draws unknown |
| TMS 250 ms after second-frame onset | Common probe pulse 250 ms after second-frame onset | Probe is a model current, not TMS |
| 120% of resting motor threshold (PEST) | Probe = 120% of the model's resting threshold, found on fixation trials (§6.4) | Analogue, not physical intensity |
| MEP peak-to-peak, 5–60 ms after TMS, FDI and ADM | Readout-population response, 5–60 ms after probe, minus a matched no-probe trial; FDI- and ADM-associated readouts | Arbitrary units, not mV |
| 20 fixation-cross trials at session start and end | Black-image fixation trials at start and end, for threshold and drift checks | — |
| Count the finger movements (attention task) | Auxiliary training task: detect/count index movements (§6.3) | — |
| Within-participant z-normalisation of MEPs | Same normalisation applied within each model instance | — |

## 3. Frozen cohort

| Item | Value |
| --- | --- |
| Archive | `data/accepted`, historical accepted batch (original masks) |
| Accepted manifest | `docs/accepted-dataset.json`, SHA-256 `f70a68c4…3ef` |
| Frozen source manifest | `data/frozen_sources.json`, SHA-256 `2495d909…faf` |
| Families / images | 480 / 2,880 archive PNGs + 960 derived silhouette PNGs; without reference 17: 456 families |
| Detached-hand conditions | Primary `detached_hand_silhouette` (`docs/accepted-silhouette.json`, SHA-256 `bcff8067…c2`; mattes `data/silhouettes.json`, `b071f424…ff`; segmentation `data/hand_segmentation.json`, `e2b8b272…9d`); sensitivity `detached_hand` (rectangle) |
| Independent unit | Source lineage group: 20 groups, each with 24 appearance variants × 8 images |
| Repository commit at decision | `a58d936` |

No new stimulus generation happens before the frozen evaluation. The silhouettes and the
control stimuli (§7.2) are masks over these frozen pixels and are not new generations.
Primary results use the silhouette; every result is repeated with the rectangle.

**Open before freezing (owner: you).** Confirm where the 20 grayscale reference photos
(`normal1–20`, `open1–20`) came from, and whether any person appears more than once.
References that share a person are merged into one lineage group. The photos show different
clothing and bodies in one shared setup; image similarity alone cannot settle identity.

## 4. Human target

Computed by [`analysis/human_target.py`](../analysis/human_target.py) from the unchanged OSF
files (hashes in `config/analysis.json`). Outputs go to `output/human_target/`.

### 4.1 Deposit audit and reproduction of the published analysis

- 11,280 MEPs, 32 participants, 346–360 per participant; the designed total is 11,520, so
  240 (2.08%) are missing, exactly the exclusion rate the paper reports.
- 93 amplitudes are below 50 µV although the paper states that cutoff; they are kept in the
  primary analysis and removed in a sensitivity scale.
- `Type` 1 = movement, `Type` 0 = static. `Filter 1` encodes the muscle, not an exclusion.
- **Table 2 is reproduced exactly** (8/8 legible coefficients and z values) by a linear mixed
  model on **raw mV** with a participant random intercept, fitted by maximum likelihood.
- **Every post-hoc z value in the Results text is reproduced exactly** (10/10) by separate
  random-intercept models fitted to each subset: one muscle × body for movement effects, one
  muscle × movement for body comparisons.

### 4.2 Discrepancies with the paper (reported, not corrected)

1. The Methods describe z-normalised MEPs; the published model used raw mV.
2. All published tests treat single trials as observations, with only a random intercept.
   Within-participant variability of the effects is ignored, which inflates z values
   (for example, ADM upper-limb movement effect: published z = 4.67; participant level
   t(31) = 1.86, p = .07 on the Methods scale).
3. Post-hoc z values come from subset models, not from the three-way model; Table 3 differs
   from the text for four FDI comparisons (9.01, 9.01, 5.59, 4.89).
4. Questionnaire scores separate full body from the other two displays
   (full body > upper limb ≈ detached hand), while MEP facilitation separates the detached
   hand (full body ≈ upper limb > detached hand). The reported mediation by ownership is not
   modelled here.
5. In the paper, full body also differed by showing a whole room, and body conditions were
   run as separate blocks, so effects that ignore movement are confounded with block order.

### 4.3 Definitions

For each participant, average the trials in each of the 12 cells `R(b, m, u)`: body b ∈
{FB, UL, DH}, movement m ∈ {static, movement}, muscle u ∈ {FDI, ADM}. Then
`Δ(b,u) = R(b, movement, u) − R(b, static, u)`.

| Family | Contrast | Definition |
| --- | --- | --- |
| Primary | `C_FDI` | `[Δ(FB,FDI) + Δ(UL,FDI)] / 2 − Δ(DH,FDI)` |
| Secondary (Holm) | Six movement effects | `Δ(b,u)` for every body and muscle |
| Secondary (Holm) | Three-way, FB vs DH and UL vs DH | `[Δ(b,FDI) − Δ(b,ADM)] − [Δ(DH,FDI) − Δ(DH,ADM)]` |
| Secondary (Holm) | FB vs UL | `Δ(FB,FDI) − Δ(UL,FDI)` |
| Exploratory (Holm) | `C_ADM`, body differences ignoring movement or within one movement type | See code |
| Model comparison | Scale-free ratios | `Δ(DH,FDI) / mean[Δ(FB,FDI), Δ(UL,FDI)]`; `Δ(UL,FDI) / Δ(FB,FDI)` |

Inference: one value per participant per contrast (n = 32); paired t test (two-sided),
Wilcoxon signed-rank as a check, Cohen's d_z, and 95% percentile bootstrap intervals
(10,000 participant resamples, paired across all 12 cells, seed 20261022). Holm correction
within the secondary and within the exploratory family; the primary family has one test.

### 4.4 Scales

Primary: **participant_z**, the paper's Methods normalisation (per participant, over all
retained MEPs of both muscles, SD with n − 1). Sensitivity: raw mV (published-model scale),
natural log, within-muscle z, and participant_z after removing MEPs below 50 µV. A human
effect counts as **robust** only if its test is significant on the primary scale (p < .05,
Holm-adjusted within its family) **and** its 95% bootstrap interval excludes 0 on every scale.

### 4.5 Reference values (primary scale)

| Contrast | Estimate [95% CI] | t(31), p | Positive participants | Robust? |
| --- | --- | --- | --- | --- |
| `C_FDI` | 0.283 [0.147, 0.419] | 4.06, .0003 | 24/32 | **Yes** (all 5 scales) |
| Δ(FB, FDI) | 0.288 [0.159, 0.413] | 4.40, .0001 (Holm .001) | 25/32 | Yes |
| Δ(UL, FDI) | 0.223 [0.118, 0.327] | 4.09, .0003 (Holm .002) | 23/32 | Yes |
| Δ(DH, FDI) | −0.028 [−0.157, 0.097] | −0.43, .67 | 15/32 | No effect detected |
| Δ(UL, ADM) | 0.114 [0.014, 0.250] | 1.86, .07 (Holm .44) | 22/32 | No: skewed, interval excludes 0 but the test does not |
| Three-way FB vs DH | 0.247 [0.078, 0.423] | 2.79, .009 (Holm .06) | 20/32 | No: Holm n.s.; interval includes 0 on log scale |
| Δ(FB,FDI) − Δ(UL,FDI) | 0.065 [−0.133, 0.251] | 0.66, .51 | 19/32 | No difference detected |
| Ratio DH / connected (FDI) | −0.11 [−0.67, 0.38] | — | — | — |
| Ratio UL / FB (FDI) | 0.77 [0.34, 1.78] | — | — | — |

Units are participant SDs. A non-significant contrast is not evidence of zero: the model is
never scored as correct merely because a human contrast was not significant.

## 5. Model endpoints and success criteria

### 5.1 Model data

Each model instance plays the role of a participant: one random initialisation, one block
order, and one held-out source group's images. Its single-trial readouts are z-normalised
within the instance (as in the paper), averaged into the 12 cells, and passed through the
same contrast code as the human data.

Units of inference: held-out **source groups** (n = 20) for image generalisation; at least
**5 initialisations** per configuration, reported separately as optimisation stability.
Trials, time steps and neurons are never treated as independent observations.

### 5.2 Criteria, fixed now

| Criterion | Requirement |
| --- | --- |
| **P1, primary** | Model `C_FDI` > 0, with the 95% bootstrap interval over held-out source groups excluding 0, and the same sign in at least 4 of 5 initialisations |
| S1, pattern | Model ratio DH / connected lies within the human interval [−0.67, 0.38] |
| S2, pattern | Model ratio UL / FB lies within the human interval [0.34, 1.78] |
| S3, beyond baselines | Model `C_FDI` exceeds the hand-only and low-level baselines (paired over source groups, 95% interval of the difference excluding 0) |
| S4, out-of-sample | For H2, the upper-limb result is a prediction: no parameter is fitted on upper-limb data |
| Exploratory | Muscle specificity (three-way contrasts), ADM pattern, controls of §7.2 |

A model **reproduces the human pattern** if P1 and S1–S4 all hold. If P1 fails for every
model, the result is reported as negative. Parameters are not changed after the evaluation
is opened to rescue a criterion.

## 6. Models

### 6.1 Baselines (no human data used)

| Baseline | Input | Definition |
| --- | --- | --- |
| Low-level | Full image | Visible area, mean luminance, RMS contrast, edge energy, and these statistics for the A − B difference |
| Hand-only | Silhouette hand, cropped to `used_hand_box` and shown at the same scale for every context | Identical input in every context, so the context effect is 0 by construction |
| DINOv2 | Full image | ViT-L/14; summary token and patch tokens; checkpoint revision pinned at extraction |
| OpenCLIP | Full image | ViT-bigG-14, `laion2b_s39b_b160k`; revision pinned at extraction |

For each baseline the movement signal is the distance between static and movement features,
divided by the between-source spread, computed per context and passed through §4.3.

### 6.2 Hypotheses

| Model | Mechanism | Human data used |
| --- | --- | --- |
| H0, no context | Visuomotor readout from hand features only | None |
| H1, salience | Movement signal scaled by low-level change in the whole image | None |
| H2, attachment gate | A context pathway estimates whether the hand is connected to an arm and gates the visuomotor mapping | At most two gate parameters, fitted on full body and detached hand only |

No condition label enters any model, and no condition receives its own gain.

### 6.3 Network and training

- Frozen image encoder → feedforward leaky integrate-and-fire network (snnTorch), then a
  recurrent variant; a non-spiking network with matched size and inputs as a twin.
- Initial constants: time step 1 ms, membrane 20 ms, synaptic 5 ms; sensitivity at 10/40 ms
  and 2/10 ms; time-step check at 0.5 ms on a subset.
- Training uses only an auxiliary task, matching the participants' instruction: detect or
  count index movements in static→static and static→movement sequences. MEP values are never
  a training target.
- Preprocessing and any dimensionality reduction are fitted on training source groups only.

### 6.4 Probe and readout

- Model resting threshold: the smallest probe amplitude that evokes a readout response above
  a fixed criterion on 50% of fixation trials, found by an adaptive staircase. Probe for the
  task = 120% of that threshold, as in the paper.
- Readout: response of the FDI- and ADM-associated populations 5–60 ms after the probe,
  minus the same window on a matched trial without a probe (same random input stream).
- Plasticity is frozen during evaluation. Silent or saturated readouts invalidate a run.

## 7. Splits, controls and sensitivity

### 7.1 Splits

- Outer loop: leave one source group out (20 folds). All 24 variants and 6 images of a
  group stay in the same fold. Tuning uses grouped inner folds within the training groups.
- Every result is repeated without reference 17.

### 7.2 Controls (masks over frozen full-body pixels)

| Control | H2 prediction |
| --- | --- |
| Arm with a black gap across the wrist | Behaves like the detached hand |
| Detached hand with a forearm stub | Behaves like the upper limb |
| Rectangle with desk patch (archive `detached_hand`) | Behaves like the detached hand: a surface is not a body |
| Equal-area windows, hand-centred vs arm-centred | Only the window containing a connected arm produces facilitation |
| Block-scrambled context, hand intact | Behaves like the detached hand |

Each control is checked like the contexts: visible pixels equal full-body pixels, hidden
pixels are black, and A/B differences stay inside the action region.

## 8. Gates

| Gate | Date | Pass | If not passed |
| --- | --- | --- | --- |
| Spec freeze | 30 Oct | Open items of §3 resolved; this document and `config/analysis.json` committed | Freeze delayed; no modelling starts |
| Gate 1, baselines | 13 Nov | Baseline table complete for all contexts and controls | If a low-level or hand-only baseline meets P1, the continuity explanation is in doubt; S3 becomes decisive |
| Gate 2, model health | 4 Dec | Stable, non-saturated activity; probe threshold found; H0–H2 make different control predictions | Drop recurrence/spiking detail; report the attempt as a negative result |
| Freeze | 11 Dec | Configurations, seeds and code hashes recorded | — |
| Evaluation | 14–18 Dec | Held-out evaluation opened once | — |

## 9. Reporting rules

- Report all 12 cells and all contrasts for humans and models, with intervals and units.
- Keep human uncertainty (participants), image generalisation (source groups) and model
  stability (initialisations) separate.
- Report failed runs, excluded runs with reasons, and all deviations from this document.

## 10. Amendment log

| Date | Change | Reason |
| --- | --- | --- |
| 22 Oct 2026 | Draft created | — |
| 22 Oct 2026 (draft) | Primary detached hand changed from the rectangle to the skin-only silhouette; rectangle kept as sensitivity | Match the paper's isolated hand on black |
| 22 Oct 2026 (draft) | Silhouette outlines re-derived with SAM-HQ on the final frames | First version clipped fingertips and left edge artifacts |
