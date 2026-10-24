# Analysis specification

**Status: FROZEN — 22 October 2026, by the project owner, before any model result exists.
No model, baseline or feature extraction had been run.**
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
readout and a peripheral MEP; generalisation beyond the 20 source hands. The 20 hands are
AI-generated synthetic identities (§3), so no claim concerns the variability of real people's
hands or bodies. In the human data, wrist continuity and the global amount of visible context
change together (full body, upper limb, detached hand), so the human data cannot show that
continuity rather than visible context drives the effect; that distinction rests on model
predictions for the matched controls, which have no human data (amended 24 October 2026).

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
| MEP peak-to-peak, 5–60 ms after TMS, FDI and ADM | Readout-population response over 55 ms from the probe (amended 24 Oct 2026; was 5–60 ms), minus a matched no-probe trial; FDI- and ADM-associated readouts | Arbitrary units, not mV; no conduction delay, so the window starts at the probe |
| 20 fixation-cross trials at session start and end | Black-image fixation trials at start and end, for threshold and drift checks; presented to the network as no visual stimulus (rest), since a frozen encoder's description of a black frame lies far outside its training images (amended 24 October 2026) | — |
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

**Reference images (resolved 22 October).** The 20 grayscale reference pairs
(`normal1–20`, `open1–20`) were generated by AI as references; according to the project owner
they depict 20 different (synthetic) people. Each reference therefore stays its own lineage
group: 20 independent units. Their independence is limited to identity: they share one
generator, one scene template and one camera geometry, so results generalise at most to
hands of this kind in this setup.

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
| Ratio DH / FB (FDI) | −0.10 [−0.67, 0.35] | — | — | — |

Units are participant SDs. A non-significant contrast is not evidence of zero: the model is
never scored as correct merely because a human contrast was not significant.

## 5. Model endpoints and success criteria

### 5.1 Model data

Each model instance plays the role of a participant: one random initialisation, one block
order, and one held-out source group's images (all 24 appearance variants as trials). Its
single-trial readouts are z-normalised within the instance (as the paper did per participant),
averaged into the 12 cells, and passed through the same contrast code as the human data.

- **Initialisations:** 10 per configuration. If Gate 2 shows this is not computable in time,
  5 are used; the choice is logged in the amendment log before the evaluation is opened.
- **Estimate:** the mean over instances (source groups × initialisations).
- **Interval:** 95% hierarchical percentile bootstrap, 10,000 resamples, seed 20261022:
  resample the 20 source groups with replacement, then initialisations within each group
  (Saravanan, Berman & Sober, 2020). Source groups carry image generalisation, initialisations
  carry optimisation variability; trials, time steps and neurons are never independent units.

### 5.2 Criteria, fixed now

**Primary model:** H2 (attachment gate, §6.2), implemented as the spiking network selected at
Gate 2 (feedforward or recurrent; logged before evaluation). H0, H1 and the non-spiking twin
are comparison models scored with the same criteria. A comparison model never replaces the
primary model after results are seen.

**Encoder (amended 24 October 2026).** The frozen image encoder of every model is the OpenCLIP
ViT-bigG-14 image embedding (primary). The DINOv2 ViT-L/14 summary token and mean patch token are
robustness encoders: the identical pipeline, initialisations and evaluation run on each, and every
result is reported for all three. OpenCLIP was chosen after the Gate 1 results because its
baselines were the least favourable to H2 ([baselines](baselines.md) R1, R4, R5), so the choice
cannot favour the hypothesis.

| Criterion | Requirement | Why |
| --- | --- | --- |
| **P1, primary** | `C_FDI` > 0, 95% interval excluding 0 | The one human effect that survives every scale (§4.5) |
| **P2, required** | Δ(FB, FDI) > 0 and Δ(UL, FDI) > 0, each 95% interval excluding 0 | The two robust human movement effects; prevents P1 passing through a negative detached-hand effect alone, and keeps the ratios below well defined |
| **P3, stability** | `C_FDI` positive in at least 80% of initialisations (averaged over source groups): 8 of 10, or 4 of 5 | One unstable optimisation cannot create or remove the result |
| **S1, consistency** | Ratio Δ(DH, FDI) / mean[Δ(FB, FDI), Δ(UL, FDI)] within the human interval [−0.67, 0.38] | No detached-hand facilitation relative to connected displays |
| **S2, consistency / prediction** | Ratio Δ(UL, FDI) / Δ(FB, FDI) within the human interval [0.34, 1.78] | Full body and upper limb facilitate similarly |
| **S3, not low-level** (amended 24 Oct 2026) | (b) The same network, training, probe and readout fed with hand-only input does **not** meet P1, and the primary model's `C_FDI` exceeds it by a paired difference over source groups whose 95% interval excludes 0. (a) The same network with its context estimate replaced by the global visible-area fraction (gate refitted by the same procedure) is run, and **on both matched control pairs** (`hand_stub` − `hand_gap_stub`, `window_arm` − `window_hand`) the primary model's FDI movement-effect difference exceeds the low-level model's by a paired difference over source groups whose 95% interval excludes 0; whether the low-level model meets P1 is reported | The context effect is not explained by the hand alone, and the model's attachment account is distinguishable from a global visible-context account; the human FB/UL/DH data cannot make this distinction, so the matched controls must |
| Exploratory | Muscle specificity (three-way contrasts), ADM pattern, controls of §7.2, sensitivity scales, all-context training | Human ADM effect is not robust; controls have no human data |

**Decision rule.** A model *reproduces the robust human pattern* if P1, P2, P3, S1, S2 and S3
all hold with the primary encoder **and** P1 holds in the same direction with both robustness
encoders (amended 24 October 2026). If P1 holds with some encoders only, the result is reported as
encoder-dependent. Every criterion is reported for every model, pass or fail. If the primary model fails
P1, the result is reported as negative. Parameters, preprocessing and readouts are not changed
after the evaluation is opened.

**Which tests are predictions.** H0 and H1 use no human data, so S1 and S2 are predictions for
them. H2's gate is fitted to the human ratio Δ(DH, FDI) / Δ(FB, FDI) (§6.2); for H2, S1 is a fit
check and S2 is the out-of-sample prediction, because no upper-limb data enter its fitting.

**Reading S1 and S2.** The human intervals are wide (n = 32), so S1 and S2 test compatibility,
not equivalence. Passing them shows that the model is not inconsistent with the human ratios;
failing them is informative. No equivalence claim is made without a margin declared in advance.
The decisive tests are P1–P3 and S3.

**Rationale.** Fixing endpoints, aggregation, primary model and decision rule before any
result follows preregistration practice (Nosek et al., 2018). Requiring the effect to exceed
low-level and hand-only versions of the same model follows from §4 and the measured confounds
(the silhouette maximises the low-level movement signal in the detached hand).

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
| H2, attachment gate | A context pathway estimates whether the hand is connected to an arm and gates the visuomotor mapping. Amended 24 Oct 2026: the estimate is the forearm visible without a break below the wrist line, in palm widths, capped at 1 (`model/context.py`); the gate is g = c + (1 − c)·a | One gate parameter c (≤ two allowed), fitted to the human ratio Δ(DH, FDI) / Δ(FB, FDI) = −0.10 (§4.5) using training source groups only; no upper-limb data |

No condition label enters any model, and no condition receives its own gain.

### 6.3 Network and training

- Frozen image encoder (OpenCLIP primary, DINOv2 for robustness; §5.2) → feedforward leaky integrate-and-fire network (snnTorch), then a
  recurrent variant; a non-spiking network with matched size and inputs as a twin.
- Initial constants: time step 1 ms, membrane 20 ms, synaptic 5 ms; sensitivity at 10/40 ms
  and 2/10 ms; time-step check at 0.5 ms on a subset.
- Training uses only an auxiliary task, matching the participants' instruction: detect or
  count index movements in static→static and static→movement sequences. MEP values are never
  a training target.
- Training images are **full-body views only**: the observer's view of their own connected
  hand. Associative accounts hold that observation–execution matching is learned from seeing
  one's own actions (Heyes, 2001; Cook et al., 2014), and sensorimotor training reconfigures
  FDI/ADM MEP specificity during index- and little-finger observation (Catmur, Walsh & Heyes,
  2007). Upper-limb and detached-hand images are never seen during training. Training on all
  contexts is an exploratory ablation.
- Preprocessing and any dimensionality reduction are fitted on training source groups only.

### 6.4 Probe and readout

- Model resting threshold: the smallest probe amplitude that evokes a readout response above
  a fixed criterion on 50% of fixation trials, found by an adaptive staircase. Probe for the
  task = 120% of that threshold, as in the paper.
- Readout: response of the FDI- and ADM-associated populations 0–55 ms after the probe (amended 24 October 2026: the human 5–60-ms window follows a corticospinal conduction delay the model does not have),
  minus the same window on a matched trial without a probe (same random input stream).
- Plasticity is frozen during evaluation. Silent or saturated readouts invalidate a run.

## 7. Splits, controls and sensitivity

### 7.1 Splits

- Outer loop: leave one source group out (20 folds). All 24 variants and 8 images of a
  group stay in the same fold. Tuning uses grouped inner folds within the training groups.
- Every result is repeated without reference 17.

### 7.2 Controls (masks over frozen full-body pixels)

Defined and checked in [controls.md](controls.md); P = palm width (landmarks 5–17).

| Control | H2 prediction |
| --- | --- |
| `hand_gap_stub`: silhouette + forearm segment starting 0.5 P below the wrist, on black, area equal to `hand_stub` (missing wrist) | Behaves like the detached hand |
| `hand_stub`: silhouette + 1 P of forearm, on black | Behaves like the upper limb |
| Rectangle with desk patch (archive `detached_hand`) | Declared at freezing: behaves like the detached hand (a surface is not a body). **Corrected 24 Oct 2026, before any model result:** the rectangle shows on average 0.80 palm widths (0.50–1.00) of forearm continuous with the hand, about as much as `hand_stub`, so H2 predicts facilitation close to the upper limb |
| `window_hand` vs `window_arm`: equal-area hand-axis rectangles, extended sideways over desk or 1 P down the arm | Only the window containing a connected arm produces facilitation |
| `scrambled`: upper-limb view in 32 px blocks, blocks touching the hand fixed | Behaves like the detached hand |
| `occluded_wrist` (exploratory): upper-limb view with an opaque band over 0.5 P of the wrist | Not declared (occlusion, not discontinuity) |

Each control is checked like the contexts: the hand and every pixel that changes between frames
equal the full-body image, hidden pixels are black, and matched pairs have equal area.

## 8. Gates

| Gate | Date | Pass | If not passed |
| --- | --- | --- | --- |
| Spec freeze | 30 Oct | Open items of §3 resolved; this document and `config/analysis.json` committed | Freeze delayed; no modelling starts |
| Gate 1, baselines | 13 Nov | Baseline table complete for all contexts and controls | If a low-level or hand-only baseline meets P1, the continuity explanation is in doubt; S3 becomes decisive |
| Gate 2, model health | 4 Dec | Stable, non-saturated activity; probe threshold found; H0–H2 make different control predictions; primary network variant and number of initialisations logged | Drop recurrence/spiking detail; report the attempt as a negative result |
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
| 22 Oct 2026 (draft) | Reference-image origin recorded (AI-generated, 20 distinct synthetic identities) | Resolves the §3 open item; restricts generalisation claims |
| 22 Oct 2026 (draft) | §5 revised: hierarchical aggregation, 10 initialisations, primary model named, P2 and P3 added, S3 redefined as same-model low-level and hand-only controls, H2 fit target excludes upper limb, full-body-only training | Audit found undefined aggregation, a P1 loophole, a vacuous S3, partial circularity of S1 for H2 and unspecified training contexts; approved by the project owner |
| 22 Oct 2026 | **Frozen** | Open items resolved, criteria approved; no model result exists |
| 22 Oct 2026 (after freeze, before any result) | §7.2 controls given exact definitions and parameters; the wrist gap is shown on black as `hand_gap_stub` with area matched to `hand_stub`; the desk-scene band is kept as exploratory `occluded_wrist` without a declared prediction | A black band on a desk scene reads as an occluder behind which the arm is completed, so it does not test discontinuity; parameters had not been fixed at freezing |
| 24 Oct 2026 (after the Gate 1 baselines, before any model result) | §5.2 and §6.3: OpenCLIP ViT-bigG-14 named as primary frozen encoder, DINOv2 summary and mean patch token as robustness encoders; the decision rule additionally requires P1 in the same direction with both robustness encoders | The spec required a frozen encoder without naming one; choosing after the baselines could favour the hypothesis, so the least favourable encoder is primary and agreement across encoders is required. Approved by the project owner |
| 24 Oct 2026 (before any model result) | §1 claim boundary on continuity vs visible context; §2 fixation presented as rest; §5.2 S3 split into (b) hand-only as written and (a) a refitted global visible-area gate compared on the matched control pairs; §6.2 H2's context estimate defined as contiguous forearm continuity at the wrist with a one-parameter gate; §7.2 rectangle prediction corrected | A low-level gate fitted to the same human ratio can match it through visible area, so S3(a) as written would reject H2 by construction and the human data cannot separate the two accounts; the encoded black frame drove the model's motor layer to 45 Hz at rest; two gate parameters cannot be identified from one ratio; the archive rectangle was measured to contain 0.80 palm widths of forearm. Decided by the assistant under the project owner's delegation of 24 October 2026 |
| 24 Oct 2026 (before any model result) | §2 and §6.4: readout window 0–55 ms after the probe instead of 5–60 ms | The human window captures an MEP that arrives after about 20 ms of corticospinal conduction; the model reads its motor population directly, which answers within 1–5 ms in synchronous volleys, so the 5-ms boundary cut through them and made the probe dose-response non-monotone (health check on training sources). The window keeps the human length; with it the dose-response is monotone (Spearman 1.00) |

## 11. References

- Catmur, C., Walsh, V. & Heyes, C. (2007). Sensorimotor learning configures the human mirror
  system. *Current Biology*, 17, 1527–1531.
- Cook, R., Bird, G., Catmur, C., Press, C. & Heyes, C. (2014). Mirror neurons: from origin to
  function. *Behavioral and Brain Sciences*, 37, 177–192.
- Heyes, C. (2001). Causes and consequences of imitation. *Trends in Cognitive Sciences*, 5,
  253–261.
- Matamala-Gomez, M., Frisco, F. et al. (2025). Virtual body continuity during action
  observation affects motor cortical excitability. *Scientific Reports*, 15, 13364.
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C. & Mellor, D. T. (2018). The preregistration
  revolution. *PNAS*, 115, 2600–2606.
- Saravanan, V., Berman, G. J. & Sober, S. J. (2020). Application of the hierarchical bootstrap
  to multi-level data in neuroscience. *Neurons, Behavior, Data analysis, and Theory*, 3(5).
