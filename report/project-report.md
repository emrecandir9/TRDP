# Can an image-driven visuomotor model reproduce display-dependent motor facilitation? A pre-specified test of attachment accounts

Muhammed Emre Candir — TRDP project report

**Status: skeleton, 11 November 2026 (final-phase plan, task A4).** Every section heading is in place. The
sections that do not depend on the held-out evaluation are drafted from the project documents; sections
marked **[stub]** are filled from `output/model/evaluation/result.json` after 15 December 2026, quoting the
fields named in each stub. Numbers in the drafted sections are copied from the documents cited beside them;
before hand-in a script checks every quoted number against its source file (plan, block E).

---

## Summary [stub]

*To be written last (21–23 December). It states, in this order: the question; the claim adopted (the
pipeline tests attachment accounts, it does not assert one); the result of the only blind prediction
(`result.main.openclip.verdicts.h2.P2`, `…S2`); the dependence of the gate's necessity on the training diet
(`result.main.openclip.gate_necessary`, `result.configurations.snn_allctx.gate_necessary`); the decision
rule's verdict (`result.decision`); and the main limitations.*

---

## 1. Question and claim

Matamala-Gomez et al. (2025) recorded motor-evoked potentials (MEPs) from the first dorsal interosseous
(FDI) and abductor digiti minimi (ADM) while 32 participants watched a left index finger abduct in three
displays: a full body, an upper limb and a detached hand. Watching the movement facilitated FDI MEPs for the
full body and the upper limb but not for the detached hand.

This project asks whether a model that receives images of the three displays, with the paper's trial timing,
can reproduce that pattern, and what such a model has to contain. It does not replicate the human experiment
and makes no claim about ownership, agency or physiological equivalence between a model readout and an MEP
([analysis spec](../docs/analysis-spec.md) §1).

**Claim adopted (11 November 2026, [review](../docs/independent-review.md) §8, answer 6).** The project
shows what a pre-specified pipeline can establish about attachment accounts of the human pattern; it does
not assert that attachment explains it. Three findings recorded before the evaluation motivate this
framing: the raw image change does not give the human pattern (§5); a network trained on full-body images
gives a human-like ordering without any gate, as a training-distribution effect (§9.2–9.3); and a
continuity gate and a visible-area gate cannot be told apart on the paper's three displays (§6.6).

## 2. The human target

Source: [analysis spec](../docs/analysis-spec.md) §4; code `analysis/human_target.py`; outputs
`output/human_target/`.

**Deposit audit.** The OSF deposit holds 11,280 MEPs from 32 participants (346–360 each); 240 of the
designed 11,520 (2.08%) are missing, exactly the exclusion rate the paper reports. The paper's Table 2 is
reproduced exactly (8 of 8 legible coefficients and z values) by a linear mixed model on raw mV with a
participant random intercept, and every post-hoc z value in its Results text (10 of 10) by separate
random-intercept models on subsets.

**Discrepancies with the paper (reported, not corrected).** The Methods describe z-normalised MEPs but the
published model used raw mV; every published test treats single trials as observations with only a random
intercept, which inflates z values (the ADM upper-limb movement effect, published z = 4.67, is t(31) = 1.86,
p = .07 at participant level); post-hoc z values come from subset models, and Table 3 differs from the text
for four FDI comparisons; questionnaire scores separate the full body from the other two displays while MEP
facilitation separates the detached hand; and display blocks were not interleaved.

**Definitions.** For each participant the trials of each of the 12 cells R(display, trial type, muscle) are
averaged, and Δ(b, u) = R(b, movement, u) − R(b, static, u). The primary contrast is
C_FDI = [Δ(FB, FDI) + Δ(UL, FDI)]/2 − Δ(DH, FDI). Inference is at participant level (n = 32): paired t test,
Wilcoxon check, d_z and 95% percentile bootstrap intervals (10,000 resamples, seed 20261022). A human effect
counts as robust only if it is significant on the primary scale (participant z) and its interval excludes 0
on all five scales (participant z, raw mV, log, within-muscle z, and participant z without MEPs below 50 µV).

**Reference values (participant z).**

| Contrast | Estimate [95% CI] | t(31), p | Robust |
| --- | --- | --- | --- |
| C_FDI | 0.283 [0.147, 0.419] | 4.06, .0003 | Yes, all five scales |
| Δ(FB, FDI) | 0.288 [0.159, 0.413] | 4.40, .0001 | Yes |
| Δ(UL, FDI) | 0.223 [0.118, 0.327] | 4.09, .0003 | Yes |
| Δ(DH, FDI) | −0.028 [−0.157, 0.097] | −0.43, .67 | No effect detected |
| Ratio Δ(DH)/mean[Δ(FB), Δ(UL)] | −0.11 [−0.67, 0.38] | — | — |
| Ratio Δ(UL)/Δ(FB) | 0.77 [0.34, 1.78] | — | — |
| Ratio Δ(DH)/Δ(FB) | −0.10 [−0.67, 0.35] | — | — |

C_FDI is the one human effect that survives every scale; it is the model's primary target. The ratio
intervals are wide (n = 32), so the ratio criteria test compatibility, not equivalence.

## 3. Stimuli

Sources: [README](../README.md), [project guide](../docs/project-guide.pdf) (11 October, partly superseded),
[silhouette](../docs/detached-hand-silhouette.md), [controls](../docs/controls.md),
[validation](../docs/validation.md).

### 3.1 Source pairs and families

Twenty grayscale reference pairs (static and abducted index, left hand, first-person view) were generated
by AI as references and, according to the project owner, depict 20 different synthetic people. They were
developed into 20 approved colour pairs ("V7") and expanded into 480 families: four desks × two seeds ×
three shared white balances per source, preserving the approved hand movement. The accepted archive holds
2,880 RGB PNGs (2048 × 2048) in three contexts. The image-plane abduction ranges from 10.5° to 24.7°
(median 18.3°). A source and its 24 appearance variants form one lineage group; the 20 groups are the
independent units, independent in identity only (one generator, scene template and camera geometry).
Reference 17 has a known clothing issue; every analysis is repeated without it (456 families).

### 3.2 Displays

- **Full body:** the whole first-person scene (desk, arm, torso, legs).
- **Upper limb:** rows above the fitted desk edge kept, black below.
- **Detached hand (primary):** a skin-only silhouette cut at the wrist on black, matching the paper's
  isolated hand; each visible pixel is the full-body pixel. Outlines come from SAM-HQ on the final frames
  (pinned model and code). The archive's rectangle with a desk patch is kept as a sensitivity display.

The decision to make the silhouette primary is spec amendment 22 October 2026 (draft); 960 silhouette PNGs
are derived from the frozen pixels and approved by the project owner with hash binding.

### 3.3 Controls

Six control displays, masks over the frozen full-body pixels, separate body continuity from visible area
and surround ([controls](../docs/controls.md); spec §7.2). P is the source's palm width.

| Control | Display | H2 prediction |
| --- | --- | --- |
| `hand_stub` | Silhouette + 1 P of forearm, on black | Like the upper limb |
| `hand_gap_stub` | Silhouette + forearm segment starting 0.5 P below the wrist; area equal to `hand_stub` | Like the detached hand |
| `window_arm` / `window_hand` | Equal-area rectangles along the hand axis, with 1 P of arm or extended over desk | Only the arm window facilitates |
| `scrambled` | Upper-limb view in 32-px blocks, blocks touching the hand fixed | Like the detached hand |
| `occluded_wrist` (exploratory) | Upper-limb view with a black band over 0.5 P of the wrist | Not declared |

In every control the hand and every pixel that changes between frames equal the full-body image, hidden
pixels are black, and matched pairs have equal area (checked).

### 3.4 Acceptance and its limits

The 480 families were visually accepted by one reviewer (the project owner) on 11 October 2026, before any
model existed; the 20 reference pairs were not screened by how the movement looked (review answer 7). A
two-rater check was prepared ([rater protocol](../docs/rater-protocol.md)) and *[decision D2: run / not run]*.
Numeric checks never approve realism.

## 4. Pre-specification

The analysis spec was frozen on 22 October 2026, before any model result, and changes since then only
through its amendment log (§10), each row stating what had and had not been computed when it was written;
`config/analysis.json` records each amendment with the spec's hash, and a test fails if they disagree.
GitHub's record of push times shows every commit reached the public repository within 65 minutes of its
commit time ([push times](../docs/push-times.json)), so the dates are externally supported. This is
pre-specification with dated amendments, not a registered preregistration (§11).

The development rule (model-method §6): until the evaluation is opened, no context contrast, ratio or
MEP-analogue summary is computed on a held-out source; development uses training sources only.

## 5. Gate 1: what the images alone contain

Source: [baselines](../docs/baselines.md). Low-level, pixel, hand-only, DINOv2 and OpenCLIP baselines
were computed on all 10,560 condition images, with no human data fitted.

- **No baseline reproduces C_FDI > 0; all show the reverse**, robustly: the movement changes the image
  representation most for the detached hand. The human pattern is not the visual change available in these
  images.
- Frozen encoders separate connected from disconnected displays only as well as a pixel code does, and do
  not generalise that separation across matched pairs: attachment is not available off the shelf.
- Context changes the direction, not only the size, of the encoded movement.
- The reversal persists in hand-centred views (Addendum A).

*Qualification (10 November 2026, review finding F2).* Gate 1 measures the size of the feature change. A
trained network's readout is direction-specific, and on training sources the full-body-trained network
responds less to the silhouette than to the full body (§9.2). Gate 1 shows that raw visual change does not
give the human pattern; it does not show that a trained network must suppress a stronger detached-hand
signal.

## 6. The model

Sources: [model method](../docs/model-method.md) §1–7; code `model/`.

### 6.1 Trials

Each model instance plays one participant: one initialisation, one block order, and one held-out source's
images (its 24 variants as trials). Blocks, 60-s familiarisation, 30 static and 30 movement trials per
block, first frame 2,250–3,250 ms, second frame 750 ms, probe 250 ms after second-frame onset, and 20
fixation trials (black, presented as rest) at start and end follow the paper (model-method §1–2). Because
the feedforward network has no state beyond about 20 ms, each trial is simulated from rest 300 ms before its
second frame, checked equal to the full session (§6.18).

### 6.2 Encoder

Frames are encoded once by a frozen encoder. **OpenCLIP ViT-bigG-14 is the primary encoder**, chosen after
Gate 1 because its baselines were the least favourable to the hypothesis; the DINOv2 ViT-L/14 summary token
and mean patch token are robustness encoders, and the decision rule requires P1 in the same direction with
both (spec amendment 24 October 2026). Standardisation is fitted on the training sources' full-body images
only.

### 6.3 Network

A visual layer of 256 current-based leaky integrate-and-fire neurons (τ_syn 5 ms, τ_mem 20 ms, Δt 1 ms)
receives the standardised encoder vector; two motor populations of 64 neurons each (FDI- and
ADM-associated) receive its spikes through a gate g that scales the visuomotor input. Each motor population
is a homogeneous pool sharing one weight vector and bias, so its firing and excitability change together.
Input noise gives membrane fluctuations of 0.1 θ at rest. A non-spiking twin (leaky integrators with the
spiking neuron's mean-field rate function) and a recurrent variant (trainable weights within the visual
layer, starting at zero) are comparison configurations (model-method §6.1, §6.14, §6.20).

### 6.4 Training

Each network trains on 18 of the 20 sources: the held-out source and one inner source (the next one,
cyclically) are left out; the inner source serves the health check and is never trained on. Training uses
full-body trials only. The objective is to detect the index movement from the FDI population's spike count
50–300 ms after second-frame onset (binary cross-entropy); MEP analogues are never a target. Regularisers
keep visual neurons at 2–40 Hz and, in every trial, each motor population near 1 Hz on static frames and at
rest and below 3 Hz on movement frames, so that the motor populations, like the paper's muscles, stay near
rest during observation. Adam, learning rate 10⁻³, batch 32, 3,000 updates; 10 initialisations per
configuration (model-method §6.2).

### 6.5 Probe and readout

The probe is a 1-ms current into every motor neuron at 120% of the resting threshold, found by bisection on
fixation trials. The readout is the population spike count 0–55 ms after the probe minus that of a matched
no-probe branch with identical state and noise, so the difference is caused by the probe alone. Readouts are
trimmed (± 2 SD within cell) and z-normalised within instance, as the paper did per participant, and passed
to the human-target contrast code (model-method §6.3).

**Readout validity.** The probe response grows with excitability only while the population is near rest; a
population driven hard reads as unexcited. Each network's validated readout range is the pre-probe rate at
which its excitability curve peaks. A held-out display on which an instance's pre-probe rate exceeds that
range is *over-driven*; every criterion is computed with those readouts removed and with them set to the
readout maximum, and also on the readouts as simulated (spec §5.1–5.2; model-method §6.17, §6.19a).

### 6.6 Hypotheses and comparison models

| Model | Gate | Human data used |
| --- | --- | --- |
| H0, no context | g = 1; input is the hand-only image, identical in every display | None |
| H1, salience | g = 1 + s/s̄, s the low-level change from the trial's first frame | None |
| **H2, attachment gate (primary)** | g = c + (1 − c)·a, a = forearm continuity at the wrist (palm widths, capped at 1) | c fitted to the human ratio Δ(DH)/Δ(FB) = −0.10 on training sources |
| Ungated network | H2's network with g = 1 | None |
| Visible-area gate (S3(a)) | As H2 with a = global visible-area fraction | c fitted as for H2 |
| H2-learned | As H2 with a learned by self-supervised forearm prediction | c fitted as for H2 |
| H4, own-body familiarity | As H2 with a = typicality of the display's features for full-body views | c fitted as for H2 |

H2's continuity estimate is 1.00 for the full body and the upper limb and 0.00 for the detached hand in every
source; for the controls it is 0.99 (stub), 0.00 (gap stub), 0.99 (arm window), 0.00 (hand window), 0.15
(scrambled) and 0.00 (occluded wrist); the archive rectangle shows 0.80 palm widths of connected forearm
(model-method §7). The estimate is computed from segmentation masks and the full-body reference image, so
H2's gate is supplied, not computed from the displayed image alone (spec §1, amended 10 November 2026). On
the paper's three displays continuity and visible area change together, so a visible-area gate fits them as
well; only the matched controls separate the two accounts, and they have no human data.

### 6.7 Fit of c

Per fold and initialisation, full-body and silhouette trials of the 19 non-held-out sources are simulated for
every c on the grid 0, 0.05, …, 1 with common random numbers; c is the value whose source-mean ratio
Δ(DH)/Δ(FB) is closest to −0.10 (model-method §6.6, §6.19). Because the human ratio's interval is wide, a
sensitivity value c⁺ is also reported: the largest c that keeps the training-source ratio inside
[−0.67, 0.35] (spec §5.2, 10 November 2026).

## 7. Networks trained, health and readout validity

Sources: model-method §6.5, §6.16–6.17, §6.20–6.23; [validation](../docs/validation.md).

| Set | Networks | Healthy on the inner source |
| --- | --- | --- |
| Campaign: H2, H0, H1 spiking and the H2 twin, three encoders, 20 folds × 10 initialisations | 2,400 | 2,016 (OpenCLIP 181–200 per configuration; DINOv2 120–188) |
| Recurrent H2, OpenCLIP | 200 | 180 (feedforward: 181) |
| Robustness variations (visual layer 128/512, learning rate 3 × 10⁻⁴/3 × 10⁻³, 6,000 updates, τ_mem 10/40 ms, τ_syn 2/10 ms), 20 folds × 3 | 540 | 52–60 of 60 per variation |
| All-context training, 20 folds × 3 | 60 | 57 |
| Time step 0.5 ms, 20 folds × 1 | 20 | 18 |
| **Total** | **3,220** | |

Health criteria (model-method §6.5) cover visual activity, motor rest, task (AUC ≥ 0.9 on the inner source),
threshold and dose-response, readouts neither silent nor saturated, readout facilitation, background
activity within the readout range, a measured readout peak and a monotone excitability curve. A network that
fails a network-level criterion is excluded; a failure on the inner source (task, facilitation, background)
does not exclude it, because such failures follow the fold, where the inner source and the training set
cannot be separated, and excluding them is never conservative (spec §5.1, model-method §6.17).

**Rehearsal (11 November 2026; model-method §6.25).** Every network ran through the evaluation's code on its
inner source with the full body, the silhouette and the controls. It exposed one fault: the spiking-network
library kept every network in GPU memory, and 411 runs ran out of memory; after the fix (no value changed)
all 3,220 ran, at about 4 s and 0.27 MB per network. It also ran the visible-area gate on the inner-source
controls, which the spec had said would wait for the evaluation; this is recorded as an amendment.

## 8. Criteria and decision rule

Source: spec §5.2; model-method §6.19, §6.19a; [criteria-status.json](../docs/criteria-status.json).

| Criterion | Requirement |
| --- | --- |
| P1, primary | C_FDI > 0, 95% interval excluding 0 |
| P2, required | Δ(FB, FDI) > 0 and Δ(UL, FDI) > 0, each interval excluding 0 |
| P3, stability | C_FDI positive in at least 8 of 10 initialisations |
| S1 | Ratio Δ(DH)/mean[Δ(FB), Δ(UL)] inside [−0.67, 0.38] |
| S2 | Ratio Δ(UL)/Δ(FB) inside [0.34, 1.78] |
| S3 | (b) H0 does not meet P1 and H2 exceeds it; (a) H2 exceeds the visible-area gate on both matched control pairs |

Intervals are 95% hierarchical percentile bootstraps (10,000 resamples, seed 20261022): source groups with
replacement, then initialisations within groups. Each criterion is combined over the two treatments of
over-driven readouts into met, not met or undetermined; undetermined never counts in the hypothesis's
favour. **Decision rule:** a model reproduces the robust human pattern if P1, P2, P3, S1, S2 and S3 hold with
the primary encoder and P1 holds in the same direction with both robustness encoders. The necessity of the
gate is reported as the paired difference C_FDI(H2) − C_FDI(ungated) and per criterion (gate necessary only
if the ungated network fails what H2 meets).

**What each criterion can still show.** The table below (from `docs/criteria-status.json`) separates criteria
built into a model, calibrated during development, already previewed on unseen inner sources, and open.

| Criterion | H2 | Ungated | H0 | H1 |
| --- | --- | --- | --- | --- |
| P1 | built in | previewed | built in | open |
| P2, full body | calibrated | calibrated | previewed | previewed |
| P2, upper limb | **open** | open | built in | open |
| P3 | built in | previewed | built in | open |
| S1 | built in | previewed | built in | open |
| S2 | **open** | open | built in | open |
| S3(a) | previewed | — | — | — |
| S3(b) | built in | — | — | — |

For H2 the upper-limb half of P2 and S2 are the only open tests: H2's estimate is exactly 1, 1 and 0 for
the paper's displays and its fitted c is 0, so its detached-hand effect is 0 by construction (spec §5.2,
10 November 2026). The decision rule is unchanged and every criterion is reported; a met decision rule is to
be read with this table.

## 9. Results on training sources, recorded before the evaluation

### 9.1 Fit of c and Gate 2

Sources: model-method §6.19. Gate 2 met on 9 November 2026.

- **The declared target is outside the reachable range.** With the gate closed (c = 0) the training-source
  ratio Δ(DH)/Δ(FB) is about 0, never near −0.10: a gate scales the visual drive and cannot push the
  detached hand below baseline. The declared rule therefore gives c = 0 in almost every OpenCLIP and twin
  network (spiking OpenCLIP median 0.00, range 0–0.05, all gates), and in all 740 recurrent and robustness
  networks. With DINOv2 mean patch the curve is flat and c is arbitrary.
- **Distinct control predictions:** H0–H1, H0–H2 and H1–H2 differ with every encoder under both treatments.
- **Probe-strength sweep:** 2,397 of 2,400 networks rise smoothly with probe strength (every primary-encoder
  network); full-body facilitation rises steadily from 1.0 to 1.5 × threshold, so the 120% probe is not on a
  knife edge.
- The feedforward spiking network stays the primary variant; the recurrent network is the ablation.

### 9.2 The ungated network already separates the displays

Without any gate, the full-body-trained network responds less to the silhouette than to the full body
(training sources; model-method §6.19):

| Configuration | Ungated ratio Δ(DH)/Δ(FB), median [5%, 95%] | Inside the human interval |
| --- | --- | --- |
| OpenCLIP, spiking | 0.49 [0.35, 0.71] | 6% of networks |
| OpenCLIP, twin | 0.37 [0.17, 0.59] | 43% |
| DINOv2 summary, spiking | 0.04 [0.01, 0.14] | 100% |
| DINOv2 mean patch, spiking | 0.00 [−0.00, 0.02] | 100% |
| OpenCLIP recurrent | 0.53 [0.35, 0.72] | 4% |

The gate is therefore probably not necessary for P1; with OpenCLIP it matters for S1, and with DINOv2 it
does nothing, so P1 with the robustness encoders does not test it.

### 9.3 The two training diets

Sources: model-method §6.23. Trained on all three paper displays (60 networks, matched by fold and seed),
the ungated network facilitates for the detached hand slightly more than for the full body: ratio 1.14
[1.03, 1.25] against 0.53 [0.35, 0.72] for the matched primary networks, with full-body facilitation
unchanged (1.80 against 1.77 instance SDs). By the reading declared before training, the primary network's
weak detached-hand response is a training-distribution effect, and with experience of every display a gate
is needed to obtain the human pattern (c⁺ 0.25 against 0.45). The silhouette ratio is largely in-sample for
these networks (18 of 19 fit sources were trained on in that display).

### 9.4 What is already known on unseen sources

Every source is the inner source of one fold, and that fold shares 17 of 18 training sources with the fold
where the source is held out. Full-body facilitation, the detached-hand ratio curves and H2's control
predictions (stub − gap stub +1.42, arm − hand window +0.30, positive in 20 of 20 inner sources) are
therefore known before the evaluation; only the upper-limb display is genuinely unseen (review finding F3).

## 10. The held-out evaluation [stub]

*Run once, 14–15 December 2026, by `model/run_evaluation.py simulate` and `score` under the freeze record
`config/freeze.json` (tag `freeze-2026-12-11`). SHA-256 of `result.json`: [to be recorded].*

### 10.1 Completeness [stub]
*Networks simulated per configuration (`result.main.*.instances`, `result.configurations.*.instances`); any
technical re-run, with reason.*

### 10.2 The blind prediction: upper-limb transfer [stub]
*H2: `main.openclip.analyses.{raw,remove,maximum}.models.h2.delta_UL` and `.S2_ratio`; verdicts
`main.openclip.verdicts.h2.P2`, `.S2`. Same for the ungated network. Outcome sentences, fixed in advance
(plan §3): if P2-UL and S2 are met, "the full-body mapping transfers to the upper-limb image at a human-like
ratio", with the caveat that the upper limb shares most of its pixels with the full body; if either is not
met, the decision rule is not met and the result is reported as negative.*

### 10.3 Necessity of the gate under the two training diets [stub]
*`main.openclip.analyses.*.necessity`, `main.openclip.gate_necessary`;
`configurations.snn_allctx.analyses.*.necessity`, `configurations.snn_allctx.gate_necessary`. Figure
`diets.png`.*

### 10.4 Continuity against visible area [stub]
*`main.openclip.analyses.*.S3a`, `main.openclip.analyses.*.controls`, verdict `main.openclip.verdicts.h2.S3a`.
Figure `controls.png`. To be read with §9.4: the outcome was effectively known.*

### 10.5 All criteria, all models [stub]
*Table `criteria.csv` (verdict with status), `effects.csv`, figure `cells.png` (human and model cells) and
`effects.png`; over-driven shares `main.openclip.over_driven`.*

### 10.6 Decision rule [stub]
*`result.decision`, with the encoders' P1 (`main.dinov2_cls.verdicts.h2.P1`,
`main.dinov2_patchmean.verdicts.h2.P1`), read with §8's status table.*

### 10.7 Robustness and declared repetitions [stub]
*Table `robustness.csv`: without reference 17, healthy only, untrimmed, rectangle detached hand; twin,
recurrent, robustness variations, 0.5 ms, all-context; H2 at c⁺.*

## 11. Development history

Source: spec §10; model-method change log; [decision audit](../docs/decision-audit.md);
[independent review](../docs/independent-review.md).

| Date | Step |
| --- | --- |
| 11 Oct 2026 | 480 families visually accepted |
| 22 Oct | Human target built; analysis spec frozen; silhouette and controls built and approved |
| 23 Oct | Gate 1 baselines: every baseline shows the reverse pattern |
| 24 Oct | Encoder chosen (least favourable to H2); network fixed before training; H2's context pathway decided; readout window, motor-rest regularisers and readout-range rule after a training-source health check found the readout over-driven |
| 25–26 Oct | Training campaign (2,400 networks); learned context pathways and twin repaired; rule for failing networks and over-driven displays; criteria, fit of c and Gate 2 checks declared before running |
| 8–9 Nov | Fit of c (c = 0) and Gate 2 checks (met); recurrent and robustness campaigns |
| 10 Nov | Independent review (sound with major revisions, 18 findings); ungated results recorded; reporting amendments (necessity, readouts as simulated, c⁺, criteria informative for H2); evaluation driver and lock; all-context ablation and 0.5-ms subset |
| 11 Nov | Claim adopted; final phase planned |
| 4 Dec | Gate 2 recorded in the amendment log *[planned]* |
| 11 Dec | Configurations frozen *[planned]* |
| 14–15 Dec | Held-out evaluation *[planned]* |

*To be expanded: the amendment table (spec §10) with, for each row, what had been computed when it was
written and its GitHub push time; the decisions delegated by the project owner (24 October) and how they were
audited; the review's findings and how each was handled.*

## 12. Limitations

- **H2's gate is supplied, and c is not identified.** The continuity estimate uses masks and the full-body
  image, and the human ratio's interval admits every c from 0 to about 0.45 (OpenCLIP).
- **Most criteria are built in or previewed for H2** (§8); only the upper-limb transfer is a blind test.
- **The gate's necessity depends on the training diet** (§9.3), which was not part of the original design.
- **The readout is not physiological:** a population spike count, in arbitrary units, with no conduction
  delay; an over-driven readout falls, unlike a human MEP, which is why the over-driven treatments exist.
- **Synthetic stimuli from one generator and scene, accepted by a single reviewer**; the 20 sources are
  independent in identity only.
- **Continuity and visible area are confounded in the human data**; the controls that separate them have
  no human data.
- **Pre-specification is documented by dated amendments and push times**, not by a registered
  preregistration, and several amendments were written after training-source results.

## 13. Reproduction

*To be completed in block E: environment files, commands from the README, which data stay local and how to
obtain them, the freeze tag, and the CREMI job scripts (`scripts/cremi_*.sh`).*

## References

The verified list shared with the paper-style report is [references.md](references.md) (task A6,
11 November 2026): the decision audit's 29 references, the spec's working paper, and the 11 added after
review answer 40, each checked against the publisher's record, with the limits on how some may be cited
(review answer 39). The final list keeps only the entries cited in the text.
