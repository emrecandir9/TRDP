# Model method: trial simulator, inputs and pre-declared choices

Code: [`model/trials.py`](../model/trials.py) (trial simulator), [`model/bank.py`](../model/bank.py)
(frozen-encoder inputs); tests in [`tests/test_trials.py`](../tests/test_trials.py). Implements
the parts of the [analysis spec](analysis-spec.md) §2, §5.1 and §6.3–6.4 that define what a model
instance sees, and fixes the details the spec and the paper leave open.

**Status, 24 October 2026: draft.** Sections 1–4 are fixed: the trial simulator is built and
tested, and nothing has been trained. Section 5 (encoder) was decided on 24 October 2026
(option A, spec amendment). Section 6 fixes the network before any training; section 7 records
the H2 context pathway and the S3 amendment; section 8 is a validation of all methods. Later changes go
to the change log at the end.

## 1. What the paper and the frozen spec fix

From the paper's Methods (Matamala-Gomez et al., 2025, "VR action observation task"), as
already transcribed into spec §2:

| Element | Paper | Model |
| --- | --- | --- |
| Blocks | One per display; order counterbalanced across participants | One per display; six orders cycled across instances |
| Familiarisation | Static stimulus for one minute before each block | 60 s of a static frame of the held-out source |
| Trials | 60 per block, 30 static and 30 movement, randomised | Same, randomised per block and instance |
| Trial timing | Static hand for 2,250–3,250 ms (jittered), then a second frame for 750 ms | Same; jitter uniform in integer ms |
| Second frame | Movement trials: abducted index; static trials: the static hand again | Same frame pairs, same family within a trial |
| Probe | TMS 250 ms after second-frame onset, 120% of resting threshold | Probe current 250 ms after onset, 120% of the model threshold (spec §6.4) |
| Fixation | Fixation-cross trials, 20 at the start and 20 at the end of the session | 20 black-display trials at start and end (spec §2: black image) |
| Task | Count the finger movements | Auxiliary training task: detect or count movements (spec §6.3) |

## 2. Rules for details neither specifies

Each rule below is an assumption, chosen to be neutral, and is tested in `tests/test_trials.py`.

| Detail | Rule | Reason |
| --- | --- | --- |
| Gap between trials | None: the next trial's static frame follows the second frame directly | The paper says each trial *started* with the static hand and describes no interval; after a movement trial the finger therefore returns at the start of the next trial, at least 2,250 ms before its probe |
| Fixation-trial timing | Same as task trials (first period 2,250–3,250 ms, second 750 ms, probe at 250 ms), black throughout | Not reported; same timing gives the threshold and drift checks the same probe context |
| Breaks between blocks | Not modelled; network state is carried through the whole session | Questionnaires were filled between blocks; with membrane and synaptic constants of 2–40 ms (spec §6.3), the 60-s familiarisation removes any carry-over from the previous block |
| Images per trial | Spec §5.1: the held-out source's 24 appearance variants are the trials. Per block and trial type, each variant once in random order, then 6 more without repeats (each variant 1–2 times) | Uses every variant, balances static and movement trials |
| Familiarisation image | Static frame of one variant drawn at random per block | The paper used one photograph per display |
| Instance and block order | Instance = (held-out source, initialisation). Order index = (source rank + initialisation) mod 6, over the six permutations in lexicographic order | Every source meets every order across its 10 initialisations, and every initialisation uses every order across sources. The naive `instance mod 6` would give each source only three orders |
| Rectangle sensitivity | Same random stream as the primary session; only the detached-hand images differ | Paired comparison: any difference is caused by the display alone |
| Control sessions | Separate sessions, one block per control display (six), same block structure; orders from a Williams square (first-order carry-over balanced) indexed as above | The controls have no human data; a balanced order prevents systematic carry-over |
| Random streams | `SeedSequence([20261022, source, initialisation, session kind])`; every trial gets its own noise seed for the matched no-probe trial (spec §6.4) | Reproducible, independent across instances |

Resulting sessions: the primary session lasts 15.8 simulated minutes (220 trials, about 145
distinct images); a control session lasts 29.3 minutes (400 trials).

**Training sequences (spec §6.3).** `training_session` builds full-body trials only, from
training sources only, with the same timing, half of them movement trials. Probes are never used
in training, and MEP values are never a target.

## 3. Inputs

- Frames are static, so each frame's encoder output is computed once (the Gate 1 features,
  `output/features/`, hash-checked on load) and held constant while the frame is displayed. The
  network supplies all temporal dynamics.
- The black fixation display is presented as rest: no visual stimulus current (§6.8). Its encoded
  features (`output/features_extra/black.npz`) lie far outside the training images and are kept for
  reference only.
- **Standardisation is fitted on the training sources' full-body images only** and applied
  unchanged to every display. Full body is the only display seen in training (spec §6.3), and no
  statistic of the held-out source or of a test display enters preprocessing (spec §6.3, §7.1).
  Detached-hand and control features therefore arrive in a distribution the network has not
  seen, which is part of the test.

## 4. What the simulator guarantees (tests)

Counts per block and session; timing bounds; probe 250 ms after onset and the readout window
(0–55 ms) inside the second frame; contiguous timeline without gaps; correct frame pairs;
every variant used, none from another source; order balance (all six orders per source and per
initialisation; Williams carry-over balance for 3, 4 and 6 displays); determinism; pairing of the
rectangle sensitivity session; time grids at 1 and 0.5 ms; training sequences limited to
full-body, training-source images; feature-bank look-ups on the real features.

## 5. Encoder: decided 24 October 2026 (option A)

**Decision.** The project owner approved option A on 24 October 2026: OpenCLIP ViT-bigG-14 is the
primary encoder, the DINOv2 summary token and mean patch token are robustness encoders, and the
decision rule requires P1 in the same direction with both. Recorded as a spec amendment (§5.2,
§10) and in `config/analysis.json` (`model`). The proposal is kept below as written.

### Proposal as written

Spec §6.3 requires a frozen image encoder but does not name one. The choice is being made
**after** the Gate 1 results are known, so it could be biased, deliberately or not, towards an
encoder that happens to favour H2. For example, the DINOv2 summary token was the only encoder
whose full-body decoding axis showed no reversed context effect (transfer gain, [baselines](baselines.md) R4).

**Proposal (recommended):**

1. **Primary encoder: OpenCLIP ViT-bigG-14**, the encoder whose baselines are *least*
   favourable to H2. It has the most negative normalised C and transfer-gain C of the encoders,
   and its attachment decoder transfers between matched pairs below chance. Choosing the hardest
   case means the choice cannot have helped the hypothesis (a severe test).
2. **Robustness encoders: DINOv2 summary token and DINOv2 mean patch token.** The identical
   pipeline, initialisations and evaluation run on each; every result is reported for all three.
3. **Amended decision rule (spec §5.2):** "reproduces the robust human pattern" requires the
   existing criteria for the primary encoder **and** P1 in the same direction for both robustness
   encoders. If P1 holds for some encoders only, the result is reported as encoder-dependent.

Alternatives: (B) DINOv2 summary token as primary, on a priori grounds (self-supervised,
purely visual, no language supervision), with the same robustness requirement; (C) all three
co-primary. Option A is recommended because it is the only choice that is demonstrably not
favourable to the hypothesis. Whichever is chosen is entered in the spec's amendment log, with
`config/analysis.json` updated, before any network is trained.

## 6. Network (fixed 24 October 2026, before any training)

Code: [`model/network.py`](../model/network.py), [`model/train.py`](../model/train.py). All values
below are initial constants (spec §6.3). They may be changed only on the basis of the health
criteria (§6.5) and the auxiliary task, evaluated on **training sources** (grouped inner folds,
spec §7.1), and every change goes to the change log.

**Development rule.** Until the held-out evaluation is opened (spec §8, 14–18 December), no
context contrast, ratio or MEP-analogue summary is computed on a held-out source. Development
diagnostics use training sources only, and H2's gate fit (§6.6) uses training sources only.

### 6.1 Architecture (shared by H0, H1 and H2)

| Part | Definition |
| --- | --- |
| Input | Standardised encoder vector of the displayed image (§3), constant within a frame; D = 1,280 (OpenCLIP) or 1,024 (DINOv2) |
| Visual layer V | 256 current-based leaky integrate-and-fire neurons (snnTorch `Synaptic`): `I ← αI + s(W_in x + b_V + σξ)`, `U ← βU + I − θS`, `S = H(U − θ)`, θ = 1, reset by subtraction, α = exp(−1/τ_syn), β = exp(−1/τ_mem), τ_syn = 5 ms, τ_mem = 20 ms, Δt = 1 ms. **Units:** s = (1 − α)(1 − β), so a constant input x settles at membrane potential x; initial b_V = 0.8 (below threshold), W_in ~ N(0, 0.5²/D) |
| Motor layer M | Two populations of 64 neurons each, FDI- and ADM-associated, same neuron model; input `s(g · w_p · S_V + b_p + σξ) + P(t)` for population p. **Each population is a homogeneous pool** (amended 24 Oct 2026, §6.10; was a separate weight row and bias per neuron): its neurons share one visuomotor weight vector w_p (initial N(0, 1/256)) and one bias b_p (initial 0.6, below threshold) and differ only by their independent noise, so the pool's firing and its excitability change together |
| Gate g | Multiplies the visuomotor input; defined per hypothesis (§6.6); never a function of a condition label |
| Noise | Independent Gaussian input noise in both layers, σ set so that the membrane potential fluctuates with SD 0.1 θ at rest (derived analytically, checked in a test); the matched no-probe twin shares it exactly |
| Probe P(t) | Raw (unscaled) current of amplitude A injected into every motor neuron for 1 ms at the probe time |
| Recurrent variant | Adds recurrent weights within V (built after the feedforward model passes §6.5) |
| Non-spiking twin | Same sizes, inputs, gate and training; each unit is a leaky integrator (τ_mem) with a softplus output rate instead of spikes |

### 6.2 Auxiliary training (spec §6.3)

- **Data:** full-body trials of the training sources only (`trials.training_session`), half of
  them movement trials. Upper-limb, detached-hand and control images are never used in training.
- **Window:** each training trial is simulated from rest for the last 300 ms of its first frame
  and the first 350 ms of its second frame; the first 300 ms let the network settle.
- **Objective:** detect the index movement from the FDI population. With `n_FDI` the FDI
  population's mean spike count 50–300 ms after second-frame onset, the logit is `κ(n_FDI − n₀)`
  (κ, n₀ trainable readout scalars) and the loss is binary cross-entropy against
  movement/static. MEP analogues are never a training target.
- **Regularisers:** visual neurons kept between 2 and 40 Hz (hinge on the mean rate); **in every
  trial**, each motor population held near 1 Hz on static second frames and at rest (quadratic,
  weight 1/5 per Hz²; eight no-stimulus rest trials run in every batch), and **below 3 Hz on movement
  frames** (hinge, same weight). Amended 24 October 2026 before any evaluation: first (§6.10) from 5 Hz
  on static frames with weight 1/50, no rest trials and no limit on movement frames, to limits on
  the batch mean (5 Hz on movement frames); then (§6.12) to per-trial limits, rest 1 Hz and 3 Hz on
  movement frames, because the batch-mean limit let single sources reach 9 Hz and a 2–4 Hz range left
  too little margin for unseen sources. The ADM population is held at the
  same rate in movement and static frames. **This builds in muscle specificity** (FDI learns the index
  movement, ADM does not), as the associative account predicts for index-movement training; ADM
  results are therefore exploratory, as in spec §5.2.
- **Observation without execution (§6.10).** The paper excluded every trial with background EMG
  in the 100 ms before the pulse, so its MEPs were recorded with the hand at rest. Observation
  modulates corticospinal neurons without producing muscle activity (Kraskov et al., 2009;
  Vigneswaran et al., 2013), and MEP facilitation during observation is a change of excitability at
  rest (Fadiga et al., 1995). The model's motor populations stand for that corticospinal output, so
  observing a movement may raise their firing only slightly. The limit also keeps the readout valid:
  the probe response of this population grows with excitability only near rest (§6.10).
- **Optimiser:** Adam, learning rate 10⁻³, batch 32 trials, 3,000 updates; surrogate gradient
  fast sigmoid (snnTorch default, slope 25); weights initialised from a seed per
  initialisation (10 per configuration, spec §5.1).

### 6.3 Probe, threshold and readout (spec §6.4)

- **Readout:** for each probe, the FDI (and ADM) population spike count 0–55 ms after the probe
  (the human window's length; it starts at the probe because the model has no conduction delay, §6.9),
  minus the same count in the matched no-probe twin. The twin is computed by branching from the
  identical network state at the probe time with the identical noise stream, so the difference is
  caused by the probe alone. The session then continues from the no-probe branch.
- **Resting threshold:** on the fixation display, presented as rest (§6.8), the smallest A for which the FDI
  response is at least 6 spikes (≈ 10% of the population) on at least 50% of 40 trials, found by
  bisection over A in [0, A_max] with 14 halvings (an adaptive staircase). The task probe is
  1.2 × threshold, as in the paper.
- **Single-trial readouts** of the task trials are trimmed as in the paper (readouts beyond ± 2 SD
  of their cell mean removed within each instance, cell = display × trial type × population),
  then z-normalised within each instance over both populations (the paper's per-participant
  normalisation; the deposited MEPs contain task trials only, 360 per participant) and passed to
  the human-target contrast code. Untrimmed readouts are a sensitivity analysis (amended 24
  October 2026, before any evaluation: the paper's trimming had not been mirrored). Fixation
  trials serve the threshold and drift checks only.
- **Readout range rule (amended 24 October 2026, before any evaluation).** The readout measures
  excitability only while the motor population is near rest (§6.10); a display that drives it
  harder would show a smaller probe response for a reason unrelated to the hypothesis, and could
  imitate the human pattern. For every model and display, the mean rate of each motor population in
  the 100 ms before the probe is reported per trial type. If it exceeds the network's validated
  readout range (the pre-probe rate at which its excitability curve peaks, §6.5, §6.12; a fixed 6 Hz,
  then 5 Hz, until the §6.12 amendments) for a display, every contrast involving that display is reported as
  **outside the validated readout range**, and a criterion that depends on it is not counted as
  met, whichever way it comes out.

### 6.4 Evaluation simulation

Each instance runs its full session (§2) at Δt = 1 ms with plasticity frozen; probes on every
task and fixation trial. Instances are simulated in parallel on the GPU. The same session is run
for the primary displays, the paired rectangle sensitivity and the controls.

**Memory of the feedforward network.** Its only state is the synaptic and membrane variables
(τ ≤ 20 ms), so its state at the probe, 250 ms after the second frame starts, depends on that
frame alone (residual influence of earlier frames < e⁻¹² ≈ 6 × 10⁻⁶). Trial jitter, familiarisation, block order and
carry-over therefore cannot affect it; they are kept for fidelity and matter only for the
recurrent variant. A simulation that starts each trial from rest 300 ms before its second frame is
equivalent up to that residual and may be used for the feedforward network if the first instance
shows identical cell means within simulation noise. The "final frame only vs sequence" ablation
(next-phase plan W7) is null by construction for the feedforward network.

### 6.5 Health criteria (Gate 2, spec §8), on training sources

| Criterion | Requirement |
| --- | --- |
| Visual activity | ≤ 10% silent neurons (no spike over all inner-source trials), ≤ 5% above 200 Hz |
| Motor rest | Mean rate of each motor population on fixation 0.5–20 Hz (amended 24 Oct 2026 from 1–20 Hz with the lower rest target, §6.10) |
| Task | Movement detection on full-body trials of an inner held-out training source: AUC of the FDI count, movement vs static, ≥ 0.9 (amended 24 Oct 2026, §6.10: was accuracy ≥ 0.9 with the trained threshold; accuracy is still reported) |
| Threshold | Found inside (0, A_max); probe dose-response monotone (Spearman ≥ 0.9 over 10 amplitudes, 0.5–2 × threshold) |
| Readouts | Not silent (mean task response > 0) and not saturated (no response ≥ 90% of the population) |
| **Readout facilitation** (added 24 Oct 2026, §6.10) | On full-body trials of the inner source, the FDI probe response at the task probe is larger on movement than on static trials (95% interval of the paired difference above 0). This checks that the readout measures the trained motor response; it uses the training display only and no context information |
| **Background activity** (added) | Each motor population's rate in the 100 ms before the probe within the network's validated readout range: at most the pre-probe rate at which its own excitability curve peaks (§6.12; was a fixed 6 Hz, then 5 Hz) on static and movement full-body trials of the inner source (the model's analogue of the paper's background-EMG exclusion) |
| **Readout peak found** (added 26 Oct 2026, §6.17) | The excitability curve's peak lies inside the tested sweep (extra current 0–0.6), so the readout range is a measured peak, not the end of the test |
| **Excitability range** (added) | At rest, raising the FDI population's excitability with a constant current raises its probe response monotonically up to the curve's peak (the validated readout range) (Spearman ≥ 0.9) |
| Distinct predictions | H0, H1 and H2 make different control predictions (spec §8) |

### 6.6 Hypotheses (spec §6.2)

- **H0, no context:** the visual input is the hand-only image of the same family and frame (the
  Gate 1 `hand_only` features), identical in every display; g = 1. Its context effect is 0 by
  construction. Its standardisation is fitted on the training sources' hand-only images, its only
  input (amended 24 October 2026; §3 describes the display models).
- **H1, salience:** display features; g = 1 + s/s̄, where s is the 64 × 64 luminance distance
  between the displayed frame and the trial's first frame (0 for first frames and static second
  frames) and s̄ its mean over training full-body movement frames. No parameter is fitted.
- **H2, attachment gate:** display features; g = c + (1 − c)·a, with a the forearm continuity of
  the display (§7, `model/context.py`) and one parameter c fitted to the human ratio
  Δ(DH, FDI)/Δ(FB, FDI) = −0.10 on training sources only.
- **Ungated network (comparison, added 24 October 2026 before any evaluation):** H2's trained
  network with g = 1 on every display. It costs no training (H2 and the ungated network share it) and is the
  direct test that the gate is necessary: it shows what the learned visuomotor mapping does with
  the display features alone, which Gate 1 found reversed (largest movement signal for the detached
  hand). Spec §1 allows claims about which components are necessary; without this model the
  necessity of the gate would rest on H0 and H1 only, which differ from H2 in their input or gate
  and in their training.
- **Fitting c (declared 24 October 2026, before any fit).** Per outer fold and initialisation, after
  training: full-body and silhouette detached-hand trials of the fold's 19 training sources
  (30 static and 30 movement trials per source and display, every variant used as in §2, task probe,
  trimming and normalisation as in §6.3 within each source) are simulated once with common random
  numbers for every c on the grid {0, 0.05, …, 1}. c is the grid value whose ratio of source-mean
  Δ(DH, FDI) to Δ(FB, FDI) is closest to −0.10; ties go to the smaller c. If Δ(FB, FDI) ≤ 0 on the
  training sources, the fit is undefined and the run fails health (§6.5). The S3(a) visible-area
  model's c is fitted by the identical procedure. The fitted c and the whole ratio curve are
  reported; the held-out source never enters.
- **Training per hypothesis.** Each hypothesis is trained with its own input and gate on full-body
  training trials: H0 on hand-only features; H1 with its salience gate; H2 with its gate, which is
  1 on every full-body image (a = 1), so H2 and the ungated network share training.
- **Fixation gate.** On the fixation display (rest) H0 and H1 use g = 1 and H2 uses g = c (no
  forearm, a = 0); with no visual stimulus the visual layer is near silent, so the gate there has
  little effect.

### 6.7 First health check (24 October 2026, training sources only)

Spiking network, OpenCLIP, initialisation seed 0; outer held-out source 1 not used at all; inner
check source 2 never trained on; 3,000 updates in 16.5 minutes on `jessica`
(`output/model/health_openclip_snn_fold1_seed0.json`).

| Criterion (§6.5) | Result |
| --- | --- |
| Task accuracy, inner source | 1.00 |
| Visual activity | 18–23 Hz; 0% silent neurons |
| Motor rest (static second frames) | 5.0 Hz |
| ADM movement − static | ≈ 0 (held by the regulariser) |
| Threshold | Found: 0.063 (raw 1-ms current) |
| Dose-response, 0.5–2 × threshold | Monotone, Spearman 1.00 |

Not yet checked: motor rate on the fixation display, saturation of task readouts, the non-spiking
twin and the recurrent variant, and distinct H0–H2 control predictions (needs §7).
**Compute:** at this speed the full design (20 folds × 10 initialisations × hypotheses × 3
encoders × spiking and rate) would need about 1,000 GPU hours if trained one network at a time;
networks are therefore to be trained in parallel batches on the GPU before the full runs.

### 6.8 Corrections from the first health check (24 October 2026, before any evaluation)

- **Fixation as rest.** The encoded black frame lies far outside the training images (mean |z|
  2.9 against 0.8 for full-body images); on it half the visual neurons were silent, 9% fired
  above 200 Hz and the FDI population fired at 45 Hz, against 0 Hz on full-body frames. A
  threshold measured there would calibrate the probe in a state unlike any task display. The
  fixation display is therefore presented as no visual stimulus: the visual layer receives no
  stimulus current, only its noise (`network.rest_drive`). This matches the paper, where the
  resting threshold was measured at rest, and a black frame delivers no light. The encoded black
  frame (`output/features_extra/black.npz`) is kept for reference but not used.
- **Rest rate per population.** The rest regulariser held only the mean of FDI and ADM near 5 Hz,
  so FDI was silent on static frames while ADM fired at 10 Hz; each population is now held near
  5 Hz on static frames.
- **Portable checkpoints.** Saved on the CPU so they load anywhere.

The health check is to be repeated with these corrections before any further step.

### 6.9 Second health check and readout window (24 October 2026)

Same fold, seed and training as §6.7, with the §6.8 corrections, on `jessica` (16.5 minutes).
Every criterion passed except the probe dose-response (Spearman 0.76): responses at 0.5–2 ×
threshold were 8.4, 2.4, 1.5, 5.7, 18, 34, 50, 56, 48, 31 spikes. A millisecond trace showed the
cause: the motor population answers the probe within 1–5 ms in synchronous volleys, and the 5-ms
start of the declared window (taken from the human MEP, which arrives after about 20 ms of
corticospinal conduction) cut through them at amplitude-dependent points. Counted from the probe,
the evoked response grows monotonically. The window was therefore set to 0–55 ms (the human length,
starting at the probe; spec amendment). The network does not depend on the readout, so the saved
network was rechecked without retraining (`train.recheck`):

| Criterion (§6.5) | Result |
| --- | --- |
| Task accuracy, inner source | 1.00 |
| Visual activity | 21–23 Hz; 0% silent neurons |
| FDI / ADM on static frames | 5.0 / 4.9 Hz |
| Motor rest at fixation | FDI 1.6 Hz, ADM 3.2 Hz |
| Threshold | Found: 0.020; response at 1 × threshold 5.9 spikes (criterion 6) |
| Dose-response, 0.5–2 × threshold | 2.5 → 16.3 spikes, monotone (Spearman 1.00) |

A test now requires the evoked response not to fall as the probe grows.

### 6.10 Pre-training validation: the readout and the motor regime (24 October 2026)

Asked to validate every step before the training campaign, the assistant checked whether the
probe readout measures what the trained network does. Every check below uses full-body trials of
training sources (inner source 2 of fold 1) or the fixation display; no other display enters a
probe, a movement contrast or a ratio. To find out whether test displays drive the network far from
its training regime, static-frame activity (no probe, no movement) was also inspected on every
display for training sources 3–20 with the §6.9 network; that inspection showed one display,
`scrambled`, driving the FDI population to 25 Hz on static frames (full body 5 Hz), which motivated
the readout range rule (§6.3).

**Finding 1: the readout did not register the trained response.** In the §6.9 network the FDI
population fired at 26 Hz before the probe on movement trials (99% of its neurons active) and at 5 Hz
on static trials, yet the probe response was the same (9.9 vs 9.4 spikes; difference −0.47, 95%
interval −1.32 to 0.38). Raising the FDI population's excitability step by step at rest showed why:
its probe response grows from 7.8 spikes at 1.6 Hz to a plateau of 13 near 6–9 Hz and falls back to
about 9 spikes at 30–40 Hz. A neuron that already fires is moved forward in time by the probe rather
than recruited, so a population driven hard reads as unexcited. Shorter windows (0–3, 0–5, 0–10,
0–20 ms) behave the same way, so this is not a window effect. (In people, MEPs grow with background
contraction; this decline is a property of the model, which is why it is kept out of its operating
range.) With the declared training, P2 would have failed in every model for a measurement reason, and
a test display that over-drives the network could have imitated the human pattern.

**Change 1: observation without execution** (§6.2): each motor population held near 2 Hz on static
frames and below 5 Hz on movement frames, as the paper's MEPs were taken at rest. Health check
(`output/model/v3/`): FDI 0.7 Hz on static and 4.5 Hz on movement frames, but the probe response
*fell* on movement trials (10.6 → 6.5 spikes; ADM 15.1 → 7.1 although its rate did not change).

**Finding 2: the population was not homogeneous.** Each motor neuron had its own visuomotor weights
and bias. On movement frames the spread of the visual input across FDI neurons grew eightfold (SD
0.028 → 0.216): 12% of neurons fired above 10 Hz while half were silent, and the mean membrane
potential fell (0.715 → 0.663 θ). The network detected the movement with a few neurons and moved the
rest away from threshold, so the population as a whole became less excitable.

**Change 2: homogeneous pools** (§6.1): the neurons of a population share one visuomotor weight
vector and one bias and differ only by their noise. A population then stands for a muscle's
corticospinal pool, its rate and its excitability move together, and the excitability curve above
describes the operating range exactly. Health check (`output/model/v4/`, OpenCLIP seeds 0 and 1,
DINOv2 summary token seed 0): FDI probe response 12.4 → 17.5, 12.3 → 17.1 and 12.8 → 19.0 spikes
from static to movement trials (differences 5.1 [4.1, 6.2], 4.9 [3.8, 5.9], 6.1 [5.0, 7.2]); ADM
changes by −0.6 to +1.0 spikes; pre-probe rates ≤ 4.4 Hz; excitability curve monotone up to 5 Hz.

**Finding 3 and change 3: rest and the task criterion.** At rest the pools fell to 0.2–0.8 Hz
(criterion 0.5–20 Hz), because nothing held the no-stimulus state; eight rest trials per batch now
hold it near 2 Hz too (§6.2), so rest, where the threshold is calibrated, is a baseline and not a
silent state. Detection accuracy on the inner source was 0.95, 0.89 and 0.50 although the FDI count
separated the trial types clearly (DINOv2: 0.22 vs 0.51 spikes per neuron): the trained threshold
had been set on the training sources, whose counts were twice as high (0.47 vs 1.41). Source-level
offsets do not affect the readout, which is z-normalised within each instance, so the task
criterion is now the threshold-free AUC (§6.5).

**Other corrections found in the same review.** The declared visual criterion "≤ 5% above 200 Hz"
was not computed, and "silent" was measured on one training batch; both are now computed over two
variants of every training source. The paper's ± 2 SD trimming had not been mirrored (§6.3). H0's
standardisation was not defined (§6.6). The ungated network was missing as a comparison model, and
c's fitting procedure was not declared (§6.6). §7 stated that a gated network cannot produce a
negative movement effect, which is not true for c > 0; corrected. The README's directory map had no
`model/` entry.

Results of the corrected design: §6.11.

### 6.11 Health check of the corrected design (24 October 2026, training sources only)

Fold 1 (outer source 1 never used), inner source 2, 3,000 updates each, on `jessica`, three to four
networks sharing the GPU (44 minutes each); `output/model/v5/`.

| Criterion (§6.5) | H2, OpenCLIP, seed 0 | H2, OpenCLIP, seed 1 | H2, DINOv2 summary, seed 0 |
| --- | --- | --- | --- |
| Task AUC (accuracy) | 1.00 (1.00) | 1.00 (0.995) | 0.999 (0.50: source offset, §6.10) |
| Visual rate; silent / > 200 Hz neurons | 17 Hz; 0 / 0 | 14 Hz; 0 / 0 | 19 Hz; 0 / 0 |
| Rest (fixation) FDI / ADM | 1.95 / 2.02 Hz | 1.98 / 2.01 Hz | 1.91 / 2.02 Hz |
| Threshold; dose-response Spearman | 0.016; 1.00 | 0.016; 1.00 | 0.017; 1.00 |
| FDI probe response, static → movement | 8.2 → 10.5 | 7.5 → 10.1 | 4.8 → 7.8 |
| **Readout facilitation** [95% CI] | **+2.3 [1.5, 3.1]** | **+2.6 [1.9, 3.4]** | **+3.0 [2.3, 3.7]** |
| ADM, static → movement | 6.6 → 6.6 | 6.5 → 7.4 | 7.0 → 7.6 |
| Highest pre-probe rate (criterion ≤ 6 Hz) | 5.5 Hz (FDI, movement) | 5.1 Hz | 2.2 Hz |
| Excitability curve Spearman up to 5 Hz | 1.00 | 1.00 | 1.00 |
| Saturated responses | 0 | 0 | 0 |
| **All criteria** | **pass** | **pass** | **pass** |

The facilitation is 28–63% of the static response and muscle-specific: ADM changes by 0 to 0.9
spikes. Its size in model units is not compared with the human effect; only signs, contrasts and
scale-free ratios are ([next-phase plan](next-phase-plan.md) §4.1). 

| Criterion (§6.5) | H2, DINOv2 mean patch, seed 0 | H0, OpenCLIP, seed 0 | H1, OpenCLIP, seed 0 | Non-spiking twin, OpenCLIP, seed 0 |
| --- | --- | --- | --- | --- |
| Task AUC (accuracy) | **0.891** (0.50) | 0.988 (0.57) | 1.00 (0.98) | **0.00** (0.50) |
| Rest FDI / ADM | 1.94 / 2.04 Hz | 1.92 / 2.00 Hz | 1.93 / 2.04 Hz | 4.1 / 5.1 Hz |
| FDI probe response, static → movement | 2.7 → 4.0 | 7.7 → 9.2 | 7.3 → 10.1 | 0.01 → 0.00 |
| Readout facilitation [95% CI] | +1.3 [0.9, 1.8] | +1.5 [0.7, 2.3] | +2.8 [2.1, 3.6] | **−0.01 [−0.01, −0.00]** |
| Highest pre-probe rate | 1.7 Hz | 3.3 Hz | 4.5 Hz | 0.1 Hz |
| **All criteria** | **fails task** (by 0.009) | **pass** | **pass** | **fails task, facilitation, excitability** |

- **DINOv2 mean patch token.** Detection on the inner source just misses the criterion while the
  readout facilitation is clear. This encoder carried the weakest full-body movement signal in Gate 1
  (single-image accuracy 0.57, R4). It is a robustness encoder; its runs carry their own health checks
  in the campaign, and nothing is tuned for it.
- **Non-spiking twin: not usable as built.** Its softplus unit, `softplus(k(U − θ))/k`, gives a large
  output below threshold (about 90 Hz at the initial membrane level, against near silence for the
  spiking neuron), so at initialisation every layer fired at 40–180 Hz. The rate regularisers then
  pushed its motor output to near zero on task frames, the task loss stayed at chance (0.70) and the
  readout carries no signal. The twin needs a rate function matched to the spiking neuron's
  rate–membrane relation (for example the noisy LIF transfer function) before its campaign. It is a
  comparison model and the Gate 2 fallback (spec §8), so it must be repaired before Gate 2, but it does
  not affect the spiking campaign.

**Readiness.** The spiking network passes every health criterion for H2 (two seeds; OpenCLIP and the
DINOv2 summary token), H0 and H1, so their training campaign can start (`model/campaign.py`). Each
run repeats the health check on its own inner source, and failed runs are reported with the failing
criteria (spec §9).

### 6.12 First campaign stopped; per-trial limits (24 October 2026, before any evaluation)

**Campaign start and stop.** The training campaign (H2, OpenCLIP, spiking, on `jessica`'s GPU; H2,
DINOv2 summary token, on its CPUs) was started with the §6.11 settings. Its first three networks
(folds 1–3, seed 0) were checked as they finished:

- Fold 1, seed 0 reproduced the §6.11 health check **bit for bit** (identical training history,
  threshold, facilitation), so training is deterministic on this machine.
- Fold 2 failed two criteria: on its unseen inner source 3 the FDI population fired at 9.0 Hz before
  the probe and the readout facilitation was not detected (+0.46 spikes [−0.39, 1.31]). Per source,
  the fold-1 to fold-3 networks held training sources at 3.7–6.3 Hz on movement trials (single
  variants up to 8.1 Hz) while the batch mean sat at the 5 Hz limit; unseen inner sources ranged from
  2.2 to 9.4 Hz. The readout of these pooled networks grows from 2 to 5.4–5.5 Hz and is flat beyond
  (excitability curves), so the declared 6 Hz background criterion lay past the readout's peak.

Both campaigns were stopped after about an hour; their three finished and four partial networks are
archived in `output/model/campaign_superseded_batch_mean_cap/` and are not used.

**Change.** The motor limits apply to every trial, as the paper excluded background EMG trial by
trial, and the background criterion is 5 Hz, below the measured peak (§6.2, §6.5; spec amendment).

**Health check with per-trial limits, rest 2 Hz, movement limit 4 Hz** (`output/model/v6/`, folds
1–4, seed 0, inner sources 2–5): facilitation +2.81 [1.93, 3.69], +1.48 [0.80, 2.16], +1.06
[0.34, 1.78] and +2.26 [1.46, 3.06], all detected; highest pre-probe rates 4.2, 6.3, 2.2 and 5.04 Hz,
so folds 2 and 4 fail the 5 Hz criterion. Unseen sources were driven up to about 2.3 Hz above the
training limit, which leaves too little margin below the readout's peak.

**Second change.** Rest 1 Hz and movement limit 3 Hz (per trial): the working range moves down and
leaves about 2.5 Hz below the readout's peak for unseen sources; a lower rest also calibrates a larger
task probe, whose readout grows over a wider range (§6.10).

**Decision rule, written before the results of this second check.** The campaign restarts with
these settings if all four networks (folds 1–4, seed 0) detect readout facilitation on their unseen
inner source (95% interval above 0) and pass the task, rest, threshold, dose, excitability and
saturation criteria. The background criterion is then reported for every campaign run and applied
at evaluation by the readout range rule (§6.3), with the share of instances above the limit
reported per display; it is not used to change the settings again. If any of the four fails a criterion, the cause is diagnosed on training sources and
corrected before the campaign restarts, as in §6.10–6.12 (amended 25 October 2026, before the
results, at the project owner's request: a measurement or code fault is fixed, not left standing).
What is never done is to change any setting in response to a context effect, a ratio or any result on
a held-out source; none is computed before the evaluation (§6, development rule).

**Result of the second check (rest 1 Hz, movement limit 3 Hz, per trial;** `output/model/v7/`, folds
1–4, seed 0, inner sources 2–5):

| Fold (inner source) | Facilitation [95% CI] | ADM static → movement | Highest pre-probe rate | Readout peak |
| --- | --- | --- | --- | --- |
| 1 (2) | +6.57 [5.65, 7.50] | 8.16 → 8.22 | 3.3 Hz | 7.9 Hz |
| 2 (3) | +2.40 [1.46, 3.33] | 7.78 → 7.99 | 5.7 Hz | 8.0 Hz |
| 3 (4) | +3.50 [2.83, 4.17] | 7.98 → 7.82 | 1.1 Hz | 8.0 Hz |
| 4 (5) | +6.48 [5.57, 7.39] | 7.62 → 7.43 | 3.2 Hz | 8.0 Hz |

Task AUC 0.98–1.00, rest 1.0 Hz, dose-response and excitability monotone, no saturation, no silent or
fast visual neurons, in all four. With the lower rest the calibrated task probe is larger (threshold
0.022–0.023 against 0.016) and, as expected (§6.10), the readout keeps growing up to about 8 Hz
instead of 5.5 Hz. The fixed 5 Hz background criterion, set below the earlier networks' 5.5 Hz peak,
was therefore out of date: fold 2's 5.7 Hz lies inside its own network's working range. **Third
change:** the background limit is each network's measured readout peak (§6.5), which keeps the
criterion's stated purpose. Near the peak the readout grows less steeply (in fold 2's network about
0.4 spikes per Hz between 4.8 and 6.3 Hz against about 2.8 between 1 and 2.5 Hz), so an instance driven close to its peak shows a
compressed facilitation; pre-probe rates are reported per display for this reason (§6.3).

With the per-network limit all four networks pass every criterion (records rechecked with the final
code; recheck is deterministic on this GPU). By the rule above, the campaign restarts with these
settings.

### 6.13 Training many networks at once (25 October 2026)

A spiking network is simulated one millisecond after another, and each millisecond is a handful of
very small operations, so a single network keeps the GPU waiting on instruction launches rather than
computing (one network: 0.41 s per update on `jessica`, three sharing the GPU: about 0.30 s per
network-update). `model/stack.py` trains K networks of identical shape together: their parameters
are stacked along a leading network dimension and every millisecond is one set of batched operations.
Nothing is shared between the networks. Each keeps its own initial weights (built by
`network.Network` with its own seed), its own trial draws and noise (the generators of `train.train`,
seeded the same way, drawn in the same order), its own readout and its own loss; the summed loss gives
each network exactly its own gradient, and Adam acts element by element. A stacked network is
therefore the network `train.train` gives alone, up to the rounding of batched matrix products: the
largest weight difference after 20 updates was 6 × 10⁻⁸ (test in `tests/test_network.py`).

| Networks trained together (alone on the GPU) | Seconds per update | Per network |
| --- | --- | --- |
| 1 | 0.41 | 0.41 |
| 10 | 0.56 | 0.056 |
| 20 | 0.85 | 0.042 |
| 40 | 1.87 | 0.047 |

The campaign trains the folds of one seed as one stack (`model/campaign.py --stack 20`); in the
running campaign a stack of 17 networks took 0.54 s per update (0.032 s per network, about nine times
the earlier throughput). Each trained network is written to `<run>.trained.pt`, and its health check
(`train.py --finish`, the same `prepare`/`finish` code as `train.health`) runs as a separate process
while the next stack trains. Stacks checkpoint and resume like single runs. Reports record
`trainer: stack` and the stack size. Folds 1–3 of seed 0 were trained one at a time before the switch;
fold 4 of seed 0 was also trained one at a time in the §6.12 check, so the stack's fold-4 network can be
compared with it.

### 6.14 Non-spiking twin repaired (25 October 2026, before any evaluation)

The twin's softplus output (§6.11) emitted about 90 Hz below threshold and did not learn. Its unit now
emits the **mean-field rate function of the network's own spiking neuron**: the steady firing rate of
the noisy leaky integrate-and-fire neuron (same constants, noise and reset) under a constant input that
settles its membrane at x, tabulated by simulation on a grid of x from −1 to 3 in steps of 0.01
(`network.lif_transfer`) and interpolated linearly, so it is differentiable. The twin computes the same
filtered input as the spiking neuron, with the same noise, and emits this rate instead of spikes. At
x = 0.8 the rate is 1.6 Hz, at 1.0 13 Hz, at 1.6 50 Hz; a test checks it against an independent
spiking simulation (within 10%). Unlike the spiking population (§6.10), the twin's probe response grows
monotonically with excitability, because it has no reset. Health check (25 October 2026, folds 1–4,
seed 0, one stack of four, `output/model/v8_rate/`): **all four pass every criterion**. The twin's
campaign (H2 with each encoder, 20 folds × 10 initialisations) is queued after the spiking
configurations.

### 6.15 Spinal motoneuron readout: tried, not adopted (25 October 2026)

To make the MEP analogue less abstract, a spinal stage was built (`network.SPINAL`, readout stage
`spinal`, used by no result): each muscle's cortical pool drives 64 motoneurons of the same neuron
model; motoneurons have no resting drive (silent at rest, as the paper required no background EMG); the
connection weight (0.0193) is the largest for which cortical spikes at the 3 Hz movement limit leave
them essentially silent (≤ 0.1 Hz; derived by bisection on synthetic input, re-derived by a test); the
readout counts recruited motoneurons, with its own resting threshold and 120% probe.

On the four §6.12 networks (training sources only) it detected full-body facilitation in three folds
(+16 to +43 recruited motoneurons), but recruitment was nearly all-or-none: at rest the dose-response
went from 0 to all 64 motoneurons between 0.5 and 2 × threshold, 43–63% of task trials recruited at
least 90% of the pool, and on fold 2's high-drive inner source motoneurons fired before the probe and
facilitation was negative. A spread of motoneuron thresholds (the size principle; resting levels from 0
to −0.25, −0.5 or −1) did not remove the saturation. The cause is structural: the probe makes the whole
cortical pool fire within a few milliseconds, a volley too large for the motoneurons' range, whereas a
human MEP at 120% of resting threshold is far below its maximum. A valid spinal readout would need
corticospinal volley structure (I-waves), heterogeneous connections and larger pools, each with
literature-based constants; it is left as future work, and "the readout is an analogy, not physiology"
remains a stated limitation. The stage stays in the code, switched off, with its tests.

### 6.16 Training campaign: health on training sources (25–26 October 2026)

**What ran.** 12 configurations × 20 folds × 10 initialisations = 2,400 networks: the spiking network
for H2, H0 and H1 and the non-spiking twin (§6.14) for H2, each with OpenCLIP, the DINOv2 summary
token (`dinov2_cls`) and the DINOv2 mean patch token (`dinov2_patchmean`). They ran on six CREMI GPUs
(`jessica`, `daniel`, `joe`, `vladimir`, `moneo`, `cheonech`), one configuration per machine at a time,
from 25 October 01:55 to 26 October 16:27. Every network trained and received its health report
(`output/model/campaign/<configuration>/`); none failed to run. As in every check before, health is
measured on each network's inner training source (the source after the held-out one), which the
network never trained on; no held-out source was used.

| Configuration | Healthy / 200 | Over-driven | No transfer | Facilitation only | Inner sources failing in ≥ 5 of 10 seeds (**none healthy**) |
| --- | --- | --- | --- | --- | --- |
| H2, OpenCLIP, spiking | 181 | 18 | 0 | 1 | 9, **18** |
| H2, DINOv2 summary, spiking | 143 | 41 | 11 | 5 | **3**, 9, **12**, **17**, **18**, **20** |
| H2, DINOv2 mean patch, spiking | 120 | 48 | 22 | 10 | **1**, **2**, **3**, **9**, **12**, **17**, 18, 20 |
| H0, OpenCLIP, spiking | 198 | 0 | 2 | 0 | — |
| H0, DINOv2 summary, spiking | 188 | 6 | 2 | 4 | **20** |
| H0, DINOv2 mean patch, spiking | 185 | 8 | 3 | 4 | 5 |
| H1, OpenCLIP, spiking | 190 | 10 | 0 | 0 | **18** |
| H1, DINOv2 summary, spiking | 138 | 21 | 40 | 1 | 3, **7**, **10**, 15, 17, **18**, 20 |
| H1, DINOv2 mean patch, spiking | 126 | 23 | 51 | 0 | **2**, **9**, **10**, 15, **17**, 18, **19** |
| H2, OpenCLIP, twin | 200 | 0 | 0 | 0 | — |
| H2, DINOv2 summary, twin | 171 | 21 | 8 | 0 | 3, 12, 18, 20 |
| H2, DINOv2 mean patch, twin | 176 | 18 | 6 | 0 | 9, 17 |

In total 2,016 of 2,400 networks pass every criterion (twin rows with the readout ranges measured
over the extended sweep, §6.17; 1,996 before). Only three criteria ever fail (task, facilitation,
background); threshold, dose-response, excitability, fixation, saturation, readout peak and the
visual-activity criteria pass in every network.

**Failures follow the fold, not the initialisation.** Where a fold fails, it usually fails for every
seed: 367 of the 404 failures before §6.17 (91%) lie in folds where at least 5 of 10 seeds fail. Each
source is the inner source of exactly one fold, so these data cannot tell a property of the inner
source from a property of the fold's training set (both exclude the held-out and the inner source):
on H1's sources 17 and 18, the two highest-salience sources, fold 17 trains without either. They take two
forms, which are the two sides of the same limit: the motor drive learned on the training sources does
not carry over to an unseen source at the right strength.

- **Over-driven** (background fails, nearly always with facilitation): on the inner source's movement
  frames the FDI population fires at 8–24 Hz before the probe (twin: 15–21 Hz; healthy spiking networks:
  median 1.6–3.1 Hz), past its readout peak (about 8 Hz spiking, 15.7 Hz twin), where the probe response falls
  (§6.10). The movement−static facilitation then shrinks to about zero or turns negative, so the
  facilitation failure is a consequence of the background failure. The per-trial limit of 3 Hz (§6.12)
  holds the trained sources only; nothing holds the rate on a source the network has not seen.
- **No transfer** (task fails): the population stays at rest (about 0.95 Hz) on the inner source's
  movement frames, so movement and static trials are not separated (AUC < 0.9) and nothing is
  facilitated.

**Where they come from.**

- **Encoder.** DINOv2 transfers between sources far less evenly than OpenCLIP: 3–8 failing sources
  for H2 against 2 (spiking) and 0 (twin). Input statistics explain this only in part: source 9's
  standardised DINOv2 movement vectors are about twice the typical length of the training sources'
  (2.3 × summary token, 2.0 × mean patch token), whereas most failing sources look ordinary. A modest
  shift in the feature vector, passed through a steep motor-pool input, is enough to move the rate
  from 3 Hz to either rest or past the peak.
- **H1's gate.** H1's salience gate is one value per source. On sources 17 and 18 it is 3.05 and 2.91,
  above the 99th percentile of the training movement frames in those folds (2.80 and 2.62), so it
  multiplies the movement drive beyond the trained range. This alone explains H1 with OpenCLIP (source
  18 over-driven in every seed); with DINOv2 it amplifies the encoder's uneven transfer in both
  directions (task AUC down to 0.09 on source 18, a reversed response).
- **H0** (hand-only input, no gate) is the most robust; its few failures are mostly one source per
  configuration.
- **The twin** has more headroom (readout peak about 15.7 Hz against 8 Hz, §6.17) and fails less with
  the same encoder: 176 against 120 healthy networks for H2 with the DINOv2 mean patch token, 200
  against 181 with OpenCLIP.

**Consequence for the evaluation.** If failing networks were excluded, a fold with no healthy network
would contribute no instance: 1 fold for H2 with OpenCLIP (spiking), 5–6 for H2 with DINOv2 (spiking),
up to 5 for H1 with DINOv2, none for the twin. How networks that fail health enter the evaluation was
decided on these training-source results only, after the readout ranges were re-measured (§6.17); no
training setting was changed in response to them.

### 6.17 Readout ranges re-measured, and networks that fail health (26 October 2026, before any evaluation)

**Curve extension (a measurement fault, fixed).** The excitability curve of §6.12 swept the extra FDI
current from 0 to 0.3 in 13 steps. Testing the premises of the readout range rule on all 2,400
campaign reports found that every spiking network peaks inside that sweep and falls after it, but no
twin network does: the twin's response was still rising at the last level, so its "readout range" was
the end of the test (about 14 Hz), not a peak. The sweep now runs to 0.6 in the same steps
(25 levels; `train.excitability_curve`), and a new criterion, `readout_peak_found`, requires the peak to
lie inside it (`train.diagnose`). Every campaign network's curve was recomputed on `joe`'s GPU
(`model/extend_curves.py`, rest display only); the first 13 levels reproduced the saved curves
exactly, and the extended curves were written into the health reports, with the 13-level values kept
under `*_sweep13` keys.

| | Spiking (1,800) | Twin (600) |
| --- | --- | --- |
| Peak inside the extended sweep | 1,800 | 600 |
| Readout range changed | 0 (median 8.0 Hz) | 575 (median 14.1 → 15.7 Hz; plateau, then a slow fall) |
| Background verdict changed | 0 | 20 fail → pass, 0 pass → fail |
| Excitability verdict changed | 0 | 0 |

**Premises of the rule, tested on all 2,400 networks (training sources only).**

| Premise | Result |
| --- | --- |
| Network-level criteria (visual activity, rest, threshold, dose-response, excitability, saturation, silent readout, readout peak) pass | 2,400 / 2,400 |
| The pre-probe rate rises with the injected current at every step, so a rate above the peak's means more excitation than at the peak | 2,400 / 2,400 |
| The readout peaks inside the tested range and falls beyond it | 2,400 / 2,400 |
| The peak is the readout's maximum: on inner-source movement trials, networks within their range never exceed it (13-level ranges) | 2,159 / 2,166 within range ≤ peak; largest ratio 1.005 |
| Over-driven networks read below their own peak (13-level ranges) | 216 / 234 (spiking 175 / 175; the 18 twins were those whose range was the end of the old sweep) |
| Networks that do not transfer read near rest (pre-probe < 2 Hz), a valid measurement | 145 / 145 (117 still facilitate, 19 reversed) |

**Physiology.** In humans the MEP input–output curve is sigmoid; background contraction raises its
gain 4–7× and leaves its plateau unchanged (Devanne, Lavoie & Capaday, 1997). An MEP does not fall
when the muscle is active. The model's falling readout above its peak is therefore a model property,
not physiology, and is reported as a limitation ([decision audit](decision-audit.md)).

**Rule (declared 26 October 2026, before any evaluation; spec §5.1 and §6.4 amended).**

1. A network that fails a network-level criterion is excluded (none does).
2. No network is excluded for a source-dependent health failure (task, facilitation or background on
   its inner source). Excluding is never conservative: if a failure reflects the inner source, it
   would drop an unrelated held-out source; if it reflects the fold's training set, it would drop the
   networks most likely to show null or invalid held-out effects and so favour P1 and P2. The two
   cannot be separated (§6.16). Health results are reported per network; the share of failing
   networks per configuration estimates how often its held-out readouts will be invalid.
3. Networks that do not respond to a held-out source stay in: their readouts are valid near rest.
4. A held-out display whose pre-probe FDI or ADM rate (mean over that instance's trials of the display
   and trial type) exceeds the instance's readout range is **over-driven** for that instance. Every
   criterion is computed twice: (a) with over-driven instance-displays removed (complete cases per
   contrast), as the paper removed trials with background EMG; (b) with their readouts set to the
   instance's readout maximum (the peak of its excitability curve), the least excitation consistent
   with being past the peak and the analogue of the human MEP plateau. Removing them favours "no
   detached-hand effect" when detached-hand displays are over-driven; the maximum works against it.
5. A criterion is **met** only if it is met under both (a) and (b), **not met** only if it fails under
   both, and otherwise **undetermined**, which never counts in the hypothesis's favour (also not for
   S3's requirement that a comparison model does not meet P1). These are worst-case bounds for
   outcomes that cannot be measured (Horowitz & Manski, 2000). The share of over-driven instances is
   reported per display and model.
6. The analyses restricted to healthy networks and without trimming are reported, not pass conditions.

### 6.18 Evaluation simulator (26 October 2026, before any evaluation)

`model/evaluate.py` runs one instance (trained network, hypothesis, initialisation) through the
session of §6.4 on one source and returns, per trial, the FDI and ADM probe readouts and both
populations' mean rate in the 100 ms before the probe.

- **Hypotheses as input and gate per displayed image.** H0 sees the hand-only image of the same
  family and frame, g = 1; H1 has g = 1 + s/s̄ on movement frames (1 elsewhere); the gated models (H2,
  the S3(a) visible-area model, H2-learned, H4) have g = c + (1 − c)·a on every frame of a display,
  with a from their own attachment estimate; the ungated network has g = 1. On the fixation display
  there is no visual drive and g = 1, or c for the gated models (§6.6).
- **Probe.** Resting threshold found on the fixation display with the model's fixation gate
  (`network.threshold`), task probe 1.2 × threshold on every task and fixation trial, read against the
  matched no-probe twin.
- **Two modes.** `continuous` runs the whole session at 1 ms, never reset, with the session continuing
  from the no-probe branch after each probe. `trialwise` starts each trial from rest 300 ms before its
  second frame and simulates all trials as one batch. Every model and gate value run on one session
  uses the same noise seed, so they are compared with common random numbers (a gated model with c = 1
  reproduces the ungated run exactly, tested).
- **Per instance**, as the paper per participant: task trials only; readouts beyond ± 2 SD of their
  cell mean removed (cell = display × trial type × population; untrimmed is the sensitivity
  analysis); z-normalised over both populations; averaged into cells. Pre-probe rates are kept per
  cell, untrimmed, for the readout range rule.
- **Aggregation.** The three primary displays' cells, in the human target's order, go through the
  human target's own contrast and ratio functions (`analysis/human_target.py`); intervals by the
  declared hierarchical bootstrap (source groups, then initialisations within each drawn group;
  10,000 resamples, seed 20261022); P3 from C_FDI per initialisation averaged over source groups.
- **Safeguard.** The simulator refuses a network's own held-out source unless called with
  `open_evaluation=True`, which only the frozen evaluation may do (development rule, §6).

**Equivalence of the two modes (§6.4), checked on trained networks without any context display:**
a full-body-only session on the network's inner training source (H2, OpenCLIP, fold 1, seed 0;
spiking and twin; `evaluate.py mode-check`, on `joe`).

| Check | Spiking | Twin |
| --- | --- | --- |
| Noise off: trials identical between modes | 100 / 100 (largest difference 0) | 100 / 100 within 9 × 10⁻⁵ (float32 rounding) |
| Noise on: continuous run vs 20 trialwise runs, Welch t per cell (3 trial types × 4 measures), Holm | No cell differs (smallest adjusted p 0.53) | No cell differs (smallest adjusted p 0.14) |

The feedforward networks are therefore simulated trialwise; the recurrent variant will be simulated
continuously. Tests: `tests/test_evaluate.py` (models and gates, reproducibility, common random
numbers, mode equivalence without noise, probe twin, held-out refusal, trimming and normalisation,
contrasts through the human code, the two-level bootstrap).

## 7. H2 context pathway and criterion S3: decided 24 October 2026

The project owner delegated this decision to the assistant on 24 October 2026, asking for the
option that best matches the paper and is scientifically and academically sound. It is recorded
as a spec amendment (§1, §2, §5.2, §6.2, §7.2 and §10 of the spec).

**Context pathway: idealised forearm continuity (option a).** a is the forearm visible without a
break below the wrist line, in palm widths, capped at 1 (`model/context.py`): 1-px cross-sections
of the source's SAM-HQ forearm mask are counted from the wrist line until the first one that is
less than half visible; "visible" means the displayed pixel equals the full-body pixel and is not
black. Reasons:

1. **It matches the paper's manipulation.** The paper varies visual continuity between the hand
   and the body, and its detached hand is cut at the wrist; the missing-wrist control follows
   Tieri et al. (2015). Continuity at the wrist is the variable the paper manipulates.
2. **It measures connection, not amount.** A visible segment after a gap does not count (tested),
   so the matched pairs are separated by connection alone, while their area is equal.
3. **It is parsimonious and transparent.** It has no learned parameter, uses only the displayed
   pixels and the hand's own geometry (never a condition label), and is identical across the 24
   appearance variants of every source (checked on all 480 families). The cap of one palm width is
   the forearm length of `hand_stub`, which the frozen spec already treats as an attached arm.
4. **Option b offers no gain in validity.** A learned arm detector would be trained with the same
   segmentation masks, so it is not more autonomous; it would add a model with its own architecture
   and training choices, require new OpenCLIP patch-token extraction, and its output on these
   displays is expected to be the same forearm-presence signal.

Measured values (mean over 20 sources; `data/context_continuity.json`):

| Display | FB | UL | DH | DH rectangle | Stub | Gap stub | Arm window | Hand window | Scrambled | Occluded wrist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | 1.00 | 1.00 | 0.00 | 0.80 | 0.99 | 0.00 | 0.99 | 0.00 | 0.15 | 0.00 |

Consequences, stated before any model result:

- The values agree with every H2 prediction declared in spec §7.2 except the archive rectangle,
  which shows 0.80 palm widths of forearm (0.50–1.00 by source); its declared prediction is
  corrected in the spec. H2 therefore predicts that the rectangle sensitivity analysis will show
  little of the FB/UL vs DH contrast, which is a prediction, not a failure of the design.
- The occluded wrist gets a = 0: a pixel-continuity estimate cannot complete an arm behind an
  occluder. H2 thus predicts no facilitation there; the spec declared no prediction.

**Gate: one parameter.** g = c + (1 − c)·a. The frozen spec allows at most two parameters, but a
single human ratio cannot identify two (a slope and a midpoint): intermediate displays such as the
scrambled view would then depend on an arbitrary choice. With one parameter, the gate is linear in
continuity, and c, the residual gain of a fully disconnected hand, is fitted on training sources
to the human ratio Δ(DH, FDI)/Δ(FB, FDI) = −0.10. The ungated network's detached-hand movement effect is expected to
be positive (Gate 1 transfer gains for the detached hand are 1.24–1.75, so the movement moves the
silhouette along the full-body readout axis at least as much as the full body), and the gate scales
it towards 0 as c falls, so the fit is expected to reach c = 0. H2 then effectively uses no human data
beyond the bound; the fitted value is reported.

**Criterion S3.** A low-level gate fitted to the same human ratio could match it through visible
area alone, since the detached hand shows far less of the image. S3(a) as frozen would then reject
H2 almost by construction, while the human data (FB, UL, DH, where continuity and visible area
change together) cannot tell the two accounts apart. S3 is therefore amended: (b), the hand-only
version, is kept as written; (a) replaces the context estimate by the global visible-area fraction
with the same gate refitted, and H2 must exceed it on **both** matched control pairs, where area is
equal and only connection differs. Whether the low-level model meets P1 is reported.

**What H2's success will and will not show.** Because a = 1 for FB and UL and 0 for DH, H2's fit to
the robust human pattern is largely built in once the network transfers the movement mapping from
full body to upper limb. Stated per criterion (added 24 October 2026, before any evaluation):

| Criterion | For H2 | What can still fail |
| --- | --- | --- |
| P2, full body | Trained (the auxiliary task) | Only the readout (checked by the readout-facilitation criterion, §6.5) |
| P1, S1 | Close to built in: with c ≈ 0 the gated detached hand has no visuomotor input, so Δ(DH) ≈ 0 | Only through Δ(UL) and Δ(FB) |
| S3(b) | Built in: H0's contrast is 0 by construction | Only through P1 |
| P2, upper limb; S2 | **Not built in**: the full-body mapping must transfer to the upper-limb features, whose movement direction has cosine ≈ 0.5 with the full-body one (Gate 1 R7) | Transfer too weak or too strong; readout out of range |
| S3(a) | **Not built in**: the two pair members get equal low-level gates and different H2 gates by construction, but H2 exceeds the low-level model only if the full-body mapping also responds to the movement in the stub and arm-window displays, whose movement directions are nearly orthogonal to the full-body one (cosine 0.05–0.25, R7) | Δ(stub) or Δ(arm window) not above 0 |
| P3, encoders | Not built in | Unstable optimisation; encoder-dependent transfer |

The informative tests are therefore S2, the upper-limb half of P2, the control-display transfer that
S3(a) requires, P3, agreement across the three encoders, and the comparison with the ungated network,
H0 and H1. The ungated network is the reference for necessity: if it already reproduced the human
pattern, the gate would not be needed. A positive result supports the claim that a wrist-continuity
gate on a learned visuomotor mapping is sufficient for the human pattern and distinguishable from a
visible-context account in its control predictions. Because the gate is computed from segmentation
masks rather than learned from the image, it does not show how such a signal could arise from vision,
and it does not show that humans use continuity rather than visible context (spec §1).

### 7.1 Learned context pathways (declared 25 October 2026, before any gate value is computed)

Requested by the project owner to address the hand-designed gate. In these stimuli, attachment and the
paper's condition coincide by design (full body and upper limb show the forearm, the detached hand does
not), so **any** correct attachment estimate gives the same gate for the three paper displays; the main
pattern stays partly built in for every attachment model. Two learned pathways are therefore added, with
different roles. Neither uses a condition label, the paper's display names, MEP analogues or human data;
both are fitted per outer fold on the fold's 19 training sources only, and applied, like H2's gate
(g = c + (1 − c)·m, c fitted by §6.6), to the trained H2 networks at evaluation (no new training).

**H2-learned, attachment learned by forearm prediction.** A self-supervised predictor learns, from
full-body images only, what the forearm just below the wrist looks like given the hand:

- Geometry from each source's hand landmarks (the controls' geometry: wrist point, hand axis, palm
  width P). Hand window: 0 to 1.5 P along the axis from the wrist towards the fingers, ±0.75 P across,
  sampled at 24 × 24 luminance. Forearm band: 0 to 1 P below the wrist line, ±0.5 P across, 16 × 16.
- Predictor: ridge regression from the hand window to the forearm band, fitted on the static and
  movement full-body frames of the 19 training sources; the ridge penalty is chosen from
  {10⁻², 10⁻¹, …, 10⁴} by leave-one-training-source-out prediction error.
- Match of a display (static frame): its prediction error e (mean squared difference between the
  predicted and the displayed band). Calibration: the leave-one-training-source-out errors of the
  full-body static frames, E. **m = min(1, 2 · P(E ≥ e))**: 1 for a display predicted at least as well
  as the median full-body view of an unseen person, falling to 0 for a display predicted worse than
  every one.
- What it adds over the rule: the continuity signal is learned from seeing a connected body and read
  from the displayed pixels, not defined by segmentation; on the controls and on unseen hands its
  values are not given by construction. Expected for the paper's displays: m(FB) = m(UL) (identical
  pixels in the band), m(DH) ≈ 0 (black band).

**H4, own-body familiarity (competing hypothesis).** A context gate that measures how much the whole
display resembles the observer's own first-person body view, learned from full-body images only:

- Features: the network's own encoder input (each encoder separately), standardised on the training
  sources' full-body static frames (each refit leaves out the same sources as the model).
- Model: principal components of the 19 training sources' full-body static frames, as many as explain
  90% of their variance; distance d = Euclidean norm of the whitened component scores.
- Calibration as above: E = leave-one-training-source-out distances of full-body static frames
  (components refitted without the source); **m = min(1, 2 · P(E ≥ d))**.
- It tests whether a generic own-body prior, rather than attachment, can produce the human pattern. It
  is not built in: Gate 1 found the upper-limb features far from the full-body ones (R6), so H4 is
  expected to give the upper limb a low gate and to fail S2 (upper limb ≈ full body in humans); if so,
  that is evidence against a familiarity account.

Under this calibration a full-body view of an unseen person gets m = 0.75 on average (1 for the half
that is at least as typical as the median, then falling evenly), so both pathways also weaken full-body
facilitation somewhat; this follows from the declared rule and applies to every display alike.

Both are scored with every criterion of spec §5.2 and reported with H0, H1 and the ungated network;
neither replaces the primary model. Their gate tables are computed and recorded before any evaluation
(they are properties of the displays, not results).

**Gate tables (computed 25 October 2026 after the definitions above were committed to the text, before
any evaluation; `model/learned_context.py`, `output/learned_context/`).** Mean m of each held-out
source's displays (static frames), averaged over the 20 folds:

| Display | FB | UL | DH | DH rectangle | Stub | Gap stub | Arm window | Hand window | Scrambled | Occluded wrist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rule a (§7, for comparison) | 1.00 | 1.00 | 0.00 | 0.80 | 0.99 | 0.00 | 0.99 | 0.00 | 0.15 | 0.00 |
| H2-learned (forearm prediction) | 0.69 | 0.69 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 |
| H4, OpenCLIP | 0.68 | 0.15 | 0.00 | 0.01 | 0.00 | 0.00 | 0.01 | 0.02 | 0.00 | 0.10 |
| H4, DINOv2 summary | 0.71 | 0.39 | 0.00 | 0.09 | 0.00 | 0.00 | 0.02 | 0.02 | 0.09 | 0.34 |
| H4, DINOv2 patch mean | 0.70 | 0.02 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.02 |

Ridge penalties chosen per fold: 0.01, 10 or 100; H4 kept 13–15 principal components.

- **H4** behaves as expected: the upper limb is far less familiar than the full body, so H4 predicts
  much weaker upper-limb facilitation and is expected to fail S2.
- **H2-learned does not isolate attachment.** It agrees with the rule on the paper's displays, but it
  also rejects the connected controls (stub, arm window) and the archive rectangle, which show a
  forearm below the wrist: the predictor learned the whole full-body scene in the band (the desk
  around the forearm included), and any display that differs from a full-body view there, even with a
  connected arm, is predicted worse than every full-body view of an unseen person. Its control
  predictions therefore differ from H2's, and it is expected to fail S3(a) on the stub pair.
- **The definitions are not changed after these values.** Narrowing the band or relaxing the
  calibration until the stub reads as connected would steer a model's control predictions towards the
  hypothesis. The result is reported as it is: a simple self-supervised continuity learner on pixels
  does not recover attachment from these images.

## 8. Validation of the methods (24 October 2026)

A review of every stage against the paper, the frozen spec and the code, made when the owner asked
for a general check before the network phase continues.

| Stage | Check | Result |
| --- | --- | --- |
| Stimuli and cohort | Hashes; Linux reproduction on CREMI; cross-machine encoder reproduction | All pass; Gate 1 features bit-identical on two GPUs |
| Image review | Independent raters | Not run (owner's decision); single unblinded review is a stated limitation |
| Human target | Published statistics reproduced; participant-level inference; five scales | Reproduced exactly; only `C_FDI`, Δ(FB) and Δ(UL) are robust; ADM exploratory |
| Gate 1 baselines and Addendum A | Method committed before data; all deviations logged; sensitivity | Reverse pattern robust (72/76 and 144/156 cells); exploratory analyses uncorrected, labelled as such |
| Trial simulator | Paper's Methods re-read against the code | Matches; assumptions listed in §2 (no inter-trial gap, fixation timing, variants as trials) |
| Encoder | Choice after seeing baselines | Least favourable encoder chosen, agreement across encoders required (spec amendment) |
| Network units | Synthetic test | Raw units amplified inputs about 110-fold; normalised before any training on image features |
| Fixation display | Health check on training sources | Encoded black frame out of distribution (FDI 45 Hz); presented as rest (§6.8) |
| Rest regulariser | Health check | Held only the mean of FDI and ADM; now per population (§6.8) |
| Readout normalisation | Against the deposit | Task trials only, as the deposited MEPs (§6.3) |
| Readout window | Dose-response in the second health check | Human 5–60 ms window cut through the model's 1–5-ms volleys; 0–55 ms from the probe (§6.9, spec amendment) |
| H2 gate | Identifiability | Two parameters not identifiable from one ratio; one parameter (§7) |
| H2 control predictions | Measured continuity on all displays | Agree with spec §7.2 except the rectangle (corrected) |
| S3 | Logic against H2 with a fitted gate | Would reject H2 by construction; amended to the matched-control comparison (§7) |
| Muscle specificity | Training objective | Built in by the ADM regulariser; ADM results exploratory (§6.2) |
| Development rule | Code paths | Health checks use training sources only; no held-out contrast is computed before the evaluation |
| **Readout validity** (pre-training validation, §6.10) | Probe readout on trained full-body trials | Did not register the trained response (−0.47 spikes): motor population over-driven (26 Hz), then heterogeneous; corrected by observation without execution and homogeneous pools; facilitation +4.9 to +6.1 spikes |
| Paper's preprocessing | Methods re-read for every step | Background-EMG exclusion and ± 2 SD trimming had not been mirrored; added (§6.3, spec amendment) |
| Necessity of the gate | Comparison models | No model isolated the gate; ungated network added (§6.6) |
| What is built in | Each criterion against H2's construction | P1, S1, S3(b) and P2 (full body) are close to built in for H2; S2, P2 (upper limb), S3(a) and P3 are not (§7) |
| Network memory | Time constants | Feedforward state at the probe depends on the second frame only; sequence effects only in the recurrent variant (§6.4) |
| Health criteria | Code against §6.5 | "> 200 Hz" not computed and "silent" measured on one batch; task accuracy confounded by source offsets; corrected (§6.10) |
| Citations | Added references checked against the publishers' records | Fadiga et al. 1995, Kraskov et al. 2009, Vigneswaran et al. 2013 verified |

**Still to do before Gate 2:** the training campaign is finished (26 October 2026, §6.16; 2,016 of
2,400 networks healthy after the readout ranges were re-measured, §6.17); the rule for networks that
fail health is declared (§6.17); the evaluation simulator is built and its two modes agree (§6.18).
Open: the criteria code applying §6.17's two treatments, the fit of c, the recurrent variant. Done 25 October 2026: the
non-spiking twin repaired (§6.14), the spinal readout tried and not adopted (§6.15), the learned
context pathways declared (§7.1).

**Next step after these (decided 25 October 2026, not yet declared in the spec):** robustness
variations of the choices that have no literature value: visual layer 128 and 512 neurons (256 in
the primary), learning rate 3 × 10⁻⁴ and 3 × 10⁻³ (10⁻³), 6,000 updates (3,000), and the time
constants and time step already listed in spec §6.3 (10/40 ms and 2/10 ms; 0.5 ms). Each would run on
all 20 folds with 3 initialisations and be evaluated once with the primary model. They are to be
declared in the spec before the evaluation is opened.

## Change log

| Date | Change | Reason |
| --- | --- | --- |
| 24 Oct 2026 | Draft: trial simulator and input rules fixed; encoder proposal | Start of the model phase |
| 24 Oct 2026 | Encoder decided: option A (OpenCLIP primary, DINOv2 robustness, amended decision rule) | Project owner's decision; spec amendment |
| 24 Oct 2026 | Network, training, probe, health criteria and H0/H1 fixed (§6) before any training; H2 context pathway and S3 raised as an open decision (§7) | Start of network construction |
| 24 Oct 2026, before any training on image features | Input units normalised (constant input x → membrane x), noise defined by a resting membrane SD of 0.1 θ, initial biases 0.8 / 0.6, probe kept as a raw 1-ms current | The declared constants were in snnTorch's raw units, where a constant input is amplified about 110-fold: in a synthetic test the visual layer fired at about 600 Hz. After the change, rates were 14–27 Hz (visual) and 5 Hz (motor rest) |
| 24 Oct 2026, after the first health check, before any evaluation | Fixation presented as rest; rest rate per population; CPU checkpoints; readout z-normalisation over task trials; training per hypothesis (§6.6, §6.8) | Health check on training sources found the encoded black frame out of distribution (FDI 45 Hz at rest) and FDI silent on static frames |
| 24 Oct 2026 | H2 context pathway (forearm continuity), one-parameter gate, S3 amendment, rectangle correction (§7); validation review (§8) | Decision delegated by the project owner; spec amendment |
| 24 Oct 2026 | Readout window 0–55 ms from the probe (§6.9, spec amendment); second health check passes every criterion | The 5-ms window start cut through the model's synchronous volleys and made the dose-response non-monotone |
| 24 Oct 2026, before any evaluation | Pre-training validation (§6.10): observation without execution (rest 2 Hz, movement limit 5 Hz, rest trials), homogeneous motor pools, readout-validity criteria, AUC task criterion, visual criteria over all training sources, ± 2 SD trimming, readout range rule, ungated comparison, H0 standardisation, declared fit of c, H0/H1 training paths, campaign driver; §7 corrected and made explicit about what is built in (spec amendment) | The probe readout did not register the trained motor response, so P2 would have failed in every model for a measurement reason; the remaining items were gaps between the spec, the paper and the code |
| 24 Oct 2026, before any campaign run | Training made interruption-safe: checkpoint with optimiser and both random streams every 250 updates, resumed exactly (tested); results written atomically, report last; reconnecting launch command in `model/campaign.py` | CREMI lab PCs kill a user's processes when the SSH session closes and can be switched off; an interrupted run previously restarted from zero, and a run killed while saving could have left a damaged file counted as finished |
| 24–25 Oct 2026, before any evaluation (first campaign stopped after 3 networks) | Motor limits per trial, rest 1 Hz, movement 3 Hz; background criterion and readout range rule use each network's measured readout peak (§6.12, spec amendment) | The batch-mean limit let unseen sources drive the motor population past the readout's peak, where facilitation was not detected |
| 25 Oct 2026, during the campaign (no evaluation) | Stacked training (§6.13), about nine times faster; equivalence tested | One network at a time left the GPU waiting on instruction launches; the full design would have taken weeks |
| 26 Oct 2026, after the campaign (no evaluation) | Campaign health recorded (§6.16): 1,996 of 2,400 networks healthy; failures follow the inner source (over-driven or no transfer); no setting changed | Record before Gate 2; how failing instances enter the evaluation is to be decided before Gate 2 |
| 26 Oct 2026, before any evaluation | Excitability sweep extended to 0.6 and `readout_peak_found` criterion added; every campaign curve recomputed (§6.17): spiking ranges unchanged, twin ranges 14.1 → 15.7 Hz, 20 twins no longer over-driven (2,016 healthy) | The twin had not peaked in the 0.3 sweep, so its readout range was the end of the test, not a measured peak |
| 26 Oct 2026, before any evaluation | Networks that fail health (§6.17; spec §5.1, §6.4): none excluded for source-dependent failures; over-driven held-out displays counted only if a result holds both with them removed and at the readout maximum | Exclusion is never conservative; the over-driven readout is not measurable, so it is bounded both ways; every premise tested on all 2,400 networks |
| 26 Oct 2026, before any evaluation | Evaluation simulator (§6.18); trialwise and continuous modes agree on trained networks | Needed for every evaluation and for the fit of c |
