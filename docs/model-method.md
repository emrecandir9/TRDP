# Model method: trial simulator, inputs and pre-declared choices

Code: [`model/trials.py`](../model/trials.py) (trial simulator), [`model/bank.py`](../model/bank.py)
(frozen-encoder inputs); tests in [`tests/test_trials.py`](../tests/test_trials.py). Implements
the parts of the [analysis spec](analysis-spec.md) §2, §5.1 and §6.3–6.4 that define what a model
instance sees, and fixes the details the spec and the paper leave open.

**Status, 24 October 2026: draft.** Sections 1–4 are fixed: the trial simulator is built and
tested, and nothing has been trained. Section 5 (encoder) was decided on 24 October 2026
(option A, spec amendment). Section 6 fixes the network before any training; section 7 is an open
decision about H2 that needs the project owner. Later changes go
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
- The black fixation display is encoded with the same encoders and settings
  (`output/features_extra/black.npz`).
- **Standardisation is fitted on the training sources' full-body images only** and applied
  unchanged to every display. Full body is the only display seen in training (spec §6.3), and no
  statistic of the held-out source or of a test display enters preprocessing (spec §6.3, §7.1).
  Detached-hand and control features therefore arrive in a distribution the network has not
  seen, which is part of the test.

## 4. What the simulator guarantees (tests)

Counts per block and session; timing bounds; probe 250 ms after onset and the readout window
(5–60 ms) inside the second frame; contiguous timeline without gaps; correct frame pairs;
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
| Motor layer M | Two populations of 64 neurons each, FDI- and ADM-associated, same neuron model; input `s(g · W_VM S_V + b_M + σξ) + P(t)`, initial b_M = 0.6 (below threshold) |
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
- **Regularisers:** visual neurons kept between 2 and 40 Hz (hinge on the mean rate); motor rate
  on static second frames held near 5 Hz; the ADM population is held at the same rate in
  movement and static frames. **This builds in muscle specificity** (FDI learns the index
  movement, ADM does not), as the associative account predicts for index-movement training; ADM
  results are therefore exploratory, as in spec §5.2.
- **Optimiser:** Adam, learning rate 10⁻³, batch 32 trials, 3,000 updates; surrogate gradient
  fast sigmoid (snnTorch default, slope 25); weights initialised from a seed per
  initialisation (10 per configuration, spec §5.1).

### 6.3 Probe, threshold and readout (spec §6.4)

- **Readout:** for each probe, the FDI (and ADM) population spike count 5–60 ms after the probe,
  minus the same count in the matched no-probe twin. The twin is computed by branching from the
  identical network state at the probe time with the identical noise stream, so the difference is
  caused by the probe alone. The session then continues from the no-probe branch.
- **Resting threshold:** on the black fixation display, the smallest A for which the FDI
  response is at least 6 spikes (≈ 10% of the population) on at least 50% of 40 trials, found by
  bisection over A in [0, A_max] with 14 halvings (an adaptive staircase). The task probe is
  1.2 × threshold, as in the paper.
- **Single-trial readouts** are z-normalised within each instance over both populations (the
  paper's per-participant normalisation) and passed to the human-target contrast code.

### 6.4 Evaluation simulation

Each instance runs its full session (§2) at Δt = 1 ms with plasticity frozen; probes on every
task and fixation trial. Instances are simulated in parallel on the GPU. The same session is run
for the primary displays, the paired rectangle sensitivity and the controls.

### 6.5 Health criteria (Gate 2, spec §8), on training sources

| Criterion | Requirement |
| --- | --- |
| Visual activity | ≤ 10% silent neurons (no spike over a training session), ≤ 5% above 200 Hz |
| Motor rest | Mean motor rate on fixation 1–20 Hz |
| Task | Detection accuracy ≥ 0.9 on full-body trials of an inner held-out training source |
| Threshold | Found inside (0, A_max); probe dose-response monotone (Spearman ≥ 0.9 over 10 amplitudes, 0.5–2 × threshold) |
| Readouts | Not silent (mean task response > 0) and not saturated (< 90% of the population) |
| Distinct predictions | H0, H1 and H2 make different control predictions (spec §8) |

### 6.6 Hypotheses (spec §6.2)

- **H0, no context:** the visual input is the hand-only image of the same family and frame (the
  Gate 1 `hand_only` features), identical in every display; g = 1. Its context effect is 0 by
  construction.
- **H1, salience:** display features; g = 1 + s/s̄, where s is the 64 × 64 luminance distance
  between the displayed frame and the trial's first frame (0 for first frames and static second
  frames) and s̄ its mean over training full-body movement frames. No parameter is fitted.
- **H2, attachment gate:** display features; g = σ(w(a − a₀)), with an attachment estimate a from
  a context pathway and the two parameters w, a₀ fitted to the human ratio
  Δ(DH, FDI)/Δ(FB, FDI) = −0.10 on training sources only. **The context pathway and the S3
  comparison are awaiting the project owner's decision (§7).**

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

## 7. Open decision: the H2 context pathway and criterion S3

**Problem.** S3 requires that "the same network, training, probe and readout fed with (a)
low-level statistics" does not meet P1. H2 has two gate parameters fitted to the human DH/FB
ratio. If the low-level version's gate is also fitted to that ratio, it can match it through
visible area alone, because the detached hand shows far less of the image than the full body.
That version would then meet P1, and H2 would fail S3 almost by construction. The human data
cover only FB, UL and DH, where attachment and visible area change together, so they cannot
separate the two. Only the matched controls (`hand_stub` vs `hand_gap_stub`, `window_arm` vs
`window_hand`, equal in area and luminance) can, and they have no human data.

**Proposed resolution (recommended, needs a spec amendment):** keep S3 as written for the
hand-only version. For the low-level version, refit its gate by the same procedure on low-level
statistics, and add a declared **discrimination test on the matched controls**: H2's predicted
facilitation differences `stub − gap stub` and `arm window − hand window` must exceed those of the
low-level version, with a source-bootstrap interval excluding 0. Alternative: keep S3 unchanged
and accept that H2 will very likely fail it.

**Context pathway options:**

- **(a) Idealised continuity (recommended):** a = the visible fraction of a band 0–0.5 palm widths
  below the source's wrist line, inside the source's forearm mask (`data/inputs/*/sam_arm.png`),
  in the displayed image. It is computed from the image and the hand's geometry, never from a
  label; it models an observer who knows where the wrist is. Transparent, but it encodes the
  hypothesis directly, so its control predictions follow almost by construction.
- **(b) Learned arm detector:** a readout of patch features trained on full-body training images
  only to predict the forearm mask (segmentation targets, no condition labels); a = predicted arm
  presence just below the wrist. A more model-like account, but it needs OpenCLIP patch tokens
  (not yet extracted) or mixing encoders.

## Change log

| Date | Change | Reason |
| --- | --- | --- |
| 24 Oct 2026 | Draft: trial simulator and input rules fixed; encoder proposal | Start of the model phase |
| 24 Oct 2026 | Encoder decided: option A (OpenCLIP primary, DINOv2 robustness, amended decision rule) | Project owner's decision; spec amendment |
| 24 Oct 2026 | Network, training, probe, health criteria and H0/H1 fixed (§6) before any training; H2 context pathway and S3 raised as an open decision (§7) | Start of network construction |
| 24 Oct 2026, before any training on image features | Input units normalised (constant input x → membrane x), noise defined by a resting membrane SD of 0.1 θ, initial biases 0.8 / 0.6, probe kept as a raw 1-ms current | The declared constants were in snnTorch's raw units, where a constant input is amplified about 110-fold: in a synthetic test the visual layer fired at about 600 Hz. After the change, rates were 14–27 Hz (visual) and 5 Hz (motor rest) |
