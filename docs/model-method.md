# Model method: trial simulator, inputs and pre-declared choices

Code: [`model/trials.py`](../model/trials.py) (trial simulator), [`model/bank.py`](../model/bank.py)
(frozen-encoder inputs); tests in [`tests/test_trials.py`](../tests/test_trials.py). Implements
the parts of the [analysis spec](analysis-spec.md) §2, §5.1 and §6.3–6.4 that define what a model
instance sees, and fixes the details the spec and the paper leave open.

**Status, 24 October 2026: draft.** Sections 1–4 are fixed: the trial simulator is built and
tested, and nothing has been trained. Section 5 (encoder) was decided on 24 October 2026
(option A, spec amendment). Section 6 lists what must be fixed before the first training run. Later changes go
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

## 6. To fix before the first training run

- Network sizes; the input projection from the encoder vector to input currents; FDI- and
  ADM-associated readout populations; the H0, H1 and H2 pathways (spec §6.2).
- Auxiliary-task objective, training-sequence length, optimiser and schedule (on training
  sources only, tuned by grouped inner folds, spec §7.1).
- Staircase criterion for the model resting threshold (spec §6.4) and the silent or saturated
  run criteria.
- Gate 2 diagnostics (spec §8): activity, probe dose-response, and distinct H0–H2 control
  predictions.

## Change log

| Date | Change | Reason |
| --- | --- | --- |
| 24 Oct 2026 | Draft: trial simulator and input rules fixed; encoder proposal | Start of the model phase |
| 24 Oct 2026 | Encoder decided: option A (OpenCLIP primary, DINOv2 robustness, amended decision rule) | Project owner's decision; spec amendment |
