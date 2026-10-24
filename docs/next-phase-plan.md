# Next phase plan — 22 October to 18 December 2026

The stimulus set is finished. This plan moves effort from images to the target,
controls and model, and ends with frozen results by **18 December**. January stays
free for writing the report.

*Status 23 Oct:* the human target, the frozen analysis spec, the silhouette and the
control stimuli were completed on 22 October; **Gate 1 (visual baselines) was completed on
23 October, three weeks early: no baseline reproduces `C_FDI`, and every one shows the
reverse pattern**, so H2 has a clean target ([baselines](baselines.md)). No model has been
run. The rater validation (issue 5) is prepared but not run: the project owner decided on 24 Oct
to skip it. Next: the trial simulator (built 24 Oct) and the spiking model (W4).
GPU work runs on the CREMI lab PC `daniel` (RTX 4000 Ada, 20 GB).

## 1. Where the project stands

| Area | Status |
| --- | --- |
| Stimuli | 480 accepted families from 20 V7 source pairs, plus the approved detached-hand silhouette. Integrity checks pass; 28 CPU tests pass |
| Action | Left index abduction, 10.5–24.7° (median 18.3°), matches the paper |
| Effective sample size | **20 distinct movements**: desk, seed and white balance are nuisance variation |
| Human target | Built: `analysis/human_target.py`; reproduces the paper's published statistics exactly ([analysis spec](analysis-spec.md) §4) |
| Analysis spec | Frozen 22 Oct; one logged amendment (§7.2 controls) before any result |
| Control stimuli | Six controls for all 20 sources, approved 22 Oct ([controls](controls.md)) |
| Rater validation | Package prepared 24 Oct ([protocol](rater-protocol.md)); **not run**, by the project owner's decision (24 Oct): recruiting naive raters was not feasible. Single unblinded review remains a stated limitation |
| Visual baselines | Done 23 Oct ([baselines](baselines.md)): method fixed before extraction; 10,560 images; every baseline shows the reverse pattern (largest movement signal for the detached hand), 0/20 sources positive; Gate 1 passed. 24 Oct, Addendum A: the reversal persists in hand-centred views where the hand is up to twice as large (28,800 images) |
| Model | 24 Oct ([model method](model-method.md)): trial simulator and inputs built; encoder decided (OpenCLIP primary, spec amendment); spiking network and auxiliary training built; first health check passes on training sources. H2 context pathway (forearm continuity) and the S3 amendment decided 24 Oct (spec amendment); methods validated ([model method](model-method.md) §8). Open: repeat the health check with the corrections, H0/H1 training, parallel training |

### Reanalysis of the paper's deposited data (22 October)

The published Table 2 and every post-hoc z value in the Results are reproduced exactly from
the deposit. They come from mixed models on **raw mV** (not the z-scored values the Methods
describe) with only a participant random intercept, so ~11,000 single trials act as
independent observations. At participant level (n = 32, paired tests, Methods z-scale):

| Effect | Participant-level result | Use in this project |
| --- | --- | --- |
| FDI movement > static, full body | t = 4.4, p = .0001, 25/32 positive | Robust |
| FDI movement > static, upper limb | t = 4.1, p = .0003, 23/32 positive | Robust |
| FDI movement effect, detached hand | t = −0.4 (none) | Robust |
| `C_FDI` (definition below) | t = 4.1, p = .0003; survives log scale | **Primary endpoint** |
| ADM movement effect, upper limb | p = .07 to .015 depending on normalisation | Exploratory |
| FDI full body / upper limb > detached hand, regardless of movement | p ≈ .02–.05; n.s. on log scale; confounded with block order | Exploratory |
| FDI full body vs upper limb | t = 0.2 | Expected ≈ 0 |

Questionnaire ratings follow full body ≫ upper limb ≈ detached hand, whereas MEPs
follow full body ≈ upper limb ≫ detached hand. The robust motor effect therefore
tracks a **visible** property, the hand being attached to an arm, more closely than
reported ownership. This is the effect a vision-driven model should explain.

### Measured low-level confounds (80 seed-9 neutral families)

| | Full body | Upper limb | Detached hand: rectangle | Detached hand: silhouette (primary) |
| --- | --- | --- | --- | --- |
| Visible (non-black) area | 100% | 62% | 7.8% | 2.1% |
| Mean luminance | 0.49 | 0.30 | 0.04 | 0.011 |
| Movement pixels / visible pixels | 1.1% | 1.7% | 13.6% | 31.8% |
| Contrast-normalised A/B distance (64 px grey) | 3.8 | 2.3 | 4.6 | 19.9 |

A generic image model sees the **largest** movement signal in the detached hand,
opposite to the human result; the paper-matched silhouette makes this even stronger.
Baselines must show this explicitly.

## 2. Do first (22–25 October)

1. **Stop stimulus work.** Do not run the fresh CUDA canary or new generation sweeps.
   They are not on the critical path. Keep the corrected-mask code for later.
   *Done 22 Oct: README updated; the roadmap on page 25 of the project guide is superseded.*
2. **Freeze the analysis cohort.** Use the historical accepted archive: 480 families,
   with every result repeated without reference 17 (456 families). Record the
   decision and the archive manifest hash in the analysis spec.
   *Done 22 Oct: `config/analysis.json` and [analysis spec](analysis-spec.md) §3.*
3. **Build the human target in this repository** (`analysis/human_target.py`): the
   12-cell participant table, all movement contrasts with paired bootstrap intervals,
   and sensitivity versions (raw mV, log, within-muscle z, excluding < 50 µV).
   It must reproduce the numbers above.
   *Done 22 Oct: reproduces the numbers above and all 18 legible published statistics; 7 tests.*
4. **Draft the analysis spec** (`docs/analysis-spec.md`): endpoints, splits, models,
   gates and what counts as success or failure. Freeze it on 30 October, before any
   model result exists.
   *Drafted 22 Oct. Open items resolved the same day: the 20 references are AI-generated,
   20 distinct synthetic identities (spec §3); the success criteria were audited, revised and
   approved (spec §5.2); the silhouettes were visually approved. **Frozen 22 Oct.***

## 3. Issues to address

| # | Issue | Action | When |
| --- | --- | --- | --- |
| 1 | Paper's statistics treat trials as independent and use raw mV despite the Methods; questionnaire z = −117 is implausible for n = 32 | Done: participant-level targets; discrepancies listed in [analysis spec](analysis-spec.md) §4.2. Contacting the authors is optional and your decision | W1 |
| 2 | Only 20 independent movements | Leave-one-source-out (20 folds) for every evaluation; never split a source's 24 variants or six conditions. Report per-source results | All weeks |
| 3 | Area, luminance and edge confounds between contexts | Control stimuli (§4.2) and low-level baselines; a continuity claim must survive them | W2–W3 |
| 4 | Detached hand is a desk rectangle with a cut wrist, not an isolated hand | Done 22 Oct: skin-only silhouette cut at the wrist is the primary condition; rectangle kept as sensitivity ([method](detached-hand-silhouette.md)). Visually approved 22 Oct ([record](silhouette-approval.json)) | W1 |
| 5 | One unblinded reviewer accepted all images | Two naive raters score a fixed sample (20 sources × 4 contexts × 2 frames, both detached-hand versions) for realism and visible artifacts; report agreement. **Decision 24 Oct (project owner): not run**; the package stays available, and the report states that one unblinded reviewer accepted all images | — |
| 6 | Archive was produced with the earlier mask; desk generation changed some hand pixels | Done 22 Oct: 144/480 families (all variants of references 10, 11, 12, 16, 18, 20), at most 645 pixels = 0.65% of the visible hand, largest difference 6/255, identical in both frames; located at the outer palm edge near the wrist. **Decision 22 Oct: keep all 480 families, no exclusion** (see validation) | W1 |
| 7 | No sex metadata per source (the paper sex-matched bodies) | Done 22 Oct: `data/source_metadata.json` records the reference sex labels (10 female, 10 male) from the generation configuration; V7 images were generated sex-neutral; you confirmed on 22 Oct that all 20 look like their label, recorded as `apparent_sex` | W1 |
| 8 | `dataset.py review` accepts `assistant` as reviewer | Done 22 Oct: only `reviewer: human_user` decisions are recorded; test added | W1 |

## 4. New work to start

### 4.1 Human target and analysis spec

Built. Definitions, families, scales, reference values and model success criteria are in
the [analysis spec](analysis-spec.md) §4–§5. Model outputs are compared on contrasts and
scale-free ratios, never on absolute millivolts.

### 4.2 Control stimuli (CPU only, built from frozen pixels)

Built 22 Oct: six controls for all 20 sources, defined, checked and illustrated in
[controls.md](controls.md); exact parameters added to spec §7.2 by amendment before any result.
Two matched pairs isolate connection: `hand_stub` vs `hand_gap_stub` (equal forearm area) and
`window_arm` vs `window_hand` (equal area, arm vs desk). `scrambled` keeps the upper-limb
pixels but scatters the arm; `occluded_wrist` is exploratory. Visually approved 22 Oct
([record](controls-approval.json)).

### 4.3 Visual baselines

*Done 23 Oct: method committed before any feature extraction, then run on all 480 families;
results and figures in [baselines](baselines.md). Paired decoding accuracy saturates at 1.00, so
single-image accuracy and an exploratory transfer gain are reported instead.*

- Encoders: low-level statistics (area, luminance, edge energy), DINOv2 ViT-L/14 (summary and patch tokens), OpenCLIP, and hand-region-only features (no context effect by construction).
- Per context and control: the movement signal, meaning the A/B feature distance relative to between-source spread.
- Leave-one-source-out decoding of static vs movement within each context, plus train-on-full-body / test-on-detached-hand transfer.
- Features are extracted once on GPU (Kaggle or cluster), cached with hashes and analysed on CPU.

### 4.4 Temporal model

- Trials follow the paper: static frame for 2,250–3,250 ms, then the same frame or the movement frame for 750 ms, with a common probe 250 ms after the second frame starts. FDI and ADM readout populations; state is not reset between the frames.
- Frozen encoder → leaky integrate-and-fire network (snnTorch), plus a non-spiking twin with matched capacity.
- Three hypotheses:
  - **H0, no context:** readout uses hand features only.
  - **H1, salience:** the movement signal is scaled by low-level visual change; expected to fail as described in §1.
  - **H2, attachment gate:** a context pathway estimates whether the hand is connected to an arm and gates the visuomotor mapping.
- Key tests: calibrate on full body and detached hand only, then **predict upper limb**. Under H2 the arm-gap control must behave like the detached hand. H2 can therefore fail on data it was not fitted to.
- No condition labels enter the network, and no condition gets its own gain.

## 5. Schedule

| Week | Dates | Work | Deliverable |
| --- | --- | --- | --- |
| W0 | 22–23 Oct | Freeze cohort; human target module; draft analysis spec | Done 22 Oct |
| W1 | 26–30 Oct | Issues 6–8 and spec freeze, all done 22 Oct; start the rater sample | Done 22 Oct: frozen `docs/analysis-spec.md` and `config/analysis.json` |
| W2 | 2–6 Nov | Rater package prepared 24 Oct, not run (owner's decision); controls built and approved 22 Oct | — |
| W3 | 9–13 Nov | Free (features, baselines and Gate 1 done 23 Oct; ratings not collected) | **Gate 1 passed 23 Oct** |
| W4 | 16–20 Nov | Trial simulator; feedforward LIF and non-spiking twin; rate and probe diagnostics | Working models with stable activity |
| W5 | 23–27 Nov | H0/H1/H2; calibrate on full body + detached hand, predict upper limb; recurrent variant | First held-out predictions |
| W6 | 30 Nov–4 Dec | Predictions for all controls; probe-strength sweep | Control predictions. **Gate 2, 4 Dec** |
| W7 | 7–11 Dec | Ablations: spiking vs non-spiking, recurrent vs feedforward, final frame only vs sequence; 10 initialisations (spec §5.1); without reference 17 | Ablation table; configurations frozen 11 Dec |
| W8 | 14–18 Dec | Single frozen evaluation; final human vs model figures; methods and results draft; supervisor review | Interim results report |

## 6. Decision gates

**Gate 1 — 13 November: do visual baselines already produce `C_FDI`?**

- If low-level or hand-only baselines reproduce it, the continuity explanation is in doubt. The report then focuses on that confound, and the model must beat those baselines.
- If generic encoders show the reverse pattern (largest effect in the detached hand), H2 has a clean target. Continue as planned.

*Outcome 23 Oct: the second branch, for every low-level, pixel, hand-only and encoder baseline
(72 of 76 sensitivity cells reversed, 4 null, none positive). Frozen encoders also show no
attachment variable beyond a pixel cue, so H2's context pathway has to compute it
([baselines](baselines.md) R5).*

**Gate 2 — 4 December: is the model scientifically usable?**

- Stable, non-saturated activity; probe responses vary smoothly with strength; H0, H1 and H2 make different predictions for the controls.
- If not, simplify: drop recurrence and spiking details, keep the non-spiking version and report the spiking attempt as a negative result. Do not add per-condition gains.

## 7. Out of scope until after 18 December

- Fresh CUDA canary or new stimulus generation
- Reward-modulated STDP or other local learning rules
- Little-finger stimuli for an independent ADM mapping, unless W7 ends early
- Biophysical TMS → EMG modelling
- Questionnaire modelling: the model has no ownership variable

## 8. Rules that apply throughout

- Leave-one-source-out splits; preprocessing fitted only on training folds.
- Report human uncertainty across participants, image generalisation across sources and model stability across initialisations, kept separate.
- Keep failed runs and negative results; open the held-out evaluation once.
- "Not significant" is not "zero": state an equivalence margin before claiming one.
