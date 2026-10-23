# Gate 1: visual baselines and latent-space analyses

Code: [`analysis/features.py`](../analysis/features.py) (rendering, low-level statistics,
encoders), [`analysis/baselines.py`](../analysis/baselines.py) (analysis),
[`analysis/baseline_figures.py`](../analysis/baseline_figures.py) (figures). Implements
[analysis spec](analysis-spec.md) §6.1 and Gate 1 (§8) and the baseline plan in the
[next-phase plan](next-phase-plan.md) §4.3.

**Status.** The method below was written and committed on 23 October 2026, **before any
feature was extracted or any baseline number was seen**. It makes the operational choices
that the spec leaves open (resolution, distance, spread, decoder, latent-space tests). The
spec itself is unchanged. Any later change to this method is listed in the change log at the
end, with its reason.

## 1. Question

Gate 1 asks whether the human pattern can already be produced by the images alone, before
any motor model exists. The human robust pattern (spec §4.5) is facilitation for full body
(FB) and upper limb (UL) but none for the detached hand (DH): `C_FDI` > 0. For a visual
baseline the analogue is the **movement signal**: how much the representation of a display
changes when the index finger abducts. If a low-level or hand-only baseline already gives a
larger movement signal for the connected displays than for the detached hand (P1 met), the
continuity explanation is in doubt and spec criterion S3 becomes decisive (spec §8). If the
generic encoders show the reverse (largest signal in the detached hand), the model must
overcome the input statistics, and H2 has a clean target.

A baseline's movement signal is a *distance*, so it is never negative. It describes the input
that a model inherits; it is not a predicted MEP. P2 (positive FB and UL effects) is therefore
satisfied trivially by any non-blind baseline and is not evaluated here.

## 2. Stimuli

All 480 accepted families (20 sources × 24 appearance variants), both frames, rendered from
the hash-checked archive (`features.Renderer`):

| Condition | Rendered from | Role |
| --- | --- | --- |
| `full_body`, `upper_limb` | Archive PNGs | Connected |
| `detached_hand_silhouette` | Archive silhouette PNGs | Disconnected (primary DH) |
| `detached_hand` | Archive rectangle PNGs | Disconnected (sensitivity DH) |
| `hand_stub`, `window_arm` | Controls of spec §7.2 over the family's full-body pixels | Connected |
| `hand_gap_stub`, `window_hand`, `scrambled` | Same | Disconnected |
| `occluded_wrist` | Same | No declared role (exploratory) |
| `hand_only` | Silhouette cropped to `used_hand_box`, padded with black to a centred square | Spec §6.1 hand-only input; identical in every context |

10,560 images. Each rendered array is hashed (`output/features/index.csv`), binding every
feature to its pixels. A test checks that the rendered controls equal `controls.render` and
that the hand-only crop keeps every silhouette pixel.

## 3. Representations

| Name | Definition | Role |
| --- | --- | --- |
| `lowlevel` | Visible area (fraction of pixels with any channel > 0), mean luminance, RMS contrast (SD of luminance), edge energy (mean squared Sobel gradient of luminance); luminance = Rec. 709 weights on stored values / 255; whole 2048² image | Low-level baseline (spec §6.1) |
| `pixels64` | Luminance box-averaged to 64 × 64 (4,096 values) | Low-level baseline: fixed pixel code |
| `dinov2_cls`, `dinov2_patchmean` | DINOv2 ViT-L/14 (`facebook/dinov2-large`), final-layer-norm summary token and mean of the 256 patch tokens, 224 px | Generic self-supervised encoder |
| `openclip` | OpenCLIP ViT-bigG-14, `laion2b_s39b_b160k`, `encode_image` before L2 normalisation, 224 px | Generic language-supervised encoder |
| `dinov2_448_*` | As DINOv2 at 448 px (32 × 32 patches) | Sensitivity: resolution |
| hand-only | Every representation of the `hand_only` input | Spec §6.1 hand-only baseline |

**Preprocessing.** The whole square image is resized to the encoder size with PIL bicubic
(antialiased) and normalised with the encoder's own mean and SD. No centre crop is applied, so
nothing at the border (the hand is near the top) is removed; this departs from the encoders'
default evaluation crop deliberately. Encoders run in float32 with TF32 disabled, because the
two frames of a family differ in a few thousand pixels and reduced precision could swamp that
difference. The model revisions are resolved at extraction and recorded with the weight hash in
`output/features/manifest.json`. A **noise floor** is measured by re-encoding two families
image by image in reverse order; it must be far below the smallest movement signal.

## 4. Movement signal (spec §6.1)

For representation R, condition c and family i with frames A (static) and B (movement):

- **Distance:** `d = ‖R(B) − R(A)‖₂`.
- **Between-source spread:** `σ_c = sqrt(E‖R(A_s) − R(A_t)‖² / 2)` over all pairs of
  different sources s, t that share an appearance variant (same desk, seed and white balance),
  using static frames. For one dimension this is the between-source SD; for an embedding it is
  the total between-source SD. Comparing sources within a variant isolates identity (hand,
  body, clothing) from desk and lighting.
- **Movement signal:** `m = d / σ_c`, computed per condition. A value of 0.1 means the finger
  movement moves the representation by a tenth of the typical difference between two people
  photographed in the same scene. For `lowlevel`, whose four statistics have different units,
  each statistic is divided by its own between-source SD before the Euclidean distance. A
  statistic with no between-source spread is left out of that distance; this happens only for
  visible area in the full-body view (1 in every image), and the code refuses to proceed if
  such a statistic changed between the frames.
- **Sensitivity measures:** raw distance `d` (embeddings share units across conditions) and
  cosine distance `1 − cos(R(A), R(B))`; not defined for `lowlevel`.

## 5. Inference

Sources are the independent units (spec §3, §5.1). Each family's movement signal is averaged
over the 24 appearance variants of its source, giving one value per source and condition.

- **Contrast** (spec §4.3 without muscles): `C = (m_FB + m_UL)/2 − m_DH`, primary DH =
  silhouette. Also every per-condition signal, `FB − UL`, and the matched-control differences
  `stub − gap stub`, `window arm − window hand`, `UL − scrambled`, `UL − occluded wrist`,
  `stub − DH`.
- **Ratios** (scale-free, spec §4.5): `DH / connected`, `UL / FB`, `DH / FB`, as ratios of
  source means.
- **Intervals:** 95% percentile bootstrap over the 20 sources, 10,000 resamples, seed
  20261022 (as for the human target). The spread `σ_c` is a fixed scale for each condition and is
  not resampled. Also reported: the number of sources with a positive contrast.
- **Gate 1 rule (spec §8):** a baseline *meets P1* if `C` > 0 and its interval excludes 0;
  it shows the *reverse pattern* if the interval lies below 0. S1 and S2 compatibility: the
  baseline's ratio lies inside the human 95% interval (`DH / connected` in [−0.67, 0.38],
  `UL / FB` in [0.34, 1.78]). Human values are reference lines only; no human data enter any
  computation.
- **Sensitivity:** rectangle DH; without reference 17 (19 sources); raw and cosine distances;
  DINOv2 at 448 px.
- **Hand-only:** one input for every context, so `C` = 0 exactly by construction (spec §6.1);
  its movement signal is reported as the size of the movement visible in the hand alone.

## 6. Linear decoding (next-phase plan §4.3)

**Static vs movement.** For each representation and display, a linear decoder is trained to
tell static from movement frames: standardisation and L2 logistic regression (balanced
classes) fitted on 19 sources; the regularisation `C ∈ {10⁻⁴ … 10²}` is chosen by grouped
5-fold cross-validation inside those 19 sources (spec §7.1). It is tested on the held-out
source (20 folds). **Primary metric: paired accuracy**, the fraction of held-out families
whose movement frame scores higher than its static frame (chance 0.5); single-image accuracy
is also reported. Each decoder is tested on every display of the held-out source, which gives
the full transfer matrix; the declared transfer is **trained on full body, tested on each
display**. Mean over held-out sources, source-bootstrap interval.

**Transfer gain (exploratory; added after the smoke test).** For each fold, the mean
static → movement shift of the decoder's score on a held-out display, divided by the mean
shift on the training display and sources: 1 means the movement moves the representation
along the learned full-body axis as much as in full-body training, 0 means not at all. It is
summarised per display and passed through the same contrasts as the movement signal. It was
added because paired accuracy reached 1.00 for every encoder in the smoke test, so accuracy
cannot show differences between displays.

**Attachment (exploratory).** Can a linear readout of a frozen encoder tell a connected from a
disconnected display when low-level properties are matched? Three tasks, same pipeline, both
frames as samples: `stub_pair` (hand stub vs gap stub, equal area), `window_pair` (arm window
vs hand window, equal area) and `main` (FB and UL vs DH). Paired accuracy compares the two
displays of the same family and frame. Within-task accuracy shows that the pair differs in the
representation; **transfer between tasks** (stub ↔ window, main → each pair) is the stronger
test, because the two pairs differ in shape, so only a shared attachment variable can carry
over. A low-level representation that transfers would mean the matched pairs are not matched
in that representation.

## 7. Latent-space geometry (exploratory)

These describe the representations; none is a Gate 1 criterion.

1. **Movement direction.** Unit vectors `u = (R(B) − R(A)) / ‖·‖`. *Alignment with full
   body*: `cos(u_c, u_FB)` within a family: is the same finger movement encoded in the same
   direction regardless of context? *Cross-source consistency*: mean cosine between the
   directions of different sources within one appearance variant: is there a shared
   "abduction axis"?
2. **Representational similarity (RSA).** Per family, a representational dissimilarity matrix
   (RDM) over the nine context displays with a declared role (static frames; each dimension
   standardised over all these images; Euclidean distance / √dim), averaged within source.
   Compared with an **attachment model RDM** (0 = same role, 1 = different) by Spearman
   correlation and by partial Spearman correlation removing visible-area (|Δ log area|) and
   luminance RDMs. Source-bootstrap intervals.
3. **Patch maps (DINOv2, 224 px).** Per 14-px patch (16 × 16 grid), the mean norm of the
   patch-token change between frames, per display. The *share outside the hand*: the fraction
   of patch-change energy in patches not touching the dilated hand (6 px) in either frame. In a
   vision transformer, attention can carry the finger movement into patches of the arm or body;
   differences within matched pairs (stub vs gap stub, arm vs hand window, UL vs scrambled)
   show whether that spread depends on connection.
4. **Variance decomposition.** The share of between-family variance of the movement signal
   explained by source, desk, seed and white balance (balanced 20 × 4 × 2 × 3 design),
   supporting the source as the unit.
5. **Example inputs.** One family at encoder resolution, with the luminance change map.

## 8. Limits declared in advance

- The movement signal and the decoders measure information in the image representations. A
  pattern in them is not a model of motor excitability, and no human data are fitted.
- Twenty sources from one generator and one scene template (spec §3): intervals describe these
  hands only.
- The controls have no human data; their predictions belong to H2, not to the baselines.
- Two encoders and two low-level codes do not exhaust the possible visual baselines.

## 9. Reproduce

```sh
pip install -r requirements-features.txt        # CUDA machine, Linux
python analysis/features.py                       # ~10,560 images; writes output/features/
python analysis/baselines.py                      # CPU; writes output/baselines/ and figures
python -m unittest tests.test_baselines
```

`python analysis/features.py --limit 2 --output <dir>` runs two appearance variants of every
source (smoke test).

## Results

Pending: no feature has been extracted at the time of writing.

## Change log

| Date | Change | Reason |
| --- | --- | --- |
| 23 Oct 2026 | Method written and committed before any feature extraction (commit `9d65ecc`) | — |
| 23 Oct 2026 | Smoke test: one appearance variant (20 families) through the whole pipeline | Check the code on real data before the full extraction |
| 23 Oct 2026, after the smoke test | Statistics without between-source spread left out of the low-level distance (§4) | Full-body visible area is 1 in every image; the declared formula divided 0 by 0 |
| 23 Oct 2026, after the smoke test | Transfer gain added (§6), labelled exploratory | Paired accuracy saturated at 1.00 for all encoders, so the declared metric cannot separate displays |
| 23 Oct 2026, after the smoke test | Figure layout corrected; no change to any computation besides the two rows above | Overlapping labels |
