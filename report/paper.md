# What would an image-driven model need to reproduce display-dependent motor facilitation during action observation? A pre-specified test of attachment accounts

Muhammed Emre Candir

**Status: skeleton, 11 November 2026 (final-phase plan, task A5).** Outline of [review](../docs/independent-review.md)
§8. Methods and limitations are drafted; the abstract and results are stubs that give, for each possible
outcome, the exact sentence the paper will contain and the `result.json` field it quotes. Fields are paths in
`output/model/evaluation/result.json`; `A` stands for each of the three analyses (`raw`, `remove`,
`maximum`). The stub sentences were fixed before the evaluation and are not edited after it, except to insert
numbers. Target length about 8,000 words.

---

## Abstract [stub]

*Template (one version is kept, chosen by the verdicts):*

> Watching a finger move facilitates motor-evoked potentials when the hand is seen attached to a body but not
> when it is seen detached (Matamala-Gomez et al., 2025). We asked what an image-driven visuomotor model
> would need to contain to reproduce this. We built a spiking network on a frozen image encoder, trained it
> only to detect the movement in full-body images of 18 synthetic sources, and read its excitability with a
> simulated probe at the paper's timing. A participant-level reanalysis of the human data showed one effect
> robust to every scale, the FDI contrast between connected and detached displays. Raw image change gave the
> reverse pattern. Without any gate, the network trained on full-body images already responded less to the
> detached hand (ratio 0.53 on training sources), while the same network trained on all displays responded more (1.14), so whether
> an attachment gate is needed depends on the training diet. On held-out sources, the one blind prediction —
> transfer to the upper-limb display — **[was met: Δ(UL) = x [ci], ratio UL/FB = y, inside the human
> interval | was not met: …]**. A continuity gate and a visible-area gate could not be distinguished on the
> paper's displays. The pipeline makes explicit which parts of the human pattern a model inherits from its
> training and which it predicts.

## 1. Question

Matamala-Gomez et al. (2025) found FDI facilitation during observation of index abduction for a full body
and an upper limb but not for a detached hand. A natural reading is that the motor system's response depends
on whether the observed hand is attached to a body. We ask a narrower question: can a visuomotor model driven
by images reproduce the display dependence, and what would it have to contain? We treat this as a test of
accounts, not as support for one: a model reproducing the pattern shows that an account is sufficient under
stated assumptions, and the comparison models show which components are necessary.

**Related work (draft; references in [references.md](references.md)).** Watching an action facilitates MEPs in
the muscles that would perform it (Fadiga et al., 1995), and corticospinal neurons modulate during observation
without producing muscle activity (Kraskov et al., 2009; Vigneswaran et al., 2013). Computational models of
the mirror system learn visuomotor mappings for grasp recognition from visual input (Fagg & Arbib, 1998;
Oztop & Arbib, 2002; review in Oztop, Kawato & Arbib, 2006), but none addresses how the body context of an
observed hand changes the motor response. Associative accounts hold that such responses arise from learned
correlations between seeing and doing (Heyes, 2001; Keysers & Perrett, 2004; Catmur et al., 2007; Cook et
al., 2014); predictive coding offers an alternative in which observation is explained by inferred motor
causes (Kilner, Friston & Frith, 2007). Our network is an associative model in this sense: it learns to
detect the movement from images and is never given an MEP target. Body continuity changes the felt ownership
of a seen hand (Perez-Marcos et al., 2012; Tieri et al., 2015), and ownership has been modelled as Bayesian
inference over multisensory cues (Samad, Chung & Shams, 2015; review in Kilteni et al., 2015); our gates
are much simpler, one scalar per display. MEP amplitude reflects many factors besides the excitability of
the neurons of interest (Bestmann & Krakauer, 2015), and circuit models of TMS are far more detailed than our
probe current (Esser, Hill & Tononi, 2005); the model's readout is an analogue of excitability only. Finally,
deep networks used as models of vision can succeed for reasons unlike human vision (Bowers et al., 2023) and
often exploit shortcuts in their training distribution (Geirhos et al., 2020). This bears directly on our
findings: the full-body-trained network's weaker response to the detached hand is such a training-
distribution effect (§6(b)).

## 2. Human target

Participant-level reanalysis of the deposited data (32 participants, 11,280 MEPs). We reproduced the
published mixed models exactly (8 of 8 Table 2 values, 10 of 10 post-hoc z values) and then analysed the
participant cell means with paired tests and bootstrap intervals on five scales. C_FDI =
[Δ(FB) + Δ(UL)]/2 − Δ(DH) is the only contrast robust on every scale: 0.283 [0.147, 0.419], t(31) = 4.06.
Δ(FB) = 0.288 and Δ(UL) = 0.223 are robust; Δ(DH) = −0.028 shows no effect. The published analysis treats
single trials as independent and used raw mV rather than the z-scores its Methods describe; at participant
level the ADM upper-limb effect is not significant (t(31) = 1.86, published z = 4.67). The model targets are
C_FDI, the two connected movement effects and two scale-free ratios, Δ(DH)/mean[Δ(FB), Δ(UL)] ∈
[−0.67, 0.38] and Δ(UL)/Δ(FB) ∈ [0.34, 1.78].

## 3. Stimuli and controls

Twenty AI-generated synthetic first-person left hands, each as a static and an abducted-index pair,
expanded into 480 families (4 desks × 2 seeds × 3 white balances) and accepted by visual review. The three
displays are built from the same pixels: full scene, upper limb (black below the desk edge) and a skin-only
silhouette cut at the wrist. Six control displays, also masks over the same pixels, separate continuity from
visible area: a hand with a 1-palm-width forearm stub against the same area of forearm after a 0.5-palm-width
gap, and equal-area windows with or without arm. The hand and every changing pixel are identical across all
displays. Limits: one generator and scene template, single-reviewer acceptance, no human data for the
controls.

## 4. Image-only baselines

Pixel, low-level, hand-only and frozen-encoder (DINOv2, OpenCLIP) representations of all 10,560 condition
images all show the reverse of the human pattern: the movement changes the representation most for the
detached hand. Frozen encoders separate connected from disconnected displays no better than a pixel code and
do not generalise across the matched pairs. This shows that raw visual change does not produce the human
pattern; it does not show what a trained, direction-specific readout does, which §6(b) addresses.

## 5. Model

**Encoder.** OpenCLIP ViT-bigG-14 (primary; chosen after the baselines as the encoder least favourable to the
hypothesis), DINOv2 ViT-L/14 summary and mean patch tokens (robustness).

**Network.** 256 leaky integrate-and-fire visual neurons; two 64-neuron motor pools (FDI, ADM) receiving the
visual spikes through a multiplicative gate g. Trained (3,000 Adam updates) to detect the index movement from
the FDI pool on full-body trials of 18 sources, with the motor pools held near rest during observation, as
the paper's muscles were. No MEP value is ever a target.

**Readout.** A 1-ms probe current at 120% of the resting threshold, 250 ms after the second frame; the pool's
spike count over 55 ms minus that of a matched no-probe branch. Single-trial readouts are trimmed and
z-scored within each model instance (one instance = one held-out source × one initialisation) and passed to
the human contrast code. Readouts on displays that drive the pool past its measured excitability peak are
flagged and treated two ways (removed; set to the maximum), alongside the readouts as simulated.

**What was calibrated.** The motor regime (rest targets, readout window, readout-range rule) was developed
against full-body facilitation on unseen training sources. No other display entered development.

**Accounts compared.** H0 (hand-only input), H1 (low-level salience gate), H2 (forearm-continuity gate
g = c + (1 − c)·a, c fitted to the human detached-hand ratio on training sources), the ungated network
(g = 1), a visible-area gate, a learned forearm-prediction gate and an own-body-familiarity gate. H2's
continuity estimate is 1, 1 and 0 on the three displays in every source; it is computed from segmentation
masks, so the gate is supplied rather than computed from the image.

**Evaluation.** Ten initialisations × 20 leave-one-source-out folds per configuration; 95% hierarchical
bootstrap over sources then initialisations. Criteria P1 (C_FDI > 0), P2 (Δ(FB), Δ(UL) > 0), P3 (8 of 10
initialisations), S1 and S2 (ratios inside the human intervals), S3 (exceeds hand-only and visible-area
versions); decision rule: all with the primary encoder and P1 with both robustness encoders. Specified
before any model result (22 October 2026), amended only through a dated log; code and configurations frozen
on 11 December 2026 and the held-out evaluation run once.

## 6. Results, in order of evidential weight [stub]

### (a) Upper-limb transfer on held-out sources: the only blind prediction

*Fields:* `main.openclip.analyses.A.models.h2.delta_UL` (estimate, ci), `….models.h2.S2_ratio`,
`main.openclip.verdicts.h2.P2`, `main.openclip.verdicts.h2.S2`; the same for `ungated`;
`main.openclip.over_driven.h2.upper_limb`.

*Outcome sentences:*

- P2 and S2 met: "On held-out sources the network's FDI readout facilitated for the upper limb
  (Δ(UL) = [x] [ci]) at a ratio to the full body of [y], inside the human interval [0.34, 1.78]: the
  full-body mapping transfers to the upper-limb image. Because the upper limb shares most of its pixels with
  the full body, this is the expected direction, and its size is the informative part."
- P2 met, S2 not met: "The network facilitated for the upper limb (Δ(UL) = [x] [ci]) but at a ratio of [y]
  to the full body, outside the human interval: the transfer is [weaker | stronger] than in participants. The
  decision rule is not met."
- P2 not met (upper-limb half): "The network did not facilitate for the upper limb (Δ(UL) = [x] [ci]); the
  full-body mapping does not transfer to the upper-limb image, and the decision rule is not met."
- Any of these undetermined: "Under the two treatments of over-driven readouts the verdict differs ([share]
  of instances over-driven on the upper limb); the criterion is undetermined and is not counted as met."

### (b) The two training diets and the necessity of the gate

*Fields:* `main.openclip.analyses.A.necessity` (estimate, ci, exceeds), `main.openclip.gate_necessary`,
`configurations.snn_allctx.analyses.A.necessity`, `configurations.snn_allctx.gate_necessary`,
`….models.ungated.S1_ratio` in both. Training-source values already known: ungated ratio 0.53 (full-body
training) against 1.14 (all-context training).

*Outcome sentences:*

- Gate not necessary with full-body training, necessary with all-context training (the outcome expected from
  training sources): "Whether the gate is needed depends on what the network was trained on. Trained on full
  bodies only, the network without a gate already gives [met criteria] (C_FDI(H2) − C_FDI(ungated) = [x]
  [ci]); trained on all displays, it facilitates for the detached hand ([ratio]) and needs the gate to meet
  [criteria]."
- Gate necessary under both diets: "The gate adds to C_FDI under both training diets ([x] [ci]; [x'] [ci'])."
- Gate necessary under neither: "Under neither diet does the gate add to C_FDI ([x] [ci]; [x'] [ci']); the
  held-out data do not show the gate to be necessary."

### (c) Continuity gate against visible-area gate

*Fields:* `main.openclip.analyses.A.S3a.pairs` (stub − gap stub, arm window − hand window: H2 − area),
`main.openclip.analyses.A.controls`, `main.openclip.verdicts.h2.S3a`; the visible-area gate's criteria
`main.openclip.verdicts.area`.

*Outcome sentences:* "On the paper's three displays the visible-area gate [meets | does not meet] the same
criteria as H2 ([list]). On the matched controls, where area is equal and only connection differs, H2
exceeds it by [x] [ci] (stub − gap stub) and [x] [ci] (arm − hand window). This separation was effectively
known before the evaluation (H2's control predictions were positive in 20 of 20 unseen training sources, and
at c = 0 the visible-area gate gives both pair members nearly the same gate); it shows that the two accounts
make different predictions, which only human data on the controls could decide."

### (d) Robustness

*Fields:* `main.dinov2_cls.verdicts.h2.P1`, `main.dinov2_patchmean.verdicts.h2.P1`, `decision`;
`configurations.{rate, snn_rec, snn_tm10, snn_tm40, snn_ts2, snn_ts10, snn_lr3e-4, snn_lr3e-3, snn_nv128,
snn_nv512, snn_u6000, snn_dt0.5}.verdicts.h2`; `without_reference_17`, `healthy_only`, `untrimmed`,
`rectangle`.

*Outcome sentences:* "With both robustness encoders P1 [holds | does not hold] ([verdicts]); the result is
[encoder-independent | encoder-dependent]. [N] of [M] comparison configurations agree with the primary
verdicts on P2 and S2; the exceptions are [list]. Removing reference 17, restricting to healthy networks,
omitting trimming and using the rectangle detached hand [change | do not change] the verdicts ([list])."
With DINOv2 the ungated network is already inside the human ratio interval on training sources, so P1 with
the robustness encoders does not test the gate.

## 7. What is built in

*Table from `docs/criteria-status.json` (generated by `analysis/report_figures.criterion_table`, joined
with the verdicts): for every criterion and model, met / not met / undetermined, and built in, calibrated,
previewed or open.* For H2, P1, P3, S1 and S3(b) are built in (continuity 1, 1, 0 and c = 0 make the
detached-hand effect 0 by construction), the full-body half of P2 was calibrated, S3(a) was previewed, and
only the upper-limb half of P2 and S2 are open.

## 8. Limitations

- **The gate is supplied and its parameter is not identified.** H2's continuity estimate is computed from
  segmentation masks and the full-body image; the declared fit target (−0.10) is unreachable by a gate, which
  gives c = 0, and every c up to about 0.45 fits the human interval with OpenCLIP.
- **Most criteria are built in or previewed for H2**; the evaluation adds one blind test.
- **The readout is not physiological**: a spike count without conduction delay, whose response falls when
  the pool is over-driven, unlike a human MEP.
- **Synthetic stimuli**: one generator and scene, 20 identities, single-reviewer acceptance.
- **Continuity and visible area are confounded in the human data**; the matched controls have no human data.
- **Pre-specification with dated amendments and GitHub push times**, not a registered preregistration;
  several amendments followed training-source results, and each is listed with what was known when it was
  written.

## 9. Conclusion [stub]

*Template:* "An image-driven network trained only on full-body views reproduces the ordering of the human
pattern without any attachment mechanism, because the detached hand lies outside its training distribution;
with experience of every display it does not, and an attachment gate is then needed. On held-out sources the
upper-limb transfer [was | was not] human-like. The paper's three displays cannot separate continuity from
visible context; the matched controls can, and measuring MEPs on them is the experiment that would decide
between the accounts."

## Data and code availability [stub]

*Repository, freeze tag `freeze-2026-12-11`, what stays local (stimulus archive, human deposit), and the
OSF deposit of the human data (Matamala-Gomez et al., 2025).*

## References

The verified list is [references.md](references.md) (task A6); the final list keeps only the entries
cited in the text, formatted for the target journal.
