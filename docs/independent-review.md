# Independent review of the TRDP project

Reviewed 10 November 2026 at commit `4fd2d50` (local Mac copy), plus the training-source outputs on CREMI (`/net/travail/mucandy/TRDP`, read from `jessica`). No held-out source and no upper-limb display was simulated for this review.

## Progress on the findings

Updated as each item is handled; the commit that handled it is given. Open items are continued from here.

| Item | Status |
| --- | --- |
| F2: record ungated results, withdraw "clean target" | **done** 10 Nov (model-method §6.19 table, §7 note, baselines note, README; `fit_gate.py --summary` prints the ungated ratio) |
| F16: document contradictions (answer 37) | **done** 10 Nov, except the items inside the spec and `config/analysis.json` (handled with the amendment below) |
| F1, F2, F7, F8: declared reporting amendments (informative criteria, necessity test, raw readouts, c sensitivity) | **done** 10 Nov (spec §5.2 and §10; model-method §6.19a) |
| F2, F7, F8: code for the necessity test, raw reporting and c sensitivity | **done** 10 Nov (`criteria.necessity`, `fit_gate.sensitivity_c`, tests); the evaluation driver must call them and `treated_cells(treatment=None)` |
| F9: amendment list and spec hash in `config/analysis.json`; "frozen" wording; external timestamp | amendment list, hash test (`tests/test_spec.py`) and wording **done** 10 Nov; external timestamp at the 11 Dec freeze **open** (commits are pushed to GitHub the same day from now on) |
| F3–F6, F11, F12: limitations written into the documents | F3, F5, F6 **done** 10 Nov (spec §1, §5.2, §8; validation "Remaining limits"); F4, F11, F12 to be carried into the report |
| All-context training ablation: specify, then train | specified and built 10 Nov (model-method §6.23, `--variation allctx`); training **not started yet** (next step: `python model/campaign.py --hypothesis h2 --representation openclip --variation allctx --seeds 0-2 --stack 20` on a lab-PC GPU, then ADM curves and `fit_gate.py --configurations h2_openclip_snn_allctx`); results **open** |
| F10: evaluation driver, tested and hashed | built, tested and rehearsed **done** 10 Nov (`model/run_evaluation.py`, model-method §6.22); its hash is recorded by `run_evaluation.py freeze` on 11 Dec (**open**, run on CREMI where the outputs are) |
| F13–F15, F17: wording and provenance fixes | F13, F14, F17 **done** 10 Nov (model-method §6.13, §6.19a; decision audit); F15 **done** 10 Nov (`train.provenance` in every new health report, fit, Gate 2 and evaluation record; feature manifest and learned gate tables pinned in `config/analysis.json` and checked at load). Outputs made before 10 Nov carry no code hash |
| 0.5-ms time step (already declared) | implemented and tested **done** 10 Nov (model-method §6.21); its 20 networks, curves and fits **open** |

Labels: **[V]** verified by me in code, data or output; **[D]** stated by the project, not independently checked; **[I]** my inference, premises given.

---

## 1. Executive verdict

**Sound with major revisions.** The engineering, the human-target reanalysis and the disclosure culture are strong. The inferential design is not yet able to support the claim the project wants to make.

The three findings that matter most:

1. **The primary endpoint cannot fail for the primary model.** H2's gate is exactly 1 for full body and upper limb and exactly 0 for the detached hand in all 20 sources, and c = 0. Any network with full-body facilitation then meets P1, P3, S1 and S3(b). Only the upper-limb half of P2, and S2, are real tests.
2. **The gate is already shown not to be needed for the main pattern.** On training sources the *ungated* network gives Δ(DH)/Δ(FB) of 0.49 (OpenCLIP), 0.04 and 0.00 (DINOv2), without being over-driven. The human-like ordering comes from weak transfer of a full-body-trained readout to out-of-distribution silhouettes. This contradicts the "clean target" premise and model-method §7's own expectation, and it is not reported.
3. **The "single held-out evaluation" is largely previewed.** Every source has already been evaluated as an unseen "inner" source, including H2's control predictions on all 20 sources. Only the upper-limb display is genuinely unseen.

None of these needs a parameter change. All need reframing and declared reporting amendments before 11 December.

---

## 2. What I could and could not access

**Read in full:** `README.md`; `docs/analysis-spec.md`, `model-method.md`, `baselines.md`, `decision-audit.md`, `validation.md`, `next-phase-plan.md`, `controls.md`; `config/analysis.json`; `analysis/human_target.py`; all of `model/` except `recurrence.py` (and `extend_curves.py` partly); `tests/test_human_target.py`; test names and docstrings of the other test files; `git log` with per-commit file statistics; the spec at `4faa1ff` and `b4e3fd7`.

**Read in part:** `analysis/baselines.py` (signal, bootstrap, decoding), `docs/detached-hand-silhouette.md` (first 40 lines), `docs/approval-receipt.json`, PPTX slide text (72 slides, extracted from the XML), both PDFs as text (`pdftotext` on `jessica`; images and layout not inspected).

**Not read:** `scripts/generate.py`, `dataset.py`, `silhouette.py`, `controls.py`, `segment_*.py`, `raters.py`, `analysis/features.py`, `handview.py`, `baseline_figures.py`, `docs/rater-protocol.md`, `scripts/cremi_*.sh`. Findings about stimuli construction are therefore [D] unless a check script confirmed them.

**Commands run (Mac, `.venv`, Python 3.12.14):**

| Command | Result |
| --- | --- |
| `python -m unittest discover -s tests -v` | `Ran 154 tests in 65.934s — OK (skipped=1)`; the skip is the GPU-only resume test |
| Same on `jessica` (RTX 4000 Ada) | `Ran 154 tests in 105.956s — OK`, none skipped; test files hash-identical to the Mac's |
| `python analysis/human_target.py --output <scratch>` | All 18 published statistics reproduced; `cells`, `contrasts`, `ratios`, `paper_comparison` CSVs byte-identical to `output/human_target/` |
| `python scripts/dataset.py check` | `Verified 20 sources, 120 condition PNGs, 100 inputs, 40 silhouette mattes; 480 families / 160 parents / 10 canaries.` |
| `python scripts/silhouette.py check` / `verify-archive` | `Verified 40 silhouette mattes…` / `Verified 480 families of silhouette images.` |
| `python scripts/controls.py check` | `Verified 6 controls x 20 sources: hand and movement untouched, areas matched, permutations valid.` |
| Hand-built synthetic check of `model/criteria.py` (scratch script) | See findings F1, F8 and answer 27 |
| Read-only summaries of `output/model/{campaign,gate_fit,gate2}` on `jessica` | See F2, F3, F7, F15 |

**Missing or limited:** campaign, fit and Gate 2 outputs exist only on CREMI (read there). `validate_dataset.py` (full archive) and the Gate 1 pipeline were not re-run. No external timestamp of any commit exists in the repository, so every "before any result" statement rests on author-controlled git dates; author and committer dates agree for 47 of 48 commits [V].

---

## 3. Findings, ranked by severity

| ID | Sev. | Area | Finding | Evidence | Label | Recommended action | Allowed before evaluation? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | Critical | A, F | For the three paper displays H2's attachment estimate is the condition itself: a = 1.0 (FB), 1.0 (UL), 0.0 (DH) in all 20 sources. With c = 0 the detached hand receives no visuomotor input, so Δ(DH) ≡ 0 and C_FDI = [Δ(FB)+Δ(UL)]/2. A synthetic model with full-body facilitation only (UL = 0, DH ≡ 0) meets P1, P3 and S1 and fails only P2 and S2. Spec §5.2 still calls P1–P3 and S3 "the decisive tests", contradicting model-method §7 | `data/context_continuity.json` (sets {1.0}, {1.0}, {0.0}); `model/evaluate.py:73-78`; scratch check: `P1 True, P2 False, P3 True, S1 True, S2 False`; `docs/analysis-spec.md:233` vs `docs/model-method.md:1001-1011` | V | Keep the decision rule as declared, but state in the spec and report that for H2 the informative criteria are P2-UL and S2 only, and correct line 233. Do not present P1 for H2 as a test | Only as a declared amendment (wording and reporting; no criterion value changes) |
| F2 | Critical | A, D, F | The ungated network (c = 1) already shows FB > DH on training sources: ratio Δ(DH)/Δ(FB) median 0.49 [5–95%: 0.35–0.71] for OpenCLIP spiking, 0.37 twin, 0.04 DINOv2 summary, 0.00 DINOv2 patch mean; no detached-hand cell is over-driven (share 0.000). In model units Δ(FB) is 1.80 and Δ(DH) 0.91 for OpenCLIP, so the ungated C_FDI is positive unless the upper-limb effect is near zero or negative (the upper limb has not been simulated). The gate is therefore unlikely to be needed for P1, and with DINOv2 it is inert (100% of ungated networks inside the human ratio interval). The human-like ordering arises from weak transfer to out-of-distribution silhouettes, not from attachment. Model-method §7 predicted the opposite ("expected to be positive… at least as much as the full body"), and the documents report the c = 1 ratio only for one configuration | `output/model/gate_fit/*/*.json` on CREMI (curve rows at c = 1); `docs/model-method.md:983-986`, `:831` | V (outputs), I (interpretation) | Report these training-source values now. Make the ungated network a co-headline comparison and declare the paired difference C_FDI(H2) − C_FDI(ungated) as the necessity test. State that "P1 with both robustness encoders" does not test the gate. Retire "clean target" | Yes (reporting); the necessity test only as a declared amendment |
| F3 | Major | G | The held-out evaluation is largely previewed. Each network trains on 18 sources; its "inner" source is never trained on, every source is the inner source of one fold, and that fold shares 17 of 18 training sources with the fold where the source is held out. On inner sources the project has already computed: FB facilitation (H2 OpenCLIP positive in 99% of networks), the DH/FB ratio curves, and H2's control predictions (stub − gap stub +1.42, arm − hand window +0.30, positive in 20/20 sources). With c = 0 the visible-area model gives both pair members a gate of about 0.03–0.06, so S3(a)'s outcome is effectively known. "S3(a) is not previewed" is true of the code, not of the information | `model/train.py:512`; `model/campaign.py:37-38`; `model/gate2.py:65-84`; `output/model/gate2/summary.json` on CREMI; `docs/analysis-spec.md:345-346` | V | Say plainly which quantities are already known on unseen sources and that only the upper-limb display (P2-UL, S2) is new. Do not describe the December run as a blind test of the controls or of S3(a) | Report as limitation only |
| F4 | Major | E, G | Training settings and health criteria were iterated five times on 24–25 Oct (`v3`–`v7`, a stopped campaign) against full-body facilitation on unseen inner sources of folds 1–4: rest 5 → 2 → 1 Hz, movement limit none → 5 → 4 → 3 Hz, batch mean → per trial, heterogeneous → pooled motor neurons, background criterion 6 Hz → 5 Hz → per-network peak. The restart rule was "all four networks detect readout facilitation on their unseen inner source". This is calibration against a quantity equivalent to P2-FB | `docs/model-method.md:341-362, 429-493`; `docs/analysis-spec.md:370-372`; file dates of `output/model/v3…v7` (24 Oct 17:15 → 25 Oct 00:55) | V (sequence), D (measurements) | Describe this as model calibration on development folds. Treat P2-FB as calibrated, not tested | Report as limitation only |
| F5 | Major | G | The frozen S3(a) required that the same network *fed with low-level statistics* not meet P1. On 24 Oct it was replaced by a visible-area *gate* compared on the control pairs, and the "must not meet P1" requirement was dropped, with the stated reason that it "would reject H2 by construction". With c = 0 the area gate (FB 1.0, UL ≈ 0.62, DH ≈ 0.02) is expected to reproduce the paper pattern as well as H2. The decision was made by the assistant under delegation, after Gate 1 and after H2's a values were computed | `git show b4e3fd7:docs/analysis-spec.md` line 182 vs current `:206`; `docs/analysis-spec.md:367`; `docs/model-method.md:989-995` | V | Report the frozen S3(a) outcome beside the amended one (the area model's P1, S1, S2 verdicts, already "reported" by the spec). State that on the paper's displays a visible-area account is not distinguishable from H2 | Yes (reporting already allowed by spec §5.2) |
| F6 | Major | A | H2 is not a model "that receives only images". Its gate needs the source's SAM-HQ forearm mask and a pixel-equality comparison with the full-body reference image. The image-computed alternative (H2-learned) does not recover attachment: m ≈ 0 for the stub and arm window, and 0 for the full body of sources 2, 3 and 11 | `model/context.py:59, 66`; `docs/model-method.md:1086, 1095-1104, 763-765`; `docs/analysis-spec.md:15`; PPTX slide "The question" | V | Reword the question for H2: "a visuomotor network with an externally supplied continuity signal". Keep H2-learned's failure in the main text | Yes (wording) |
| F7 | Major | F | c is not identified. The target −0.10 has a human interval of [−0.67, 0.35] and is −0.008 on raw mV; a gate cannot reach negative values; for OpenCLIP every c from 0 to about 0.45 (median) keeps the training-source ratio inside the human interval; for DINOv2 the ratio is non-monotone in c (peaks near 0.15 at c ≈ 0.3–0.4, about 0 at both ends), so the fit is arbitrary. "One human-derived parameter" misdescribes the model: H2 as evaluated has no fitted parameter and a binary gate | `output/human_target/ratios.csv` (re-derived); `gate_fit` curves on CREMI; `model/fit_gate.py:36-53`; `docs/model-method.md:824-834` | V | Keep c as declared. Report the boundary as an identification failure. Declare one sensitivity analysis from training sources only: the largest c whose training-source ratio stays inside the human interval | Only as a declared amendment (sensitivity analysis; primary unchanged) |
| F8 | Major | E, G | The over-driven rule is not a Horowitz–Manski bound. "Remove" is a complete-case analysis and "maximum" imputes a single high value; the lower extreme is never imputed and the measured value is not a pass condition. In a hand-built instance whose over-driven full-body movement cell reads −1.82 as measured, "remove" drops Δ(FB) and "maximum" turns it into +2.12. Both treatments are neutral or favourable to P2 relative to the measurement. The readout itself falls with excitation, unlike a human MEP | Scratch check of `model/criteria.py:50-76`; `docs/model-method.md:681-692`; `docs/decision-audit.md:71, 91` | V | Call the two treatments a sensitivity analysis. Declare that every criterion is also reported on the readouts as simulated, and report the over-driven share per display | Only as a declared amendment (adds a reported analysis) |
| F9 | Major | G | "Frozen" overstates. The spec was frozen 1 h 44 min after its first draft, then amended in 12 logged rows. `config/analysis.json` lists 5 amendments, all `before_any_result: true`; the current spec hash (`e38b0f58…`) is not recorded and no code checks it. Definitions and the values they produce share one commit for H2's a (`3cfe210`) and for the learned gates (`8c57621`), so the claimed order cannot be verified. No external timestamp exists | `git log`; spec hashes per commit vs `config/analysis.json` `frozen_spec`; `git show 3cfe210`, `8c57621` | V | Describe the work as "pre-specified before the held-out evaluation, developed on training folds". At the 11 December freeze create an external timestamp (OSF registration or a signed tag pushed to a public remote) | Yes |
| F10 | Major | H | No evaluation driver exists. `criteria.py` has scoring functions but nothing assembles instances, models, encoders and verdicts; `evaluate.simulate_instance` has no caller; the held-out guard is a boolean argument, and `simulate_grid` has none | `grep open_evaluation` (only `model/evaluate.py:20, 330, 334`); `model/evaluate.py:190` | V | Write the driver, test it end to end on synthetic data and on inner sources, and record its hash before 11 December | Yes |
| F11 | Major | C | Stimulus validity rests on one unblinded reviewer; the rater protocol was not run. Movement frames are edited images, so a network can in principle detect editing traces rather than abduction. Because every display reuses the same hand pixels this cannot bias a between-display contrast directly, but it weakens any claim about "movement" | `docs/validation.md:229-233`; `docs/approval-receipt.json` (`visual_quality_certified: false`) | D | Report as a limitation; if time allows, run the prepared two-rater check (it uses no model result) | Yes |
| F12 | Minor | B, G | Folds and initialisations are not independent in the way the bootstrap assumes. The 20 networks of a seed share initial weights, trial draws and noise (seeded by the seed only) and 17 of 18 training sources; "initialisation" is crossed with source, but resampled as nested. Intervals are conditional on one generator and scene | `model/network.py:106`; `model/train.py:220-221`; `model/campaign.py:118`; `model/criteria.py:84-94` | V | Report as a limitation (Bengio & Grandvalet, 2004, JMLR, on the variance of cross-validation) | Report as limitation only |
| F13 | Minor | G | Gate 2 "distinct predictions" passes by construction: H0 sees the same hand-only input in every display and H2 zeroes the gap stub and hand window. Holm-adjusted p is at its floor (0.0008) for most of the 8 quantities | `model/gate2.py:105-118`; `gate2/summary.json` on CREMI | V | State that this check confirms the code paths differ, not that the hypotheses are empirically separable | Yes (wording) |
| F14 | Minor | H | "Same networks, ~9× faster" is not true of individual networks. The stack-trained and singly trained fold-4 seed-0 networks differ in weights (max |Δ| 0.067 in `w_in`, mean |w| 0.014) while their health measures agree (threshold identical, facilitation 6.46 vs 6.48). The equivalence test covers 5 updates on small CPU networks; the text says 20 | CREMI `v7/` vs `campaign/h2_openclip_snn/…fold4_seed0`; `tests/test_network.py` (stack test); `docs/model-method.md:507` | V | Say "equivalent training procedure; networks differ by floating-point divergence" | Yes |
| F15 | Minor | H | Provenance gaps: health reports carry no code hash or commit; the CREMI copy is at `920092d` with uncommitted changes (file contents equal HEAD; verified by hash); the feature manifest is checked only against itself, not against a tracked hash | Report keys on CREMI; `git status` there; `model/bank.py:36-39` | V | Record code hashes in every output from now on; pin the feature hashes in `config/analysis.json` | Yes |
| F16 | Minor | I | Documentation contradictions (list in answer 37) | — | V | Fix | Yes |
| F17 | Minor | J | Several "L" entries of the decision audit cite a general principle for a specific choice (answer 39) | `docs/decision-audit.md:45, 55, 57, 78, 91` | I | Reclass as "L (principle) / C (choice)" | Yes |
| F18 | Note | B | The human target is sound; the "robust" rule is ad hoc but conservative; the endpoint was chosen after the human reanalysis (same commit) but it is the paper's headline effect | `git show 4faa1ff`; `docs/analysis-spec.md:201` | V | Disclose in one sentence | Yes |

---

## 4. Amendment timeline (question 30)

"Results existing" is reconstructed from commit times, output file dates on the Mac, and the documents. Every "before any model result" in the spec means "before any held-out contrast", which is true throughout.

| Date (commit) | Change | Results existing at that moment | Could it favour H2? | Judgement |
| --- | --- | --- | --- | --- |
| 22 Oct 17:09 (`4faa1ff`) | Draft spec; C_FDI primary | Human reanalysis (same commit) | No (target side) | Defensible; disclose that the endpoint followed the human reanalysis |
| 22 Oct 18:48–18:53 (`41890c4`, `b4e3fd7`) | P2, P3 added; S3 redefined; freeze | Human data only | No; tightens | Sound |
| 22 Oct 19:54 (`85c5757`) | Control definitions and parameters | Low-level statistics of the controls | No | Sound |
| 23 Oct 21:14–21:37 (`9d65ecc`, `e3f087c`) | Gate 1 method; smoke-test changes; transfer gain added | Smoke test on 20 families | No | Sound; `features.npz` dated 22:11, after both commits [V] |
| 24 Oct 01:45 (`be4f5b2`) | Hand-view scales {2, 4} → {1.5, 2, 3} | Real low-level and pixel signals of 20 families at s = 2 and 4 had been seen (logged) | No | Defensible, exploratory; GPU extraction ran 01:56–02:47, after the commit [V] |
| 24 Oct 03:00 (`a1c59b8`) | OpenCLIP primary; P1 required on all three encoders | Gate 1 and Addendum A | Intended not to | Defensible but must be disclosed; see answer 16 |
| 24 Oct 03:54 (`3cfe210`) | H2's a defined; one-parameter gate; S3(a) replaced; rectangle prediction reversed; §1 boundary; fixation as rest | Gate 1; first health check; a values on all displays (same commit) | Yes: removes a criterion H2 was expected to fail; turns the rectangle from a challenge into a prediction | **Garden-of-forking-paths risk**; delegated to the assistant; disclose prominently (F5) |
| 24 Oct 16:35 (`1005ce6`) | Readout window 5–60 → 0–55 ms | Second health check: non-monotone dose-response | Neutral | Justified fault fix |
| 24 Oct 20:05 (`d0fcb0f`) | Observation without execution; pooled motor neurons; trimming; ungated model; AUC criterion; readout range rule | `v3`–`v5` on fold 1 inner source; static activity of all displays on sources 3–20 | Yes, indirectly: makes FB facilitation measurable | Defensible fault fix in origin, calibration in effect (F4); delegated |
| 24–25 Oct (`e2e84a3`) | Limits per trial; 4 Hz then 1/3 Hz; background criterion → per-network peak | Stopped campaign (folds 1–3), `v6`, `v7` on unseen inner sources | Yes, as above | **Garden-of-forking-paths risk** (three criteria changes after failures); disclose as calibration |
| 25 Oct 02:08 (`8c57621`) | H2-learned and H4 added; twin repaired; spinal readout not adopted | Gate tables of both pathways in the same commit; spinal readout tried on 4 networks | No (comparison models) | Sound for the twin and spinal stage (reported honestly); ordering of definition and values not verifiable |
| 26 Oct 19:58 (`bc3cafd`) | No exclusion for source-dependent health failures; remove/maximum rule | Health of all 2,400 networks | Mixed: keeps failing networks (against H2), but both treatments spare P2 (F8) | Defensible but must be disclosed as chosen after the health results |
| 26 Oct 21:05 (`3c280cd`) | Criteria code; fit under treatments; Gate 2 checks; cross-fitting | In-sample learned-gate values | No | Sound; cross-fitting is a genuine fix |
| 8 Nov 21:02–21:09 (`57f59d6`, `21de913`) | Recurrent variant and selection rule; robustness variations | Fit of c and Gate 2 checks were running (README: 599/600 fits by 23:40) | No | Sound rule; "before any fit had a result" is not verifiable |

By Nosek et al. (2018, PNAS), an amendment made before the confirmatory data are seen is legitimate if logged. By Gelman & Loken (2013, "The garden of forking paths"), the problem is not intent but that choices were contingent on data. Here the confirmatory data are statistically close to the development data (F3), so the 24–25 Oct rows carry more weight than their "before any result" labels suggest.

---

## 5. Answers to questions 1–41

**A. Question and inference**

1. Well posed, weakly falsifiable. Failure is: P1, P2, P3, S1, S2 or S3 not met with OpenCLIP, or P1 not met with a DINOv2 encoder. For H2 only two failures can occur in practice: Δ(UL, FDI) not above 0, or Δ(UL)/Δ(FB) outside [0.34, 1.78] (`docs/analysis-spec.md:201-214`) [V, F1].
2. Agree with model-method §7, and it understates the case. P1, P3, S1, S3(b) are built in; P2-FB is calibrated (F4); S3(a) is known (F3); encoder agreement on P1 does not test the gate (F2). What remains is P2-UL and S2. A positive result can show that a full-body-trained readout transfers to the upper-limb image within a factor of 0.34–1.78. It cannot show that continuity, rather than image similarity to the training display, produces the pattern [I from F1–F3].
3. An oracle variable (F6). H2-learned does not rescue the claim: it fails on the connected controls [D, `docs/model-method.md:1095`].
4. Spec §1 is careful. Overclaims: "Can a model that only sees images…" (spec §1, PPTX, short PDF) for H2; "visual change alone cannot explain the human effect" (PPTX Gate 1 outcome) and "H2 has a clean target" (`docs/next-phase-plan.md:10`, baselines R1), which F2 contradicts; "frozen… like preregistration" (PPTX). `project-guide.pdf` makes no physiological claim [V, text only].
5. Not sound as a premise. Gate 1 measures an unsigned feature distance, and the silhouette-on-black construction inflates it (changed pixels are 29% of the visible image; `docs/baselines.md:249-255`). The trained readout is direction-specific and responds *less* to the silhouette (F2). Gate 1 correctly shows that raw feature change does not give the pattern; it does not show that a trained network must suppress a stronger detached-hand signal [V].

**B. Human target**

6. Matches spec §4: 12 cells per participant with pairing enforced (`analysis/human_target.py:106-112`); participant_z over all retained MEPs with n − 1 (`:80-91`); contrasts as declared (`:125-145`); participant bootstrap paired across cells, seed from config (`:167-170`); Holm within secondary (9 tests) and exploratory families (`:190-194`). "Robust" is only computed in code for C_FDI (`:309`); the table column is hand-derived. The rule is ad hoc but conservative [V].
7. Yes: 8/8 Table 2 values and 10/10 post-hoc z values within 0.01, re-run by me. The discrepancies in §4.2 are fairly stated; items 1–3 follow from the refit itself [V]. Table 3 values were not checked against the paper [not verifiable here].
8. Chosen after the reanalysis: the first draft and the human-target code are one commit, and the spec's own reason is "the one human effect that survives every scale". It is also the paper's headline effect, and it concerns the target, not the model [V].
9. Percentile bootstrap of a ratio of means. Denominators are far from 0 (Δ(FB): t = 4.4), so the intervals are stable, though a Fieller interval (Fieller, 1954, JRSS B) would be the standard check. S1 cannot discriminate (any ratio from −0.67 to 0.38 passes, and H2 gives 0 by construction); S2 excludes only gross failure. The criteria compare a model point estimate with the interval and ignore model uncertainty (`model/criteria.py:127-128, 144-145`). "Compatibility, not equivalence" is the right label [V].

**C. Stimuli and controls**

10. Twenty identities from one generator, template and camera support a claim about this stimulus set only, as spec §3 says. Source is the unit in the splits (`model/train.py:512`), the bootstrap (`model/criteria.py:84`), Gate 1 (`analysis/baselines.py`, `source_means`) [V]; Addendum A [D].
11. See F11. The 144-family hand-pixel deviation is identical in both frames and the reference-17 flaw has a declared sensitivity analysis [D, `docs/validation.md:108-126`]. None of these can differ between displays, because displays are masks over the same pixels (`controls.py check`, `silhouette.py verify-archive` pass) [V].
12. Area matching holds (check passes). The pairs differ in more than connection: the gap segment shows a more proximal forearm part and extends further; the windows differ in shape (acknowledged). H2's predictions were declared at freeze except the rectangle, which was reversed after a was measured (0.80); that correction is logged but changes the rectangle from a sensitivity challenge into a predicted null. `docs/controls.md` still shows the old prediction [V].
13. Deterministic and hash-bound [V by check; method D]. One mask per source for all 24 variants [D]. It does create a low-level difference (the finger moves against black), which the project documents as the source of the Gate 1 reversal.

**D. Gate 1**

14. Signal, spread, source bootstrap and leave-one-source-out decoding match the document (`analysis/baselines.py:79-135, 228-262`); the held-out source never trains and regularisation is chosen by grouped inner folds [V]. The spread is computed over all 20 sources and not resampled (declared). RSA and variance decomposition not checked. Commit order: `9d65ecc` 21:14, `e3f087c` 21:37, features 22:11, results commit 22:37 [V by file date].
15. Verified for GPU features (commit 01:45; extraction manifest 02:47 after a 51-minute run). Low-level and pixel signals at the old scales had been seen, as the change log admits [V].
16. Conservative with respect to the baselines only. For the trained network OpenCLIP is the one encoder where the gate changes anything (ungated ratio 0.49); with DINOv2 the ungated network already gives no detached-hand facilitation. So requiring P1 on all three encoders protects against nothing (F2) [V].

**E. Model**

17. LIF dynamics, surrogate gradient (fast sigmoid, slope 25), time constants, analytic noise and the probe as a raw 1-ms current match the document (`model/network.py:19, 87-94, 199-232`). The readout is the summed spike difference over steps 0–54 against a twin branched from the same state with the same noise (`:260-279`; `model/evaluate.py:217-220`) [V]. The window change followed a failed dose-response check and has a mechanistic reason; a fault fix, to be disclosed.
18. Legitimate as a modelling choice, but it is an engineered constraint on the measured population and it was tuned (F4). The changes responded to facilitation and background results on unseen inner sources.
19. Serious for the MEP analogy: the readout measures net extra spikes in 55 ms, which falls when neurons already fire, whereas the human input–output curve is sigmoid with a plateau (Devanne et al., 1997). The project states this. The bounding rule does not handle it correctly (F8): partial-identification bounds need a defined true outcome with known support (Manski, 2003, *Partial Identification of Probability Distributions*); here the "true" value belongs to a model that was not built.
20. The associative account supports learning from self-observation in general; it does not single out a full first-person scene, and people mostly see their hand and forearm. Training on full body alone is what makes the silhouette out of distribution, and that is the mechanism behind F2. The all-context ablation is listed as exploratory with no procedure (`docs/analysis-spec.md:207, 284-285`); it is the single most informative missing analysis and should be specified [I].
21. Twin: failed, diagnosed, replaced by a tabulated rate function, then checked; no criterion result could have informed it [D]. Spinal readout: reported as tried and not adopted, with the failure described [D, `docs/model-method.md:540-559`]. Honest.
22. The selection rule is sound and prefers the simpler model. Robustness variations with 20 folds × 3 initialisations are adequate for sign and direction, not for small differences; P3 with three initialisations is a weak criterion. Recurrent mode check was run on 1 of 200 networks.

**F. Fit of c and H2**

23. (a) A design constant in effect; the fit was predicted to hit the boundary before it ran. (b) No (F7). (c) Yes. (d) Yes: with DINOv2 patch mean the gate does nothing and c is arbitrary; H2 and the ungated network are the same model there. (e) The owner's plan is right. Legitimate before 11 December: report the boundary, report the ungated values, declare the sensitivity analysis of F7 and the necessity test of F2. Not legitimate: a new target, a new gate form, or a different c.
24. Linear form and the one-palm cap are project choices with no outside justification; for the paper's displays neither matters, because a is 0 or 1. The 0.5 visibility threshold affects only the rectangle (0.50–1.00) and scrambled (0.15) displays. Identical across variants: enforced in the build (`model/context.py`, `build`) [V in code, not re-run].
25. The in-sample scoring was a comparability fault, not a leak of held-out data; the fix leaves out both the fold's source and the scored source (`model/learned_context.py`, `_crossfit_pair`) and is complete for the fit of c [V]. With m = 0 for the full body of three sources, H2-learned is a weak comparison and should be presented as a negative result about learning attachment from pixels.
26. Verified: `gate2.py` builds only continuity-gated H2 models (`:67-70`), and no committed code simulates the area gate on a control or any display of a held-out source. The information is nonetheless available (F3) [V].

**G. Evaluation and integrity**

27. Checked by hand-built cases: trimming is applied after the treatment and before normalisation; normalisation is over both populations; `criteria.treated_cells(None)` equals `evaluate.instance_cells`; the two bootstrap implementations give identical draws; the hierarchical standard error equals the standard error of source means (0.068 vs 0.065; naive 0.032); ratios are ratios of means; a non-positive denominator gives "not met"; verdict logic as declared [V]. Reservation: F8.
28. No leak into fitted quantities: standardisation, H1's mean salience and training use 18 sources (`model/train.py:512-513`); the fit of c uses the 19 non-held-out sources (`model/fit_gate.py:100`); learned pathways exclude the fold; thresholds and readout ranges use the rest display. `simulate_instance` refuses the held-out source (`model/evaluate.py:334`, tested), but it is a flag (F10). Design-level exposure: Gate 1, the encoder choice and every calibration decision used all 20 sources. The documents say "19 training sources" where networks train on 18 [V].
29. Mathematically sound for the feedforward network: 250 ms is 12.5 membrane time constants (e^−12.5 ≈ 4 × 10⁻⁶), and the noise-free test shows identical trials. For the recurrent network it is an empirical finding on one network. The 60-s familiarisation and block order have no effect by construction, so that part of the paper's procedure is nominal in this model; spec §2 still says "model state carried" [V/I].
30. Section 4.
31. No; see F9. An external reviewer would accept the log as a transparent development record, not as a preregistration.
32. Yes, nearly always passes (F13). The three non-smooth networks are reported honestly; by the declared wording ("every network") those two H1 configurations fail, which the text says.

**H. Code**

33. 154 tests, all pass (one GPU-only test skipped on the Mac, passed on `jessica`). Many tests check scientific properties: contrast arithmetic, pairing, held-out exclusion in decoding, common random numbers, probe twin, treatments, verdicts. Untested: the complete decision pipeline (no driver); that training never touches the held-out or inner source on real folds; stack equivalence beyond 5 updates; the over-driven rule on real reports; recurrent mode equivalence with noise.
34. No bug found that changes a result. Notes: per-trial `noise_seed` is generated and never used (`model/trials.py:111`); the session noise seed ignores the session kind, so fit, Gate 2 and evaluation sessions of one source and seed start from the same stream (`model/evaluate.py:153-156`); stack claims are keyed by fold range, so two machines computing different pending lists could train a network twice (harmless, since outputs are written atomically, but results then depend on stack composition, F14); a claim held by a dead process on another machine is never retaken; `Network.__init__` reseeds the global generator on every load (`model/network.py:106`). Superseded runs live in separate folders and are not globbed by the fit or Gate 2 code [V].
35. Human data hashes are enforced at load (`analysis/human_target.py:42-46`); feature files against their own manifest; the continuity table against its code hash; the spec hash against nothing (F9). Package versions are pinned in all four requirement files. A third party can reproduce the human target from the repository and the OSF files (done here); Gate 1 needs the local archive and a GPU.
36. Evidence: a 5-update CPU test, and one fold-4 network trained both ways with equal health and different weights (F14).

**I. Documentation**

37. Contradictions and stale statements [V]:
    - README "all 43 decisions… 30 references" vs 48 decisions and 29 listed references in the audit.
    - README links to `model-method.md` without `docs/` (lines 269, 297, 301); "Done on 9 November" precedes "Done on 8 November".
    - Spec line 233 vs model-method §7 (F1); spec §2 "model state carried" vs trialwise evaluation; "19 training sources" vs 18; the §10 table is broken by a blank line at 369.
    - `docs/controls.md` rectangle prediction "like the detached hand" vs the corrected spec.
    - Model-method header still "24 October 2026: draft"; the robustness results sit inside §7 (line 1020); "after 20 updates" vs a 5-update test.
    - Next-phase plan: "28 CPU tests", "one logged amendment".
    - `config/analysis.json`: 5 of 12 amendments; `comparison_models` omits H2-learned and H4.
    - PPTX and short PDF were committed on 9 November but describe the state before the fit of c.
38. Mostly listed. Undeclared in §2: networks train on 18 sources; evaluation trials start from rest 300 ms before the second frame instead of 2,250–3,250 ms; no familiarisation is simulated; trimming is applied after the over-driven treatment.
39. Sample: Saravanan et al. (2020), Kriegeskorte et al. (2009), Nosek et al. (2018), Holm (1979), Mehrer et al. (2020), Varoquaux (2018) support their entries. Overstated: Yamins & DiCarlo (2016) concerns task-trained networks and ventral-stream object responses, not CLIP or DINOv2 as the front end of action observation; Heyes (2001), Cook et al. (2014) and Catmur et al. (2007) support learned visuomotor associations, not full-body-only training; Tsakiris & Haggard (2005) does not address wrist continuity; Fourcaud & Brunel (2002) is an analytic theory while the twin uses a simulated table; Horowitz & Manski (2000) (F8). The numerical claim attributed to Devanne et al. (gain 4–7×) is not verifiable here.

**J. Literature and positioning**

40. Missing, all of which I am confident exist: computational mirror-system models (Fagg & Arbib, 1998, *Neural Networks*; Oztop & Arbib, 2002, *Biological Cybernetics*; Oztop, Kawato & Arbib, 2006, *Neural Networks*); alternative accounts (Keysers & Perrett, 2004, *TiCS*; Kilner, Friston & Frith, 2007, *Cognitive Processing*); what MEPs measure (Bestmann & Krakauer, 2015, *Experimental Brain Research*); TMS circuit models (Esser, Hill & Tononi, 2005, *J Neurophysiol*); body-ownership models (Samad, Chung & Shams, 2015, *PLoS ONE*; Kilteni et al., 2015, *Frontiers in Human Neuroscience*); limits of deep networks as vision models and shortcut behaviour (Bowers et al., 2023, *BBS*; Geirhos et al., 2020, *Nature Machine Intelligence*). The last two bear directly on F2.
41. Not in its current framing. Reviewer 2 would attack, in order: (1) the result is built in (F1); (2) the detached-hand null is a distribution-shift artefact that appears without the gate (F2); (3) the gate is an oracle (F6); (4) the readout is not an MEP analogue and was calibrated until it showed the base effect (F4, F8); (5) synthetic stimuli with one unblinded reviewer (F11); (6) "preregistration" with twelve amendments (F9). A methods-oriented paper that states F1–F3 itself, leads with the ungated and all-context comparisons, and presents the pipeline and the human reanalysis as contributions would be reviewable.

---

## 6. Strengths (keep)

- **Human-target reanalysis.** Exact reproduction of 18 published statistics, participant-level inference, five scales, hash-checked inputs; reproducible from the repository in one command [V].
- **Source-level statistics.** Hierarchical bootstrap correctly implemented and equivalent across two modules; leave-one-source-out everywhere; no preprocessing leak found [V].
- **Common random numbers and the matched no-probe twin.** Zero-amplitude probe gives exactly zero; a gate of 1 reproduces the ungated run exactly (tests) [V].
- **Disclosure.** The project itself wrote down that P1 and S1 are close to built in, that the readout is non-physiological, that H2-learned fails, that the spinal stage failed, and that the fit of c is at a boundary. Most findings above extend statements the project already made.
- **Negative results kept.** Superseded campaigns are archived, not deleted; 384 unhealthy networks stay in the analysis.
- **Integrity tooling.** Hash-bound stimuli and controls, deterministic segmentation, 154 tests, pinned requirements, atomic writes [V].

---

## 7. Pre-evaluation action list

**Before the 11 December freeze (in priority order)**

1. Record, in model-method and the README, the ungated (c = 1) training-source ratios for every configuration, and withdraw the "clean target" statement (F2).
2. Declare by amendment, with reasons, and without changing any value: (a) the necessity test C_FDI(H2) − C_FDI(ungated), paired by instance; (b) that for H2 the informative criteria are P2-UL and S2; (c) criteria also reported on readouts as simulated (F8); (d) the c-sensitivity analysis (F7). Each can only make the claim harder to support.
3. Specify the all-context training ablation (procedure, networks, what is compared) and, if compute allows, train it before the freeze. It is the direct test of whether the pattern is a distribution-shift artefact.
4. Write and test the evaluation driver; dry-run it on inner sources; hash it (F10).
5. Replace "frozen" and "preregistered" wording; create an external timestamp of the frozen state; complete the amendment list and spec hash in `config/analysis.json` (F9).
6. Fix the document contradictions of answer 37.
7. Implement the 0.5-ms time step as already declared.

**To be written into the report as limitations**

- P1, P3, S1, S3(b) are built in for H2; P2-FB was calibrated; S3(a) and the control predictions were known on unseen sources before the evaluation (F1, F3, F4).
- H2's gate is supplied, not computed from the image; a visible-area gate is not distinguishable on the paper's displays (F5, F6).
- c is not identified (F7); the readout is not physiological and the over-driven rule is a sensitivity analysis (F8).
- Twenty synthetic identities, one generator, one unblinded reviewer (F11); intervals conditional on the training sets (F12).

**For a future study only**

- A context signal computed from the image; human data for the matched controls; a readout with a plateau (volley-based or spinal); real photographs with independent raters; a lockbox of sources never used in development; training on mixed displays as the primary design.

---

## 8. Questions for the author

1. What did the delegation of 24 October cover, and did you review the S3 amendment and the rectangle reversal before they were committed?
2. Were commits pushed to GitHub on the day they were made? Push times would be the only external timestamps available.
3. Was any upper-limb or held-out-source simulation ever run outside the committed code (notebooks, interactive sessions on CREMI)?
4. For `3cfe210` and `8c57621`: were the definitions written before the values were computed, and is there any record (shell history, chat log) of that order?
5. Who saw the ungated ratios on 9 November, and why is only the DINOv2 patch-mean value in the text?
6. Is the claim you want to defend "attachment explains the human pattern" or "this pipeline can test such accounts"? The evidence supports the second.
7. Were the 20 reference pairs screened in any way that depended on how the movement looked, and by whom?
8. What is the supervisor's requirement for the January report: a confirmatory result, or a methods and feasibility report?
