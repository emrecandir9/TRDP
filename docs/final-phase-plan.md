# Final phase plan: 11 November 2026 to the end of the project

Written 11 November 2026. It replaces the schedule rows W5–W8 of the [next-phase plan](next-phase-plan.md),
which were completed early or overtaken by the [independent review](independent-review.md).

**Fixed dates** (spec §8): Gate 2 on 4 December, freeze on 11 December, single held-out evaluation 14–18
December. **Assumed date:** final reports handed in on Friday 29 January 2027. The plan's last block moves
with that date; confirm it with the supervisor (decision D1).

## 1. Where the project stands today

| Item | State |
| --- | --- |
| Networks trained, checked and fitted | 3,220: campaign 2,400, recurrent 200, robustness 540, all-context 60, 0.5-ms 20 |
| Evaluation program | `model/run_evaluation.py`: simulate, score, decision rule; locked until `config/freeze.json` exists |
| Open tests for H2 | Upper-limb half of P2, and S2. Everything else is built in, calibrated or already known on unseen sources |
| Claim for the report | The pipeline tests attachment accounts of the human pattern; it does not assert one (review §8, answer 6) |
| Review findings | All handled except the freeze record |
| Tests | 176 pass |

## 2. Rules that hold until the end

1. No model setting, criterion, target or decision rule changes. A code or measurement fault is fixed, logged
   in the model-method change log, and re-checked on training sources only.
2. No upper-limb readout and no held-out source before 14 December. Rehearsals use inner sources and the
   displays already simulated there (full body, silhouette, controls).
3. The spec changes only through §10, and `config/analysis.json` records each change with its hash
   (`tests/test_spec.py` fails otherwise).
4. Every step ends with the status documents updated in the same commit, and the commit pushed the same
   hour, so GitHub's push record dates it.
5. After the evaluation is opened, every new analysis is labelled exploratory in the text and in its output
   file name.

## 3. Schedule

### Block A — Wednesday 11 to Friday 20 November: prepare everything that does not need results

| # | Task | Output | Done when |
| --- | --- | --- | --- |
| A1 | Report figures and tables as code: `analysis/report_figures.py` reading `output/model/evaluation/result.json` and the human target | Functions for: human vs model cell plot (12 cells); C_FDI and movement effects per model with intervals; the two training diets side by side; criterion table (met / not met / undetermined, with the built-in / calibrated / previewed / open label); control-display effects; robustness table | Runs end to end on a synthetic `result.json` made by the test fixtures; tests added |
| A2 | Criterion-status table as data: `docs/criteria-status.json` listing for every criterion and model whether it is built in, calibrated, previewed or open, with the document section that shows it | One file used by A1 and by both reports | Agrees with spec §5.2 (10 November amendment) and model-method §6.19a |
| A3 | Full rehearsal of `score` on real records: run `simulate_network` on every network's inner source with the full-body and silhouette pair and the controls (no upper limb), save under `output/model/rehearsal/`, and run the scoring functions that need only those displays (necessity on Δ(DH), control pairs, over-driven shares) | Timing per network, memory, file sizes; a list of any network that fails to run | All 3,220 networks produce a record; expected about 13 GPU hours in total, so one night on two lab PCs with `--claim` |
| A4 | Project report skeleton: `report/project-report.md` with every section heading, the methods sections that do not depend on results (stimuli, human target, Gate 1, model, training, health, criteria, development history with the amendment table and push times) | Draft of about 60% of the final text | Supervisor can read it without the repository |
| A5 | Paper-style skeleton: `report/paper.md` following the outline in review §8; abstract and results left as stubs with the exact sentences each possible outcome would produce | Draft methods and limitations | Each results stub names the `result.json` field it will quote |
| A6 | Related work: add the missing literature named in review answer 40 to both skeletons, each reference checked against the publisher's record | Reference list | Every entry verified, as the decision audit did |
| A7 | Decision D2 (rater check): if two naive raters are available, run the prepared protocol on the fixed sample this week | `output/raters/` results and one paragraph in validation, or a recorded decision not to run | Decided by Monday 16 November |

### Block B — Monday 23 November to Friday 4 December: complete the pre-evaluation record, Gate 2

| # | Task | Output | Done when |
| --- | --- | --- | --- |
| B1 | Ablation table of the plan's W7, defined before the evaluation: spiking vs twin, feedforward vs recurrent, full-body vs all-context training, gated vs ungated, with and without reference 17; "final frame vs sequence" stated as null by construction for a network without memory | A table in model-method naming, for each row, the two configurations and the `result.json` fields compared | Committed; no new training needed |
| B2 | Completeness audit on CREMI: every one of the 3,220 networks has a health report, FDI and ADM readout ranges inside the sweep, a fit of c under the three analyses, and (for the 1,800 H0/H1/H2 spiking networks) Gate 2 predictions | A script `scripts/audit_outputs.py` and its printed counts in validation | All counts equal the declared numbers |
| B3 | Fix what A3 found, on training sources only; re-run the affected rehearsal records | Change-log rows | No network fails to run |
| B4 | Send both skeletons, the review and this plan to the supervisor | Feedback on framing and on the assumed hand-in date (D1) | Sent by Friday 27 November |
| B5 | **Gate 2, Friday 4 December.** Spec amendment logging what §5.1 and §5.2 require before the evaluation: primary variant (feedforward spiking), 10 initialisations per configuration, the list of configurations scored, and that Gate 2's conditions were met on 9 November | §10 row; `config/analysis.json` updated; Gate 2 entry in validation | Committed and pushed on 4 December |

### Block C — Monday 7 to Friday 11 December: freeze

| # | Task | Done when |
| --- | --- | --- |
| C1 | Last pass over the code the evaluation uses (`model/*.py`, `analysis/human_target.py`): no change after Wednesday 9 December unless a test fails | Wednesday 9 December |
| C2 | Full test suite on the Mac and on a lab-PC GPU (the GPU-only test included); `dataset.py check`, `silhouette.py check`, `verify-archive`, `controls.py check`; `human_target.py` output byte-identical to the stored one | Thursday 10 December |
| C3 | CREMI working copy at the same commit as GitHub, `git status` clean | Thursday 10 December |
| C4 | Final spec amendment row "Frozen for evaluation", with the number of amendments and the statement of what is built in | Friday 11 December, morning |
| C5 | `python model/run_evaluation.py freeze` on CREMI; copy `config/freeze.json` to the Mac; commit, push, and tag `freeze-2026-12-11`; refresh `docs/push-times.json` so the push time of the freeze commit is on record | Friday 11 December |
| C6 | Optional external snapshot: a Zenodo or OSF deposit of the tagged repository (decision D3) | Friday 11 December |
| C7 | Check that the lock holds: `run_evaluation.py simulate` starts only with the unchanged freeze record; changing one listed file makes it refuse | Friday 11 December |

### Block D — Monday 14 to Friday 18 December: the evaluation, once

| Day | Task |
| --- | --- |
| Mon 14 | Start `run_evaluation.py simulate --claim` on the lab PCs that are free (two workers per GPU). About 13 GPU hours in total; finished overnight on two or more machines. Sessions kept open from the Mac |
| Tue 15 | Completeness: 3,220 records, none missing, none written by changed code (each record carries its code hashes). A network that failed to run for a technical reason is re-run with the same frozen code; this is logged. No record is ever re-simulated because of its values |
| Tue 15 | `run_evaluation.py score` once. Record the SHA-256 of `result.json` in validation, commit and push the same hour |
| Wed 16 | Write the result into model-method (new section), README, validation and the review's progress table, exactly as the stubs of A5 prescribed. State the decision-rule verdict, the two open tests (P2-UL, S2), the necessity of the gate under both training diets, and every repetition (without reference 17, healthy only, untrimmed, rectangle) |
| Thu 17 | Figures and tables from A1 on the real result. Any analysis not declared before the freeze goes into a separate file `result_exploratory.json` and is labelled exploratory |
| Fri 18 | Interim results note to the supervisor: two pages, the verdicts, the main figure, the limitations |

**What each outcome means, fixed now.**

| Outcome | Reading |
| --- | --- |
| P2-UL and S2 met | The full-body mapping transfers to the upper-limb image at a human-like ratio. Reported as the one confirmed blind prediction, with the caveat that the upper limb shares most of its pixels with the full body |
| P2-UL or S2 not met | The decision rule is not met. Reported as negative; nothing is changed or re-run |
| Gate not necessary with full-body training, necessary with all-context training | The expected outcome from training sources. The headline is the dependence on the training diet |
| Any criterion undetermined | Reported as undetermined, with the over-driven shares; never counted in the hypothesis's favour |

### Block E — Monday 21 December to Friday 29 January 2027: the two reports

| Dates | Task |
| --- | --- |
| 21–23 Dec | Fill the results sections of both reports from `result.json`; no new analysis |
| 24 Dec–3 Jan | Break |
| 4–8 Jan | Project report complete draft: all figures, the development history, the limitations, the reproduction instructions. Internal consistency check of every number against its source file (a script that extracts the numbers quoted and compares them) |
| 11–15 Jan | Paper-style report complete draft (about 8,000 words): claim of review answer 6, results in order of evidential weight, criterion-status table, limitations. Send both to the supervisor on Friday 15 January |
| 18–22 Jan | Revision after feedback. Replace the two slide and PDF summaries in `docs/` with versions that match the final results |
| 25–28 Jan | Final checks: tests pass; documents agree with each other; repository tagged `final-report`; data-availability statement (what is in the repository, what stays local, how to obtain the OSF deposit) |
| Fri 29 Jan | Hand in |

## 4. Decisions needed from the project owner

| # | Decision | Needed by | Default if none |
| --- | --- | --- | --- |
| D1 | Hand-in date and required format of the final report | 27 November | 29 January 2027; both documents as PDF |
| D2 | Run the two-rater check of the stimuli | 16 November | Not run; stays a stated limitation |
| D3 | Deposit the frozen repository on Zenodo or OSF | 9 December | GitHub tag and push record only |
| D4 | Which lab PCs are started for 14 December | 11 December | `jessica` and one more |
| D5 | Whether the authors of the reference paper are told about the reanalysis discrepancies before the report is shared | 15 January | Not contacted; discrepancies reported neutrally |

## 5. Risks

| Risk | Effect | Response |
| --- | --- | --- |
| Lab PCs off or sessions dropped during the evaluation | Simulation stops | Every step skips finished records; relaunch. Keep 14–15 December for it, score on the 15th |
| A fault found in frozen code after 11 December | The lock refuses to run | Fix, log the reason in §10 with the diff, delete and rewrite the freeze record in a commit that states it, and re-run from the start. Not done after any held-out value has been looked at |
| A fault found after scoring | The result may be wrong | Report it; correct in a dated erratum section with both versions shown |
| CREMI storage quota (45 GB soft) | Writes blocked | Evaluation records are about 0.3 MB per network, 1 GB in total; rehearsal records the same. Delete rehearsal records after B3 |
| The result is negative | No reproduction claim | Already covered: the claim is about what the pipeline shows. A negative P2-UL is a result |
| Time lost in January | Paper-style report unfinished | The project report has priority; the paper-style report is cut to its methods, the criterion-status table and the main result |

## 6. What is out of scope

New stimuli, a context signal computed from the image, a physiological readout, human data for the matched
controls, and any change of training settings. Each is named in the reports as future work.
