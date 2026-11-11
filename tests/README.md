# tests/

192 unit and regression tests. They need no dataset or GPU. Tests that use local data or the OSF
files skip when those are absent. The closed-evaluation test skips by design after the freeze.

```sh
python -m unittest discover -s tests
```

| File | Covers |
| --- | --- |
| `test_dataset.py` | Stimulus pipeline regression checks, including failure cases |
| `test_silhouette.py` | Silhouette geometry, registration, seams, wrist cut, frozen mattes |
| `test_controls.py` | Control geometry, equal areas, scramble, frozen controls |
| `test_human_target.py` | Human target; reproduction of the published statistics |
| `test_baselines.py` | Baseline statistics, leave-one-source-out decoding, rendering |
| `test_handview.py` | Hand-centred windows and analysis |
| `test_raters.py` | Rater-validation package |
| `test_trials.py` | Trial simulator and feature bank |
| `test_context.py` | Forearm-continuity estimate |
| `test_learned_context.py` | Learned context pathways |
| `test_network.py` | Network units, probe readout, training, health measures, campaign |
| `test_evaluate.py` | Evaluation simulator |
| `test_criteria.py` | Criteria and verdicts |
| `test_fit_gate.py` | Fit of c |
| `test_gate2.py` | Gate 2 checks |
| `test_run_evaluation.py` | Evaluation driver and its freeze lock |
| `test_rehearse.py` | Rehearsal |
| `test_audit_outputs.py` | Completeness audit |
| `test_report_figures.py` | Report figures and tables |
| `test_criteria_status.py` | Criterion-status table |
| `test_spec.py` | The analysis spec equals its recorded hash |
