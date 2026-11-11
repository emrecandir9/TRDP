# model/

The spiking visuomotor model, its training, the training-source checks (Gate 2) and the held-out
evaluation. The method and every intermediate result are in [docs/model-method.md](../docs/model-method.md).
Every Python file here is hash-locked by `config/freeze.json`.

| File | Contents |
| --- | --- |
| `trials.py` | Trial simulator following the paper's procedure: sessions, timing, block orders |
| `bank.py` | Frozen-encoder inputs, hash-checked; standardisation on training sources only |
| `context.py` | H2 context pathway: visible forearm continuity below the wrist, in palm widths (capped at 1) |
| `learned_context.py` | Learned context pathways: forearm prediction (H2-learned) and own-body familiarity (H4) |
| `network.py` | Spiking visuomotor network, non-spiking rate twin, probe readout, threshold search |
| `train.py` | Auxiliary training for each hypothesis, the health check, and job claims for shared GPUs |
| `stack.py` | Trains many networks at once on one GPU (same procedure, about 9× faster) |
| `campaign.py` | Trains every fold × initialisation of a configuration through the health-checked path |
| `extend_curves.py` | Readout excitability curve of every network over the extended sweep (FDI and ADM) |
| `fit_gate.py` | Fit of the gate parameter c on training sources; sensitivity value c⁺ |
| `gate2.py` | Gate 2 checks on training sources: control predictions (`predictions`), probe-strength sweep (`sweep`), `summary` |
| `recurrence.py` | How much a recurrent network uses its recurrent weights |
| `evaluate.py` | Evaluation simulator: sessions, probes, ± 2 SD trimming, cells, hierarchical bootstrap |
| `criteria.py` | Criteria of spec §5.2, treatment of over-driven readouts, verdicts, necessity of the gate |
| `run_evaluation.py` | The held-out evaluation: `freeze`, `simulate`, `score`; locked until `config/freeze.json` matches |

Order of a full run: `campaign.py` → `extend_curves.py` (then `--apply`) → `fit_gate.py` →
`gate2.py` → `run_evaluation.py freeze` → `simulate` → `score`. Most steps take `--shard/--shards`
and `--claim`, so several GPUs can share the work without repeating a network. Outputs go to
`output/model/` (local).
