# config/

Parameters and frozen records that the code reads. Hashes in these files bind the code to its
inputs, so edit them only through the step that owns them.

| File | Contents | Written by |
| --- | --- | --- |
| `recipe.json` | Generation recipe: frozen-source hashes, desks, seeds, white balances, canary IDs, prompts, render settings, pinned models, silhouette hashes | Stimulus production (11 Oct 2026) |
| `analysis.json` | Frozen cohort, OSF file hashes, human-target settings, published statistics to reproduce, spec hash, control settings, model decisions and the dated list of all spec amendments | Analysis spec and its amendments |
| `freeze.json` | sha256 of the 23 files the held-out evaluation depends on (model code, human target, spec, features manifest, training-source outputs) | `python model/run_evaluation.py freeze`, 11 Nov 2026 (tag `freeze-2026-11-11`) |

`model/run_evaluation.py` refuses to simulate held-out sources unless every file in `freeze.json`
still has its recorded hash. `tests/test_spec.py` checks `docs/analysis-spec.md` against the hash in
`analysis.json`.
