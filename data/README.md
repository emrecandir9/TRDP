# data/

Tracked files are compact metadata and hashes. The images and the OSF deposit are local only (in
`.gitignore`). Copy them from the project's local copy into the same paths, and do not regenerate
approximations of the frozen inputs.

**Tracked**

| File | Contents |
| --- | --- |
| `frozen_sources.json` | Metadata and sha256 of the 220 frozen source and input files |
| `source_metadata.json` | Reference and confirmed apparent sex per source, with provenance |
| `hand_segmentation.json` | SAM-HQ hand masks: provenance, prompts, hashes |
| `arm_segmentation.json` | SAM-HQ forearm masks for the controls: provenance, prompts, hashes |
| `silhouettes.json` | Silhouette matte hashes, parameters and per-source diagnostics |
| `controls.json` | Control masks, permutations, parameters and diagnostics |
| `context_continuity.json` | Visible forearm continuity per display, the input of H2's context pathway |

**Local only**

```text
sources/                 20 approved V7 source pairs in all three contexts
inputs/                  source image, Canny control, desk mask and weight, hand matte,
                         case geometry, SAM-HQ masks, mattes
human/                   unchanged OSF files of the reference paper
accepted/                the 480-family accepted dataset (2,880 PNGs, RGB, 2048 × 2048)
├── _provenance/         original desk weights and the 160 diffusion parent records
├── r12_birch_s9_warm/   one family: reference 12, birch desk, seed 9, warm balance
│   ├── full_body/       static.png, movement.png
│   ├── upper_limb/      static.png, movement.png
│   ├── detached_hand/   rectangle with desk patch (sensitivity condition)
│   └── audit.json       per-family provenance and image hashes
└── validation.json      result of scripts/validate_dataset.py
accepted_silhouette/     <family>/detached_hand_silhouette/{static,movement}.png (960 PNGs;
                         the primary detached hand: skin only, cut at the wrist)
```

Contexts keep their original positions, and hidden regions are black. A family's images are paired
views, not independent observations. Approval records and image hashes are in `docs/accepted-dataset.json`,
`docs/accepted-silhouette.json` and `docs/approval-receipt.json`.
