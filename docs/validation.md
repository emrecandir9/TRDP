# Validation — 11 October 2026

The original accepted archive passes its **historical integrity checks**. The corrected
input masks pass their own checks. These are different mask versions: the accepted
images were preserved unchanged, as requested. A corrected-mask generation run has
not been sampled or visually approved.

## Results

| Scope | Result |
| --- | --- |
| CPU regression suite | 10 tests passed, including protected composition, changed-file rejection, original-mask provenance, immutable checkpoints, canary gating, idle timing and pinned model resolution |
| Frozen local bundle | All 220 hashes match: 120 RGB 2048-square source PNGs and 100 preparation inputs; all formats, dimensions and derived context pixels checked |
| Corrected masks | Binary mask equals weight > 0; no overlap with the original action. All 20 hash-bound historical hand mattes were resized into their native crops and independently confirmed protected |
| Approved archive | All 480 unique families, 2,880 original PNG hashes and 480 original audit records checked |
| Original provenance | All 160 parent JSONs match the family audit hashes; all 20 original desk weights match the parent input hashes and recorded previous versions |
| Historical pixel invariants | Zero protected-pixel errors under the original masks; action differences stay inside original support; retained context pixels match; hidden pixels are black |
| Corrected-guard comparison | All 480 families have deviations: 4,116,516 changed pixel positions summed across the 960 full-body A/B images. This includes the larger protected contact buffer; it is not a count of distinct people, anatomical defects or corrupted files |
| Worked example | Reference 6, birch, seed 9: composition from the original saved proposal and original weight matches the saved native composite exactly; warm export pixels also match |
| Illustrated guide | 25 pages rendered and visually inspected; bookmarks, text extraction, page dimensions and image provenance checked |
| Large local review PDF | 480 pages, 960 form fields, readable and unencrypted; retained unchanged. Its 480 pages were not freshly re-rendered |
| Publication boundary | Dataset, original-mask provenance, frozen input images, environment, new outputs and large review PDF are ignored by Git; no Git LFS |

The detailed local result is `data/accepted/validation.json`, including every family ID,
audit hash and corrected-guard deviation count. The original archive contains six PNGs
plus `audit.json` per family. Its `_provenance/` holds only the 20 original weights and
160 parent records needed to validate it. No replacement images or redundant preview
exports were added.

An additional check of the 20 beech / seed-9 / neutral static exemplars found changed
pixels inside the opaque source-hand matte (alpha >= 250) for references 10, 11, 12,
16, 18 and 20: respectively 136, 107, 352, 6, 162 and 78 pixels. The other 14 exemplars
were exact there. This confirms that the earlier mask issue can affect hand pixels,
in addition to the contact buffer. This exemplar check is not an exhaustive anatomy
assessment; the approved originals remain preserved.

## Project files checked

| File | Check |
| --- | --- |
| `README.md` | Directory map, local links and documented CLI entry points checked |
| `requirements.txt` | Pinned NumPy 2.3.5 / Pillow 12.0.0 installed by the review setup |
| `.gitignore` | Required local-only paths confirmed ignored |
| `.gitattributes` | PDF binary handling; no LFS rules |
| `config/recipe.json` | Valid JSON; 20 sources / 160 parents / 480 jobs / 10 canaries; frozen-manifest hash matches |
| `data/frozen_sources.json` | Valid JSON; every referenced local image rehashed and checked; correction retains previous input hashes |
| `scripts/setup.sh` | Shell syntax and actual Mac review-environment setup passed |
| `scripts/models.py` | Syntax, CLI and pinned/cache-only/download behavior tested with a mock hub |
| `scripts/generate.py` | Syntax, CLI, imports, existing-output and incomplete-checkpoint behavior checked; sampling untested |
| `scripts/run.py` | Syntax, CLI, moved worker path and continuous-idle reset behavior checked |
| `scripts/dataset.py` | Protected composition, white balance, exports, source/context checks and hash-bound review safeguards exercised |
| `scripts/validate_dataset.py` | Full downloaded archive checked; corrupt approval/weight cases covered by regression tests |
| `tests/test_dataset.py` | Syntax checked; all 10 CPU tests passed |
| `docs/project-guide.pdf` | Full visual layout review and saved-image/parameter checks completed |
| `docs/accepted-dataset.json` | 480 approvals, unique IDs, all 2,880 PNG hashes and 24 reference-17 flags checked |
| `docs/approval-receipt.json` | Historical approval scope, counts and known issue checked; original receipt preserved |
| `docs/validation.md` | Results reconciled with the local verification record and command outputs |

## Recheck

Run from the repository root after `./scripts/setup.sh review`:

```sh
source .venv/bin/activate
python -m unittest discover -s tests -v
python scripts/dataset.py check
python scripts/validate_dataset.py
```

Checks need the local bundle described in the README; CPU tests work without it.
The archive validator reports historical success and separately prints the corrected-
guard note. It does not rewrite the original accepted PNGs or audit records.

Current frozen-source manifest SHA-256:
`2495d9098ad6b7b8df014b7e56223c76ebf9cb5295fd88d0aa26c748c8e38faf`.

Local large-review PDF SHA-256:
`bc53cede317c3733c5bd5078b310f7d7d650e41593539ef6fe6bc99f31841447`.

## Remaining limits

No fresh CUDA inference or actual model download was run during this cleanup. No new
human review of every downloaded image is implied. The accepted reference-17 clothing
issue remains flagged. A future corrected-mask run requires new canary and full-batch
review. Upstream source independence, anatomical ground truth and physiological
replication remain unvalidated.
