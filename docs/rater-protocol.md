# Rater validation: realism and visible artifacts

24 October 2026. Code: [`scripts/raters.py`](../scripts/raters.py). Tests:
[`tests/test_raters.py`](../tests/test_raters.py). Outputs: `output/raters/` (local). Addresses
issue 5 of the [next-phase plan](next-phase-plan.md) §3.

**Status: not run.** On 24 October 2026 the project owner decided to skip the rater validation,
because recruiting naive raters was not feasible. The package below is complete and can still be
used unchanged if raters become available. Prepared on 24 October 2026, **before any rating exists**. The sample, the
presentation orders, the instructions and the analysis below are fixed now. Any change before
the first session is listed in the change log at the end, with its reason; nothing changes after
the first session. **This document is for the experimenter: do not show it to the raters.**

## Purpose

All 480 families were accepted by one reviewer, the project owner, who knew the design, the
hypothesis and which frame of each pair was edited (issue 5). This validation asks two people
who know none of this to rate a fixed sample, and reports how far they agree. It answers four
questions:

1. How realistic do the stimuli look to naive observers, in each context?
2. Which images show a visible error, according to either rater?
3. Is the edited frame (movement) rated as less realistic, or as flawed more often, than the
   static frame of the same image? Every endpoint of the [analysis spec](analysis-spec.md) is a
   movement − static contrast within a context, so a visible flaw that differs between the two
   frames would be a confound.
4. How well do two raters agree, and so how much weight can a single reviewer's judgement carry?

## Raters

Two raters, A and B. A rater counts as **naive** if all of the following hold:

| Criterion | Requirement |
| --- | --- |
| Involvement | Not involved in the project: not the project owner, a supervisor, or anyone who has worked on it |
| Prior exposure | Has not seen the stimuli, the project guide, the review PDF or the figures in `docs/figures/` |
| Information | Not told the hypothesis, the reference study, how the pictures were made, that they come in static/movement pairs, or which frames are edited |
| Independence | Does not see the other rater's answers and does not discuss the task with anyone until both have finished |
| Vision | Adult with normal or corrected-to-normal vision (self-report) |

Experience with photography, image editing or AI image generation is recorded but does not
exclude anyone. If a rater withdraws or turns out not to be naive, a new naive rater takes the
same letter and starts from the first picture; the incomplete file is kept but not analysed.
Ratings are stored under the letter only, never under the rater's name.

## Sample

One family per source: 20 sources × 4 contexts (`full_body`, `upper_limb`,
`detached_hand_silhouette`, `detached_hand` rectangle) × 2 frames = **160 images**. Both
detached-hand versions are included. The families use 20 different appearance variants out of
24, each once, drawn with seed 20261022 until every balance rule held (58th draw):

| Rule | Result |
| --- | --- |
| Desk | Each of beech, birch, maple, oak 5 times; seed 9 / 123 split 2/3 or 3/2 per desk |
| Seed | 9 and 123 10 times each |
| White balance | cool 6, neutral 7, warm 7; 1 or 2 per desk; 3 or 4 per seed |
| Variants | 20 distinct (desk, seed, balance) combinations |

| Ref | Family | Desk | Seed | White balance |
| --- | --- | --- | --- | --- |
| 01 | `r01_maple_s123_warm` | maple | 123 | warm |
| 02 | `r02_maple_s123_neutral` | maple | 123 | neutral |
| 03 | `r03_beech_s9_warm` | beech | 9 | warm |
| 04 | `r04_beech_s9_cool` | beech | 9 | cool |
| 05 | `r05_oak_s123_neutral` | oak | 123 | neutral |
| 06 | `r06_birch_s9_warm` | birch | 9 | warm |
| 07 | `r07_oak_s9_neutral` | oak | 9 | neutral |
| 08 | `r08_maple_s9_neutral` | maple | 9 | neutral |
| 09 | `r09_oak_s9_warm` | oak | 9 | warm |
| 10 | `r10_beech_s123_cool` | beech | 123 | cool |
| 11 | `r11_beech_s123_warm` | beech | 123 | warm |
| 12 | `r12_maple_s9_warm` | maple | 9 | warm |
| 13 | `r13_birch_s9_cool` | birch | 9 | cool |
| 14 | `r14_maple_s9_cool` | maple | 9 | cool |
| 15 | `r15_oak_s123_cool` | oak | 123 | cool |
| 16 | `r16_oak_s123_warm` | oak | 123 | warm |
| 17 | `r17_birch_s123_neutral` | birch | 123 | neutral |
| 18 | `r18_birch_s123_cool` | birch | 123 | cool |
| 19 | `r19_birch_s9_neutral` | birch | 9 | neutral |
| 20 | `r20_beech_s123_neutral` | beech | 123 | neutral |

Reference 17 carries the accepted clothing issue ([approval receipt](approval-receipt.json)); it
stays in the sample, and every result is repeated without it, as for every analysis.

## Packet and blinding

`python scripts/raters.py sample` checks every archive PNG against its SHA-256 in
[`accepted-dataset.json`](accepted-dataset.json) or [`accepted-silhouette.json`](accepted-silhouette.json)
(and those manifests against `config/analysis.json`), then writes:

| Path | Content | Given to raters |
| --- | --- | --- |
| `output/raters/packet/images/img_001.png` … `img_160.png` | The 160 images |
| `output/raters/packet/rate.html` | Offline rating form | Yes |
| `output/raters/key.csv` | Code → family, source, context, frame, original and exported SHA-256, positions in both orders | **No** |
| `output/raters/manifest.json` | Code hash, seed, sample, balance, orders, every file hash, package versions | **No** |

| Export setting | Value |
| --- | --- |
| Size | 1024 × 1024 px: the mean of each 2 × 2 block of the 2048 px original (PIL `Image.reduce(2)`) |
| Format | Lossless PNG, 8-bit RGB, no metadata; no compression artifacts can be rated |
| Names | `img_NNN.png`, numbers assigned at random, files written in number order |
| Packet | 161 files, about 100 MB; packet id `c9d840032ec0`; `key.csv` SHA-256 `7d468243…a7f1` |

Blinding measures:

- File names and numbers carry no information about family, context or frame.
- The key and manifest live outside the packet. `python scripts/raters.py check` re-hashes every
  file and refuses a packet that holds any other file (for example a copy of the key) or whose
  form contains a context, frame, desk, family or source name.
- The form calls every image a "picture" and shows only "Picture 12 of 160"; it never names a
  condition and gives no hint that a frame was edited.
- What cannot be blinded: raters see the context, and they can notice that scenes recur and that
  the index finger differs between some pictures. They are not told why.

## Presentation order

Each rater has a separate random order (A and B, independent streams of seed 20261022). Both
orders follow the same rules, checked by `order_problems` before the packet is written:

| Rule | Design |
| --- | --- |
| Rounds | 8 rounds of 20 pictures; every round shows every source once and every context 5 times |
| Frames | Every source shows each context once in rounds 1–4 and once in rounds 5–8, with the other frame, so the two frames of an image are at least 61 positions apart |
| Frame order | Static frame first for 10 of the 20 sources in every context; for every source in 2 of its 4 contexts; every round has 10 static and 10 movement pictures (2 or 3 of each per context) |
| Spacing | Two pictures of the same source are at least 10 positions apart, so the frames and contexts of a family are never adjacent |
| Randomised per rater | Grouping of sources, cycle of contexts, which frame comes first, order within each round |

In both orders the mean position of each context and each frame lies between 79.8 and 81.4
(midpoint 80.5), so fatigue or drift affects all contexts and both frames alike.

## Instructions

The experimenter reads the instructions aloud; the form shows the same
text before the first picture and the questions beside every picture.

> You will see 160 pictures, one at a time. Some show a whole scene; others show only part of
> it, and the rest of the picture is black. The black areas and the cropping are intentional and
> are not errors in themselves: judge how the visible part looks.
>
> For each picture, answer two questions. First: How realistic does this photograph look?
> Answer from 1 to 7, where 1 means clearly artificial and 7 means indistinguishable from a real
> photograph. All seven values may be used.
>
> Second: Do you see any visible error or artifact (e.g. distorted fingers, smeared skin, wrong number of fingers, unnatural edges)?
> Answer none, minor or major. Minor means a small flaw that you notice only when looking
> closely. Major means an error that is obvious or that makes the picture look wrong.
>
> You may add a comment, for example to say where an error is. There are no right or wrong
> answers: we want your own impression. Many pictures look alike; rate each one on its own.
>
> You can click a picture to enlarge it, and go back to change an answer. Work at a steady pace;
> about 10 to 15 seconds per picture is typical, and you may take a short break at any time. At
> the end, download your ratings and give the file to the experimenter.

Beside every picture the form shows (no answer is preselected; *Next* stays disabled until both
questions are answered):

| Field | Text |
| --- | --- |
| Question 1 | How realistic does this photograph look? |
| Scale | Buttons 1–7; 1 = clearly artificial; 7 = indistinguishable from a real photograph |
| Question 2 | Do you see any visible error or artifact (e.g. distorted fingers, smeared skin, wrong number of fingers, unnatural edges)? |
| None | No visible error |
| Minor | A small flaw, noticed only when looking closely |
| Major | An obvious error, or one that makes the picture look wrong |
| Comment | Comment (optional), for example where the error is |

**Questions from raters.** The experimenter answers only by repeating the instructions. Questions
about how the pictures were made, what they are for, or whether some are edited get: "I can tell
you after the session".

## Procedure

| Item | Requirement |
| --- | --- |
| Duration | 30–45 minutes (160 pictures at 10–15 s, plus instructions); breaks allowed after any picture |
| Computer | Laptop or desktop, screen at least 13 inches and 1280 × 800 px; preferably the same computer for both raters |
| Browser | Current Chrome, Firefox, Safari or Edge; `rate.html` opened directly from disk; browser zoom 100 %, full-screen window |
| Display | Brightness fixed at a comfortable level; automatic adjustments (Night Shift, True Tone, f.lux, auto-brightness) off |
| Room | Normal indoor light, no glare or direct sunlight on the screen; viewing distance about 50–70 cm |
| Supervision | One rater at a time; the experimenter stays available but does not watch the answers or comment on pictures |

Steps:

1. Before the first session, run `python scripts/raters.py check` and copy **only**
   `output/raters/packet/` to the rating computer. The key never leaves the project computer.
2. Open `rate.html` by double-clicking it in its folder, so it opens in Chrome or Safari (a
   preview, e.g. in an email, a chat or a code editor, cannot load the pictures; the form then
   says so). Check that the first picture appears, choose the rater's letter on the start screen, and press
   Continue. The experimenter chooses the letter; the rater does not need to know what it means.
3. Read the instructions aloud; the rater then reads them on screen and starts.
4. Progress is saved in the browser after every picture; if the window closes, reopening
   `rate.html` with the same letter resumes at the first unrated picture.
5. At the end the rater clicks *Download ratings (CSV)*. If the download fails, *Show the data
   as text* gives the same content to copy into a file. The experimenter checks that the file has
   160 rows, stores it on the project computer (for example `output/raters/ratings/A.csv`), and
   then chooses the same letter and *Start again* to delete the copy kept in the browser.
6. Fill in the session record below. Debrief each rater only after both have finished.

| Session record | A | B |
| --- | --- | --- |
| Date, start and end time | | |
| Computer, screen size and resolution, browser | | |
| Naive criteria confirmed (table above) | | |
| Experience with photography, image editing, AI images | | |
| Breaks, interruptions, deviations from this protocol | | |
| File name and SHA-256 of the downloaded ratings | | |

## Pre-declared analysis

Fixed on 24 October 2026, before any rating exists. Command:
`python scripts/raters.py analyse --ratings A.csv B.csv` (either order); results in `output/raters/`.

**Validation.** The analysis runs only on two files that pass every check against the key: the
columns `code, realism, artifact, comment, rater, position, timestamp`; one rater letter per file
and one file per letter; all 160 codes, each exactly once, no unknown code; realism an integer
1–7; artifact one of `none`, `minor`, `major`; each code at its position in that rater's order.
`key.csv` is checked against the hash in `manifest.json`. Values are never corrected by hand; a
missing or invalid answer is completed by the rater in the form.

**Units.** Sources are the independent units, as in the [analysis spec](analysis-spec.md) §7.1.
Every interval is a 95% percentile bootstrap over the 20 sources (10,000 resamples, seed
20261022): a resampled source brings all its images. Every result is computed for all 20 sources
and again without reference 17 (19 sources).

| Quantity | Definition | File |
| --- | --- | --- |
| Per image | Both raters' realism, artifact, comment and position; mean realism; difference A − B; major by either |
| Mean realism per context | Per image, the mean of the two raters; per source, the mean of its two frames; then the mean over sources, with interval. Also per rater | `contexts.csv` |
| Artifact rates per context | Share of ratings that are minor or major (any artifact), and major; same aggregation and interval; also per rater | `contexts.csv` |
| Movement − static per context | Per source, the movement frame's value minus the static frame's, for realism, any-artifact and major rates (raters averaged, and each rater); mean over sources with interval; number of sources with a higher, lower or equal value for movement | `frames.csv` |
| Realism agreement | Quadratic-weighted Cohen's kappa over the 7 categories (Cohen, 1968); ICC(2,1), two-way random effects, absolute agreement, single rater (Shrout & Fleiss, 1979; McGraw & Wong's ICC(A,1)); exact and within-one agreement (%); mean difference A − B | `agreement.csv` |
| Artifact agreement | Cohen's kappa over none / minor / major, unweighted (Cohen, 1960); raw percent agreement | `agreement.csv` |
| Agreement scope | All 160 images, and within each context (40 images) | `agreement.csv` |
| Flagged images | Every image that either rater called major, with both raters' answers and comments | `flagged.csv` |
| Answer distributions | Counts per rater, context and category | `distributions.csv` |
| Record | Hashes of the rating files, key, manifest, code and every output; headline values | `summary.json` |

The ICC is implemented in `scripts/raters.py` and tested against the Shrout & Fleiss (1979)
Table 2 example (ICC(2,1) = .29); the weighted kappa against its closed form
2·cov / (var A + var B + (mean A − mean B)²) and the unweighted kappa against a textbook 2 × 2
example (κ = .40). A bootstrap resample in which a kappa is undefined (both raters gave every image the same
single category) is left out of that interval and counted in `undefined_resamples`.

**Reading rules, fixed now.**

- **Within-context agreement is the main agreement result.** Agreement over all 160 images is
  inflated by differences between contexts (cut-outs are expected to look less realistic than
  full scenes); it is reported but not interpreted on its own.
- Agreement labels follow Landis & Koch (1977) for kappa (below 0 poor, ≤ .20 slight, ≤ .40 fair,
  ≤ .60 moderate, ≤ .80 substantial, above that almost perfect) and Koo & Li (2016) for the ICC
  (< .50 poor, < .75 moderate, < .90 good, above that excellent). They describe the point
  estimate and are always given with its interval; with 40 images per context the intervals are
  wide.
- When almost all answers fall in one category, the artifact kappa can be low despite high
  agreement (Feinstein & Cicchetti, 1990); it is read together with the percent agreement and
  the answer distributions.
- A movement − static difference is called **detected** when its 95% interval excludes 0.
  Otherwise it is "not detected", which is not evidence of no difference: with 20 sources only
  large differences can be detected.
- Realism is compared between frames within a context, not between contexts (see Limits);
  context means are descriptive.
- No other test is run. Analyses not listed here are labelled exploratory.

## How results are used

- **Report only.** The results are added to [validation](validation.md) by the project owner,
  with the hashes from `output/raters/summary.json` and the session records.
- **No exclusion.** The ratings do not exclude images, families or sources and do not change the
  frozen cohort. Any exclusion would require an amendment of the [analysis spec](analysis-spec.md)
  (§10) approved by the project owner, with its reason, before the held-out evaluation is opened.
- Flagged images are inspected by the project owner and listed with the results; they stay in
  the cohort unless such an amendment is made.
- A detected movement − static difference in a context is reported as a limitation of that
  context's stimuli in the final report.

## Limits

- **Two raters.** Agreement comes from one pair; it cannot separate a rater from the pair, and
  the mean ratings describe these two people, not observers in general.
- **Cut-outs are unnatural by design.** The silhouette and the rectangle are hands on black;
  as photographs they cannot look like a real scene, so their realism is not comparable with the
  full-body or upper-limb views. Realism is compared within a context only.
- **Context is visible.** Raters know which kind of display they rate; context cannot be blinded.
- **Pairs are visible.** Raters see both frames of every image and eight pictures of every
  scene; the 61-position minimum between the two frames of an image reduces direct comparison
  but does not prevent it.
- **Viewing differs from the experiment.** 1024 px on a monitor, not the paper's VR display;
  detail finer than 2 × 2 original pixels is averaged, and the hand covers a smaller visual angle.
  Raters can enlarge a picture.
- **One family per source.** 20 of the 24 appearance variants appear, each once; effects of desk,
  seed or white balance cannot be estimated, and the other 460 families are not rated.
- **Clustered images.** The 8 images of a source are not independent; agreement statistics use
  images as units, and the intervals resample sources.
- **No practice pictures.** Showing chosen examples would anchor the scale, so there are none;
  the first ratings may be less settled. Every round contains every source and context, so this
  affects all contexts alike.
- **Realism is not ownership.** The raters are not the paper's participants, and realism ratings
  are not embodiment or ownership ratings.

## Commands

```sh
python scripts/raters.py sample           # once: verified packet, key and manifest (refuses to overwrite)
python scripts/raters.py check            # before each session: hashes and blinding
python scripts/raters.py analyse --ratings output/raters/ratings/A.csv output/raters/ratings/B.csv
python -m unittest tests.test_raters      # 21 CPU tests; the sample test needs the local archive
```

`sample` is deterministic: a rerun with `--force` reproduces the same sample, orders, images and
key (with the recorded NumPy and Pillow versions); a test checks this against the packet in
`output/raters/`.

## References

- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and
  Psychological Measurement*, 20, 37–46.
- Cohen, J. (1968). Weighted kappa: nominal scale agreement with provision for scaled
  disagreement or partial credit. *Psychological Bulletin*, 70, 213–220.
- Feinstein, A. R. & Cicchetti, D. V. (1990). High agreement but low kappa: I. The problems of
  two paradoxes. *Journal of Clinical Epidemiology*, 43, 543–549.
- Koo, T. K. & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation
  coefficients for reliability research. *Journal of Chiropractic Medicine*, 15, 155–163.
- Landis, J. R. & Koch, G. G. (1977). The measurement of observer agreement for categorical data.
  *Biometrics*, 33, 159–174.
- McGraw, K. O. & Wong, S. P. (1996). Forming inferences about some intraclass correlation
  coefficients. *Psychological Methods*, 1, 30–46.
- Shrout, P. E. & Fleiss, J. L. (1979). Intraclass correlations: uses in assessing rater
  reliability. *Psychological Bulletin*, 86, 420–428.

## Change log

| Date | Change | Reason |
| --- | --- | --- |
| 24 Oct 2026 | Protocol, sample, packet and analysis prepared |
| 24 Oct 2026 | Form and instructions in English only; French version removed | Project owner's decision; raters will be given the English form |
| 24 Oct 2026 | Form tells the user when it is opened in a preview that cannot load the pictures; procedure says to open it from its folder | The form was opened in a preview, where no picture loads |
| 24 Oct 2026 | Validation not run | Project owner's decision: naive raters could not be recruited; single unblinded review stays a limitation |
