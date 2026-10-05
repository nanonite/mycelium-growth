# Triage: recovering dish tracks from the images (#23)

**Question.** Phase 1 of PLAN.md asks: *can the images be grouped into dish tracks* — using
**inoculum position**, **lid marks** (the orange rim dot, pen marks on the lid) and
**image-embedding clustering**? Issue comment #17 flagged the handle: *"Each dish has an orange
rim dot at varying angles + occasional pen marks — possible dish fingerprint for track
recovery."*

**Answer. No.** The rim dot is real and measured to ~0.1°, but it does not identify dishes. The
angles form a smooth, mildly non-uniform distribution — all 720 half-degree bins occupied,
chi²/df 7.4, and no single angle shared by more than 37 of the 16,513 detections — and the rim
embeddings find no structure beyond the byte-identical duplicates: out of 203,525,400 pairs, every
one of the 80 pairs above 0.99 is a byte-identical copy. There is no track structure to recover,
and what few pairwise links exist could not anchor a longitudinal model even if they were all
genuine.

The one thing the images *do* group by is byte-identical duplicates — 133 of 20,176 images (89
of them across an archive or folder boundary, 44 of them the same frame under two ids inside
`unlabeled-GS_PO_TS.zip`) — and that is a **labelling and numbering error**, not a dish track:
the `PO` block is a byte-for-byte copy of the `MYG` block, and the `PDA` block is a byte-for-byte
copy of `labeled-GL`.

This was the last in-data route to Phase 1. With #21 (no timestamps) and #22 (shuffled
numbering), the MyceliumSeg gate is **C: neither dish nor time recoverable** (#26).

Deliverables: `data/triage/dish_tracks.csv` (one row per image, 23 columns),
`data/triage/dish_tracks_summary.csv`, `data/triage/dish_embeddings.npz`,
`notes/triage_dish_tracks.json` (machine-readable stats). Code:
`scripts/triage/dish_tracks.py` (driver) +
`scripts/triage/dish_geometry.py` (I/O-free geometry/feature primitives), tests in
`tests/test_dish_geometry.py` and `tests/test_dish_tracks.py`.

---

## 1. The dish is findable, and the camera is fixed

`find_disc` locates the rim from the steepest outward falloff in the angular-median radial
profile, taking the **outermost** falloff whose outside is bare stage. Two things were not
obvious and were found by looking at failures:

* **Absolute brightness thresholds do not work.** Stage illumination is visibly uneven (broad
  vertical falloff bands in some frames), so a fixed cut-off lands somewhere different on every
  image. The derivative of the radial profile is stable; the threshold is not.
* **The sharpest edge is not the rim.** A colony is a bright disc, so its own edge is a sharp
  falloff and on a well-covered plate it is *sharper* than the rim. The rim is distinguished by
  being the **last** edge before the stage — past it the profile sits at stage level for good.

With that, the geometry is highly reproducible. `dish_r / image_width`:

| Archive | median | IQR |
|---|---|---|
| `labeled-GL` | 0.30222 | 0.00012 |
| `unlabeled-GL1` | 0.30222 | 0.00040 |
| `unlabeled-GL2` … `GL7` | 0.30222 | 0.00036 – 0.00075 |
| `labeled-MYG_PDA_TEMP` | 0.30222 | 0.02659 |
| `labeled-GS_PO_TS` | 0.32722 | 0.10931 |
| `unlabeled-GS_PO_TS` | 0.33000 | 0.00588 |

The camera is fixed and the dish is nearly centred (`cx = 455 ± 13`, `cy = 330 ± 10`, mean ± sd
over the 20,176 fits, on a 900-px working image).

**But the scale is not constant across condition blocks**, which matters for #25. Grouping by
condition folder rather than by archive (px/mm is for the 900-px working image, dish = 90 mm):

| Condition block | n | median r/W | IQR | implied px/mm |
|---|---|---|---|---|
| GL (incl. `labeled-GL`, `unlabeled-GL1..7`) | 18,427 | 0.30222 | 0.00039 | 6.044 |
| `unlabeled-GS_PO_TS` (flat, mixed) | 1,689 | 0.33000 | 0.00588 | 6.600 |
| GS | 10 | 0.22241 | 0.00258 | 4.448 |
| PO | 10 | 0.33146 | 0.00340 | 6.629 |
| TS | 10 | 0.33019 | 0.00631 | 6.604 |
| MYG | 10 | 0.33146 | 0.00340 | 6.629 |
| PDA | 10 | 0.30210 | 0.00505 | 6.042 |
| TEMP15 | 10 | 0.30222 | 0.00000 | 6.044 |

The GS dishes are visibly smaller in frame (checked by eye, not only by the fit), so the blocks
were shot at different camera heights or zoom. **A single global px→mm constant would be wrong
by up to ~35%.** #25 should calibrate per block from the per-image `dish_r` column in
`dish_tracks.csv`.

Note two rows that are not yet explained here and are explained in §5: `MYG` reproduces `PO`'s
median *and IQR* to five decimals, and `PDA` reproduces `GL`'s scale. The dish geometry is
independently confirming the byte-identity of those blocks — the fit was not told about the
duplicates.

---

## 2. The rim dot is real — and does not identify dishes

The marker is unambiguous: a small, highly saturated yellow blob at `rho ≈ 0.90` of the rim
radius (median 0.900, p10 0.860), crisp at full resolution. `find_rim_dots` keeps, per angle, the
*most yellow* sample along the radial band rather than the mean — the dot is only a few pixels
wide and averaging dilutes it below any useful threshold.

Detected in **16,513 / 20,176 images (81.8%)**, with tight angular concentration (median
resultant 0.999) and 13,703 images yielding exactly one dot.

**It does not partition the release.** All numbers below come from `dot_angle_structure` in
`dish_tracks.py` and are in `notes/triage_dish_tracks.json` under `dot_angles`.

| Test | Result | Reading |
|---|---|---|
| Distinct half-degree angle bins occupied | **720 / 720 (100%)** | If the release were *K* reused dishes, at most *K* bins could be occupied |
| Distinct angle values at 0.01° | **3,606** | the same bound from the other side *if the angle repeats exactly*: *K* ≥ 3,606 → ≤ 5.6 images per dish |
| Most repeated single angle | **37 images (0.22%)**; median angle seen once | reuse with a fixed dish orientation piles ~n/K images onto one value |
| Pairs within 0.05° / 0.10° | 97,402 / 114,812 observed vs 37,870 / 75,740 for uniform angles | **not** uniform — 2.6× / 1.5× |

The last row is deliberately *not* a "the angles are at chance" claim. The collision counts are
excess over uniform, and they should be: they carry the same broad bias as the chi-square below.
Two things stop them from meaning "reused dishes":

* **45% of the angles land exactly on the detector's own 0.5° sampling grid** (`find_rim_dots`
  samples the rim on `RIM_DOT_N_ANG = 720` steps; a peak spanning one step reports that step, a
  peak spanning two reports their mean). Anything measured below that resolution is quantisation,
  so collisions at 0.05° are the wrong scale to test at.
* **the multiplicity is what the local density already predicts.** The most crowded 0.01° cell
  holds 37 images while its ±0.25° neighbourhood holds 89 — pile-up onto a grid point, not a
  spike. A handful of reused dishes would instead show a large share of all images sitting on a
  few values; the median angle is seen once and the largest share is 0.22%.

So the angle is at best a ~9-bit label (720 slots, unevenly filled) — with 20,176 images that is
~30 images per slot even if the slots were used perfectly. Under the strongest reuse the numbers
allow — *K* ≥ 3,606 dishes at a fixed orientation — it would separate images by at most a factor of
5.6, far short of what a longitudinal model needs. If the dishes were re-oriented between shots the
angle is not even that, and identity has to come from the embeddings and the colony position, which
is where §3 and §4 look.

The chi² of the angle histogram against uniform is 5,338 on 719 df (7.4 per df), so there *is* a
broad bias — roughly a 2× excess around 150–210° and a deficit near 300°. That bias is **not**
discreteness: a handful of reused dishes would give sharp spikes with empty arcs between them,
and every one of the 720 bins is occupied. Its cause is not identified (it is not explained by
detections simply being stronger there — in the 150–210° band the p10 saturation is 53 against 134
for the release as a whole, i.e. the peak-band detections are slightly *weaker*), but whatever
produces it carries no per-dish information.

---

## 3. Embedding clustering: no structure beyond the duplicates

Two rim embeddings were computed per image, both detrended against the stage's broad
illumination (a circular moving average; the undetrended version is dominated by an angular
ramp and makes any two frames look alike):

* `rim` — 360-dim, rotation-**sensitive** (detrended yellow-ness and brightness vs angle).
* `rim_rot` — 16-dim, rotation-**invariant** (magnitudes of the angular Fourier harmonics), so
  turning the dish between shots cannot defeat it.

Two nulls are available, and the second one is the test that matters:

* **Nearest neighbours vs a covariance-matched Gaussian** (`gaussian_null`): 20,176 rows drawn
  from a Gaussian with the observed feature covariance, i.e. the shared low-frequency structure
  that makes any two frames look alike and nothing else. This is the baseline, not the verdict:
  for `rim` the observed nearest-neighbour median is 0.629 against 0.312 for the null, which
  says the real images are heavier-tailed than a Gaussian (duplicates, quantisation, similar
  colony stages) rather than that they group.
* **The duplicate split**: of every pair over a threshold, how many are byte-identical copies?
  `clustering_report` reports both halves, so "is the tail dish tracks or is it files filed
  twice" is answered directly rather than assumed.

| Threshold | pairs above it | of which byte-identical | non-duplicate |
|---|---|---|---|
| > 0.90 | 137 | 80 | 57 |
| > 0.95 | 83 | 80 | **3** |
| > 0.99 | 80 | **80** | **0** |

Out of 203,525,400 image pairs, every pair that `rim` puts above 0.99 is a byte-identical
duplicate — all 80 of them (the 80 unordered pairs formed by the 133 duplicate images of §5),
with zero extras. `rim_rot`
gives 3,867 / 355 / 81 against the same 80 duplicates, so its one extra pair above 0.99 is the
eyeball candidate below.

This also disposes of the "maybe the dish was rotated" objection: the rotation-invariant
embedding finds nothing the rotation-sensitive one misses.

A caveat worth recording: because the tail is broad, a handful of pairs look convincing by eye
(same dot angle, same pen marks, different colony stage — e.g.
`unlabeled-GL4/00009715` ↔ `unlabeled-GL7/00016658`, sim 0.9922, the only non-duplicate pair
above 0.99 in either embedding). It is a *candidate*, not a finding: its dot angle differs by
3.25° (7.75° vs 11.00°), its colony covers 0.28 R against 0.08 R, and there is no ground truth
to settle it. What settles the plan is the arithmetic: even if **every** one of the 57
non-duplicate pairs above 0.90 were the same dish photographed twice, they would cover at most
~110 images — 0.5% of the release — at two time points each. That is not a growth curve. The
counts, not the anecdotes, are the result.

---

## 4. Inoculum position and lid marks: also null

**Inoculum position.** Colony centroid offset from the dish centre, normalised by dish radius:
median 0.028 R, p90 0.064 R (per-archive medians 0.024 – 0.063 R). The colonies sit close to the
centre, and the *direction* of the residual offset is spread around the circle — circular
resultant 0.150 over all images and 0.06 – 0.27 per archive, against ≈ 1.0 for a single
inoculum spot that gets re-imaged. There is a weak preferred direction (the 60–120° bins carry
about 1.2× the uniform share), which is far too broad to identify a dish and does not repeat
consistently per archive.

**Lid marks.** Pen marks on the lid are visible on many plates (clearly in
`labeled-MYG_PDA_TEMP` and `TEMP15`) but are sparse. The high-pass angular signature of the lid
surface is exactly what the `rim` embedding of §3 clusters, and it shows no per-dish structure —
what structure it has is dominated by the colony edge and the stage illumination.

**Colony size vs file order.** Equivalent colony radius ordered by file id is a shuffle:
`colony_radius_fwd_steps` runs 0.48 – 0.55 and `dot_angle_fwd_steps` 0.49 – 0.54 across the
eleven archives (the one outlier, 0.71, is `labeled-GS_PO_TS` with only 18 dot angles) — see
`dish_tracks_summary.csv`. A genuine series would be ≈ 1.0. This reproduces #22's verdict from
an independent signal — measured colony area rather than file numbering or zip mtimes.

**Near-duplicate search (ad-hoc probe, not part of the pipeline).** 600 `unlabeled-GL1` images
compared by a polar+Cartesian thumbnail descriptor: top pair 0.969, no pair above 0.999. Nothing
hiding under another filename.

---

## 5. What the images *do* group by: 133 byte-identical duplicates

MD5-hashing every image shows **133 of 20,176 are exact duplicates** — 62 hashes with more than
one copy. They split into two different problems:

**40 hashes (89 images) cross an archive or folder boundary** — a file filed twice under two
different labels:

* **All 10 images in `labeled-GS_PO_TS/PO/` are byte-identical to the 10 in
  `labeled-MYG_PDA_TEMP/MYG/`.** Same md5, same byte count. The `PO` folder does not contain
  *P. ostreatus* images.
* **All 10 images in `labeled-MYG_PDA_TEMP/PDA/` are byte-identical to 10 images in
  `labeled-GL`** (GL on MYG at 25 °C), so the `PDA` medium label is also wrong there.
* **29 of the 30 `labeled-GS_PO_TS` images reappear inside `unlabeled-GS_PO_TS`** (the one
  exception is `labeled-GS_PO_TS/PO/image/00018441.jpg`), so the "unlabeled" shard overlaps the
  labeled set.
* The 10 `TEMP15` images — the only 15 °C set, and the entire basis of the primary estimand —
  have **no** duplicates. That block is clean.

**22 hashes (44 images) are two *different file names inside `unlabeled-GS_PO_TS.zip`*** — e.g.
`00018526.jpg` and `00018845.jpg` are the same bytes. The shard carries each of these images
twice under two ids, which is a numbering defect rather than a labelling one: it inflates the
image count, and it puts the same frame on both sides of any split taken by filename (#28/#29
must de-duplicate *within* an archive as well as across them, and #22's id ranges assume ids
are unique).

The dish geometry of §1 corroborates the cross-archive half without being told about it: the
`MYG` block's `dish_r / width` reproduces `PO`'s median and IQR to five decimals (0.33146 /
0.00340), and `PDA` sits exactly on `GL`'s scale. Two independently computed measurements
agreeing to that precision is what byte-identical files look like.

Consequences beyond this issue:

1. **#24's per-image species/medium table is wrong for 20 images.** `data/triage/known_labels.csv`
   derives labels from folder structure; for these 20 the folder and the pixels disagree, and the
   pixels are literally the same files in two archives. #34 (count the 15 °C dishes for power) is
   unaffected — `TEMP15` is clean — but should be re-checked once this is resolved.
2. **#28 (SegFormer baseline) and #29 (inference on the unlabeled images) must de-duplicate
   before splitting** — across archives *and within* `unlabeled-GS_PO_TS.zip`, which carries 22
   frames twice under two ids — or the same image lands in both train and test.
3. Worth putting to the authors alongside the metadata request in #26.

---

## 6. Verdict for the gate (#26)

| Signal | Result |
|---|---|
| Rim dot angle | detected in 81.8% of images; fills 720/720 half-degree bins; largest single angle shared by 37 images (0.22%) |
| Inoculum position | median 0.028 R from the centre, direction spread around the circle (resultant 0.15) — no repeatable spot |
| Lid / pen marks | sparse, no per-dish structure |
| `rim` embedding (rotation-sensitive) | all 80 pairs > 0.99 are byte-identical duplicates; 3 non-duplicate pairs > 0.95 of 203,525,400 |
| `rim_rot` embedding (rotation-invariant) | adds nothing but one candidate pair (0.9922) |
| Colony size vs file order | ≈50% forward steps — shuffled, independently confirming #22 |
| Exact duplicates | **133 images / 20,176** — a labelling/numbering error, not a track |

**Dish tracks cannot be recovered from the images.** MyceliumSeg is cross-sectional only. With
this, every in-data route to Phase 1 is closed, so the gate is **C**, and the recommendation
already made in `data/INVENTORY.md` stands: anchor the longitudinal growth model on the
**Fricker/Boddy grazer series** (Zenodo 14931953), which ships real capture times, replicate
identity, a randomized grazer treatment and per-image colony measurements. Keep MyceliumSeg for
cross-sectional size-by-condition work and segmentation, and *Podospora* for the growth-form /
nutrient transport test.

Left behind for other issues:

* **#25** must calibrate px→mm per condition block (§1) — a global constant is wrong by up to 35%.
* **#28/#29** must de-duplicate before splitting — across archives and inside
  `unlabeled-GS_PO_TS.zip` (§5).
* **#24**'s species/medium table needs the 20 corrected labels (§5).
* **#22**'s id ranges are not unique: 22 frames appear twice inside `unlabeled-GS_PO_TS.zip`
  under two ids (§5).

## Reproducing

```bash
.venv/bin/python scripts/triage/dish_tracks.py --jobs 11     # full scan, ~34 min on 11 cores
.venv/bin/python scripts/triage/dish_tracks.py --limit 40     # smoke test, ~20 s
.venv/bin/python scripts/triage/dish_tracks.py --report-only  # rebuild the JSON + summary
.venv/bin/python -m pytest tests -q                           # 118 tests
```

Flags: `--jobs N` (archives scanned in parallel), `--limit N` (images per archive),
`--report-only` (skip the scan; reuse an existing `--out`, and the clustering block from
`--embeddings` if it is there — without the embeddings the JSON would be rewritten with an empty
`clusters` section, which is why the file is reloaded rather than assumed). `--report-only`
rewrites `notes/triage_dish_tracks.json` and `dish_tracks_summary.csv`; it never touches this
`.md`, which is written by hand. Nothing is extracted to disk — the
archives are streamed with `zipfile`, and images are decoded at 900 px on the long side via
`Image.draft` (a DCT-domain downscale, ~3× cheaper than decoding 4608×3456).