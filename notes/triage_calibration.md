# Calibration: px → mm from the 90-mm dish rim (#25)

**Question.** PLAN.md Phase 1 asks for *mask area in pixels → mm²*, "dish rim = 90 mm",
with the warning from #23 that the apparent dish size differs by condition block
(r/W 0.302 / 0.332 / 0.224), so one global constant is off by ~35%.

**Answer.** The calibration is one number per condition block — and inside one block
(`unlabeled-GS_PO_TS`) one number per camera setup — because the fitted rim radius is
a property of *how the frame was shot*, not of the image. It is published as
`data/triage/calibration.csv` (one row per block/setup) and
`data/triage/image_scale.csv` (one row per image: which constant applies to it, and
whether its own fit agrees). On the GS block a single global constant is 36% too
large, which puts every derived length 26% too small and every area 46% too small —
and area is what Phase 1 asks for.

Deliverables: `data/triage/calibration.csv`, `data/triage/image_scale.csv`,
`scripts/triage/dish_scale.py` (I/O-free primitives), `scripts/triage/calibrate.py`
(driver), `tests/test_dish_scale.py`, `tests/test_calibrate.py`, this note.

---

## 1. What the constant is

A standard plate is 90 mm across, so the rim radius `find_disc` already fits is the
calibration itself:

```
px_per_mm = 2 × rim_radius_px / 90 mm        # measured at the 900-px working width
```

`dish_tracks.csv` carries a per-image `pixels_per_mm` computed exactly this way. It is
right for 19,616 of 20,176 images and wrong — sometimes by 45% — for the 560 where the
rim fit failed (§4). The published constant is therefore a **median over the images of
one setup**: a median cannot be moved by the failures it exists to repair.

| block | setup | n | off | r/W median | IQR | px/mm @900 | spread | one global constant errs by |
|---|---|---|---|---|---|---|---|---|
| GL | 1 | 18,427 | 435 | 0.30222 | 0.00034 | 6.0444 | 0.11% | +0.00% |
| GS | 1 | 10 | 0 | 0.22241 | 0.002578 | 4.4482 | 1.16% | **+35.88%** |
| GS_PO_TS_flat | 1 | 1,598 | 122 | 0.33111 | 0.004255 | 6.6222 | 1.29% | −8.73% |
| GS_PO_TS_flat | 2 | 91 | 0 | 0.22222 | 0.001195 | 4.4444 | 0.54% | **+36.00%** |
| MYG | 1 | 10 | 1 | 0.33173 | 0.00299 | 6.6345 | 0.90% | −8.89% |
| PDA | 1 | 10 | 0 | 0.302095 | 0.005047 | 6.0419 | 1.67% | +0.04% |
| PO | 1 | 10 | 1 | 0.33173 | 0.00299 | 6.6345 | 0.90% | −8.89% |
| TEMP15 | 1 | 10 | 0 | 0.30222 | 0.0 | 6.0444 | 0.00% | +0.00% |
| TS | 1 | 10 | 1 | 0.33254 | 0.00555 | 6.6507 | 1.67% | −9.12% |

(the "global constant" column is `(global / this block) − 1`: a positive number means
the global constant is too large here, and the derived lengths come out too small.
The global constant is the median over the whole release, 6.0444 px/mm — i.e. GL's
value, because GL is 91% of the images. `off` counts images whose own fit disagrees
with the constant it was given by more than 5%; they are members of the group, they
just do not vote on its constant.)

The spread column is the honest precision of each number: **0.11% for GL** (18,427
images, one camera setup) and 0.00–1.67% for the rest. The ten-image blocks should not
be quoted past three significant figures.

Three cross-checks that could have failed and did not:

* **PO and MYG are byte-identical files** (#23 §5), so they *must* calibrate to the
  same number: both give 6.6345 px/mm.
* **PDA is ten byte-identical copies of labeled-GL images**, so it must sit on the GL
  scale: 6.0419 vs 6.0444, −0.04%.
* **TEMP15 — the only 15 °C block, and the whole basis of the primary estimand — has
  zero spread** (all ten fits identical) and sits exactly on the GL scale, 6.0444.

## 2. One block is two camera setups

The flat `unlabeled-GS_PO_TS` shard is not one scale: 1,598 images sit at r/W 0.331 and
91 at r/W 0.222, **49% apart**. Handing those 91 frames the block's dominant constant
would put their lengths 33% too small and their areas 55% too small (2.2×) — the same
error class this issue exists to remove. Four independent lines of evidence say the
second group is real and not a cluster of failed fits:

1. **Byte-identity.** All ten images that are byte-for-byte copies of the *labeled-GS*
   images (#23 §5, scale 0.22241, checked by eye there) land in group 2, with r/W
   0.21940–0.22332. The pixels are the same, so the scale is the same — the split was
   not told about the labels.
2. **Rim-dot rate.** The yellow rim dot is detected in 69% of group 1 but 1% of group 2
   — and in 0/10 of the labeled-GS images themselves. Group 2 looks like the GS
   acquisition in every column, not just in the radius.
3. **The fit agrees with the dish.** Overlaying the fitted circle on
   `00018495`, `00019790`, `00019866`, `00020132` (group 2) shows it lying on the rim of
   a dish that fills ~44% of the frame width; the 0.331-W reference circle falls far
   outside the dish.
4. **An independent silhouette measure.** For a sample of frames I measured the
   outermost radius at which most of the annulus is brighter than stage + 45 — a crude
   dish-silhouette estimate that shares nothing with `find_disc`:

   | image | archive | fit r/W | silhouette r/W | reading |
   |---|---|---|---|---|
   | 00019511 | flat | 0.3411 | 0.3422 | fit correct (group 1) |
   | 00018495 | flat | 0.2222 | 0.2311 | genuine small dish (group 2) |
   | 00019866 | flat | 0.2200 | 0.2289 | genuine small dish (group 2) |
   | 00019750 | flat | 0.1934 | 0.2289 | **fit failed**; the dish is group 2's size |
   | 00018635 | flat | 0.2804 | 0.3444 | **fit failed** (inner reflection ring) |
   | 00018550 | flat | 0.3091 | 0.3356 | **fit failed** (inner reflection ring) |
   | 00016696 | GL7 | 0.1656 | 0.3044 | **fit failed** (colony edge) |
   | 00002564 | GL1 | 0.2042 | 0.2600 | **fit failed** (colony edge) |

   The silhouette reads ~0.01 high on small dishes (it catches the lid edge) and can
   undershoot when the agar is dark, so it is a cross-check of *which side of the
   argument an image is on*, not a second measurement of the same quantity.

The split itself is not hard-coded: `dish_scale.scale_modes` clusters the fitted r/W
within each block (density in log space, then a tightness check), which is what finds
group 2 and rejects the smears in between. Its two shape thresholds are set by the
numbers above — a real setup is tighter than 3% IQR (observed: 0.00–1.91%) while a
smear is wider than 5% (observed: 5.5% and 6.1%), and setups are 49% apart while the
tight failure clusters sit 8–20% from their own block's mode.

## 3. Which images get which constant

Every image is assigned to the nearest setup it *agrees* with (within 5% of its r/W),
so a group-2 frame keeps 4.4444 px/mm even though its block's dominant scale is 6.6222.
An image further than 5% from every setup has no readable scale of its own — that is a
failed fit, not a fourth camera height — so it takes its block's dominant constant and
is flagged `fit_status=off_scale` in `image_scale.csv`.

**560 images (2.8%) are flagged:** 435 in GL (r/W 0.166–0.287, median 0.214), 122 in
the flat shard, 1 each in MYG/PO/TS. The rim dot is detected in only 15% of the flagged
GL images, against 85% of the good ones — which is what you expect when the circle is
drawn around the colony instead of the dish.

Not every flag is harmless: of the flat shard's 122, the 79 at r/W ≥ 0.27 are clearly
dominant-setup frames whose fit locked onto an inner ring (assignment correct), while
**43 sit below 0.27, where the setup they were photographed at cannot be recovered from
the CSV alone** — `00019750` is one of them (silhouette says group 2, the rule assigns
group 1). That is 0.2% of the release. Work that needs per-image precision should
filter `fit_status == "ok"` rather than accept the prior.

## 4. Why the rim fit fails

`find_disc` takes the outermost falloff before bare stage, and two failure modes are
visible by eye (circle overlaid on the decoded frame):

* **the colony edge**, when the colony's own falloff is sharper than a washed-out rim
  (`00016696`: fit 0.1656, the dish is nearly twice that);
* **an inner reflection ring inside the dish** (`00018635`, `00018550`: fit 0.280 and
  0.309, the rim is at 0.34).

Both always *under*-report the radius, which is why the median — not the mean — is the
statistic behind every published constant, and why a failed fit can be detected as a
disagreement with the constant instead of being trusted. `edge_sharpness` does not
separate the two failure modes consistently (the low-r/W failures are as dull as good
fits, −7.4 against −7.7 in the flat shard, while the inner-ring ones are sharper, −13.2),
so it is not used here.

## 5. Using it

```python
import csv
from scripts.triage.dish_scale import area_px_to_mm2, rescale_px_per_mm

scale = {r["id"]: r for r in csv.DictReader(open("data/triage/image_scale.csv"))}
row = scale[image_id]
assert row["fit_status"] == "ok"            # or accept the dominant-scale prior

px_per_mm = rescale_px_per_mm(float(row["px_per_mm"]), width)   # width = this frame
area_mm2 = area_px_to_mm2(mask_area_px2, px_per_mm)
radius_mm = radius_px / px_per_mm
```

**Worked example** — the README's mask, `labeled-GS_PO_TS` image `00018428`, 1,214,257
px² at full resolution (4608 px wide): its block is GS, so

```
px/mm = 4.4482 × 4608/900 = 22.7748      area = 1214257 / 22.7748² = 2341 mm²
                                             r_eq = √(1214257/π) / 22.7748 = 27.3 mm
```

Using the global constant instead (30.9473 px/mm) gives 1,268 mm² and r_eq = 20.1 mm —
**46% low in area**, because an error in a length scale is squared by an area.

The constants in `calibration.csv` are for the **900-px working images** of
`dish_geometry.py`; multiply by `width/900` for any other resolution (×5.12 for the
4608-px originals). `reference_width` in the table says so explicitly.

## 6. Limits

* **The 90 mm comes from the dataset description, not from a ruler.** If the dishes are
  not standard plates, every constant scales with them — as does the 45 mm saturation
  radius in PLAN.md's growth model.
* **A block is a proxy for a camera setup.** It holds because the camera is fixed and
  each condition folder was shot in one session (#23 §1). An image rescaled *within* a
  block shows up as `off_scale`, which is the signal, not the fix.
* **The split is deliberately conservative.** Clusters closer than 25% are merged when
  their union is still tight and kept apart when it is not — merging two real
  populations would park the constant in the gap, where no image lives. The cost falls
  on the failure side: a failed-fit cluster riding on a mode is absorbed into it rather
  than published as its own setup. That is the right call for the flat shard's 57 images
  at 0.307 and 30 at 0.275 (§4 says they are failed fits, and they do end up inside
  group 1, flagged `off_scale`). The only genuine split in the release is the 49% one.
* **Duplicated frames do not move the numbers.** The only within-block duplicates are
  22 pairs inside the flat shard (1,689 → 1,667 unique md5); its median r/W is
  0.33000 either way. Cross-block duplicates (PO ≡ MYG, PDA ≡ GL) are *supposed* to
  agree, and do.
* **No attempt is made to convert a failed fit into a good one.** Repairing `find_disc`
  is #23's code and out of scope here; this issue only refuses to let a bad fit into
  the calibration.

## Reproducing

```bash
.venv/bin/python scripts/triage/calibrate.py          # rewrites both CSVs, ~1 s
.venv/bin/python -m pytest tests -q                   # 156 tests
```

Flags: `--tracks` (input rim fits), `--out` (calibration table), `--per-image`
(per-image table); all three default to the committed paths. The run needs no images
and no archives — only `data/triage/dish_tracks.csv` — and
`tests/test_calibrate.py::test_committed_tables_reproduce_from_the_rim_fits` fails if
the committed CSVs ever stop matching a re-run on those fits.
