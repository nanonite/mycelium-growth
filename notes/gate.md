# Gate C: dish identity and capture time are not recoverable from MyceliumSeg (#26)

**Date:** 2026-10-05
**Status:** decided — Gate **C** (neither dish identity nor capture time recoverable)
**Evidence:** #21 (EXIF), #22 (numbering), #23 (dish tracks), #24 (labels), #25 (px→mm calibration)
**Blocks:** #20, #34 (any growth-curve interpretation of MyceliumSeg output)
**Assumption record:** #20 falsified — `notes/assumptions.md` (cascade: #34, #41, #42, #44, #52)

**Decision.** The released MyceliumSeg files do not identify dishes and do not date
images, and no route inside the release recovers either. Therefore:

* **Near-term use includes standalone segmentation, cross-sectional morphology,
  morphology modeled across area-coverage levels, and validated frame-indexed
  measurements.** This analysis is image-level and does not supply
  chronology or elapsed time.
* **Do not fit causal temperature effects, causal medium effects, or physical
  growth-rate models (mm/day, lag time, saturation) to MyceliumSeg.**
* **Time-resolved and intervention claims fall back to the Fricker/Boddy grazer
  series, the *Podospora* movies, or a controlled rig** (§ Fallback).
* **This decision must be complete before any MyceliumSeg output is interpreted as a
  temporal growth curve** (§ Preconditions).

This is a decision record, not a request to discard MyceliumSeg: it stays the anchor
dataset for segmentation, measurement-error calibration, and cross-sectional
morphology (PLAN.md § 0, Phase 1).

---

## 1. Evidence

| Check | Issue | Result |
|---|---|---|
| Capture time (EXIF/XMP/IPTC/COM/vendor tags) | #21 | **0 / 20,176** images carry any acquisition signal. Every file has an identical 623-byte bare-JFIF header with no APP1 segment at all, so the absence is conclusive. PNG mask text/`tIME` chunks: 0 / 567. |
| Zip entry mtimes | #21 | **Export sessions, not capture times.** 9 calendar dates across 11 archives, bursts at copy throughput (up to 6.6 img/s), a single `com.apple.quarantine` instant (2025-10-08 02:25:03 UTC), uncorrelated with file numbering. |
| Sequential numbering → time or dish? | #22 | **Neither.** Forward steps ID→mtime 0.489–0.547 in the nine archives of ≥ 507 images (the two 30-image labeled archives sit at 0.79 and 0.90 because they were copied folder-by-folder, #21 §5); forward steps ID→bright 0.41–0.59 across all eleven; brightness autocorrelation at lag 1 between −0.075 and 0.057. Verdict in `triage_numbering.md`: "Numbering encodes: NEITHER". |
| Dish tracks from the images | #23 | **Not recoverable.** Rim dot detected in 81.8% of images but fills 720/720 half-degree bins and the largest shared angle is 37 of 16,513 detections (0.22%); inoculum position median 0.028 R from centre with resultant 0.15; rim embeddings — all 80 pairs above 0.99 of 203,525,400 are byte-identical duplicates; colony size vs file order ≈ 50% forward steps. |
| Per-image stage / sclerotium-colour labels | #24 | **Absent.** Paper Tables 2–3 are aggregate counts only; 0 images have stage or colour labels. Species/medium unknown per image for the 1,689 `unlabeled-GS_PO_TS` images. |
| Duplicate and mislabelled files | #23 §5 | **133 / 20,176 byte-identical images.** `PO` ≡ `MYG`, `PDA` ≡ ten `labeled-GL` images (20 folder-derived species/medium labels are wrong), 29 of 30 labeled images reappear in the flat shard, and 22 frames are carried twice under two ids inside `unlabeled-GS_PO_TS.zip`. `TEMP15` is the one clean block. |
| px → mm scale | #25 | **Recoverable, and it is.** `data/triage/calibration.csv` + `image_scale.csv`: one constant per condition block/setup (6.0444 / 6.6222 / 4.4444 px/mm @900; global constant errs by +35.9% on the GS block, i.e. areas 46% too small). 560 images (2.8%) flagged `off_scale`; 1,728 images are 1382×1037 rather than 4608×3456, so two resolutions exist. |

Two facts follow directly, and they are the whole basis of this gate:

1. **Dish identity is not recoverable.** Nothing in the release assigns an image to a
   physical dish: no dish id, no fingerprint that separates dishes, no replicate count.
2. **Capture times are not recoverable.** No timestamp survives in any container, the
   zip mtimes are packaging dates, and the file numbering is shuffled relative to both
   export time and growth progression. A frame index is therefore an *index*, not a
   time coordinate.

The only remaining route to true metadata would be the authors. Issue #70 was
closed by user decision because the required information is unavailable to the
user; no email was sent. Gate C remains in force unless verifiable metadata arrive.

---

## 2. Gate outcome

| Gate | Condition | Outcome |
|---|---|---|
| A | Dish **and** time recoverable | Longitudinal growth-curve model (best case) — **not this** |
| B | Only time or only dish | Pooled growth curves with weaker identification — **not this** |
| **C** | **Neither** | **This is the decision:** cross-sectional and frame-indexed work only, with the fallback below |

All in-data routes to A or B were tested and failed (#21–#23), so the gate is **C**.

---

## 3. Consequences for each estimand

| Estimand | Status | Why |
|---|---|---|
| Standalone mycelium mask segmentation; downstream mask/area statistics and measurement error | **Allowed** | Train from released image/mask pairs and evaluate on held-out masks; area fraction is calculated after segmentation and needs no time or dish sequence |
| Cross-sectional morphology by condition — area fraction, equivalent radius, perimeter, roughness, calibrated mm² | **Allowed, associational** | Condition labels and per-block scale exist; no time or dish needed |
| Other morphology features modeled across low-to-high `area_fraction` levels | **Allowed, associational** | Area fraction is an image-level coverage coordinate; it does not assign time, dish identity or sequence order |
| Frame-indexed change (Δ per frame, boundary-event counts) | **Allowed where order is validated** (#61), reported in **change-per-frame** units | Order within an image id is an assumption to be validated, not a capture interval |
| Radial growth rate (mm/day), lag τ, saturation at 45 mm | **Not identifiable — do not fit** | Requires dish + time; neither exists |
| Causal temperature effect (15 vs 25 °C) | **Not identified — do not fit** | Only 10 images at 15 °C, no dish identity, no replication count, no time axis; at most a cross-sectional association |
| Causal medium effect (MYG vs PDA) | **Not identified at all — do not estimate** | Medium is aliased with species by design (GL/GS on MYG, PO/TS on PDA), and 20 folder-derived labels are wrong |
| Growth-stage or sclerotium-colour effects | **Not estimable per image** | Labels do not ship per image; stratifying on stage would also condition on a downstream variable |
| Dish-level varying effects | **Not estimable** | No dish identity → no grouping → no per-dish random effects |

**Selection bias (#35).** The paper's Methods / Data collection section
*describes* the collected release — "We collected 20,176 mycelium images with
distinctive edge morphology" — but documents **no** edge-morphology inclusion
rule (Yuan et al. 2025, doi:10.1038/s41597-025-06265-1; PLAN.md §0). Under
Gate C assumption #35 is therefore recorded with two unresolved statuses, and
that is a scope fact, not a new estimand: morphology-based selection is
**undocumented** (no candidate / included / excluded frame, so the release
cannot be shown to over-represent the estimands' own features and the bias
direction is unknown), and dependence on latent time/frame is **untestable**
(no dish identity, no timestamps, no ordered sequences — § 1 above; the paper's
periodic fixed-position acquisition is process-level context, not per-image
linkage). The consequence is bounded scope: every morphology summary describes
the selected release and no selection model is identifiable. Testing selection
dependence would need the documented inclusion rule, the candidate / included /
excluded image frame, and linked dish/time or stage metadata. The prohibitions
below are unaffected. Full record: `notes/assumptions.md` §5.

---

## 4. Permitted near-term work

1. **Standalone segmentation:** train an image-to-mask model from the released
   image/mask pairs to mark the mycelium region inside the dish; evaluate on labeled
   hold-outs after de-duplicating (across archives *and* within
   `unlabeled-GS_PO_TS.zip`). The model predicts masks only; it makes no growth claim.
2. **Cross-sectional morphology:** image-level and mask-level distributions of area,
   area fraction, equivalent radius, perimeter, roughness and intensity, calibrated
   with `data/triage/calibration.csv`, reported as associations with species and
   known condition — with species and medium kept separate in interpretation because
   they are aliased.
3. **Coverage-state morphology:** after full-release segmentation and QA, rank or bin
   de-duplicated images by area fraction within known species/condition/scale strata;
   model other morphology features against coverage. Treat this as a cross-section of
   the selected release, not a chronology or same-dish trajectory.
4. **Frame-indexed measurements:** change per frame and boundary-event counts, only
   for sequences whose order is independently validated, and only in per-frame units.
5. **Provenance discipline:** every row keeps archive, image id, duplicate status,
   condition source (folder-derived vs derived), the calibration constant and its
   `fit_status`, and whether a physical time interval is known (it is not).

**Prohibited regardless of how the numbers look:** fitting causal temperature or
medium effects; fitting physical growth-rate models; calling any sequence of
MyceliumSeg images a temporal growth curve; inferring elapsed time or same-dish
trajectories from file numbering, zip mtimes, or area-fraction rank; reporting candidate
boundary events as biological hyphal tips. These limits do not prohibit segmentation
training or a cross-sectional model of morphology versus area coverage.

---

## 5. Fallback for time-resolved and intervention claims

| Need | Go to | Why it works |
|---|---|---|
| Longitudinal growth with real time | **Fricker/Boddy grazer series** (Zenodo 14931953, 10.2 GB) | Explicit capture times, replicate identity, processed network data + R code; use area/extent over time (cord-forming, not circular) |
| Grazer-treatment contrast | **Fricker/Boddy** again | Treatment labels and replicate-linked time series are available; the 2011 methods say trays were repositioned randomly each week, but do not document randomized treatment allocation, so causal interpretation requires additional assignment/exchangeability evidence |
| M0/M2 nutrient-condition contrast at hyphal scale | ***Podospora anserina* movies** (Figshare 22666138) | Three movies per condition; Figshare specifies 18 min/frame, 1.63 µm/source-pixel, and CC BY 4.0. The movie comparison is descriptive because treatment assignment is not documented; see `notes/podospora_growth_model.md`. |
| Species × medium causal contrast under our own control | **Controlled rig** (RPi camera + LED timelapse, `/mnt/sharedOs/archive/mycelium-workspace`) | Cross species × medium (e.g. oyster on both PDA and MYG) breaks the confound; randomize dish position, log temperature, fixed-interval imaging, pre-register the predicted rate ratio; publishing template: Harrington & Harrington *Physarum* scanner set (Zenodo 22868368) |

The author-metadata route (#70) was closed by user decision because the required
information is unavailable to the user; no email was sent. If dish identity and a
capture interval are supplied in the future, this gate can be reopened and the record
amended — that would move the project to A or B, and only then does a MyceliumSeg
growth curve become a legitimate target.

---

## 6. Preconditions before interpreting MyceliumSeg output as a growth curve

These preconditions apply to temporal growth curves. They are not preconditions for
training/evaluating a segmenter or modeling morphology across area-coverage levels. All
five must be true for a temporal interpretation. Today, none is.

- [ ] Capture interval known for the images in question (not file numbering, not zip mtime)
- [ ] Dish identity known, with a per-condition replicate count
- [ ] Frame order validated independently of the id number
- [ ] Condition labels verified (the 20 mislabelled images fixed or dropped)
- [ ] Measurement error from the labeled masks attached to every derived quantity

Until every box is ticked, MyceliumSeg results may include segmentation,
cross-sectional morphology, area-coverage relationships, or validated frame-indexed
change — associational, never a temporal growth rate.

---

## Reproducing

```bash
.venv/bin/python scripts/triage/exif_scan.py --jobs 11    # #21, ~10 s
.venv/bin/python scripts/triage/dish_tracks.py --report-only   # #23 report from stored CSV/NPZ
.venv/bin/python scripts/triage/label_triage.py           # #24
.venv/bin/python scripts/triage/calibrate.py              # #25, ~1 s
.venv/bin/python -m pytest tests -q                        # full suite
```

Full evidence and flags: `notes/triage_exif.md`, `notes/triage_numbering.md`,
`notes/triage_dish_tracks.md`, `notes/triage_labels.md`, `notes/triage_calibration.md`.
Gate definitions and the surrounding plan: PLAN.md Phase 1.
