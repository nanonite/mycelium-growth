# Measurement contract: observable growth estimands and units (#60)

**Date:** 2026-10-05
**Status:** defined — the contract every reported number has to satisfy *before* inference
**Evidence:** #21 (EXIF), #22 (numbering), #23 (dish tracks + duplicates), #24 (labels), #25 (px→mm), #26 (Gate C)
**Sibling:** [provenance specification](provenance.md) (#61) — this note says
*what* is estimated, `notes/provenance.md` says *where each row came from*
**Blocks:** the measurement table and error model (PLAN.md Phase 2 steps 4–5) and any estimand in Phase 3

**Contract.** Before any inference, every number that will be reported carries a
name, a definition, a unit, a population, an inclusion rule, a duplicate rule, a
calibration and an uncertainty target. Two families of estimand exist and no
others: an **image-level morphology distribution** (primary) and, only where
frame order has been demonstrated, a **change per ordered frame** (conditional).
Nothing in this contract converts a frame change into a rate per unit time.

Machine-readable mirror: `scripts/measurement_contract.py` (`CONTRACT`,
`IMAGE_ESTIMANDS`, `FRAME_ESTIMANDS`, and the guards `log_area`,
`frame_delta_log_area`, `to_physical_rate`). The two halves are kept honest by
`tests/test_measurement_contract.py`, which fails if this note and the module
disagree on a name, a unit, a rule or a number.

---

## 1. Population

The estimand is a property of a **finite, enumerable set of images**, not of a
sample of dishes. Gate C (#26) established that nothing in the release assigns
an image to a dish or a date, so there is no sampling frame to generalise over.

| | Count | Source |
|---|---|---|
| Released JPGs (11 archives, 90-mm dishes) | 20,176 | release manifest |
| Labeled masks (507 GL + 10 each GS/PO/TS/MYG/PDA/TEMP15) | 567 | labeled zips |
| Unlabeled images | 19,609 | release manifest |
| Species **and** medium known per image (`known_labels.csv`) | 18,487 | folder + paper Table 1 |
| Species/medium unknown (flat `unlabeled-GS_PO_TS` shard) | 1,689 | blank in `known_labels.csv` |
| Byte-identical duplicates (`data/triage/dish_tracks.csv`) | 133 in 62 md5 groups | #23 §5 |
| Distinct images after de-duplication | 20,105 | 20,176 − 71 repeated copies |
| Images flagged `fit_status = off_scale` | 560 (2.8%) | `image_scale.csv` |

**Scope of inference.** Summaries describe *this release*: associational,
within-release, by species and known condition. The paper describes collecting
images "with distinctive edge morphology" and documents no inclusion rule, so
the release is not a random sample of plates. Nothing here estimates a species
population, a treatment effect or a growth curve.

**Selection bias (#35).** The release's Methods / Data collection section
*describes* the collected set — "We collected 20,176 mycelium images with
distinctive edge morphology" (Yuan et al. 2025,
doi:10.1038/s41597-025-06265-1) — but documents **no** edge-morphology inclusion
rule, and ships no per-image dish, time, order or stage link (full record:
`notes/assumptions.md` §5; machine-readable: `SELECTION_BIAS` in
`scripts/measurement_contract.py`). So neither half of the independence
assumption is settled: morphology-based selection is **undocumented** (the
release cannot be shown to over-represent the estimands' own features, and the
bias direction is unknown) and dependence on latent time/frame is **untestable**
under Gate C (no dish identity, no timestamps, no ordered sequences; the paper's
periodic fixed-position acquisition is process-level context, not per-image
linkage). Testing either would need the documented inclusion rule, the
candidate / included / excluded image frame, and linked dish/time or stage
metadata. Every summary under this contract therefore carries the scope label
**selected release**, never "plates" or "population".

## 2. Inclusion rule

An image enters an estimand when all of the following hold; each exclusion is
counted and published next to the summary it affects.

1. It is in the release **and** has a segmentation to measure (labeled mask or
   inference output).
2. It is the **representative of its duplicate group** (§ 7).
3. Any stratum it is reported by comes from `known_labels.csv` (folder + paper
   Table 1). The 20 mislabelled images — `PO` ≡ `MYG`, `PDA` ≡ ten
   `labeled-GL` images (#23) — are corrected from the md5 map or dropped, never
   taken at folder value.
4. Anything that reads the dish disc or the px→mm scale — `area_fraction`,
   `log_area` in mm², `equivalent_radius`, `perimeter` — additionally requires
   `image_scale.csv` to say `fit_status = ok`. The 560 `off_scale` images (2.8%)
   have a fitted dish disc that disagrees with their block, which is exactly the
   denominator and the scale these estimands read, so they are excluded from
   those four and keep their flag; they stay in the ratio shape features,
   `mean_intensity`, and `log_area` in px².
5. Boundary-derived shape features are measured at the **900-px working width**,
   the resolution the calibration was fit at.

**Reporting:** every summary publishes N included and N excluded, per estimand
and per stratum. An exclusion that moves a headline number is a finding, not a
housekeeping detail.

## 3. Primary image-level estimands

The primary estimand is the **image-level morphology distribution**, stratified
by species (GL/GS/PO/TS), by known condition (medium, temperature — where the
release labels them), and by condition block (the nuisance stratum the px→mm
calibration hangs off). Medium is aliased with species by design, so a medium
"effect" is not estimated — that is a property of the design, not of the
estimator (#26 §3).

| Estimand | Definition | Unit | Status |
|---|---|---|---|
| `area_fraction` | `A_colony / A_dish`, both read in the same frame (mask px over fitted 90-mm dish disc px); the ratio is identical in mm² and px² | 1 (also %) | primary |
| `log_area` | `ln(A_colony / 1 mm²)` of the calibrated mask area | ln(mm²) | primary |
| `equivalent_radius` | `r_eq = √(A_colony/π)` | mm | primary |
| `perimeter` | `P` of the mask boundary at the 900-px working width | mm | primary |
| `circularity` | `4πA / P²` | 1 | primary |
| `edge_roughness` | `P / (2π·r_eq)` — 1.0 = circle | 1 | primary |
| `solidity` | `A_colony / A_convex_hull` | 1 | primary |
| `mean_intensity` | mean 8-bit grey level inside the mask | grey level | secondary |

Why these: colony **area** is the quantity the release measures reliably
(boundary IoU is only 11–30% even for the best model, but area error is far
smaller — PLAN.md §0), `log_area` makes the right-skewed, multiplicatively
scaled distribution additive, and the shape features are the calibrated,
dimensionless descriptors Phase 3 asks for. `mean_intensity` is a density proxy
only: dark incubation and fixed exposure are per-block facts, so it is stratified
by block alone and never compared across blocks.

## 4. Conditional frame-level estimand

**`delta_log_area_per_frame`** — `ln(A_{f+1}/1 mm²) − ln(A_f/1 mm²)` for
consecutive frames of one sequence, **in demonstrated frame order**; unit
*log units per frame*; strata: species, condition, sequence.

It is *conditional*, and the condition is evidence, not an assumption:

- **Precondition:** frame order demonstrated independently of the file numbering
  (PLAN.md Phase 2 step 3; `notes/provenance.md` defines the provenance
  evidence). For MyceliumSeg
  today the condition **fails**: forward steps ID→mtime are 0.489–0.547 and
  colony size vs file order ≈ 50% (#22, #23), so the estimand is *defined but
  not estimated* on this release.
- **Unit stays per frame.** The denominator is one frame step, not a duration.
  `to_physical_rate()` raises without a measured capture interval, and Gate C
  says MyceliumSeg has none — so no mm/day, log/day or events/hour ever comes
  out of this release.
- **De-duplicate first.** Two byte-identical copies of one frame differenced
  against each other would report `Δ log(area) = 0` as "no growth"; § 7 removes
  them before any differencing.
- **Detectability.** A change is reported only when `|Δ|` exceeds 2 × the
  frame-to-frame repeatability SD (`min_reportable_delta`); below that it is
  measurement error, not a change.

If order cannot be demonstrated, the cross-sectional estimand of § 3 stands on
its own and is labelled associational — never a growth curve.

## 5. Units

| Quantity | Unit | Rule |
|---|---|---|
| Area `A` | mm² (calibrated) or px² (raw) | never mixed in one summary; px² always records the frame width |
| Area fraction | 1, also % | both areas read in the same frame, so the scale cancels — it still needs a readable dish disc (§ 2.4), and it is the `colony_area_frac` column `dish_geometry` already computes |
| Log area | ln(mm²) | reference unit mandatory: log(mm²) and log(px²) differ by `2·ln(px/mm)`; `log_area()` refuses a missing unit |
| Equivalent radius, perimeter | mm | from the calibrated scale; a length-scale error is squared by an area (§ 6) |
| Circularity, edge_roughness, solidity | 1 | dimensionless **but resolution-locked**: measured at the 900-px working width, and mixed widths are prohibited |
| Mean intensity | grey level (8-bit, 0–255) | comparable within a condition block only |
| `delta_log_area_per_frame` | log units per frame | never per unit time |
| Any rate | log per time / mm per time | requires a capture interval; unavailable for MyceliumSeg (#26) |

## 6. Calibration

Anchored on one physical fact: the dish rim is 90 mm across (#25).
`data/triage/calibration.csv` gives one constant per condition block/setup, and
`data/triage/image_scale.csv` assigns each image the constant it agrees with:

| Block / setup | px/mm @ 900 px | Notes |
|---|---|---|
| GL | 6.0444 | 18,427 images |
| GS | 4.4482 | one global constant is +35.9% here → **areas 46% too small** |
| GS_PO_TS_flat setup 1 | 6.6222 | 1,598 images |
| GS_PO_TS_flat setup 2 | 4.4444 | 91 images, second camera height |
| MYG / PO | 6.6345 | |
| PDA | 6.0419 | |
| TEMP15 | 6.0444 | |
| TS | 6.6507 | |

Rules: constants are for 900-px working images; px/mm scales linearly with frame
width (`rescale_px_per_mm`) and **area with its square** (`area_px_to_mm2`), so
the 4608-px originals go through the rescale rather than the constant. The
`px_per_mm_p25..p75` spread (0.0–1.7% in length) is carried into every mm²
quantity as 0.0–3.4% in area. Images flagged `off_scale` take their block's
prior and are flagged, not silently trusted.

## 7. Duplicate handling

- **Criterion:** byte-identical MD5, already recorded per image in
  `data/triage/dish_tracks.csv` — 133 images in 62 hash groups (#23 §5).
- **Rule:** one representative per group — the lowest `(archive, id)` — kept for
  estimation, for uncertainty, and for train/test splitting. The other 71 copies
  are counted and dropped; they never appear in a confidence interval and never
  land on both sides of a split.
- **Labels:** duplicates cross condition folders (`PO` ≡ `MYG`, `PDA` ≡ ten
  `labeled-GL` images), which is where the 20 wrong folder-derived labels come
  from — condition follows the md5 map, not the folder name.
- **Sensitivity:** headline numbers are re-run with every copy included to show
  they do not move (`notes/triage_calibration.md` already does this for the
  scale).
- **Sequence rule:** de-duplicate *before* differencing (§ 4).

`deduplicate()` in `scripts/measurement_contract.py` implements the rule and is
checked against the committed CSV: 20,176 rows in, 20,105 representatives out.

## 8. Uncertainty target

| Target | Limit |
|---|---|
| Interval level | 95%, on the **deduplicated** image (a duplicate group is one observation) |
| Size estimands (`area_fraction`, `log_area`, `equivalent_radius`, mm² area) | relative bias ≤ **3%**, relative SD ≤ **10%** on de-duplicated held-out labeled masks |
| Boundary estimands (`perimeter`, `circularity`, `edge_roughness`, `solidity`) | relative bias ≤ **5%**, relative SD ≤ **20%** |
| Frame-level change | report only `|Δ log(area)| > 2 ×` frame-to-frame repeatability SD |
| Calibration spread | carried as ±(p25, p75) of px/mm → ±2× that in area |
| Coverage label | intervals describe the release (associational, within-release), never a plate population |

Measured error is compared with these limits through
`meets_uncertainty_target(kind, ...)`: **both** bias and SD have to clear. On
failure the number is published with the target marked unmet — never dropped,
never rescued by retuning the threshold until it passes. These targets are the
acceptance bar for the measurement-error layer (PLAN.md Phase 2 step 5), which
is a separate deliverable: this note fixes the bar, not the measurement of it.

## 9. Reporting rules — what the contract forbids

1. A per-time rate (mm/day, log/day, events/hour) computed from per-frame change
   without a measured capture interval. `to_physical_rate()` raises instead.
2. Calling a MyceliumSeg sequence a growth curve, or inferring elapsed time from
   file numbering or zip mtimes.
3. Causal temperature or medium effects (Gate C): medium is aliased with
   species, only 10 images are at 15 °C, and there is no dish identity or
   replication count.
4. Stratifying on growth stage or sclerotium colour: no per-image labels ship
   (#24), and stage is downstream of growth, so conditioning on it biases the
   very quantity being estimated.
5. Boundary events reported as biological hyphal tips.
6. A log of area without its reference unit, px² and mm² areas in one summary,
   or shape features from mixed working widths.
7. Duplicate copies counted as independent observations anywhere.

## 10. What this contract does not claim

It does not claim that frame order exists in MyceliumSeg (it does not, #22), that
dish identity is recoverable (it is not, #23), or that any number here is a rate.
It defines what *would* be measured and under which precondition each estimand
becomes available; the preconditions that are not met today are recorded as not
met, not quietly assumed. Gate C (#26) remains the gate: when dish identity, a
capture interval and validated order are supplied by the authors, § 4 gains a
capture interval and this contract is amended alongside it.

---

## Reproducing

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); \
    import measurement_contract as m; print(m.validate_contract())"   # []
.venv/bin/python -m pytest tests/test_measurement_contract.py -q       # contract guards
.venv/bin/python -m pytest tests -q                                    # full suite
ruff check scripts/measurement_contract.py                             # lint
```

Surrounding record: `notes/gate.md` (Gate C), `notes/triage_calibration.md`
(px→mm), `notes/triage_dish_tracks.md` (duplicates, #23), PLAN.md Phase 2 steps
1 and 4–5.
