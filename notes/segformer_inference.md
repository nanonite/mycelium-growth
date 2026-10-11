# Inference on labeled hold-outs and an auditable unlabeled sample (#29)

**Date:** 2026-10-06
**Status:** implemented — masks and provenance produced with the validated #28
checkpoint, plus the exploratory area/area-fraction smoke test
**Evidence:** #28 (validated SegFormer-B0 checkpoint), #60 (measurement
contract), #61 (provenance specification), #26 (Gate C), #21–#25 (triage)
**Machine-readable:** `scripts/segformer_inference.py`,
`tests/test_segformer_inference.py`, `notes/segformer_inference_evidence.json`,
`data/triage/inference_explore.csv`
**Blocks:** #30 (measurement table) and #31 (measurement-error model)

Phase 2 step 3 of [PLAN.md](../PLAN.md): after the estimand and provenance
contracts and the validated baseline, run inference **first** on the labeled
hold-outs and a small, fully auditable sample of the unlabeled release, keep
each row's provenance, and produce masks for observable morphology.  It does
not retrain, and it never treats the file numbering as elapsed time.

---

## 1. What runs

`scripts/segformer_inference.py` loads the #28 GPU baseline directly:

| Fact | Value |
| --- | --- |
| Config | `work_dirs/segformer-b0-repro/full-gpu/config.py` (SHA-256 `e3d3e6f6…`) |
| Checkpoint | `work_dirs/segformer-b0-repro/full-gpu/train/iter_50000.pth` (SHA-256 `bcf66b6c…`) |
| Model | SegFormer-B0 (`mit_b0`), the authors' slide test config (crop 1024, stride 768) |
| Inference | `mmseg.apis.inference.inference_segmentor`, the same test pipeline the #28 eval used |
| Retraining | none — `predict` refuses a checkpoint whose SHA-256 is not the #28 hand-off (`--allow-other-checkpoint` overrides for a future model) |

The three input sets:

| Set | Images | Source | True mask? |
| --- | --- | --- | --- |
| `test` | 50 | `data/segmentation/release/test/` (GL testset, the paper's Table 6 hold-out) | yes |
| `conditions` | 60 | `data/segmentation/release/conditions/` (GS/PO/TS/MYG/PDA/TEMP15 robustness hold-outs) | yes |
| `unlabeled` | 15 | a deterministic sample of `unlabeled-GL2.zip` and `unlabeled-GS_PO_TS.zip` | no |

Predicted masks are saved as 0/1 `uint8` PNGs (the labeled-mask convention) under
`work_dirs/segformer-b0-repro/inference/masks/<set>/<id>.png`; the run also
writes `records.json`, `exploration.json` and `evidence.json` there.  Those
files are gitignored (masks are large); the committed record is the exploration
table plus the evidence JSON.

### The unlabeled sample is deterministic and auditable

`select_unlabeled_sample()` takes a fixed stride through two archives and adds
two deliberate byte-copy probes:

* `unlabeled-GL2.zip`: every 280th image, first 8 — block `GL`, known labels;
* `unlabeled-GS_PO_TS.zip`: every 420th image, first 5 — the flat shard whose
  species/medium are **unknown** (`known_labels.csv` blank), so the
  condition-unknown path is exercised;
* probes `00019773` and `00019301` — byte copies of the labeled conditions
  images `00018428` and `00018429` (#23 md5 groups), so the duplicate-status
  column is checked against a labeled hold-out rather than only within the
  unlabeled release.

Every sampled row records the exact zip member (`archive`, `entry`) it came
from, so the sample can be regenerated from the committed triage tables without
the raw zips.  `sample/sample_manifest.json` in the work directory stores the
selection rule.

## 2. Provenance preserved per row

`build_provenance()` joins the committed triage tables — `dish_tracks.csv`
(archive, entry, frame size, dish-disc geometry, md5), `image_scale.csv`
(block, px/mm, `fit_status`), `known_labels.csv` (species/medium/temperature)
and the `myceliumseg` row of `sequence_inventory.csv` (capability).  Every
record carries:

| Column group | Fields | Source |
| --- | --- | --- |
| Archive / image | `archive`, `entry`, `id` | `dish_tracks.csv` |
| Duplicate status | `md5`, `duplicate_role` (`representative`/`copy`), `duplicate_group_size`, `duplicate_representative`, `label_conflict` | `dish_tracks.csv`, #60 § 7 |
| Condition | `species`, `medium`, `temp`, `condition`, `condition_known`, `condition_source` | `known_labels.csv` |
| Block / scale | `block`, `scale_group`, `px_per_mm`, `fit_status`, `dish_disc_radius_900` | `image_scale.csv`, `dish_tracks.csv` |
| Frame order | `frame_order_status` = `not_demonstrated`, `ordered_frames` = `no` | `sequence_inventory.csv`, #61 |
| Time unit | `time_unit_status` = `no_capture_interval`, `capture_interval` = `none` | `sequence_inventory.csv`, #61 |
| Estimand capability | `estimands_identifiable` = `image_morphology`; `estimands_blocked` = `delta_per_ordered_step;physical_rate;intervention_effect` | `sequence_inventory.csv` |

A `label_conflict` is an md5 group whose copies disagree on a non-blank label
(the `PO`≡`MYG`, `PDA`≡`labeled-GL` collisions of #23): condition follows the
md5 map, never the folder name.  `assert_no_inferred_time()` is a hard guard:
a record that grows an `elapsed_time`, `growth_rate`, `dish_id` or
`hyphal_tip_count` field fails the run.

## 3. Area and area fraction

Masks are measured with `derive_measurements()`, under the #60 contract:

| Quantity | Definition | Precondition |
| --- | --- | --- |
| `predicted_area_px` | foreground pixel count | always |
| `predicted_area_mm2` | `A_px / (px_per_mm · W/900)²` | px/mm present |
| `predicted_area_fraction` | `A_px / (π · (dish_r_900 · W/900)²)` | `fit_status == ok` and a readable disc |
| `predicted_equiv_radius_px/mm` | `√(A/π)` | as above |
| `predicted_log_area_px2/mm2` | `mc.log_area(A, reference_unit=…)` | positive area |

`W` is the frame width the mask was counted in.  `dish_tracks.csv` and
`image_scale.csv` publish geometry at the 900-px working width, so the disc
radius and the px/mm constant rescale linearly with `W/900` and area with its
square.  An `off_scale` image keeps its px² numbers and reports the calibrated
ones as null, with `area_exclusion` naming the reason — the inclusion rule of
`notes/estimands.md` § 2.4.  For the labeled hold-outs the record also carries
`reference_mask_area_px` and `predicted_over_reference_area` as a smoke sanity
check; the bias/SD measurement-error model is **#31**, not this task.

## 4. Area-fraction rank and the exploratory smoke test

`explore_records()` sorts the inferred `area_fraction` **inside provenance-safe
`(archive, block)` strata** — never across archives or scale blocks, because a
byte copy or a mislabelled folder would otherwise look like a real difference.
The current 125-image output uses this rank for smoke-test inspection. After
full-release inference, duplicate handling and label/scale QA, the same area
fraction can serve as an area-coverage coordinate for cross-sectional models of
other features. It is not an acquisition-time or same-dish sequence.
Inside each stratum it reports:

* the full sorted order with ranks;
* **gap-detected bands**: a break sits where the neighbour gap exceeds
  `gap_factor` (default 4) times the median positive gap.  With fewer than two
  positive gaps no break is claimed — for a small sample the honest answer is
  one band.  These gap bands are a descriptive smoke check, not a clustering model;
* **duplicate groups** (shared md5), with whether the predicted mask hashes are
  identical — the byte-copy check;
* **consecutive-id runs** — maximal runs of adjacent release ids.  These are an
  index/export artefact candidate, **not** evidence of acquisition time (zip
  mtimes are export sessions, #21).

The committed table is `data/triage/inference_explore.csv`; the JSON is
`exploration.json`.  Every row carries the provenance of § 2, and the CSV
writer refuses any column whose name contains `elapsed`, `timestamp`,
`growth_rate`, `per_day`, `per_hour`, `dish_id`, `timelapse` or `hyphal_tip`.

### Results

The full run inferred **125 images** — 50 `test`, 60 `conditions`, 15 `unlabeled`
— with no empty masks, and sorted them into **9 provenance-safe
`(archive, block)` strata**.  The committed table is
`data/triage/inference_explore.csv` (125 rows); the evidence JSON is
`notes/segformer_inference_evidence.json`.

**Hold-out sanity check (smoke QA only, not the #31 error model).**  All 110
labeled hold-outs have a true mask to compare against.  The predicted/reference
area ratio has median **1.008** (GL test 1.001, conditions 1.020), 90/110 within
±10% and 104/110 within ±25%; the range is 0.28–1.80, i.e. the segmenter is
unbiased on the median but has a heavy tail on individual images — exactly the
bias/SD question #31 will model, and not a quality gate here.

**Off-scale exclusions.**  Three hold-out images (`00018438` PO, `00018456` TS,
`00018458` MYG) have `fit_status = off_scale`: their calibrated `area_fraction`
and `area_mm2` are reported as null with `area_exclusion = fit_status=off_scale`
and their px² numbers are kept, per the #60 inclusion rule.  The other 122 rows
carry an `area_fraction`.

**Duplicate groups — the headline exploratory finding.**  The 125 rows form
**22 byte-identical groups**, and in **every one the predicted-mask SHA-256 is
identical**, which both reproduces #23 and confirms inference is deterministic
on identical pixels:

| Groups | Collision | Consequence |
| --- | --- | --- |
| 10 | `PO` (labeled-GS_PO_TS) ≡ `MYG` (labeled-MYG_PDA_TEMP) | the PO/MYG condition contrast is aliased, not two conditions |
| 10 | `PDA` (labeled-MYG_PDA_TEMP) ≡ ten `labeled-GL` **test** images | the conditions hold-out is **not independent of the GL test set** |
| 2 | the deliberate probes `00019773`≡`00018428`, `00019301`≡`00018429` | the cross-archive duplicate path is exercised and flagged |

62 of the 125 rows sit in an md5 group whose copies disagree on a non-blank
label, so their `label_conflict` flag is set and their folder-derived condition
is not trustworthy (condition follows the md5 map, #60 § 7).  Duplicate copies
are one observation, not two, and must be reduced to the representative before
any uncertainty is computed.

**Gap-detected bands** (exploratory; `gap_factor = 4`).  The labeled `GL` test
stratum sorts into 7 bands (0.030–0.664), `TEMP15` into 3 (0.004–0.176),
`PDA` into 2 (0.030–0.112 and 0.360–0.557), `TS` into 2, `GS` into 2, and the
stride sample of `unlabeled-GL2` into 3 (0.019–0.796); `MYG`, `PO` and the flat
shard each show one band.  These bands are candidate morphology/selection
structure inside a single archive × block — they are **not** time, growth or
condition effects, and the blocks are small (n = 7–50).

**Consecutive-id runs.**  `id_run` is non-empty only inside the labeled
hold-outs (their folder extraction is id-contiguous); the strided unlabeled
sample has none.  A run is an index/export artefact candidate, not evidence of
acquisition time.

Every row records `frame_order_status = not_demonstrated` and
`time_unit_status = no_capture_interval`; nothing in the sort was converted to
time.


### What the exploratory groupings do and do not mean

A coherent band, a duplicate group or an id run is a **hypothesis about the
data-processing pipeline**, not a biological finding:

* a band can be an acquisition/selection artefact, a condition block boundary,
  or a real size difference — the cross-sectional data cannot tell them apart;
* an id run is a property of the release's numbering, and the numbering is
  shuffled relative to time (#22);
* a duplicate group is one observation, not two, and must be reduced to its
  representative before any uncertainty is computed (#60 § 7).

The sorted order is meaningful as coverage order. It does not identify elapsed
time, dish identity, timelapse membership, a same-dish trajectory, a growth rate,
or a biological hyphal tip.

## 5. What this supports and what it does not

Supported: predicted masks and image-level morphology (area, area fraction,
equivalent radius, log area) for the hold-outs and the 125-image sample, each row
carrying its archive, duplicate status, condition label, block, frame-order status
and time-unit status; a held-out smoke comparison against the true masks. After
full-release inference and QA, area-coverage models may use area fraction to
predict other image features within the selected release.

Not supported (Gate C, `notes/gate.md`): any temporal growth curve, elapsed time, rate,
dish identity, timelapse membership, causal temperature/medium effect, or
biological tip event.  Boundary IoU is weak by construction (published 28.60%)
and HD95/ASSD are in pixels (#28), so mask QA carries both caveats and the
boundary features stay out of this task.

## 6. Reproduce

```bash
# Full run: 50 + 60 labeled hold-outs + 15 unlabeled, masks + records +
# exploration + evidence (~6 min on the RTX 4070 Ti)
nix develop .#gpu --command env PYTHONPATH=scripts/segformer_shim \
    python scripts/segformer_inference.py run

# A two-image-per-set smoke of the same path
nix develop .#gpu --command env PYTHONPATH=scripts/segformer_shim \
    python scripts/segformer_inference.py run --limit 2

# Re-run just the pure-python half from an existing records.json
nix develop --command python scripts/segformer_inference.py explore
nix develop --command python scripts/segformer_inference.py evidence
```

Tests and lint (no GPU needed):

```bash
.venv/bin/python -m pytest tests -q
nix develop --command ruff check scripts/segformer_inference.py \
    tests/test_segformer_inference.py
```

Surrounding record: `notes/segformer_baseline.md` (#28, the checkpoint),
`notes/estimands.md` (#60, the contract), `notes/provenance.md` (#61, the
capability matrix), PLAN.md Phase 2 steps 3–5.
