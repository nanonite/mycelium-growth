# Observable measurement table from the validated masks (#30)

**Date:** 2026-10-07  
**Status:** implemented — one provenance-rich row per #29 image, with
the primary size features separated from the secondary boundary and
intensity features  
**Evidence:** #28 (validated checkpoint), #29 (inference outputs),
#31 (measurement-error model), #60 (measurement contract), #61
(provenance)  
**Machine-readable:** `scripts/measurements.py`,
`tests/test_measurements.py`,
`notes/measurements_evidence.json`, `data/measurements.csv`  
**Blocks:** #41 (any growth model)  

PLAN.md Phase 2 step 4: per image, extract colony area, area fraction, calibrated equivalent radius, perimeter, compactness or roughness, intensity summaries and density proxies, and write one provenance-rich `data/measurements.csv` with image-level and mask-level provenance kept distinct. This note documents the table; it does not retrain, and it does not infer elapsed time, a growth rate, dish identity or a biological tip.

## 1. Inputs and guards

| Fact | Value |
| --- | --- |
| Checkpoint | #28 GPU baseline `iter_50000.pth` (SHA-256 `bcf66b6c…`) |
| Predicted masks | #29 inference outputs (`work_dirs/…/inference/masks/`) |
| Images | #29 inference inputs (released 4608-px JPGs + the unlabeled sample) |
| Working width | 900 px (the contract's boundary-feature width) |
| Retraining | none — `model_meta.json` must record the #28 SHA-256 |
| Perimeter | 4-connected boundary-pixel count (city-block), as in #31 |

The driver refuses to run if `model_meta.json` does not record the #28 checkpoint (`--allow-other-checkpoint` overrides), and it refuses any row that grows an elapsed-time, rate, dish-id or hyphal-tip field (Gate C). Every row keeps `frame_order_status = not_demonstrated` and `time_unit_status = no_capture_interval`.

## 2. Population

| Group | N |
| --- | --- |
| Images measured (one row each) | 125 |
| Calibrated (`fit_status = ok`) | 122 |
| Off-scale (calibrated estimands excluded) | 3 |
| &nbsp;&nbsp;source set `conditions` | 60 |
| &nbsp;&nbsp;source set `test` | 50 |
| &nbsp;&nbsp;source set `unlabeled` | 15 |
| &nbsp;&nbsp;block `GL` | 58 |
| &nbsp;&nbsp;block `GS` | 10 |
| &nbsp;&nbsp;block `GS_PO_TS_flat` | 7 |
| &nbsp;&nbsp;block `MYG` | 10 |
| &nbsp;&nbsp;block `PDA` | 10 |
| &nbsp;&nbsp;block `PO` | 10 |
| &nbsp;&nbsp;block `TEMP15` | 10 |
| &nbsp;&nbsp;block `TS` | 10 |

The 125 rows are the #29 inference population: 50 labeled-GL test images, 60 conditions hold-outs (GS/PO/TS/MYG/PDA/TEMP15) and the 15-image auditable unlabeled sample. Three images (`00018438` PO, `00018456` TS, `00018458` MYG) are `off_scale`: their calibrated estimands (mm² area, mm radius, mm perimeter, area fraction, ln(mm²)) are excluded per the #60 inclusion rule, while their px² and dimensionless shape features are kept.

## 3. Feature tiers and the #31 uncertainty they carry

The #60 contract assigns each estimand a status; the #31 error model quantifies how much error each feature carries on the de-duplicated held-out masks. The two agree on what is primary and what is secondary:

| Tier | Features | #31 error model |
| --- | --- | --- |
| **primary** | `area_px`, `area_mm2`, `area_fraction` | low bias, but 14.3% relative SD exceeds the #60 size target of ≤10%; target **unmet** (coverage-stratified audit: `notes/area_fraction_precision.md`, #66) |
| **primary** | `equiv_radius_px`, `equiv_radius_mm`, `log_area_px2`, `log_area_mm2` | bias and spread inside the #60 size budget (relative bias ≤ 3%, relative SD ≤ 10%) |
| **secondary** (boundary) | `perimeter_px`, `perimeter_mm`, `edge_roughness`, `circularity`, `solidity` | error well outside the size budget (published Boundary IoU 28.60%, #28); they carry the looser boundary budget and are not size proxies |
| **secondary** (density) | `mean_intensity`, `intensity_median`, `intensity_std`, `intensity_min`, `intensity_max` | density proxy only; comparable within a condition block, never across |

Reading: equivalent radius and log-area meet the size target. Area and area fraction are near-unbiased on average but have 14.3% relative SD, above the 10% target. The sorted coverage coordinate remains useful, while per-image percentages are approximate. Perimeter and boundary-derived shape features carry much larger, heavy-tailed error, so they are published only as shape descriptors at the fidelity the boundary metrics allow. `mean_intensity` is a density proxy only: dark incubation and fixed exposure are per-block facts.

## 4. Column dictionary

Provenance columns (image-level and mask-level) are carried from the #29 records verbatim; the table joins on `id` with `data/triage/measurement_error.csv` (#31) and `data/triage/inference_explore.csv` (#29).

| Column | Tier | Definition |
| --- | --- | --- |
| `area_px` | primary | colony mask area in px^2 at the 900-px working width |
| `area_mm2` | primary | calibrated colony area in mm^2 (fit_status == ok only) |
| `area_fraction` | primary | A_colony / A_dish, both in the same frame (fit_status == ok only) |
| `equiv_radius_px` | primary | sqrt(area_px / pi) at the 900-px working width |
| `equiv_radius_mm` | primary | calibrated equivalent radius in mm (fit_status == ok only) |
| `log_area_px2` | primary | ln(area_px) -- log area with its reference unit stated |
| `log_area_mm2` | primary | ln(area_mm^2) -- calibrated log area (fit_status == ok only) |
| `perimeter_px` | secondary | 4-connected boundary-pixel count at the 900-px working width |
| `perimeter_mm` | secondary | calibrated perimeter in mm (fit_status == ok only) |
| `edge_roughness` | secondary | P / (2 pi r_eq) -- 1.0 is a circle; resolution-locked |
| `circularity` | secondary | 4 pi A / P^2 -- dimensionless, resolution-locked |
| `solidity` | secondary | A_colony / A_convex_hull -- dimensionless |
| `mean_intensity` | secondary | mean 8-bit grey level inside the mask -- density proxy |
| `intensity_median` | secondary | median 8-bit grey level inside the mask |
| `intensity_std` | secondary | SD of the 8-bit grey level inside the mask -- texture proxy |
| `intensity_min` | secondary | minimum 8-bit grey level inside the mask |
| `intensity_max` | secondary | maximum 8-bit grey level inside the mask |

## 5. What this table supports and what it does not

**Supported (image-level morphology, this release):**

- `equivalent_radius` and `log_area` clear the size budget on the de-duplicated hold-outs. `area_fraction` still supports approximate low-to-high coverage sorting and cross-sectional indexing, but its 14.3% held-out relative SD exceeds the #60 target; see the completed #66 audit in `notes/area_fraction_precision.md`.
- `area_fraction` can also index low-to-high area-coverage levels for cross-sectional models of other features. The current table has 125 rows, so a full-release inference and QA pass is needed before treating its ranks as the dataset-wide coverage distribution.
- `perimeter`, `edge_roughness`, `circularity` and `solidity` are published with their (larger) boundary budget; they describe shape only at the fidelity the boundary metrics allow and are not size proxies.
- `mean_intensity` and the intensity summaries are density proxies, comparable within a condition block only.

**Not supported (Gate C, #60 § 9):**

- any temporal growth curve, elapsed time, rate per unit time, dish identity or timelapse membership — there is no capture interval and no demonstrated frame order. Area-fraction rank is a coverage coordinate, not chronology;
- any hyphal-tip, nuclear or branching claim — #62 assesses candidate stability under resampling and finds substantial mask disagreement; the colony masks are not independent annotations of biological tips;
- any causal temperature or medium effect — medium is aliased with species, only 10 images are at 15 °C, and there is no dish identity or replication;
- any physical growth rate in mm/day — a frame index is not a duration, and `to_physical_rate()` refuses the conversion.

## 6. Reproduce

```bash
# measurement table from the saved masks (no GPU, no retrain)
nix develop --command python scripts/measurements.py run

# tests and lint
.venv/bin/python -m pytest tests -q
nix develop --command ruff check scripts/measurements.py tests/test_measurements.py
```

Surrounding record: `notes/measurement_error.md` (#31, the error model), `notes/segformer_inference.md` (#29, the masks), `notes/estimands.md` (#60, the contract and uncertainty target), `notes/provenance.md` (#61, the capability matrix).
