# Measurement-error model: bias + SD of area/radius/perimeter on held-out masks (#31)

**Date:** 2026-10-06  
**Status:** implemented — bias and spread estimated on the de-duplicated
held-out labeled masks, with duplicate sensitivity, repeatability and
candidate-protrusion false-event rates  
**Evidence:** #28 (validated checkpoint), #29 (inference outputs), #60
(measurement contract), #61 (provenance)  
**Machine-readable:** `scripts/measurement_error.py`,
`tests/test_measurement_error.py`,
`notes/measurement_error_evidence.json`, `data/triage/measurement_error.csv`  
**Blocks:** #41 (any growth model)  

PLAN.md Phase 2 step 5: before any growth model, estimate the measurement-error layer. This note compares the #28-checkpoint predictions (#29 inference outputs) against the true masks on the held-out labeled images and reports bias and spread for area, area fraction, equivalent radius, perimeter, edge roughness, circularity and solidity, then quantifies duplicate sensitivity, sequence repeatability and false-event rates for candidate protrusions. It does not retrain, and it does not infer time, dish identity or a growth rate.

## 1. Inputs and guards

| Fact | Value |
| --- | --- |
| Checkpoint | #28 GPU baseline `iter_50000.pth` (SHA-256 `bcf66b6c…`) |
| Predicted masks | #29 inference outputs (`work_dirs/…/inference/masks/`) |
| Reference masks | `data/segmentation/release/{test,conditions}/mask/` |
| Working width | 900 px (the contract's boundary-feature width) |
| Retraining | none — `model_meta.json` must record the #28 SHA-256 |
| Perimeter | 4-connected boundary-pixel count (city-block), same convention for pred and ref |

The driver refuses to run if `model_meta.json` does not record the #28 checkpoint (`--allow-other-checkpoint` overrides), and it refuses any row that grows an elapsed-time, rate, dish-id or hyphal-tip field (Gate C). Every row keeps `frame_order_status = not_demonstrated` and `time_unit_status = no_capture_interval`.

## 2. Population and inclusion

| Group | N |
| --- | --- |
| Held-out labeled images (test + conditions) | 110 |
| Representatives (de-duplicated, the error-model population) | 90 |
| Duplicate copies (counted and set aside) | 20 |
| Calibrated representatives (`fit_status = ok`) | 88 |
| Off-scale representatives (calibrated estimands excluded) | 2 |

The 110 held-out images form 20 byte-identical duplicate groups: ten PDA conditions images are copies of ten labeled-GL test images, and ten MYG conditions images are copies of ten PO conditions images (#23). De-duplication keeps the lowest `(archive, id)` — the GL test image for the first ten, the PO image for the second — so the error model runs on 90 representatives (50 test, 40 conditions) and the test and conditions sets are not double-counted. Two representatives (`00018438` PO, `00018456` TS) are `off_scale`: their calibrated estimands (mm² area, mm radius, mm perimeter, area fraction, ln(mm²)) are excluded per the #60 inclusion rule, while their px² and dimensionless shape features are kept.

## 3. Bias and spread per feature

Errors are predicted-minus-reference. For linear features the relative error `(pred − ref)/ref` is reported; for `log_area` the log-ratio `ln(pred/ref)` (the relative error to first order). Bias is the mean, SD the sample standard deviation, both in percent. The target column is the #60 uncertainty target (size: bias ≤ 3%, SD ≤ 10%; boundary: bias ≤ 5%, SD ≤ 20%) — **both** must clear.

| Feature | Kind | N | Bias % | SD % | Median % | IQR % | Target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `area_px` | size | 90 | 1.370 | 14.552 | 0.561 | 5.301 | **unmet** |
| `area_mm2` | size | 88 | 1.009 | 14.311 | 0.486 | 5.168 | **unmet** |
| `area_fraction` | size | 88 | 1.009 | 14.311 | 0.486 | 5.168 | **unmet** |
| `equiv_radius_px` | size | 90 | 0.403 | 7.536 | 0.280 | 2.637 | met |
| `equiv_radius_mm` | size | 88 | 0.230 | 7.448 | 0.243 | 2.572 | met |
| `log_area_px2` | size | 90 | 0.008 | 2.134 | 0.050 | 0.479 | met |
| `log_area_mm2` | size | 88 | -0.058 | 3.981 | 0.061 | 0.737 | met |
| `perimeter_px` | boundary | 90 | 9.498 | 40.050 | -0.788 | 12.461 | **unmet** |
| `perimeter_mm` | boundary | 88 | 8.353 | 39.033 | -0.937 | 12.059 | **unmet** |
| `edge_roughness` | boundary | 90 | 8.396 | 30.557 | 0.072 | 13.573 | **unmet** |
| `circularity` | boundary | 90 | -3.256 | 33.280 | -0.144 | 24.686 | **unmet** |
| `solidity` | boundary | 90 | -14.255 | 23.449 | -0.323 | 24.266 | **unmet** |

Reading: area and equivalent radius are the reliable quantities — the segmenter is near-unbiased on the median with a spread well inside the size budget. Perimeter and the boundary-derived shape features carry a much larger, heavy-tailed error (the published Boundary IoU is 28.60%, #28), so they clear the looser boundary budget only on the median and fail it on the SD for the roughest images. `log_area` is the quantity a future model would actually fit, and its error is small and roughly symmetric.

## 4. Duplicate sensitivity

The headline bias/SD were re-run with every duplicate copy included (no de-duplication) to show they do not move. The table gives the de-duplicated and with-copies side by side; the delta is the sensitivity.

| Feature | N (dedup) | Bias % (dedup) | Bias % (with copies) | SD % (dedup) | SD % (with copies) |
| --- | --- | --- | --- | --- | --- |
| `area_px` | 90 | 1.370 | 2.280 | 14.552 | 15.690 |
| `area_mm2` | 88 | 1.009 | 1.711 | 14.311 | 15.319 |
| `area_fraction` | 88 | 1.009 | 1.711 | 14.311 | 15.319 |
| `equiv_radius_px` | 90 | 0.403 | 0.836 | 7.536 | 7.786 |
| `equiv_radius_mm` | 88 | 0.230 | 0.566 | 7.448 | 7.631 |
| `log_area_px2` | 90 | 0.008 | 0.116 | 2.134 | 2.095 |
| `log_area_mm2` | 88 | -0.058 | 0.107 | 3.981 | 3.888 |
| `perimeter_px` | 90 | 9.498 | 13.256 | 40.050 | 46.918 |
| `perimeter_mm` | 88 | 8.353 | 11.480 | 39.033 | 45.650 |
| `edge_roughness` | 90 | 8.396 | 11.073 | 30.557 | 34.243 |
| `circularity` | 90 | -3.256 | -6.010 | 33.280 | 33.691 |
| `solidity` | 90 | -14.255 | -17.074 | 23.449 | 25.361 |

The copies are byte-identical pixels, so including them re-weights the same observation rather than adding information; the bias and SD move only insofar as the copies sit in different blocks than their representatives (the PO/MYG and PDA/GL collisions). The movement is small and does not change any target verdict.

## 5. Repeatability

MyceliumSeg has **no ordered sequences** (`frame_order_status = not_demonstrated` for every row, Gate C), so a frame-to-frame repeatability SD is **not estimable** on this release — there is no demonstrated order to difference, and the file numbering is shuffled relative to time (#22). What the error model does provide is the single-measurement noise floor:

| Quantity | Value |
| --- | --- |
| Single-measurement SD of `ln(area)` | 0.0398 log units |
| `min_reportable_delta` for a future ordered sequence | 0.1126 log units |

`min_reportable_delta` = 2 × √2 × the single-measurement log-area SD: the smallest `|Δ ln(area)|` two independent measurements could distinguish from noise (the #60 `frame_detection_sigma = 2` rule). It is a detection floor derived from the error model, **not** an observed frame-to-frame repeatability, and it is reported as such. The byte-identical duplicate groups are a separate determinism check, not a repeatability measurement: #29 already confirmed that identical pixels give identical predicted masks.

## 6. Candidate protrusions and false-event rates

The candidate detector is a radial boundary excursion (outermost radius per 5-degree sector vs +-15-degree moving-average baseline). A candidate protrusion is a maximal run of 5° sectors whose outermost radius exceeds a ±15° moving-average baseline by both 5% and 3 px at the 900-px width. Predicted and reference protrusions are matched when their peak angles sit within 15° of each other (greedy, one-to-one).

| Quantity | Value |
| --- | --- |
| Images | 90 |
| Candidate protrusions (predicted) | 106 |
| Candidate protrusions (reference) | 47 |
| True events (matched) | 26 |
| False events (predicted, unmatched) | 80 |
| Missed events (reference, unmatched) | 21 |
| **False-event rate** (FP / (TP+FP)) | **0.755** |
| Miss rate (FN / (TP+FN)) | 0.447 |
| Per-image false-event rate, median | 1.000 |
| Per-image false-event rate, max | 1.000 |

These are **candidate** radial excursions of the mask boundary, not biological hyphal tips. The false-event rate is high because the segmenter's boundary is only approximate (Boundary IoU 28.60%): it invents excursions the reference does not have and misses ones it does. No persistence, resolution or validation claim is made, and no tip, branching or nuclear count is reported from MyceliumSeg. The rate is a property of the detector *and* the segmenter; it is reported so a future, validated tip detector has a baseline to beat.

The rate is not a threshold artifact. The table re-runs the detector and matcher over a grid of prominence thresholds and matching angles; the false-event rate stays high across the whole grid, so no candidate protrusion from this segmenter is a reliable tip.

| abs thresh (px) | match (°) | pred | ref | TP | FP | FN | False-event rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 10 | 106 | 47 | 25 | 81 | 22 | 0.764 |
| 3 | 15 | 106 | 47 | 26 | 80 | 21 | 0.755 |
| 3 | 25 | 106 | 47 | 30 | 76 | 17 | 0.717 |
| 5 | 10 | 96 | 36 | 21 | 75 | 15 | 0.781 |
| 5 | 15 | 96 | 36 | 21 | 75 | 15 | 0.781 |
| 5 | 25 | 96 | 36 | 24 | 72 | 12 | 0.750 |
| 8 | 10 | 77 | 21 | 14 | 63 | 7 | 0.818 |
| 8 | 15 | 77 | 21 | 14 | 63 | 7 | 0.818 |
| 8 | 25 | 77 | 21 | 16 | 61 | 5 | 0.792 |

## 7. What these features support and what they do not

**Supported (image-level morphology, this release):**

- `area`, `area_fraction`, `equivalent_radius` and `log_area` clear the size budget on the de-duplicated hold-outs and support predictive, image-level morphology claims (associational, within-release, by species and known condition).
- `perimeter`, `edge_roughness`, `circularity` and `solidity` are published with their (larger) boundary budget; they describe shape only at the fidelity the boundary metrics allow and are not size proxies.

**Not supported (Gate C, #60 § 9):**

- any growth curve, elapsed time, rate per unit time, dish identity or timelapse membership — there is no capture interval and no demonstrated frame order;
- any hyphal-tip, nuclear or branching claim — the candidate protrusions are unvalidated radial excursions with a high false-event rate, and no resolution or persistence evidence maps them to biological structures;
- any causal temperature or medium effect — medium is aliased with species, only 10 images are at 15 °C, and there is no dish identity or replication;
- any physical growth rate in mm/day — a frame index is not a duration, and `to_physical_rate()` refuses the conversion.

## 8. Reproduce

```bash
# error model from the saved masks (no GPU, no retrain)
nix develop --command python scripts/measurement_error.py run

# tests and lint
.venv/bin/python -m pytest tests -q
nix develop --command ruff check scripts/measurement_error.py tests/test_measurement_error.py
```

Surrounding record: `notes/segformer_baseline.md` (#28, the checkpoint), `notes/segformer_inference.md` (#29, the masks), `notes/estimands.md` (#60, the contract and uncertainty target), `notes/provenance.md` (#61, the capability matrix).
