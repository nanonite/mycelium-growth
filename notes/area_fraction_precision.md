# Area-fraction precision audit (#66)

This audit separates the error in predicted colony area from sensitivity in the fitted dish denominator. It uses the 90 deduplicated #31 labeled hold-outs; 88 have calibrated area fractions.

## Overall held-out precision

| N | Relative bias | Relative SD | Median relative error | Mean absolute relative error | 95th percentile absolute error | #60 target |
|---:|---:|---:|---:|---:|---:|---|
| 88 | 1.009% | 14.311% | 0.486% | 6.598% | 15.080% | **unmet** (bias ≤3%, SD ≤10%) |

Area-fraction relative error is numerically the same as predicted/reference mask area relative error: the predicted and reference masks for one image use the same dish-disc denominator, so it cancels from `(pred − ref) / ref`. The maximum absolute difference from `area_px_err` in this table is `2.22e-16` (floating-point roundoff). Thus the 14.3% relative SD is a segmentation-area error conditional on the shared dish fit; it does not include absolute error in the fitted dish area.

## Error by reference coverage

Equal-count quartiles are defined from human reference `area_fraction`, not from predictions.

| Reference coverage bin | N | Reference coverage range | Relative bias | Relative SD | Mean absolute error (percentage points) | 95th percentile absolute error (percentage points) |
|---|---:|---:|---:|---:|---:|---:|
| reference_coverage_q1 | 22 | 0.004–0.101 | 3.771% | 27.021% | 0.412 | 1.225 |
| reference_coverage_q2 | 22 | 0.104–0.243 | 2.150% | 6.731% | 0.660 | 2.037 |
| reference_coverage_q3 | 22 | 0.250–0.436 | -0.871% | 3.489% | 0.870 | 2.877 |
| reference_coverage_q4 | 22 | 0.440–0.611 | -1.012% | 6.597% | 2.285 | 6.277 |

## Error and sort agreement by condition

Rows with duplicate groups are kept as label-conflict diagnostics. They measure the mask errors for those pixels but do not support clean species-specific claims.

| Condition | Label status | N | Relative bias | Relative SD | Pairwise sort agreement | Kendall τ |
|---|---|---:|---:|---:|---:|---:|
| GL/MYG/15C | label_clean | 10 | -3.691% | 24.648% | 100.0% | 1.000 |
| GL/MYG/25C | label_clean | 40 | -0.497% | 5.691% | 95.9% | 0.918 |
| GL/MYG/25C | label_conflict | 10 | -2.607% | 6.423% | 95.6% | 0.911 |
| GS/MYG/25C | label_conflict | 10 | -1.382% | 1.606% | 95.6% | 0.911 |
| PO/PDA/25C | label_conflict | 9 | 13.370% | 25.578% | 97.2% | 0.944 |
| TS/PDA/25C | label_conflict | 9 | 7.243% | 20.106% | 97.2% | 0.944 |

## Dish-disc denominator sensitivity

The table below compares each accepted image's fitted px/mm with its block/setup calibration constant. The area-fraction shift is the change produced by substituting the per-image fit for the block constant, `(block_scale / fitted_scale)^2 − 1`. It is a sensitivity measure, not a validated denominator-error distribution, because no independent ruler or second dish annotation is available.

| Condition | Label status | N | Rim fit vs block scale: mean ± SD | Area-fraction shift: median | 5th–95th percentile |
|---|---|---:|---:|---:|---:|
| GL/MYG/15C | label_clean | 10 | -0.020% ± 0.054% | 0.000% | 0.000% to 0.221% |
| GL/MYG/25C | label_clean | 40 | -0.219% ± 0.718% | 0.000% | -0.640% to 4.469% |
| GL/MYG/25C | label_conflict | 10 | -0.684% ± 1.067% | 0.083% | 0.000% to 4.639% |
| GS/MYG/25C | label_conflict | 10 | -0.223% ± 0.682% | 0.000% | -0.739% to 2.513% |
| PO/PDA/25C | label_conflict | 9 | -0.030% ± 0.894% | 0.000% | -1.731% to 2.900% |
| TS/PDA/25C | label_conflict | 9 | -0.603% ± 1.152% | 0.000% | -1.233% to 4.480% |

## Interpretation

The held-out segmentation model misses the planned 10% SD target for area fraction even though mean bias is only about 1%. The largest relative error is in the lowest-coverage quartile, where small denominators inflate relative error; its mean absolute error is 0.41 percentage points. In the clean GL/MYG/25C stratum (n=40), 95.9% of image pairs keep the same area order as the reference masks. Thus coverage sorting is useful, but values close together can be swapped and absolute percentages need uncertainty; this is not an exact rank guarantee for every pair.

Dish fit adds a separate denominator sensitivity. It is not part of the #31 predicted-versus-reference relative error because the same fitted radius is used for both masks. The per-image fit versus block scale shows how much reported percentages could move under that alternate calibration; it does not establish which denominator is true. The current 14.3% SD should therefore be described as segmentation-only under a shared dish fit, with dish calibration uncertainty reported separately.

Scope is the selected release and the #31 held-out checkpoint. It does not establish a plate-population distribution, chronology, or growth rate. The sorted coordinate remains cross-sectional area coverage.

Artifacts: `area_fraction_precision_summary.csv`, `area_fraction_precision_by_image.csv`, and `area_fraction_precision_evidence.json` under `data/triage/area_fraction_precision/`.
