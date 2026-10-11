# MyceliumSeg morphology across area coverage (#41)

This is an exploratory cross-sectional model of predicted-mask shape by area coverage.
It uses no time, dish, or frame-order assumption. All results describe the selected release.
Boundary features come from predicted masks and inherit the segmenter's boundary error.

Models cover 2 large strata, from 20105 deduplicated feature rows.

| Stratum | Target | N | Spline RMSE | Constant RMSE | RMSE improvement | Spline R² |
|---|---|---:|---:|---:|---:|---:|
| `known_GL_MYG_25C_scale1` | `circularity` | 17496 | 0.18715 | 0.20549 | 8.9% | 0.114 |
| `known_GL_MYG_25C_scale1` | `solidity` | 17496 | 0.13128 | 0.13803 | 4.9% | 0.040 |
| `known_GL_MYG_25C_scale1` | `edge_roughness` | 17496 | 0.19739 | 0.20538 | 3.9% | 0.025 |
| `known_GL_MYG_25C_scale1` | `bbox_fill` | 17496 | 0.12743 | 0.13364 | 4.6% | 0.031 |
| `known_GL_MYG_25C_scale1` | `eccentricity` | 17496 | 0.11687 | 0.13674 | 14.5% | 0.233 |
| `unknown_species_25C_flat_scale1` | `circularity` | 1433 | 0.28519 | 0.31923 | 10.7% | 0.174 |
| `unknown_species_25C_flat_scale1` | `solidity` | 1433 | 0.1393 | 0.14858 | 6.2% | 0.014 |
| `unknown_species_25C_flat_scale1` | `edge_roughness` | 1433 | 0.90004 | 0.93533 | 3.8% | 0.014 |
| `unknown_species_25C_flat_scale1` | `bbox_fill` | 1433 | 0.12345 | 0.13388 | 7.8% | 0.043 |
| `unknown_species_25C_flat_scale1` | `eccentricity` | 1433 | 0.16856 | 0.18927 | 10.9% | 0.127 |

Approximate measurement-error sensitivity (30 draws resampling joint GL relative errors from #31):

| Target | Median RMSE improvement | 5th–95th percentile |
|---|---:|---:|
| `circularity` | 3.8% | 3.5% to 4.1% |
| `solidity` | 4.9% | 4.7% to 5.1% |
| `edge_roughness` | 2.2% | 2.0% to 2.6% |

Validation holds out contiguous area-coverage quantile groups. The constant baseline is the training-fold median.
The sensitivity draw transfers held-out #28 residuals to the all-labeled model; it is approximate, not a production-checkpoint validation interval.
Training includes only unlabeled images so the all-labeled segmenter is not evaluated on its own mask-training images. Condition-holdout scores use #31 reference-mask features and #28 predicted area on 90 deduplicated labeled images; groups of about ten are stress checks, not strong transport claims.
The model does not use area fraction as its own response. It predicts shape descriptors from coverage.

Artifacts: `morphology_features.csv`, `coverage_model_cv.csv`, `coverage_model_condition_holdout.csv`, `coverage_model_oof_predictions.csv`, and `coverage_model_final.joblib` in this work directory.
