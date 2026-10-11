# Area-rank pseudo-time mask model (#65)

This is a sequence-learning experiment on cross-sectional masks, not a biological time series.
The synthetic coordinate `t_star` is the area-coverage quantile from 0 to 1 within GL/MYG/25C.
Each of 100 pseudo-steps is the median shape of a coverage-ordered bin of unrelated release images.
No mask pixels are differenced directly: specimen identity and image registration are unavailable.

Unlabeled input images: 17496; pseudo-bins: 100; train/test bins: 66/30.
The AR(3) predicts a change in the next bin's median shape feature from the previous three bins and t_star.
Rolling AR forecasts use previous observed pseudo-bins; recursive AR forecasts feed their own prior predictions.
The recursive persistence baseline repeats the last training-bin value. The cubic t* model is a direct multi-step baseline evaluated over the same holdout, without feedback.

| Mask feature | Persistence rolling RMSE | Cubic t* RMSE | AR(3) rolling RMSE | Persistence recursive RMSE | Cubic t* direct RMSE | AR(3) recursive RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `circularity` | 0.035336 | 0.19154 | 0.047601 | 0.11704 | 0.19154 | 0.13808 |
| `solidity` | 0.00088247 | 0.011615 | 0.0014742 | 0.0040402 | 0.011615 | 0.0083574 |
| `edge_roughness` | 0.022148 | 0.09987 | 0.028855 | 0.064991 | 0.09987 | 0.078665 |
| `bbox_fill` | 0.0018731 | 0.0075111 | 0.0026338 | 0.0048351 | 0.0075111 | 0.0078389 |
| `eccentricity` | 0.0099236 | 0.040842 | 0.013469 | 0.020191 | 0.040842 | 0.036113 |

Compare rolling AR(3) with rolling persistence and the direct t* trend; compare recursive AR(3) with repeated-value persistence and the same direct trend over the full holdout.
A better score means only that adjacent coverage bins have learnable shape-feature changes in this selected release.
It does not estimate elapsed time, a per-frame change, same-dish growth, a rate, or causal dynamics.
The experiment tests whether this ordering can prototype sequence-model features and validation code before using verified time-lapse data.

Artifacts: `pseudo_time_trajectory.csv`, `pseudo_time_forecasts.csv`, `pseudo_time_model_summary.csv`, and `pseudo_time_ar3_surrogate.joblib`.
