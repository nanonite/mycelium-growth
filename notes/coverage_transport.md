# Held-out transport of morphology across area coverage (#44)

This is cross-sectional prediction within the selected release. It does not estimate chronology, dish-level growth, or physical rates.

Training: 17496 unlabeled GL/MYG/25C/scale1 images. 
Reference holdout: 88 deduplicated, calibrated #31 images; 
38 have conflicting archive labels.
The model learns mask-shape features from the all-labeled checkpoint predictions; test targets come from human reference masks.
Primary test area inputs are independent #28 predicted area_fraction values. Reference-mask area_fraction is included as an input sensitivity check.

| Held-out condition / label status | Target | N | Area input | Model RMSE | Train-median RMSE | RMSE change vs median | R² | Area extrapolation |
|---|---|---:|---|---:|---:|---:|---:|:---:|
| `GL/MYG/15C|label_clean` | `circularity` | 10 | #28_predicted_area_fraction | 0.25402 | 0.30748 | 17.4% | 0.317 | no |
| `GL/MYG/15C|label_clean` | `circularity` | 10 | reference_mask_area_fraction_sensitivity | 0.25463 | 0.30748 | 17.2% | 0.314 | no |
| `GL/MYG/15C|label_clean` | `solidity` | 10 | #28_predicted_area_fraction | 0.04588 | 0.018787 | -144.2% | -7.875 | no |
| `GL/MYG/15C|label_clean` | `solidity` | 10 | reference_mask_area_fraction_sensitivity | 0.047292 | 0.018787 | -151.7% | -8.430 | no |
| `GL/MYG/15C|label_clean` | `edge_roughness` | 10 | #28_predicted_area_fraction | 0.27875 | 0.30102 | 7.4% | 0.090 | no |
| `GL/MYG/15C|label_clean` | `edge_roughness` | 10 | reference_mask_area_fraction_sensitivity | 0.2788 | 0.30102 | 7.4% | 0.090 | no |
| `GL/MYG/15C|label_clean` | `bbox_fill` | 10 | #28_predicted_area_fraction | 0.047959 | 0.021636 | -121.7% | -4.911 | no |
| `GL/MYG/15C|label_clean` | `bbox_fill` | 10 | reference_mask_area_fraction_sensitivity | 0.049614 | 0.021636 | -129.3% | -5.325 | no |
| `GL/MYG/15C|label_clean` | `eccentricity` | 10 | #28_predicted_area_fraction | 0.10473 | 0.086375 | -21.2% | -0.484 | no |
| `GL/MYG/15C|label_clean` | `eccentricity` | 10 | reference_mask_area_fraction_sensitivity | 0.10246 | 0.086375 | -18.6% | -0.420 | no |
| `GL/MYG/25C|label_clean` | `circularity` | 40 | #28_predicted_area_fraction | 0.37376 | 0.46014 | 18.8% | -3.051 | no |
| `GL/MYG/25C|label_clean` | `circularity` | 40 | reference_mask_area_fraction_sensitivity | 0.37073 | 0.46014 | 19.4% | -2.985 | no |
| `GL/MYG/25C|label_clean` | `solidity` | 40 | #28_predicted_area_fraction | 0.030725 | 0.027876 | -10.2% | -1.073 | no |
| `GL/MYG/25C|label_clean` | `solidity` | 40 | reference_mask_area_fraction_sensitivity | 0.030654 | 0.027876 | -10.0% | -1.063 | no |
| `GL/MYG/25C|label_clean` | `edge_roughness` | 40 | #28_predicted_area_fraction | 0.44198 | 0.51354 | 13.9% | -1.681 | no |
| `GL/MYG/25C|label_clean` | `edge_roughness` | 40 | reference_mask_area_fraction_sensitivity | 0.43911 | 0.51354 | 14.5% | -1.646 | no |
| `GL/MYG/25C|label_clean` | `bbox_fill` | 40 | #28_predicted_area_fraction | 0.033337 | 0.030899 | -7.9% | -0.732 | no |
| `GL/MYG/25C|label_clean` | `bbox_fill` | 40 | reference_mask_area_fraction_sensitivity | 0.033297 | 0.030899 | -7.8% | -0.727 | no |
| `GL/MYG/25C|label_clean` | `eccentricity` | 40 | #28_predicted_area_fraction | 0.0818 | 0.084876 | 3.6% | 0.039 | no |
| `GL/MYG/25C|label_clean` | `eccentricity` | 40 | reference_mask_area_fraction_sensitivity | 0.081671 | 0.084876 | 3.8% | 0.042 | no |
| `GL/MYG/25C|label_conflict` | `circularity` | 10 | #28_predicted_area_fraction | 0.4035 | 0.48854 | 17.4% | -1.326 | no |
| `GL/MYG/25C|label_conflict` | `circularity` | 10 | reference_mask_area_fraction_sensitivity | 0.40129 | 0.48854 | 17.9% | -1.301 | no |
| `GL/MYG/25C|label_conflict` | `solidity` | 10 | #28_predicted_area_fraction | 0.044567 | 0.04887 | 8.8% | -0.859 | no |
| `GL/MYG/25C|label_conflict` | `solidity` | 10 | reference_mask_area_fraction_sensitivity | 0.04484 | 0.04887 | 8.2% | -0.882 | no |
| `GL/MYG/25C|label_conflict` | `edge_roughness` | 10 | #28_predicted_area_fraction | 0.58797 | 0.65393 | 10.1% | -0.974 | no |
| `GL/MYG/25C|label_conflict` | `edge_roughness` | 10 | reference_mask_area_fraction_sensitivity | 0.58738 | 0.65393 | 10.2% | -0.970 | no |
| `GL/MYG/25C|label_conflict` | `bbox_fill` | 10 | #28_predicted_area_fraction | 0.048357 | 0.052581 | 8.0% | -0.523 | no |
| `GL/MYG/25C|label_conflict` | `bbox_fill` | 10 | reference_mask_area_fraction_sensitivity | 0.048708 | 0.052581 | 7.4% | -0.545 | no |
| `GL/MYG/25C|label_conflict` | `eccentricity` | 10 | #28_predicted_area_fraction | 0.15684 | 0.17441 | 10.1% | -0.083 | no |
| `GL/MYG/25C|label_conflict` | `eccentricity` | 10 | reference_mask_area_fraction_sensitivity | 0.15722 | 0.17441 | 9.9% | -0.088 | no |
| `GS/MYG/25C|label_conflict` | `circularity` | 10 | #28_predicted_area_fraction | 0.13814 | 0.096636 | -42.9% | -1.109 | no |
| `GS/MYG/25C|label_conflict` | `circularity` | 10 | reference_mask_area_fraction_sensitivity | 0.13815 | 0.096636 | -43.0% | -1.110 | no |
| `GS/MYG/25C|label_conflict` | `solidity` | 10 | #28_predicted_area_fraction | 0.02213 | 0.0074889 | -195.5% | -13.003 | no |
| `GS/MYG/25C|label_conflict` | `solidity` | 10 | reference_mask_area_fraction_sensitivity | 0.021772 | 0.0074889 | -190.7% | -12.555 | no |
| `GS/MYG/25C|label_conflict` | `edge_roughness` | 10 | #28_predicted_area_fraction | 0.094095 | 0.055003 | -71.1% | -1.954 | no |
| `GS/MYG/25C|label_conflict` | `edge_roughness` | 10 | reference_mask_area_fraction_sensitivity | 0.094201 | 0.055003 | -71.3% | -1.960 | no |
| `GS/MYG/25C|label_conflict` | `bbox_fill` | 10 | #28_predicted_area_fraction | 0.026617 | 0.014754 | -80.4% | -2.267 | no |
| `GS/MYG/25C|label_conflict` | `bbox_fill` | 10 | reference_mask_area_fraction_sensitivity | 0.02643 | 0.014754 | -79.1% | -2.221 | no |
| `GS/MYG/25C|label_conflict` | `eccentricity` | 10 | #28_predicted_area_fraction | 0.074959 | 0.066865 | -12.1% | -0.258 | no |
| `GS/MYG/25C|label_conflict` | `eccentricity` | 10 | reference_mask_area_fraction_sensitivity | 0.075195 | 0.066865 | -12.5% | -0.266 | no |
| `PO/PDA/25C|label_conflict` | `circularity` | 9 | #28_predicted_area_fraction | 0.1864 | 0.22591 | 17.5% | -0.022 | no |
| `PO/PDA/25C|label_conflict` | `circularity` | 9 | reference_mask_area_fraction_sensitivity | 0.1872 | 0.22591 | 17.1% | -0.031 | no |
| `PO/PDA/25C|label_conflict` | `solidity` | 9 | #28_predicted_area_fraction | 0.040881 | 0.033823 | -20.9% | -0.933 | no |
| `PO/PDA/25C|label_conflict` | `solidity` | 9 | reference_mask_area_fraction_sensitivity | 0.041377 | 0.033823 | -22.3% | -0.981 | no |
| `PO/PDA/25C|label_conflict` | `edge_roughness` | 9 | #28_predicted_area_fraction | 0.13425 | 0.1663 | 19.3% | -0.066 | no |
| `PO/PDA/25C|label_conflict` | `edge_roughness` | 9 | reference_mask_area_fraction_sensitivity | 0.13467 | 0.1663 | 19.0% | -0.073 | no |
| `PO/PDA/25C|label_conflict` | `bbox_fill` | 9 | #28_predicted_area_fraction | 0.052372 | 0.050957 | -2.8% | -0.254 | no |
| `PO/PDA/25C|label_conflict` | `bbox_fill` | 9 | reference_mask_area_fraction_sensitivity | 0.05247 | 0.050957 | -3.0% | -0.258 | no |
| `PO/PDA/25C|label_conflict` | `eccentricity` | 9 | #28_predicted_area_fraction | 0.19731 | 0.2335 | 15.5% | -0.839 | no |
| `PO/PDA/25C|label_conflict` | `eccentricity` | 9 | reference_mask_area_fraction_sensitivity | 0.19555 | 0.2335 | 16.3% | -0.807 | no |
| `TS/PDA/25C|label_conflict` | `circularity` | 9 | #28_predicted_area_fraction | 0.37955 | 0.40036 | 5.2% | -1.184 | no |
| `TS/PDA/25C|label_conflict` | `circularity` | 9 | reference_mask_area_fraction_sensitivity | 0.38294 | 0.40036 | 4.3% | -1.223 | no |
| `TS/PDA/25C|label_conflict` | `solidity` | 9 | #28_predicted_area_fraction | 0.035378 | 0.057072 | 38.0% | 0.002 | no |
| `TS/PDA/25C|label_conflict` | `solidity` | 9 | reference_mask_area_fraction_sensitivity | 0.036719 | 0.057072 | 35.7% | -0.076 | no |
| `TS/PDA/25C|label_conflict` | `edge_roughness` | 9 | #28_predicted_area_fraction | 0.42615 | 0.45872 | 7.1% | -0.762 | no |
| `TS/PDA/25C|label_conflict` | `edge_roughness` | 9 | reference_mask_area_fraction_sensitivity | 0.42841 | 0.45872 | 6.6% | -0.781 | no |
| `TS/PDA/25C|label_conflict` | `bbox_fill` | 9 | #28_predicted_area_fraction | 0.055786 | 0.061912 | 9.9% | -0.200 | no |
| `TS/PDA/25C|label_conflict` | `bbox_fill` | 9 | reference_mask_area_fraction_sensitivity | 0.055906 | 0.061912 | 9.7% | -0.205 | no |
| `TS/PDA/25C|label_conflict` | `eccentricity` | 9 | #28_predicted_area_fraction | 0.18115 | 0.23156 | 21.8% | -0.153 | no |
| `TS/PDA/25C|label_conflict` | `eccentricity` | 9 | reference_mask_area_fraction_sensitivity | 0.18089 | 0.23156 | 21.9% | -0.150 | no |

Clean GL/MYG/15C rows are the only uncontested held-out condition group. GL/MYG/25C reference rows are a same-condition independent-image control, not condition transport.
GS, PO, and TS archive-labeled rows are reported separately as label-conflict diagnostics; they cannot support clean species-specific transport claims. The unlabeled flat GS/PO/TS block is excluded from species transport because its species are unknown and it has no reference masks.
Duplicate copies were excluded before scoring; only representatives remain. Off-scale rows, empty masks, invalid area fractions, and unresolved label-conflict rows are not pooled with clean groups. All test groups are tiny, so per-group scores are exploratory stress checks.
Training area/features come from the all-labeled production checkpoint; #31 test areas come from the independent #28 checkpoint. The two area-input sources differ, and predicted-mask shape targets inherit segmentation error. Reference-area scoring is a sensitivity check, not the deployment score.
All conclusions describe the selected release. The paper's image inclusion rule is undocumented, so morphology-selection bias is unresolved.

Artifacts: `coverage_transport_holdouts.csv`, `coverage_transport_predictions.csv`, `coverage_transport_models.joblib`, and `coverage_transport_evidence.json` in the full-release work directory.
