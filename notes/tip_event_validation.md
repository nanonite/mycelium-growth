# Candidate boundary-event resolution and scale stability (#62)

This analysis measures how radial boundary candidates change when the same labeled hold-out masks are resampled. It does not infer time or validate biological hyphal tips.

Population: 90 de-duplicated image pairs, with 34 reference-mask and 112 predicted-mask candidates at native width.

## Method

Each predicted/reference pair was resampled to widths 300, 450, 900, 1800, 4608 px using the same bilinear resize and 127 threshold used by #31. The detector kept its 5% relative prominence threshold; its 3 px absolute threshold at 900 px was scaled with width to preserve the same nominal image-space cutoff. Candidate masks were matched within 15° at each width.

Scale persistence is the fraction of the five widths where a candidate detected in the native 4608 px mask has a one-to-one candidate match within one 5° detector bin. This is algorithmic stability under resampling of a single image; it is not persistence over time.

## Results

| Width (px) | Predicted candidates | Reference candidates | Matched | Mask disagreement FP | Mask disagreement FN | FP fraction | Reference recall | Predicted native event retention | Reference native event retention |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 108 | 50 | 28 | 80 | 22 | 0.741 | 0.560 | 0.688 | 0.765 |
| 450 | 103 | 47 | 28 | 75 | 19 | 0.728 | 0.596 | 0.777 | 0.882 |
| 900 | 106 | 47 | 26 | 80 | 21 | 0.755 | 0.553 | 0.875 | 0.971 |
| 1800 | 104 | 37 | 22 | 82 | 15 | 0.788 | 0.595 | 0.911 | 0.971 |
| 4608 | 112 | 34 | 21 | 91 | 13 | 0.812 | 0.618 | 1.000 | 1.000 |

At native width, the calibrated 90 mm dish fit implies a nominal image sampling pitch of 32.3 µm/px (10th–90th percentile 29.4–43.7, n=88). This is geometric sampling derived from the dish, not a measured optical resolution.

## Interpretation

Across all five widths, 73/112 predicted-mask candidates (65.2%) and 25/34 reference-mask candidates (73.5%) were detected at every width.

The resampling sweep quantifies whether a geometric boundary candidate is stable at a chosen working width and whether the predicted mask agrees with the human colony mask. The reference mask is not an independent hyphal-tip annotation, so a matched candidate is not evidence of a real hyphal tip.

No resolution in this dataset can be designated as the point where an event may be called a biological hyphal tip: the labels describe colony extent, and there is no independent microscopic annotation or temporal sequence. MyceliumSeg supports candidate boundary morphology only. The available-resolution question therefore remains biologically unresolvable from this release; #62 is complete as a bounded mask/resampling validation, with tip claims explicitly excluded.

The 900 px results are the closest comparison with the earlier #31 false-event estimate. Differences arise from applying the detector at multiple scales while preserving its nominal absolute cutoff; these false and missed events are disagreements with the reference mask, not biological event error rates.

Artifacts: `tip_event_resolution.csv`, `tip_event_persistence.csv`, and `tip_event_resolution_evidence.json` in `work_dirs/segformer-b0-repro/tip-event-resolution/`.
