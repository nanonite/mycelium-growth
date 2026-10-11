# Podospora thallus image-derived growth model (#47)

## Result

Issue #47 is complete as a dataset-specific, exploratory analysis. Across the
first shared 19.8 h of the six movies, the ridge-occupancy signal increased in
all six replicates. The mean log-occupancy slope was 0.312 h⁻¹ for M2 (three
movies) and 0.242 h⁻¹ for M0 (three movies), a difference of 0.070 h⁻¹. On the
image-derived signal scale, this corresponds to an M2/M0 per-hour multiplier
ratio of 1.072. It is an association in these movies, not an identified causal
nutrient effect.

The growth-form comparison favored exponential occupancy over a power law in
the later-frame holdout for all six movies. This supports exponential growth as
a compact description of this measured signal over the observed post-onset
window; it does not establish a universal hyphal growth mechanism or reproduce
the paper's apex-count model.

## Data and provenance

The analysis uses all six M0/M2 triplicate movies (464 frames total) from the
Figshare *Podospora anserina* thallus release. The Figshare record specifies
bright-field nine-tile mosaics at 27 °C, an 18-minute capture interval, a scale
of 1.63 µm per source pixel, and CC BY 4.0. The 5 fps AVI value is playback
rate, not capture cadence. Each movie begins at its first available frame;
germination time is not given. Frame spans are 20.4–25.2 h. The primary paper
independently describes roughly 18-minute sampling during the first 20 hours.
See [the Figshare record](https://figshare.com/articles/media/6-movies_of_P_anserina_thallus_growth_obtained_from_experimental_work_Used_in_the_paper_Prediction_and_experimental_evidence_of_different_growth_phases_of_the_Podospora_anserina_hyphal_network_to_be_published/22666138),
[Ledoux et al. (2023)](https://doi.org/10.1038/s41598-023-35327-w), and
[`SOURCE.md`](../data/raw/podospora_thallus/SOURCE.md).

For a fair condition window, the slope comparison uses frames 0–66 in every
movie, or 0–19.8 h since its first frame. The final 30% of each movie's
post-onset observations in that shared window is held out when comparing the
two candidate growth forms.

## Measurement and model

[`scripts/podospora_growth_model.py`](../scripts/podospora_growth_model.py)
streams each movie at half spatial resolution. It uses a bright-ridge Hessian
response and a per-frame threshold set at the 99th percentile of response in
the four outer corners. Candidate pixels are counted in the central 70% of the
field. The initial five frames establish a median/MAD baseline; onset is the
first run of three frames above that robust baseline threshold. The model fits
the log of baseline-subtracted candidate-pixel area against hours since this
estimated onset. A power-law alternative fits log area against log elapsed
time, with a one-frame offset. Per-movie slopes are the independent observations
for the M2/M0 summary; frames are repeated observations, not replicates.

This is a deterministic image-derived ridge detector, **not a trained or
manually validated semantic segmentation model**. Red overlay pixels often
follow visible hyphae but miss faint portions and include occasional texture.
Therefore, its pixel count is called a ridge-occupancy proxy. It is not
validated hyphal area, total length, branch count, or tip/apex count. Its
converted µm² value is only the physical-scale equivalent of detected pixels,
not an independently validated biological area measurement.

## Per-movie results

The exponential slope describes log baseline-corrected ridge occupancy per
hour after estimated signal onset. Tail RMSE is computed on the final 30% of
active observations after fitting each form on the earlier 70%; lower is
better.

| Movie | Onset after first frame (h) | Exponential slope (h⁻¹) | Doubling time (h) | Exponential log-R² | Exponential tail RMSE | Power-law tail RMSE |
|---|---:|---:|---:|---:|---:|---:|
| M0_1 | 8.7 | 0.226 | 3.06 | 0.842 | 0.362 | 0.606 |
| M0_2 | 4.2 | 0.222 | 3.12 | 0.987 | 0.082 | 0.883 |
| M0_3 | 2.4 | 0.279 | 2.48 | 0.982 | 0.460 | 0.942 |
| M2_1 | 6.0 | 0.297 | 2.33 | 0.978 | 0.327 | 0.857 |
| M2_2 | 4.8 | 0.318 | 2.18 | 0.992 | 0.325 | 1.067 |
| M2_3 | 6.0 | 0.322 | 2.15 | 0.994 | 0.282 | 1.019 |

## M2/M0 contrast and robustness

| Quantity | Estimate |
|---|---:|
| Mean M2 slope (n=3 movies) | 0.312 h⁻¹ |
| Mean M0 slope (n=3 movies) | 0.242 h⁻¹ |
| Difference, M2 − M0 | 0.070 h⁻¹ |
| M2/M0 per-hour multiplier ratio | 1.072 |
| Pooled-variance t 95% CI for slope difference | [0.014, 0.125] h⁻¹ |
| Exact two-sided label-permutation diagnostic | p = 0.10 |

With three movies per condition, the exact two-sided permutation calculation
has only 20 possible label assignments, so 0.10 is its smallest attainable
two-sided p-value. The t interval depends on a small-sample pooled-variance
assumption, and the methods reviewed do not document randomized condition
assignment. Read both as exploratory summaries, not causal evidence.

Threshold sensitivity retained a positive M2-minus-M0 point estimate for all
tested background quantiles. Using q=.995, where all six movies remain
estimable, the difference was 0.104 h⁻¹ and its t interval included zero. At
the stricter q=.9995 threshold, one M0 movie no longer met the onset criterion.
The magnitude is therefore threshold-dependent, and the primary q=.99 contrast
should not be treated as a precise treatment estimate.

## Artifacts

- [`analysis_summary.json`](../data/derived/podospora_growth_model/analysis_summary.json) — method, estimates, and threshold sensitivity.
- [`movie_growth_fits.csv`](../data/derived/podospora_growth_model/movie_growth_fits.csv) — one row per movie.
- [`frame_measurements.csv`](../data/derived/podospora_growth_model/frame_measurements.csv) — frame-level signal and spread measurements.
- [`podospora_growth_trajectories.svg`](../data/derived/podospora_growth_model/podospora_growth_trajectories.svg) — shared-window trajectories and movie slopes.
- [`qc_overlays/`](../data/derived/podospora_growth_model/qc_overlays/) — four image/mask overlays per movie.

An example overlay is [`M0_1, frame 56`](../data/derived/podospora_growth_model/qc_overlays/M0_1_frame056.png).

## Limits and handoff

- The source has no released, human-labeled hyphal masks. Semantic mask
  validation and a learned segmenter are outside this result.
- The detected foreground is sensitive to contrast, threshold, and cellophane
  texture; threshold sensitivity is reported rather than hidden.
- The six films provide three movie replicates per medium. Treatment allocation
  randomization is not documented, and initial germination time is unknown.
- The paper's published apex-count growth analysis uses reconstructed network
  graphs and apex locations. This image-derived occupancy model measures a
  different quantity and cannot reproduce apex counts or hyphal-tip dynamics.
- The within-dataset result was compared in #37 with the Fricker/Boddy series.
  Their outcomes remain dataset-specific; the cross-dataset assessment does not
  justify transporting one growth form to Fricker/Boddy or MyceliumSeg.
