# Cross-dataset growth-form assessment (#37)

## Decision

The available evidence does **not support treating lag → constant extension →
saturation as an invariant growth law across the Fricker/Boddy and *Podospora*
datasets**. The bounded predictive checks find different best-supported
descriptions on different measured outcomes, and neither dataset observes a
well-identified biological lag followed by a complete plateau. Close #37 as an
assessment of the assumption, not as proof that all fungi follow different
mechanisms. Keep the shared form as an unverified hypothesis for future data.

MyceliumSeg was not fit as a time series: it has no linked dish identities or
capture times, so it cannot identify lag, extension, or saturation.

## What was compared

Each dataset was evaluated on its own linked time axis and primary image-derived
outcome. The outcomes are not interchangeable: Fricker provides workbook
measurements for wood/soil mycelial networks; *Podospora* supplies microscopic
movies measured here by a deterministic bright-ridge occupancy proxy. The
Fricker check uses 5-fold cross-validation holding out complete replicate
series. The *Podospora* check holds out each movie's final 30% of post-onset
frames after fitting on its earlier 70%.

For a direct test of the proposed three phases, the Fricker response is
`sqrt(mycelial_area / D1_area) - 1`, a dimensionless equivalent-radius change.
It uses 125 D1-linked series and 492 observations through the D8 label. A
shared curve shape is fit across the training data while extension rates vary
by species × grazer group. The candidate curves are no growth, linear
extension, lag then extension, and lag then extension capped at a plateau.

For *Podospora*, the same q=.99 baseline-subtracted ridge-occupancy signal as
#47 is used. Its square root is only an equivalent-size coordinate, not a
measured hyphal-front radius. The robust signal-onset threshold supplies a
movie-relative transition into the extension phase; it does not estimate
germination time. The held-out comparison includes exponential occupancy,
power-law occupancy, linear equivalent-size extension, and extension capped at
a fitted plateau.

## Held-out results

### Fricker/Boddy

| Shape | 5-fold series-held-out RMSE | MAE |
|---|---:|---:|
| No growth from baseline | 0.4441 | 0.2564 |
| Linear extension | 0.2271 | 0.1118 |
| Lag then extension | 0.2267 | 0.1122 |
| Lag, extension, plateau | 0.2260 | 0.1134 |

The full triphasic curve reduces RMSE by only 0.5% relative to linear
extension, while MAE is slightly worse. In every fold, the estimated plateau
falls at about 6.75 days after the first linked image, near the end of the D8
observation window; the fitted lag is only 0.25–0.50 days on the chosen grid.
These boundary estimates do not show a stable observed plateau or a biological
lag. The original grouped-CV comparison in #45 also found the saturating
`t/(tau+t)` curves no better than linear/day-factor models for mycelial area
(log-RMSE 0.6059 at tau=8 d vs 0.6034 linear). The #68 model found the
observed-day factor modestly better than EXIF-linear prediction for area
(0.3289 vs 0.3359 log-change RMSE), consistent with some nonlinearity at the
sampled labels, not specifically with saturation.

### *Podospora*

| Shape | Mean held-out log-signal RMSE across six movies | Movies with best/tied score |
|---|---:|---:|
| Exponential occupancy | 0.3064 | 5/6 |
| Linear equivalent-size extension | 0.4210 | 1/6 |
| Lag/onset, extension, plateau | 0.8160 | 0/6 |
| Power-law occupancy | 0.8956 | 0/6 |

The candidate plateau curve predicted the held-out tail substantially worse
than exponential occupancy. The observed signal-onset times were 2.4–8.7 h
after each movie's first frame, but the frame's relation to germination is
unknown. No biological lag can be inferred from those times. The movies share
only a 19.8 h comparison window; they do not show a validated hyphal-area
plateau.

## Interpretation and limits

- The Fricker fit describes changes in network workbook measurements through
  D8. The paper/archive discrepancy means D1 is an archive-relative reference,
  not a securely known time since inoculation; later colony-edge approach also
  limits the analysis window. These are wood/soil colonies rather than circular
  agar plates.
- The *Podospora* response is a threshold-sensitive candidate ridge-pixel
  count. It can include substrate texture and miss faint hyphae; it is not
  validated area, length, or tip count.
- The studies use different organisms, environments, image scales, outcomes,
  and time origins. A curve that predicts one outcome is not evidence of a
  shared mechanism. The two treatment contrasts also remain associational:
  random assignment is not documented in the reviewed methods.
- Saturation after the observed windows is not identified in either dataset.
  The current evidence is therefore insufficient to accept or falsify a
  universal asymptotic form; it supports only dataset-specific predictive
  descriptions.

## Reproducible artifacts

- Fricker triphasic candidate: [`scripts/fricker_triphasic_growth.py`](../scripts/fricker_triphasic_growth.py),
  [`fricker_triphasic_cv.json`](../data/derived/growth_form_invariance/fricker_triphasic_cv.json).
- *Podospora* triphasic candidate: [`scripts/podospora_triphasic_growth.py`](../scripts/podospora_triphasic_growth.py),
  [`podospora_triphasic_cv.json`](../data/derived/growth_form_invariance/podospora_triphasic_cv.json).
- Prior dataset-specific comparisons: [`notes/fricker_grazers_analysis.md`](fricker_grazers_analysis.md),
  [`notes/fricker_series_model.md`](fricker_series_model.md), and
  [`notes/podospora_growth_model.md`](podospora_growth_model.md).
- Data cadence and design sources are documented in those notes, including
  the [Fricker/Boddy 2011 paper](https://doi.org/10.1007/s00442-011-2005-1),
  [Zenodo record 14931953](https://zenodo.org/records/14931953),
  the [*Podospora* Figshare release](https://figshare.com/articles/media/6-movies_of_P_anserina_thallus_growth_obtained_from_experimental_work_Used_in_the_paper_Prediction_and_experimental_evidence_of_different_growth_phases_of_the_Podospora_anserina_hyphal_network_to_be_published/22666138),
  and [Ledoux et al. 2023](https://doi.org/10.1038/s41598-023-35327-w).

## Evidence needed to revisit invariance

Use a controlled time-lapse dataset with known biological start, a calibrated
common spatial observable, independent replicates, and observations spanning
lag through a visible plateau. Fit the same observation model and held-out
prediction protocol across taxa/conditions. Until then, compare forms within
each dataset and do not transport a fitted lag, extension rate, or saturation
parameter as a universal biological prior.
