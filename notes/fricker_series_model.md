# Fricker dataset-specific repeated-measures model (#68)

**Status:** model fitted; #68 scope complete. This is a dataset-specific
trajectory analysis of the archived Fungal Network Analysis measurements.
It is a predictive and descriptive model, not a causal estimate of grazing.

## Model

The model uses the workbook's network outcomes and the image-linked EXIF
capture times in `data/derived/fricker_grazers/network_series.csv`. For each
series with a linked D1 image, it models change relative to that series' D1
measurement:

```text
log(Y_it / Y_i,D1) = beta[fungus, grazer] * EXIF_elapsed_days_it + error_it
```

There is a separate proportional-change rate for each archived fungus ×
grazer group. `ctrl1` and `ctrl2` are pooled as controls. Within-series
log-differencing removes series-specific starting level. The uncertainty
bootstrap resamples whole replicate series within fungus × grazer cells, so
the repeated observations from one series stay together. The fitted rate is
reported as `100 * (exp(beta) - 1)` percent change per EXIF day; it summarizes
the observed window and is not a long-range biological law.

For model-shape selection, five-fold cross-validation holds out complete
series. It compares (1) a species × grazer linear slope on EXIF elapsed days,
(2) the same slope using numeric filename-day differences, and (3) a
species × grazer × observed-day factor. The day-factor model predicts only the
observed archive labels; it should not be used to interpolate or extrapolate.
The continuous-time linear fit provides a compact rate summary.

## Included data and timing handling

- The source inventory contains 778 workbook records across 155 series. The
  main trajectory fit uses 125 core-treatment series with linked D1 images and
  492 observations through the D8 label (367 post-baseline observations for
  area, length, volume, tips, width, and edges; 358 for loop count).
- Fifteen of the 140 core-treatment series start at the D2 label and have no
  linked D1 image. They are listed separately in
  `data/derived/fricker_grazers/day2_start_series.csv`; their D2-to-D8 changes
  are not mixed into the D1-baseline model.
- Five same-session `Pi_ctrl2` D1/D2 rows are excluded. They have identical
  EXIF seconds and four are byte-identical images.
- D8-label images are typically about 6.9 EXIF days after the linked D1 image.
  The archive's filename labels and elapsed EXIF time remain separate. The
  first interval still conflicts with the paper's stated pre-addition to
  day-2 schedule, so D1 is called the archive baseline, not a verified
  post-inoculation time origin.
- The main trajectory comparison stops at D8 because the later analysis
  reported that colonies in some trays approach tray boundaries. Later archive
  labels remain available in the inventory but are outside this model's
  primary window.

## Results

The flexible observed-day model had the lowest held-out-series RMSE for all
seven outcomes. Its reduction versus the EXIF-linear model was 2.1% for
mycelial area, 5.0% for total length, 0.1% for volume, 7.2% for tip count,
14.6% for mean width, 8.6% for edge count, and 6.2% for loop count. For
mycelial area, the elapsed-time linear, numeric-label linear, and day-factor
RMSEs were 0.3359, 0.3338, and 0.3289 log-change units. Thus the area shape
comparison is close; there is little evidence that EXIF-time and filename-day
linear slopes differ in this early window. The day factor is retained as the
best predictor at sampled labels, while the EXIF-time slope is retained as
the interpretable rate summary.

As an example of fungal-specific rates, the fitted control mycelial-area
changes were Hf 16.2%, Pi 11.9%, Pv 30.5%, and Rb 12.4% per EXIF day. The full
species × grazer × outcome estimates and series-bootstrap intervals are in
`data/derived/fricker_grazers/series_growth_rates.csv`.

Baseline-adjusted D8-label contrasts compare the log change from each series'
D1 image to its D8-label image, then average species-specific log contrasts
with equal species weight. For mycelial area, the treatment/control ratios of
those multiplicative changes were 0.647 for millipedes (four species), 0.535
for woodlouse group 1 (four), 0.559 for woodlouse group 2 (four), 0.845 for
nematodes (four), and 1.118 for collembola (three: Hf, Pv, Rb). For example,
0.647 means the estimated D1-to-D8-label growth multiplier in millipede
series was 0.647 times the matched control multiplier; it is not a 35.3%
reduction in absolute area. Mite coverage is only Hf and Pi, and no
D1-baseline-adjusted enchytraeid contrast is available. All outcomes and
bootstrap intervals are in
`data/derived/fricker_grazers/baseline_adjusted_d8_contrasts.csv`.

The source methods report that trays were randomly repositioned weekly, but do
not document random allocation to grazer treatments. Accordingly, the
contrasts describe associations in this archive, not randomized grazer
effects. The baseline-adjusted bootstrap intervals also have limited
replicate support, are not adjusted across outcomes, and are descriptive; they
are not randomization-based inference.

## Artifacts and rerun

- `scripts/fricker_series_model.py`
- `data/derived/fricker_grazers/series_model_summary.json`
- `data/derived/fricker_grazers/series_trajectory_cv.csv`
- `data/derived/fricker_grazers/series_growth_rates.csv`
- `data/derived/fricker_grazers/baseline_adjusted_d8_contrasts.csv`
- `data/derived/fricker_grazers/day2_start_series.csv`

Rerun the extraction first with `python scripts/fricker_grazers_analysis.py`,
then run `python scripts/fricker_series_model.py` from the workspace root.

## Sources

- Crowther, Boddy & Jones (2011), [Species-specific effects of soil fauna on
  fungal foraging and decomposition](https://doi.org/10.1007/s00442-011-2005-1).
  The methods give the scheduled captures, report random weekly tray
  repositioning, and do not state randomized treatment allocation.
- [Fungal Network Grazer Analysis archive, Zenodo record 14931953](https://zenodo.org/records/14931953).
- Aguilar-Trigueros et al. (2025), [Not Extremely Plastic: Testing the Limits
  of Morphological Plasticity in Fungal Mycelia in Response to Soil
  Grazers](https://pmc.ncbi.nlm.nih.gov/articles/PMC12675975/), which limits
  its main analyses to day 8 because some colonies approach tray edges later.
