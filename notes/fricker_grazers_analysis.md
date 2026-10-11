# Fricker/Boddy time-series and grazer analysis (#45)

**Status:** first-pass analysis complete; #45 closed. The dataset-specific
repeated-measures refinement #68 is also complete; see
`notes/fricker_series_model.md` for its model and outputs.

## Question answered: does the paper state a time step?

Yes. The original Crowther, Boddy & Jones (2011) paper says trays were
photographed before invertebrates were added, then after 2, 4, 8, 16, 24, 38,
52, 66, and 80 days. It also says trays were randomly repositioned weekly.
That documents capture cadence and position handling; it does **not** document
random allocation of trays to grazer treatments in the reviewed methods.

The later reanalysis, *Not Extremely Plastic*, describes photographs just
before grazer addition and then after 2, 4, 8, 16, and 32 days. It restricts
its main analyses to day 8 because some colonies approach tray edges later.
The Zenodo record says its files are data from the 2011 study and contains four
fungal species: *Hypholoma fasciculare* (Hf), *Phanerochaete velutina* (Pv),
*Phallus impudicus* (Pi), and *Resinicium bicolor* (Rb).

The archive's filenames have day labels 1, 2, 4, 8, 16, 24, and 38. Each of
the 777 linked JPEGs has EXIF capture time; filename order and capture-time
order agree across all 155 series, with no reversals. Actual median intervals
by filename-label gap are:

| Filename gap | Capture transitions | Median EXIF gap |
|---:|---:|---:|
| 1 day | 135 | 0.923 days |
| 2 days | 152 | 1.965 days |
| 4 days | 152 | 4.028 days |
| 6 days | 3 | 5.946 days |
| 8 days | 150 | 7.889 days |
| 14 days | 30 | 14.208 days |

Thus most adjacent labels match actual elapsed time closely. The first label
gap is the exception: the paper describes baseline then a 2-day capture, while
the archive's day-1 to day-2 images are typically about one day apart. Five
`Pi_ctrl2` day-1/day-2 pairs have identical EXIF seconds; four are byte-for-byte
duplicates. A processed workbook (`Pv_ctrl1_d4_4`) has no matching JPEG.
Keep the filename day label and EXIF elapsed time as separate variables. Use
elapsed time from the first linked image for the exploratory curve comparison;
do not silently label it “days since grazer addition.”

The original article and later reanalysis therefore agree that a pre-grazer
baseline and repeated images exist, but their reported late schedules differ
(the original lists day 24 and 38; the reanalysis lists day 32), and the
archive's first EXIF interval does not match the reported two-day interval.
The available files do provide genuine longitudinal time in days. The exact
mapping between every filename label and time since grazer addition remains
uncertain at the first step.

## Image-linked measurements and exploratory model comparison

`scripts/fricker_grazers_analysis.py` extracts the Fungal Network Analysis
workbook's `Summary table` and `Parameters` fields, then links them to JPEG
dimensions, EXIF capture time, and image hash. The output inventory is
`data/derived/fricker_grazers/network_series.csv`; timing evidence is in
`data/derived/fricker_grazers/timing_audit.json`.

The initial temporal comparison uses 537 observations in 140 complete common
species × treatment × replicate series, through filename day label 8. It
includes five common grazer groups (coll, mill, nema, wl1, wl2) and pooled
controls (ctrl1/ctrl2). The five same-session `Pi_ctrl2` day-2 records are
excluded. 125 series have a linked day-1 frame; 15 start at day label 2, and
their first capture is not treated as a verified pre-grazer baseline.

For each positive-valued network outcome, the script compares log-linear,
log1p, several saturating forms, and a flexible day-label factor using five-fold
cross-validation grouped by series. Numeric curves use EXIF elapsed days from
each series' first linked frame. Day-factor models use filename labels. Models
include species and treatment terms and are ridge regressions; this is a
predictive shape comparison, not a reproduction of the published multivariate
PERMANOVA.

| Outcome | Best grouped-CV model | RMSE on log outcome | Next closest |
|---|---|---:|---|
| Mycelial area | Day factor | 0.6033 | Linear, 0.6034 |
| Total length | Day factor | 0.4987 | Linear, 0.5001 |
| Volume | Day factor | 0.7938 | Linear, 0.7970 |
| Tip count | Day factor | 0.4686 | Saturating, tau 8 d, 0.4763 |
| Mean width | Day factor | 0.1589 | Saturating, tau 4 d, 0.1665 |
| Edge count | Day factor | 0.6698 | Saturating, tau 8 d, 0.6739 |
| Loop count | Linear | 1.0379 | Saturating, tau 8 d, 1.0392 |

The flexible and linear fits predict almost equally well for the main size
outcomes. This pass gives no compelling predictive reason to prefer a
saturating model. It does not establish that the true biological process is
linear or that these curves extrapolate beyond the observed window.

## Day-8 treatment-group contrasts

The script also compares precomputed workbook outcomes at filename day label 8
with the pooled controls. It computes an equal-weight average of species-level
log ratios and a stratified bootstrap interval. These are descriptive
treatment-group contrasts, not randomized-effect estimates; the source methods
do not document treatment allocation. Mite has two-species coverage and enchy
one-species coverage, so those rows are explicitly partial.

For mycelial area, the geometric-mean ratios versus control were 1.015 for
collembola, 0.778 for millipedes, 0.805 for nematodes, 0.599 for woodlouse 1,
and 0.578 for woodlouse 2 (all four species). Corresponding bootstrap intervals
and other outcomes are in `data/derived/fricker_grazers/treatment_day8_effects.csv`;
species-level rows are in `treatment_day8_effects_by_species.csv`. No
randomization-based permutation p-values are reported.

## Current interpretation and next work

- The papers do state a time schedule, so this is not a synthetic-time series.
- Use actual EXIF elapsed days and retain filename day labels; flag the first
  interval discrepancy and the 15 missing linked day-1 baselines.
- Exclude or explicitly collapse the five `Pi_ctrl2` same-session pairs before
  computing changes.
- The papers identify the four archived fungal species. The original 2011
  article covers more fungi/strains than the four-species Zenodo reanalysis
  subset; archive-level claims should stay within the four named taxa.
- The 2011 methods say trays were randomly repositioned weekly, not that
  treatment assignment was randomized. A causal grazer effect requires a
  documented allocation procedure or a justified exchangeability assumption.
- Follow-up #68 fits baseline-relative species-by-grazer trajectories on EXIF
  elapsed days, compares the model shapes by held-out series, and estimates
  baseline-adjusted D8-label contrasts. See `notes/fricker_series_model.md`.
  Those contrasts remain associational because the reviewed methods do not
  document treatment allocation.

## Sources

- Crowther, Boddy & Jones (2011), [*Species-specific effects of soil fauna on fungal foraging and decomposition*](https://link.springer.com/article/10.1007/s00442-011-2005-1). The [author-uploaded full text](https://www.researchgate.net/publication/51117765_Species-specific_effects_of_soil_fauna_on_fungal_foraging_and_decomposition) contains the methods schedule and the random-repositioning statement.
- [Fungal Network Grazer Analysis dataset, Zenodo 14931953](https://zenodo.org/records/14931953): source attribution, species list, files, and processed workbook description.
- Aguilar-Trigueros et al. (2025), [*Not Extremely Plastic: Testing the Limits of Morphological Plasticity in Fungal Mycelia in Response to Soil Grazers*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12675975/): later time-schedule description and day-8 analysis limit.
