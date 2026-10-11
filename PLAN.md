# Mycelium growth — observable dynamics and causal modeling plan

Approach: Statistical Rethinking workflow (McElreath) — estimand → DAG → generative
simulation → statistical model → validate on simulated data → fit real data →
test transport to other datasets.

Anchor dataset: **MyceliumSeg** (Yuan et al. 2025, Sci Data, doi:10.1038/s41597-025-06265-1;
Zenodo 15224240; code github.com/yuanqianguang/MyceliumSeg-benchmark).

**Minimum dataset deliverable:** a reproducible image-to-mask segmenter trained on
the released annotated pairs and evaluated on held-out masks. Area summaries,
coverage-level models and temporal growth analyses are downstream work, not
preconditions for delivering the segmenter.

The model work has three distinct targets:

1. **Segmentation:** train an image-to-mask model from the released image/mask pairs.
   This is a standalone computer-vision task; it needs no dish IDs or capture times.
   Our reproduced SegFormer baseline (#28) is trained on 457 released *G. lucidum*
   image/mask pairs and judged against 50 held-out GL masks, with 60 condition masks
   as a separate robustness check. It predicts the mycelium region mask inside the
   dish and makes no growth claim. Area is counted and normalized afterward, outside
   the segmentation objective. For the full-data deliverable, train a separate final
   model on all 547 unique labeled image/mask pairs; the 20 duplicate records are
   collapsed after verifying their masks agree. That fit includes the #28 evaluation
   masks, so #28 remains the independent benchmark score. The other release photos
   have no manual masks and cannot enter supervised training without a separate
   pseudo-label method.
2. **Morphology by area coverage:** calculate `area_fraction = A_colony / A_dish` per
   image, then rank or bin images from low to high dish coverage within known species,
   condition and scale strata. This supports models of how other image features vary
   with coverage. The rank is a cross-sectional coverage level, not acquisition
   time or a same-dish trajectory.
3. **Temporal growth:** fit sequence or mechanistic models against actual ordered
   observations with specimen identity and time. MyceliumSeg cannot identify those
   dynamics or physical rates, but its segmenter and area-coverage morphology can be
   tested as transferable image-level components on time-resolved datasets.

**Execution order:** record the Gate C decision (done — `notes/gate.md`); define the
observable estimands (done — `notes/estimands.md`) and sequence capabilities (done —
`notes/provenance.md`); train and validate the per-image segmenter; run it over the
release and build the measurement table; estimate measurement error; fit morphology
against area-coverage level; then fit temporal growth and intervention models on data
with identifiable time and specimen identity, testing transfer of the image-level
components.

### Identifiability update (2026-10-10)

For the radial model `R = min(R_max, k * max(0, t - tau))`, an unknown time scale
makes the absolute rate and lag non-identifiable from unordered radii. For any
`c > 0`, replacing `t` and `tau` by `c*t` and `c*tau`, and `k` by `k/c`, leaves
every radius unchanged. Thus MyceliumSeg cannot identify `k` in mm/day or `tau`
in days from its cross-section, regardless of sample size.

Dimensionless `tau/T` and `k*T/R_max` are mathematically identifiable only under
an assumed sampling-time distribution and selection independent of observed
morphology. For example, assuming `t ~ Uniform(0, T)` gives a radius CDF slope
and a wall mass tied to those two ratios. The MyceliumSeg cadence and selection
rule are unknown, however, so this is an assumption-based sensitivity exercise,
not a data-identified estimate. It is not on the active analysis path. A
relative rate ratio would additionally require comparable cadence and selection
across temperature groups; the ten 15 °C images, absent dish replication, and
unknown selection make that contrast too weak and bias-prone to fit.

The ordering routes have also been assessed: dish grouping was not recovered
(#23), and the area-rank pseudo-sequence failed to beat persistence forecasts
(#65). Area rank is a coverage coordinate, not chronology. MyceliumSeg therefore
remains limited to segmentation and cross-sectional morphology unless the
linked dish/time metadata become available in the future. The author-metadata issue
(#70) is closed; no contact was made.

Temporal modeling stays dataset-specific. Fricker/Boddy provides actual times
and network outcomes; *Podospora* provides ordered movies and ridge-occupancy
proxies. Neither currently supplies a validated, time-resolved equivalent
colony-radius measurement for the radial model. The controlled own-rig study
(#49) remains the eventual route to identified intervention and rate claims,
but is currently parked at the user's request. If resumed, its pre-registration
(#52) must use the rig design and time-resolved pilot evidence, not MyceliumSeg.
The optional assumed-cadence CDF model is deferred unless a specific scientific
question justifies its assumptions.

The bounded-radius likelihood comparison and provisional #31 error calibration
are recorded in [`notes/bounded_radius_likelihood.md`](notes/bounded_radius_likelihood.md).
They prepare a future eligible radius fit but do not supply a target-specific
measurement model or make any dataset eligible by themselves.

**Backlog synchronization (2026-10-10):** #39 and #40 are complete, and the
current Phase 4 epic #36 is closed after the #69 likelihood comparison. The
completed setup, MyceliumSeg triage, measurement, and estimand parent issues
(#2, #19, #27, #32) are also closed so they no longer appear as ready work.
Issue #70 was closed by user decision because the requested dish/time and label
metadata are unavailable; no email was sent, and Gate C remains in force. The
Fricker network snapshot assessment (#46) is complete: use its
traits only for cautious, study-specific network comparisons, after noting the
33-pair Zenodo versus 36-row author-table mismatch. It does not supply a prior
for MyceliumSeg masks/area fractions or any growth-time model. The E6 transport
epic (#43), #48, and #70 are closed. #48 was dropped because pursuing private
data would require an unreasonable access burden; no request was sent.
The controlled own-rig epic (#49) remains the main eventual route to identified
intervention and physical-rate claims, but it is deferred until the user is ready.
The parent and children (#50–#53) are closed in Chainlink as parked, not completed;
reopen them together when the experiment is resumed. When active, #52 must use
the rig design and time-resolved pilot, not a MyceliumSeg-derived rate ratio.

---

## 0. What MyceliumSeg actually contains (and what it constrains)

| Fact | Consequence for causal work |
|---|---|
| 20,176 JPGs, 4608×3456, 90-mm dishes, dark incubation, fixed-position camera (FPheno2000) | Constant geometry → pixel area can become mm² with per-setup calibration |
| Species: *G. lucidum* (GL), *G. sinense* (GS), *P. ostreatus* (PO), *Trametes* (TS) | 4 levels — enough for partial pooling, but not much |
| Medium: GL, GS on **MYG**; PO, TS on **PDA** | **Medium is completely confounded with species.** The medium effect can't be identified from this data |
| Temperature: GL at 15 °C and 25 °C; others 25 °C only | A possible condition contrast, but not a growth-rate estimand without dish/time identity and adequate replication |
| 567 labeled masks (507 GL, 10 each GS/PO/TS/MYG/PDA/TEMP15); 19,609 unlabeled | Train a segmenter on the labeled masks, then measure colony size on the unlabeled images |
| Growth-stage and sclerotium-colour counts reported (Tables 2–3) | We don't yet know whether these labels ship per image |
| Files numbered sequentially (`00000001`…); the paper describes images "with distinctive edge morphology" and periodic fixed-position acquisition | **Dish ID and timestamp may be missing**; whether an edge-morphology selection rule was applied is undocumented, and per-image time/frame dependence is untestable under Gate C (#35, `notes/assumptions.md` §5) |

Boundary IoU is only ~11–30% even for the best model (SegFormer). Colony *area* is
measured more reliably than the edge/radius. Any feature derived from the boundary,
including tip-like events, needs its own validation and measurement-error model.

### What counts as an observable claim

The project will distinguish three levels of interpretation:

1. **Image observables:** mask area, area fraction, equivalent radius, perimeter,
   roughness, intensity summaries, and boundary protrusion or skeleton-endpoint
   candidates.
2. **Area-coverage observables:** image-level morphology as a function of the fraction
   of the dish covered. Sorting by `area_fraction` gives a low-to-high coverage level;
   it does not assign elapsed time, specimen identity or a temporal trajectory.
3. **Sequence observables:** frame-to-frame changes, growth increments, boundary-event
   counts, and persistence of candidate tips, when frame order is known.
4. **Biological mechanisms:** hyphal tips, nuclei, branching, radial extension in mm/day,
   temperature effects, and medium effects. These require resolution, timing, and design
   evidence that the current MyceliumSeg release does not provide.

The pipeline may report a candidate tip event. It must not call that event a biological
hyphal tip until a resolution check or a higher-resolution dataset validates the mapping.

---

## Phase 1 — Data triage and identifiability gate

Download in this order (total archive 56.5 GB; check free disk first):
1. `labeled-MYG_PDA_TEMP.zip` (108 MB), `labeled-GS_PO_TS.zip` (127 MB)
2. `labeled-GL.zip` (1.5 GB)
3. One unlabeled shard, e.g. `unlabeled-GL2.zip` (6.2 GB)

Checks (write to `data/metadata.csv`: file, species, medium, temp, dish_id, timestamp, stage):
- [x] EXIF `DateTimeOriginal` present? → **no**, 0 / 20,176 (#21)
- [x] Does sequential numbering follow time within a dish? → **no**, ~51% forward steps (#22)
- [x] Can images be grouped into dish tracks? → **no**: the rim dot is measurable but does not
      fingerprint dishes (largest shared angle: 37 images; 720/720 bins occupied), and
      rim-embedding clustering finds no structure beyond the byte-identical duplicates (#23)
- [x] Are stage and sclerotium-colour labels included anywhere in the release? → **no** (#24);
      note the folder-derived species/medium labels are wrong for 20 duplicated images (#23)
- [x] Mask area in pixels → mm² (dish rim = 90 mm) — **calibrated per condition block** in
      `data/triage/calibration.csv` (plus a per-image `image_scale.csv`): apparent dish size
      differs by block (r/W 0.302 / 0.332 / 0.224), so one global constant is off by ~36%, and
      the flat `unlabeled-GS_PO_TS` shard needed a second constant for its 91 GS-scale images (#25)

**Gate:**
- **A. Dish + time recoverable** → longitudinal growth-curve model (the best case)
- **B. Only time, or only dish** → pooled growth curves with weaker identification
- **C. Neither** → use MyceliumSeg for segmentation and cross-sectional morphology;
  its author-metadata route (#70) is closed, with no contact made. If linked dish/time
  metadata become available later, reassess the gate. For temporal analyses, use the
  Fricker/Boddy grazer time series (Zenodo 14931953, see Phase 5), which has explicit
  capture times, treatment labels, and ready-made network data; causal treatment
  claims depend on assignment details.

**Recorded decision (2026-10-05, #26): Gate C.** Dish identity and capture time are
not recoverable from the released MyceliumSeg files, so MyceliumSeg work is
segmentation, morphology, and cross-sectional or frame-indexed measurement work —
never a causal temperature study. The full record lives in `notes/gate.md`.

### Gate C deliverable — complete

`notes/gate.md` records the evidence for missing dish identity and capture time, the
consequences for each estimand, the permitted cross-sectional analysis, and the
fallback to Fricker/Boddy, Podospora, or a controlled rig for time-resolved and
intervention claims. Treat any MyceliumSeg output as a temporal growth curve only when that
note's checklist of preconditions is satisfied. This is a decision record, not a
request to discard MyceliumSeg.

---

## Phase 2 — Measurement pipeline

1. The Gate C record is done (`notes/gate.md`, #26), the measurement contract is
   defined (`notes/estimands.md`, #60): population, inclusion rule, duplicate
   handling, units, calibration and uncertainty target for the image-level and the
   conditional frame-level estimand, with the reporting guards in
   `scripts/measurement_contract.py` and `tests/test_measurement_contract.py`.
   The provenance specification is done too (`notes/provenance.md`, #61): every
   dataset audited for ordered frames, specimen identity, capture interval,
   resolution, duplicates and condition labels, with the capability matrix in
   `data/triage/sequence_inventory.csv` mapping each dataset to the estimands it
   can identify — MyceliumSeg has no usable longitudinal sequence, so dynamics and
   interventions are anchored on Fricker/Boddy, *Podospora* and a controlled rig.
2. Treat segmentation as its own deliverable: train an image-to-mask model on the
   released labeled pairs and evaluate it on held-out masks. The #28 SegFormer run is
   our trained reproduction baseline for this task. Any additional architecture or
   training recipe is another image-to-mask experiment, still independent of growth
   modeling. Fit on training data, keep the GL test images untouched for final evaluation,
   and use the condition masks as a separate robustness check.
3. After the labeled hold-outs and auditable sample pass QA, run inference over the full
   release. The existing #29 artifacts cover 125 images, not the full 20,176. Preserve
   archive, image ID, duplicate status, condition provenance, frame-order status and
   time-unit status. Do not infer elapsed time from file numbering.
4. Per image or frame, extract colony area A, area fraction, equivalent radius
   `r = √(A/π)`, perimeter, edge roughness, intensity summaries, and density proxies.
   Write one provenance-rich `data/measurements.csv` with image-level and mask-level
   records kept distinct.
5. **Error model for the measurement:** on held-out labeled masks, compare predicted
   and true area, radius, perimeter, and boundary features to estimate bias and SD.
   Quantify duplicate sensitivity and, only where ordered sequences exist,
   frame-to-frame repeatability. This is the measurement-error layer (Ch. 15).
6. **Candidate boundary-event validation (#62, completed as a bounded mask/resampling analysis):** compare predicted and reference colony-mask candidates across 300–4608 px and report scale stability and mask disagreement. Some candidates persist across resampling, but this does not establish temporal persistence or biological tips. MyceliumSeg has no independent hyphal-tip annotations, so no resolution supports a biological tip label; see `notes/tip_event_validation.md`.

---

## Phase 3 — Observable estimands and causal DAG

**Near-term MyceliumSeg estimand:** first define the image-level distribution of an
observable feature, such as log area fraction, equivalent radius, or a calibrated
shape feature, by species and known condition. If ordered observations are validated,
add the expected `Δ log(area)` or fractional change per frame. If the capture interval
is known, convert it to a physical rate; otherwise report only per-frame change. If
frame order and dish identity cannot be recovered, retain the cross-sectional
estimand and label it associational rather than calling it a temporal growth curve.

**Model by area coverage:** `area_fraction` is also a useful image-level coordinate:
rank or bin images from low to high covered-dish fraction, within known species,
condition and scale strata, then model other features (for example edge shape, texture
or learned image embeddings) against that coordinate. This can describe how morphology
varies across observed coverage levels in the selected release. It does not make the
sorted images a time series, and area itself should not be modeled as an outcome of its
own rank.

That definition now exists as the measurement contract — `notes/estimands.md`
(machine-readable mirror: `scripts/measurement_contract.py`, #60). Every number
reported in this phase has to name one of its estimands, carry its unit, and
satisfy its precondition; the guards refuse the per-frame → per-time conversion
while no capture interval exists.

**Future causal estimand:** the effect of a controlled intervention, such as temperature
or medium, on a time-resolved growth feature. This moves to a dataset with identifiable
dish, time, treatment assignment, and replication.

**Secondary:** species differences in observable morphology at 25 °C, with extension
rate and lag time reserved for datasets that provide time and specimen identity
(descriptive: species and medium can't be separated here).

Draft DAG (edit in dagitty.net; code in `R/dag.R`):

```
Species S ──► Medium M            (fixed by design → S and M are aliased)
Treatment T ──► latent dynamics Z ──► observable feature Y
Frame index f / time t ──► Z      (only when sequence order is known)
Dish D ──► Z                      (varying effect when dish identity is known)
Imaging I ──► Y*                 (segmentation and boundary measurement error)
Selection: "distinctive edges" ──► inclusion   (hypothesized collider Z -> Sel <- I;
                                              mechanism UNDOCUMENTED; time/frame
                                              dependence UNTESTABLE — #35,
                                              notes/assumptions.md §5)
Duplicate or mislabeled file ──► observed condition label
```

Implications to check with `dagitty::adjustmentSets`:
- Do not claim T → k is identified from MyceliumSeg until treatment allocation,
  dish identity, time, and replication are established.
- A frame index is not a duration. Do not report mm/day or tip events per hour without
  a capture interval.
- Stratifying by growth stage can bias a treatment effect because stage is downstream
  of growth.
- The medium effect is not identified from **MyceliumSeg**, where medium is aliased
  with species; don't estimate that contrast from the anchor release.
- Duplicate images must be removed or clustered before splitting data or estimating
  uncertainty.

---

## Phase 4 — Generative model (write it first, fit it to simulated data)

### 4A. Observable morphology model

Start with the outcome that the release can identify: an image-level morphology
feature with segmentation and calibration error. Use partial pooling across species
and condition blocks, while keeping species and medium separate in the interpretation
because they are aliased in this dataset. The first results should be predictive
distributions for area fraction, equivalent radius, and selected shape features.

For an area-conditioned model, let `p[i]` be measured area fraction and `M[i]` be a
different morphology feature or image embedding:

```
M[i] ~ Normal(f(p[i]) + alpha[species[i]] + beta[condition[i]], sigma_measurement[i])
```

Use a smooth or monotone `f` only when diagnostics support it; propagate segmentation and
calibration uncertainty and remove duplicate copies before fitting. This is a model of
morphology across **area-coverage levels** in a cross-section, not a model of change over time.
Do not use `area_fraction` as both the sorting variable and the response.

### 4B. Observable sequence model when provenance exists

Only use this model after the provenance inventory identifies ordered frames and
the unit of observation (`notes/provenance.md`, #61).
For a feature such as area or area fraction, use a bounded or transformed observation
model with image-level measurement error and sequence-level dependence. For example:

```
Y_true[i, f+1] ~ Normal(Y_true[i, f] + g(Y_true[i, f], f), σ_process)
Y_obs[j]      ~ Normal(Y_true[i, f] + bias_condition[j], σ_measurement[j])
```

For count-like boundary events, use a Poisson or negative-binomial observation model
only after validating that the event detector is stable across resolution and masks.
The initial outputs are predictive distributions for feature changes and event counts,
not claims about nuclei, individual hyphae, or biological branching.

Workflow:

1. Define the feature and unit: per frame, per known interval, or cross-sectional.
2. Simulate sequences with known growth changes, segmentation noise, missing frames,
   duplicates, and false tip-like events.
3. Fit the model and check parameter and predictive recovery.
4. Use labeled masks and higher-resolution sequences to calibrate the measurement model.
5. Validate held-out sequences or datasets with posterior predictive checks.

**Exploratory pseudo-sequence training experiment (#65; completed under revised #42):**
the deduplicated unlabeled MyceliumSeg masks were sorted within GL/MYG/25C/scale1,
assigned `t*` as the unitless area-coverage quantile, and aggregated into 100 equal-count
bins. An AR(3) model predicted next-bin changes in five median mask-shape features on
30 held-out upper-coverage bins, with rolling and recursive comparisons against
persistence and a direct cubic `t*` trend. AR(3) RMSE was 30–67% worse than rolling
persistence and 18–107% worse than recursive repeat-last persistence on all five
features. The result does not support useful shape forecasting from this pseudo-index;
the saved model is retained as a preprocessing and evaluation prototype for future
verified time-lapse data. Bins contain different, unlinked images; `Δ` is a
cross-sectional coverage contrast, `t*` is not elapsed time, and this experiment
cannot estimate growth or chronology. Do not predict area from `t*`, because area
fraction defines the ordering itself. See `notes/pseudo_time_model.md` for metrics.

### 4C. Mechanistic growth model as a later component

Mycology prior knowledge: on agar, colonies grow radially at a near-constant rate after
a lag (Trinci), then saturate at the dish wall (45 mm).

```
R_true[d,t] = min( 45 , k[d] * max(0, t - τ[d]) )          # or a smooth version (softplus)
R_obs[i]    ~ Normal( R_true[d[i], t[i]] + bias , σ_seg )  # σ_seg, bias from Phase 2
log k[d]    = α[S] + β_T * (T == 15) + z_k[d] * σ_k         # varying effects per dish
log τ[d]    = γ[S] + δ_T * (T == 15) + z_τ[d] * σ_τ
```

Estimand = `exp(β_T)`, the rate ratio from 25 → 15 °C (also computed as a contrast on
the outcome scale, mm/day).

Workflow:
1. `R/sim.R` (#38, implemented): simulate dishes using known parameters; the
   reproducible default dataset and parameter/data dictionary are in
   `data/derived/dish_growth_simulation/` and `notes/simulated_growth_data.md`.
2. Candidate prior predictive checks (#39, implemented) in `R/prior_predictive.R`;
   priors, wall-contact times and the bounded-radius likelihood caveat are recorded
   in `notes/prior_predictive_growth.md`.
3. Fit with `ulam` (Rethinking + compatible RStan backend) and check recovery
   against known values (#40, implemented; see `notes/ulam_parameter_recovery.md`).
4. Fit a real time series only when eligible. Diagnostics: R-hat, n_eff, trace plots,
   posterior predictive curves per dish. Issue #69 compares bounded likelihoods and
   records provisional #31 error scales; its note requires target-specific paired-mask
   calibration and endpoint checks before any measured-radius fit. No current
   time-resolved dataset qualifies for this radial measurement model.
5. Alternatives to compare with PSIS/WAIC: logistic area growth; a Gompertz-type curve;
   an ODE version (Ch. 16 "generalized linear madness"). This connects to the Boswell
   kinetics paper in the archive

The mechanistic model is eligible for fitting only when a dataset supplies identifiable
time and a defensible mapping from image features to the biological quantity being
modeled. It can then become one component in a composition of models: segmentation and
measurement from MyceliumSeg, growth dynamics from time-resolved data, and causal
intervention effects from randomized or controlled experiments.

---

## Phase 5 — Does the model transport?

### What "applies to others" means

Transport needs a claim about which parts of the model are **invariant** and which are
**dataset-specific**. Make it explicit before testing:

| Invariant (the hypothesis) | Allowed to differ (re-estimated per dataset) |
|---|---|
| Growth form: lag τ, then constant tip-driven radial extension k, then saturation (Trinci peripheral growth zone) | Absolute k, τ (species, substrate, scale) |
| Temperature/nutrient act multiplicatively on k (log-scale effects) | Measurement model (camera vs scanner vs microscope) |
| Dish/colony-level variation ~ partial pooling | Units (mm/day vs µm/h), sampling interval |

A dataset "fits" if posterior predictive curves, using pooled priors from earlier datasets,
cover held-out colonies. Score this with out-of-sample predictive error (PSIS/LOO),
not in-sample fit.

### Transport ladder (from the curated dataset report)

Ordered from closest to MyceliumSeg to furthest:

| # | Dataset | Why it's useful | Main difference / risk |
|---|---|---|---|
| 1 | **MyceliumSeg hold-outs** (GS/PO/TS) | Same lab, rig and units → pure species transport | Only 10 labeled each; needs segmenter to generalize |
| 2 | **Fricker/Boddy grazer series** (Zenodo 14931953, 10.2 GB, CC BY 4.0) — *H. fasciculare, P. velutina, P. impudicus, R. bicolor* | Real time series with capture times and treatment labels; processed network data + R code (github.com/aguilart/FungalGrazers); causal interpretation depends on treatment-assignment evidence | Wood block → soil, cord-forming; radial extension isn't circular. Use area or extent over time |
| 3 | **Fricker network dataset** (Zenodo 5725751, 3.2 GB) | Cross-sectional, study-specific summaries of author-defined network traits | Snapshot only. #46 found 33 archived pairs versus 36 author-table rows; do not transfer to MyceliumSeg segmentation/area or use for growth-time claims |
| 4 | ***Podospora anserina* movies** (Figshare 22666138) | Microscopic, 18 min frames at 1.63 µm/source-pixel; three movies per M0/M2 medium | Relative timing from first frame; each film spans 20.4–25.2 h. A ridge-mask occupancy model is in #47; no labels to validate tip counts. CC BY 4.0. |
| 5 | ***Aureobasidium*** (Zenodo 14001310) | Microscopy time-lapse, used for growth simulation | Yeast-like dimorphic fungus; check sizes |
| — | **MDPI 110-species time-lapse** (26,451 images, every 30 min, 5–7 days) | Potential petri-dish, macroscopic, cross-species comparison | **Not public; excluded by decision (#48 closed)** — access request not worth pursuing |
| — | AMF package (Figshare 27889143) | Network/flow mechanism | Only a demo subsample; low priority for growth curves |

Excluded as non-fungal (fine for testing tools, not for testing the model): mycelyso
(*Streptomyces*), *Phytophthora* (oomycete), *Physarum*. Synthetic sets can be used for
segmenter pre-training only.

Steps:
1. **Within MyceliumSeg (#44; completed):** fit on unlabeled GL/MYG/25C/scale1 masks and
   score against deduplicated #31 reference masks from held-out condition/species blocks.
   There are 88 calibrated reference rows; only GL/MYG/15C (n=10) is a clean held-out
   condition group. GS/PO/TS rows (n=9–10 per group) all have label conflicts, so they
   remain diagnostics rather than species-transport evidence. On the clean GL/15C group,
   the spline beats the train-median baseline for circularity and edge roughness, but
   loses for solidity, bbox fill, and eccentricity. The broader scores are mixed and
   groups are small. See `notes/coverage_transport.md`. This is cross-sectional transport,
   not a growth curve.
2. **Dataset model-capability inventory (#67):** use the sequence/provenance audit (#61)
   plus each dataset's actual measurements and design to state which image, cross-sectional,
   temporal, intervention, species-prior, or network models it supports. This was the
   prerequisite for dataset-specific fits; the first-pass Fricker/Boddy analysis (#45)
   and *Podospora* analysis (#47) are complete. The cross-dataset growth-form
   comparison (#37) is also complete; it does not support a shared invariant form.
3. **Fricker grazer series (#45 and #68; complete):** the 155 identified series have
   filename day labels and actual EXIF elapsed times plus curated network measurements.
   The paper cadence, archive mismatch, grouped model comparison, and descriptive day-8
   contrasts are documented in `notes/fricker_grazers_analysis.md`. Treatment assignment
   is not documented in the reviewed primary methods, so causal interpretation remains
   conditional. The dataset-specific, baseline-relative trajectory model, held-out-series
   comparison, and baseline-adjusted contrasts are documented in
   `notes/fricker_series_model.md`. Do not assume this form transports yet.
4. ***Podospora* (#47; complete):** the six ordered M0/M2 movies support an
   exploratory ridge-occupancy growth model and replicate-level contrast. The 18-minute
   cadence, 1.63 µm/source-pixel scale, and CC BY 4.0 licence are attributed to Figshare,
   not inferred from AVI playback. The ridge proxy is unvalidated and cannot count tips;
   see `notes/podospora_growth_model.md`.
5. **Fricker network snapshots (#46; assessed):** the paper supports descriptive
   cross-sectional network-trait comparisons under its own measurement pipeline.
   The staged Zenodo bundle has 33 pairs, while the author table has 36 real rows
   across 12 taxa with unmatched IDs and unbalanced replication. The two imaging
   scales and cultivation regimes align with taxonomic groups. Do not use these
   data for time series or as a transferable MyceliumSeg segmentation/area prior;
   see `notes/fricker_network_snapshots.md`.
6. **Growth-form invariance (#37; assessed):** held-out checks found no support
   for treating lag → extension → saturation as invariant across Fricker/Boddy and
   *Podospora*. The outcomes are distinct; lag and saturation are not identified over
   the observed windows. See `notes/growth_form_invariance.md`.
7. **Your own rig** (`/mnt/sharedOs/archive/mycelium-workspace`: RPi camera + LED
   timelapse). This is the strongest test, because *you* control the intervention:
   - Cross species × medium (e.g. oyster on both PDA and MYG) to break the confound in MyceliumSeg
   - Randomize dish position; log temperature; image every N hours with fixed lighting
   - Pre-register the predicted rate ratio from Phase 4, then compare
   - Publishing template: Harrington & Harrington 2026 *Physarum* scanner set (Zenodo
     22868368, OME-Zarr, CC BY 4.0). Either six dishes on a flatbed scanner or the
     existing Pi camera. Publish the result on Zenodo, since open time-lapse data is the field's main gap

### Access decisions (2026-10-10)
- MDPI 110-species data (#48): excluded; obtaining the private dataset is not worth
  the access burden. No request was sent.
- MyceliumSeg dish/time metadata (#70): unavailable to the user; no request was sent.
  Gate C remains in force, so MyceliumSeg supports image-level and cross-sectional
  morphology work only.

---

## Tooling

- R: `rethinking`, `cmdstanr` (+ `install_cmdstan()`), `dagitty`, `posterior`
- Python: MMSegmentation / benchmark repo for masks; `exiftool`; `scikit-image` for shape metrics
- Book: *Statistical Rethinking* 2nd ed. — yes, get it. Key chapters: 1–6 (DAGs, confounds,
  colliders), 9 (MCMC), 11 (GLMs), 13–14 (multilevel, varying slopes), 15 (measurement
  error), 16 (custom scientific models / ODEs)
- Course: stat_rethinking_2026 (lectures recorded; beginner = first half of the book;
  experienced = starts at multilevel models)

## Suggested layout

```
data/raw/        zenodo zips (gitignored)
data/metadata.csv
py/segment.py    inference + shape metrics → data/measurements.csv
R/dag.R  R/sim.R  R/fit_growth.R  R/transport.R
notes/           decisions, author emails, gate outcomes
```

## Open questions
1. Which available datasets contain ordered frames, reliable intervals, and repeatable
   specimen identity? — **answered:** the capability matrix in
   `notes/provenance.md` / `data/triage/sequence_inventory.csv` (Fricker/Boddy
   qualifies for all three; *Podospora* for order + identity; Physarum for all
   three but non-fungal; MyceliumSeg for none)
2. At what image resolution can a boundary protrusion or skeleton endpoint be validated
   as a tip-like event? — **answered for the current release:** #62 measured algorithmic
   stability across 300–4608 px, but biological validation remains unresolvable because
   the released labels mark colony extent and contain no independent hyphal-tip annotations
   (`notes/tip_event_validation.md`).
3. Can MyceliumSeg support frame-indexed observables, or only cross-sectional
   morphology? — **answered:** cross-sectional only; frame-indexed change is
   defined but not estimated until order, identity and interval exist
   (`notes/provenance.md` § 4.1, Gate C)
4. How many 15 °C GL dishes exist? That count sets the power for any temperature contrast.
5. Is image selection independent of morphology and latent time/frame? — **closed as
   unresolvable from the current release (#35, 2026-10-09):** the paper *describes* collecting images "with distinctive edge
   morphology" and periodic fixed-position acquisition, but documents no edge-morphology
   inclusion rule (doi:10.1038/s41597-025-06265-1). Morphology-based selection is
   therefore **undocumented** — the release cannot be shown to over-represent the
   estimands' features and the bias direction is unknown — and time/frame dependence is
   **untestable** under Gate C (no dish identity, no timestamps, no ordered sequences;
   periodic acquisition is process-level context, not per-image linkage). Testing either
   needs the documented inclusion rule, the candidate / included / excluded image frame,
   and linked dish/time or stage metadata. Cross-sectional interpretation is bounded to
   the selected release; the Gate C prohibition on causal temperature/medium and physical
   growth-rate claims is preserved. Closure records that the needed evidence is not in
   the current release; it does not establish selection independence. `notes/assumptions.md` §5.
6. Which parts of the observable measurement model transport across datasets, and which
   must be re-estimated for each camera, scale, and resolution?
