# Mycelium growth — observable dynamics and causal modeling plan

Approach: Statistical Rethinking workflow (McElreath) — estimand → DAG → generative
simulation → statistical model → validate on simulated data → fit real data →
test transport to other datasets.

Anchor dataset: **MyceliumSeg** (Yuan et al. 2025, Sci Data, doi:10.1038/s41597-025-06265-1;
Zenodo 15224240; code github.com/yuanqianguang/MyceliumSeg-benchmark).

Near-term objective: model the growth signals that the available images can actually
identify. The first target is observable colony dynamics — area fraction, equivalent
radius, boundary shape, roughness, and carefully validated tip-like boundary events —
measured per frame or image. A frame index is a time coordinate only when the sequence
order is known; without a capture interval, results stay in change-per-frame units.

The mechanistic lag → radial extension → saturation model remains a later model
component. It should be fitted only to data with identifiable time and biological
units. MyceliumSeg is useful for segmentation, cross-sectional morphology, and
measurement-error calibration even though its released files do not identify dish
tracks or capture times.

**Execution order:** record the Gate C decision (done — `notes/gate.md`); define the
observable estimands (done — the measurement contract in `notes/estimands.md`) and the
sequence capability (done — the provenance inventory in `notes/provenance.md`);
validate the segmentation environment;
build the measurement table; estimate measurement error; fit a cross-sectional or
frame-indexed observable model; then transport dynamic or intervention models to data
with identifiable time and specimen identity.

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
| Files numbered sequentially (`00000001`…); images "selected for distinctive edge morphology" | **Dish ID and timestamp may be missing**, and the image selection is a possible selection bias — assessed under Gate C (#35): falsified for observed morphology, untestable for latent time/frame (`notes/assumptions.md` §5) |

Boundary IoU is only ~11–30% even for the best model (SegFormer). Colony *area* is
measured more reliably than the edge/radius. Any feature derived from the boundary,
including tip-like events, needs its own validation and measurement-error model.

### What counts as an observable claim

The project will distinguish three levels of interpretation:

1. **Image observables:** mask area, area fraction, equivalent radius, perimeter,
   roughness, intensity summaries, and boundary protrusion or skeleton-endpoint
   candidates.
2. **Sequence observables:** frame-to-frame changes, growth increments, boundary-event
   counts, and persistence of candidate tips, when frame order is known.
3. **Biological mechanisms:** hyphal tips, nuclei, branching, radial extension in mm/day,
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
- **C. Neither** → email the authors (Weizhen Liu / Yongping Fu). Meanwhile, run a
  cross-sectional model of colony size and stage by condition (much weaker inference).
  If the authors can't supply the metadata, **switch the anchor dataset** to the Fricker/Boddy
  grazer time series (Zenodo 14931953, see Phase 5). It has explicit time series, a
  randomized grazer treatment, and ready-made network data

**Recorded decision (2026-10-05, #26): Gate C.** Dish identity and capture time are
not recoverable from the released MyceliumSeg files, so MyceliumSeg work is
segmentation, morphology, and cross-sectional or frame-indexed measurement work —
never a causal temperature study. The full record lives in `notes/gate.md`.

### Gate C deliverable — complete

`notes/gate.md` records the evidence for missing dish identity and capture time, the
consequences for each estimand, the permitted cross-sectional analysis, and the
fallback to Fricker/Boddy, Podospora, or a controlled rig for time-resolved and
intervention claims. Treat any MyceliumSeg output as a growth curve only when that
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
2. Establish a reproducible SegFormer environment and baseline with the benchmark repo
   (MMSegmentation); issue #28 is split into environment, GPU smoke, baseline protocol,
   and reporting deliverables.
3. Run inference on labeled hold-outs and a small auditable unlabeled sample. Preserve
   archive, image ID, duplicate status, condition provenance, frame order, and whether
   a physical time interval is known. Do not infer elapsed time from file numbering.
4. Per image or frame, extract colony area A, area fraction, equivalent radius
   `r = √(A/π)`, perimeter, edge roughness, intensity summaries, and density proxies.
   Write one provenance-rich `data/measurements.csv` with image-level and mask-level
   records kept distinct.
5. **Error model for the measurement:** on held-out labeled masks, compare predicted
   and true area, radius, perimeter, and boundary features to estimate bias and SD.
   Quantify duplicate sensitivity and, only where ordered sequences exist,
   frame-to-frame repeatability. This is the measurement-error layer (Ch. 15).
6. Add candidate tip-like boundary events only as a secondary analysis. Require
   persistence, resolution checks, and false-event estimates before reporting them;
   never label them biological hyphal tips from MyceliumSeg alone.

---

## Phase 3 — Observable estimands and causal DAG

**Near-term MyceliumSeg estimand:** first define the image-level distribution of an
observable feature, such as log area fraction, equivalent radius, or a calibrated
shape feature, by species and known condition. If ordered observations are validated,
add the expected `Δ log(area)` or fractional change per frame. If the capture interval
is known, convert it to a physical rate; otherwise report only per-frame change. If
frame order and dish identity cannot be recovered, retain the cross-sectional
estimand and label it associational rather than calling it a growth curve.

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
Selection: "distinctive edges" ──► inclusion   (collider on Z -> Sel <- I;
                                              #35: falsified for observed
                                              morphology, untestable for latent
                                              time/frame — notes/assumptions.md §5)
Duplicate or mislabeled file ──► observed condition label
```

Implications to check with `dagitty::adjustmentSets`:
- Do not claim T → k is identified from MyceliumSeg until treatment allocation,
  dish identity, time, and replication are established.
- A frame index is not a duration. Do not report mm/day or tip events per hour without
  a capture interval.
- Stratifying by growth stage can bias a treatment effect because stage is downstream
  of growth.
- The medium effect is not identified. Say so explicitly; don't estimate it
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
1. `R/sim.R`: simulate dishes using known parameters
2. Fit with `ulam` (rethinking + cmdstanr) and check that the known values are recovered
3. Prior predictive checks: curves must stay within 0–45 mm and realistic time scales (days)
4. Then fit the real data. Diagnostics: R-hat, n_eff, trace plots, posterior predictive curves per dish
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
| 2 | **Fricker/Boddy grazer series** (Zenodo 14931953, 10.2 GB, CC BY 4.0) — *H. fasciculare, P. velutina, P. impudicus, R. bicolor* | Real time series; **randomized grazer intervention** (a second causal estimand); processed network data + R code (github.com/aguilart/FungalGrazers) | Wood block → soil, cord-forming; radial extension isn't circular. Use area or extent over time |
| 3 | **Fricker network dataset** (Zenodo 5725751, 3.2 GB) | Many taxa (asco/basidio/zygo) → species-level variation prior | Mostly snapshots? Check whether time series exist before using |
| 4 | ***Podospora anserina* movies** (Figshare 22666138; 9897056, 10626968) | Microscopic, ~18 min frames; **M2 vs M0 medium triplicates = a manipulated nutrient contrast MyceliumSeg lacks** | µm scale, first ~20 h only (mostly lag/early phase); licence unverified |
| 5 | ***Aureobasidium*** (Zenodo 14001310) | Microscopy time-lapse, used for growth simulation | Yeast-like dimorphic fungus; check sizes |
| — | **MDPI 110-species time-lapse** (26,451 images, every 30 min, 5–7 days) | Best possible transport target: petri dish, macroscopic, many species | **Not public** — request from the authors now (long lead time) |
| — | AMF package (Figshare 27889143) | Network/flow mechanism | Only a demo subsample; low priority for growth curves |

Excluded as non-fungal (fine for testing tools, not for testing the model): mycelyso
(*Streptomyces*), *Phytophthora* (oomycete), *Physarum*. Synthetic sets can be used for
segmenter pre-training only.

Steps:
1. **Within MyceliumSeg:** hold out species or condition blocks and predict observable
   morphology distributions using partial pooling. Report predictive error, not just fit.
   Do not call these growth curves unless frame order and time are established.
2. **Fricker grazer series:** refit the observable dynamics and growth form, using pooled MyceliumSeg
   posteriors as priors for the growth form. Separately estimate the grazer effect on
   extension rate (this was randomized, so no adjustment is needed beyond species and block)
3. ***Podospora*:** test whether the lag → linear growth form holds at hyphal scale, and
   whether the nutrient effect is multiplicative on k (compare with the log-scale temperature effect)
4. **Your own rig** (`/mnt/sharedOs/archive/mycelium-workspace`: RPi camera + LED
   timelapse). This is the strongest test, because *you* control the intervention:
   - Cross species × medium (e.g. oyster on both PDA and MYG) to break the confound in MyceliumSeg
   - Randomize dish position; log temperature; image every N hours with fixed lighting
   - Pre-register the predicted rate ratio from Phase 4, then compare
   - Publishing template: Harrington & Harrington 2026 *Physarum* scanner set (Zenodo
     22868368, OME-Zarr, CC BY 4.0). Either six dishes on a flatbed scanner or the
     existing Pi camera. Publish the result on Zenodo, since open time-lapse data is the field's main gap

### Do now (long lead time)
- [ ] Email the MDPI 110-species time-lapse authors for access
- [ ] Email the MyceliumSeg authors about dish/time metadata (in parallel with Phase 1, don't wait for it)

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
   as a tip-like event?
3. Can MyceliumSeg support frame-indexed observables, or only cross-sectional
   morphology? — **answered:** cross-sectional only; frame-indexed change is
   defined but not estimated until order, identity and interval exist
   (`notes/provenance.md` § 4.1, Gate C)
4. How many 15 °C GL dishes exist? That count sets the power for any temperature contrast.
5. Is image selection ("distinctive edge morphology") time-dependent? If so, it biases
   feature distributions and any inferred dynamics. — **answered (#35, 2026-10-07):**
   the observed-morphology half is falsified by construction (selection reads the true and
   imaged morphology, Z → Sel ← I in `R/dag.R`); the time-dependence half is untestable
   under Gate C (no dish identity, no timestamps, no ordered sequences), and no evidence
   for temporal independence is invented. Cross-sectional interpretation is bounded to the
   selected release; the Gate C prohibition on causal temperature/medium and physical
   growth-rate claims is preserved. `notes/assumptions.md` §5.
6. Which parts of the observable measurement model transport across datasets, and which
   must be re-estimated for each camera, scale, and resolution?
