# Mycelium growth — causal modeling plan

Approach: Statistical Rethinking workflow (McElreath) — estimand → DAG → generative
simulation → statistical model → validate on simulated data → fit real data →
test transport to other datasets.

Anchor dataset: **MyceliumSeg** (Yuan et al. 2025, Sci Data, doi:10.1038/s41597-025-06265-1;
Zenodo 15224240; code github.com/yuanqianguang/MyceliumSeg-benchmark).

---

## 0. What MyceliumSeg actually contains (and what it constrains)

| Fact | Consequence for causal work |
|---|---|
| 20,176 JPGs, 4608×3456, 90-mm dishes, dark incubation, fixed-position camera (FPheno2000) | Constant geometry → pixel area can become mm² with one calibration |
| Species: *G. lucidum* (GL), *G. sinense* (GS), *P. ostreatus* (PO), *Trametes* (TS) | 4 levels — enough for partial pooling, but not much |
| Medium: GL, GS on **MYG**; PO, TS on **PDA** | **Medium is completely confounded with species.** The medium effect can't be identified from this data |
| Temperature: GL at 15 °C and 25 °C; others 25 °C only | **Temperature is the only clean within-species contrast** → primary estimand |
| 567 labeled masks (507 GL, 10 each GS/PO/TS/MYG/PDA/TEMP15); 19,609 unlabeled | Train a segmenter on the labeled masks, then measure colony size on the unlabeled images |
| Growth-stage and sclerotium-colour counts reported (Tables 2–3) | We don't yet know whether these labels ship per image |
| Files numbered sequentially (`00000001`…); images "selected for distinctive edge morphology" | **Dish ID and timestamp may be missing**, and the image selection is a possible selection bias |

Boundary IoU is only ~11–30% even for the best model (SegFormer). Colony *area* is
measured well, but the *edge/radius* is noisy. That noise has to be modeled, not ignored.

---

## Phase 1 — Data triage (go/no-go gate, ~1 week)

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

---

## Phase 2 — Measurement pipeline

1. Reproduce the SegFormer baseline with the benchmark repo (MMSegmentation)
2. Run inference on all unlabeled images. Per image, extract colony area A,
   equivalent radius r = √(A/π), perimeter, edge roughness (fractal dim / perimeter²/A),
   and hyphal density (mean intensity inside the mask)
3. **Error model for the measurement:** on the test set, compare predicted vs true area
   and radius to estimate bias and SD. This becomes the measurement-error layer (Ch. 15)

---

## Phase 3 — Estimand and DAG

**Primary estimand:** the causal effect of incubation temperature (15 vs 25 °C) on the
radial extension rate of *G. lucidum* on MYG.

**Secondary:** species differences in extension rate and lag time, at 25 °C
(descriptive: species and medium can't be separated).

Draft DAG (edit in dagitty.net; code in `R/dag.R`):

```
Species S ──► Medium M            (fixed by design → S and M are aliased)
S, M, Temp T ──► rate k, lag τ
Inoculum vigour V ──► τ, k        (V proxied by sclerotium colour C: V ──► C)
Dish D (unobserved) ──► k, τ      (varying effects)
k, τ, time t ──► true radius R
R ──► growth stage G              (G is a CONSEQUENCE → never condition on it for T's effect)
R, imaging I (condensation, glare) ──► observed radius R*
Selection: "distinctive edges" ──► inclusion   (possible collider)
```

Implications to check with `dagitty::adjustmentSets`:
- T → k is identified without adjustment, because T was assigned by design. Adding C
  (sclerotium colour) may improve precision
- Stratifying by growth stage biases the temperature effect (post-treatment variable)
- The medium effect is not identified. Say so explicitly; don't estimate it

---

## Phase 4 — Generative model (write it first, fit it to simulated data)

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
1. **Within MyceliumSeg:** hold out GS/PO/TS and predict their curves using partial
   pooling on species (α[S] ~ Normal(ᾱ, σ_S)). Report predictive error, not just fit
2. **Fricker grazer series:** refit with the same structure, using pooled MyceliumSeg
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
1. Dish/time metadata (Phase 1 gate) — this determines everything downstream
2. How many 15 °C GL dishes exist? That count sets the power for the primary estimand
3. Is image selection ("distinctive edge morphology") time-dependent? If so, it biases the curve shape
