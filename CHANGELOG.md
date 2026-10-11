# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Fit the #38 synthetic dish-growth observations with `ulam` under the #39
  priors (#40). All 14 truth-bearing fixed parameters fall within their 95%
  posterior intervals; 95.0% of dish rates and 93.3% of dish lags are covered.
  Record curve error, convergence diagnostics, RStan/CmdStan compatibility,
  artifacts and the unbounded-likelihood limitation in
  `notes/ulam_parameter_recovery.md` and
  `data/derived/growth_parameter_recovery/`. This fit remains synthetic and
  does not estimate growth from MyceliumSeg.
- Implement prior predictive checks for the latent radial-growth model (#39):
  10,000 draws, bounded 0–45 mm trajectories, rate/lag summaries and wall-contact
  times through day 60. The candidate priors span week-scale to several-week
  plate filling; assumptions and outputs are in `notes/prior_predictive_growth.md`
  and `data/derived/growth_prior_predictive/`. Track the unbounded Normal
  observation-likelihood refinement for future measured radii as #69.
- Implement the known-parameter dish-growth simulator in `R/sim.R` (#38) and
  save a reproducible 60-dish, 960-observation example under
  `data/derived/dish_growth_simulation/`. The generating values and assumptions
  are documented in `notes/simulated_growth_data.md`; the scenario is synthetic
  and is not fitted to or inferred from MyceliumSeg.
- Assess the cross-dataset growth-form assumption (#37) with a series-held-out
  Fricker/Boddy lag/extension/plateau comparison and a within-movie *Podospora*
  tail holdout. The common lag → constant extension → saturation form is not
  supported as invariant: the Fricker gain over linear extension is negligible,
  and the *Podospora* plateau model predicts worse than exponential occupancy.
  Biological lag and late saturation remain unidentified; see
  `notes/growth_form_invariance.md` and `data/derived/growth_form_invariance/`.
- Fit an exploratory image-derived ridge-occupancy model to all six *Podospora*
  M0/M2 movies (#47): verified Figshare capture cadence/scale/licence, compared
  exponential and power-law forms on later-frame holdouts, estimated per-movie slopes
  and a replicate-level nutrient association, and recorded threshold sensitivity and
  mask limitations in `notes/podospora_growth_model.md`. This is not a validated
  semantic segmentation or a causal nutrient estimate.
- Fit a dataset-specific Fricker/Boddy repeated-measures model (#68): paired log
  changes from linked D1 images, fungus-by-grazer trajectories on EXIF elapsed
  time, held-out-series shape comparison, series-bootstrap rate intervals, and
  baseline-adjusted D8-label contrasts. Keep the 15 D2-start series separate,
  exclude five same-session Pi control rows, and preserve associational treatment
  interpretation because allocation is undocumented. Results are in
  `notes/fricker_series_model.md` and `data/derived/fricker_grazers/`.
- Close #35 as unresolvable from the current MyceliumSeg release. The evidence statuses
  remain morphology selection undocumented and time/frame dependence untestable; closure
  records the limit of available data and does not assert selection independence. The
  segmentation fit uses image/mask pairs only and does not establish species-specific
  performance.
- Clarify the independent model tracks after #35: the per-image segmentation model
  is trained from labeled image/mask pairs and evaluated without dish/time metadata;
  the model predicts the mycelium mask only. Full-release inference can produce
  `area_fraction` afterward for cross-sectional modeling of other features across
  area-coverage levels. Such models describe the selected release and do not turn
  area rank into elapsed time or a same-dish trajectory. Temporal growth models remain
  available on datasets with linked specimens and real times. The Gate C, estimand,
  measurement and inference notes now state this distinction; #41 is re-scoped to
  morphology by area coverage and #42 to temporal alternatives on linked sequences.
- Set the minimum MyceliumSeg dataset deliverable to a reproducible image-to-mask
  segmenter trained on released annotations and evaluated on held-out masks. Run a
  fresh GPU smoke training/evaluation in the `issue35-minimum-v2` work directory; the
  two-iteration run validates the pipeline only, while the existing 50,000-iteration
  #28 checkpoint remains the evaluated baseline.
- Add a separate final-fit path for all available MyceliumSeg annotations. It verifies
  and collapses 20 duplicate image/mask records to 547 unique supervised pairs, keeps
  the #28 benchmark checkpoint untouched, and records that the all-data fit has no
  independent holdout score. The all-labeled loader/training preflight passed. After a
  host interruption at iteration 21,400, the 50,000-iteration fit resumed from the
  latest saved checkpoint at iteration 18,750 and completed. The final checkpoint and
  environment/data evidence are in the tagged work directory; the final training loss
  is 0.0085, not a held-out quality metric.
- Assess the "distinctive edge morphology" selection assumption under Gate C
  (#35) — `notes/assumptions.md` §5 records the two unresolved statuses:
  morphology-based selection is **undocumented** (the paper's Methods / Data
  collection *describes* collecting 20,176 images "with distinctive edge
  morphology" but documents no edge-morphology inclusion rule, so the release
  cannot be shown to over-represent the estimands' own features and the bias
  direction is unknown), and dependence on latent time/frame is **untestable**
  (no dish identity, no timestamps, no ordered sequences; periodic
  fixed-position acquisition is process-level context, not per-image linkage),
  with no evidence for temporal independence invented. The public Zenodo v1.0
  record lists ten data ZIP archives and no separate metadata file; the local
  archive inventory has no per-image dish, capture-time, acquisition-order or
  stage link. Testing either needs the
  documented inclusion rule, the candidate / included / excluded image frame,
  and linked dish/time or stage metadata. Cross-sectional interpretation is
  bounded to the selected release; the Gate C prohibition on causal
  temperature/medium and physical growth-rate claims is preserved. The
  machine-readable `SELECTION_BIAS` record and guards in
  `scripts/measurement_contract.py` are checked by
  `tests/test_measurement_contract.py`; `notes/estimands.md` §1, `notes/gate.md`
  §3, PLAN.md (§0, Phase 3 DAG, open question 5) and `R/dag.R` carry the same
  statuses

### Fixed
- Correct the #35 follow-up on plate sampling: `population.sampling_frame`,
  `notes/estimands.md` §1 and `notes/image_count_15c.md` §5 no longer assert the
  release is a (non-)random sample of plates — the release documents no
  plate-level sampling design, so plate representativeness is unknown and no
  plate-population inference is supported, while the selected-release
  descriptions are preserved. `notes/assumptions.md` §5's final conditional now
  names the documented inclusion rule, the candidate / included / excluded image
  frame, and linked dish/time or stage metadata, and states those inputs are
  necessary but do not by themselves identify a selection model.
  `scripts/measurement_contract.py` gains `unsupported_sampling_claims()` and a
  `validate_contract()` guard; regression tests reject random/non-random
  sampling assertions in the record and the notes (#35)
- Correct the #35 selection-assumption record (commit 6610ebb) to match the
  source: the paper's Methods / Data collection section states the 20,176
  images were collected *with* distinctive edge morphology and describes
  periodic fixed-position acquisition; it does **not** document selecting
  images by an edge-morphology inclusion rule
  (doi:10.1038/s41597-025-06265-1). Removed the unsupported claims that
  morphology independence is falsified by construction, that the release
  over-represents the morphology estimands, and that the bias direction is
  known; `SELECTION_BIAS` now records morphology selection as undocumented and
  time/frame dependence as untestable, with `validate_selection_record()` and
  `tests/test_measurement_contract.py` rejecting the retracted statuses
- Audit eligible *G. lucidum* 15 °C images for cross-sectional morphology (#34) —
  `scripts/triage/audit_15c_images.py` verifies the 10 TEMP15 candidates
  (`labeled-MYG_PDA_TEMP.zip`, IDs 00018478–00018487) against the source
  archive, duplicate map, label triage, px→mm calibration and measurement
  table: all 10 are unique (distinct md5), correctly labelled (GL/MYG/15 °C),
  calibrated (`fit_status=ok`), and free of cross-archive collisions and label
  conflicts. Reports a validated 25 °C GL context count (18,437 unique; 18,001
  calibrated). Gate C binding: descriptive feasibility only — no dish-level
  power, no causal temperature effect, no physical growth rate. Writes
  `data/triage/audit_15c.json`; `tests/test_audit_15c_images.py` (25 tests)
  and `notes/image_count_15c.md` document the audit
- Observable measurement table from the validated #29 masks —
  `scripts/measurements.py` writes `data/measurements.csv` (one provenance-rich
  row per image: area, area fraction, calibrated equivalent radius, perimeter,
  edge roughness / circularity / solidity, intensity summaries and density
  proxies, with image-level and mask-level provenance, calibration block,
  duplicate status, frame-order status and time-unit status) plus
  `notes/measurements_evidence.json` and `notes/measurements.md`. Area,
  log-area and equivalent radius are primary; the boundary-derived features and
  the intensity/density proxies are secondary and carry the #31 error model's
  uncertainty. Guards refuse any elapsed-time, rate, dish-id or hyphal-tip
  field (Gate C). `tests/test_measurements.py` covers the intensity arithmetic,
  the off_scale exclusion and the #31 value agreement (#30)
- Inventory time-resolved image sequences and frame provenance (#61):
  `scripts/triage/sequence_inventory.py` audits every dataset under `data/raw`
  for the six provenance facts — ordered frames, specimen identity, capture
  interval, resolution, duplicates, condition labels — and writes the capability
  matrix `data/triage/sequence_inventory.csv` plus the per-file evidence
  `notes/sequence_inventory.json`; the write-up and estimand mapping are in
  `notes/provenance.md` (MyceliumSeg has **no usable longitudinal sequence**, so
  the cross-sectional fallback stands and Fricker/Boddy, *Podospora* and a
  controlled rig are nominated for dynamic and intervention models). New
  findings: 4 byte-identical `Pi_ctrl2` day-1 ≡ day-2 pairs in the grazer
  series, and `podospora_m0_raw/M0_1.tif` carries only 15 content pages behind
  an `images=85` header. Guards in `validate_rows()` are re-run against the
  committed matrix by `tests/test_sequence_inventory.py`
- Define the measurement contract before inference — `notes/estimands.md` with the
  machine-readable `scripts/measurement_contract.py` (image-level morphology estimands
  by species and condition, the conditional per-frame `delta_log_area_per_frame`,
  population, inclusion rule, duplicate handling, units, px→mm calibration and the
  95% uncertainty target) and `tests/test_measurement_contract.py` guards that refuse a
  per-time rate without a capture interval (#60)
- Reproduce the SegFormer baseline: `scripts/segformer_repro.py` (prepare / config / train / evaluate / evidence),
  the `scripts/segformer_shim/` compatibility shim the pinned benchmark needs on this stack,
  and the results write-up in `notes/segformer_baseline.md` (#28). The `evidence`
  subcommand writes `evidence.json` (device, driver, CUDA/torch/mmcv versions, batch
  size, peak GPU memory, checkpoint SHA-256, metrics, data hashes) for the GPU
  baseline run (`--tag gpu`), which trains SegFormer-B0 for the published 50,000
  iterations at the published effective batch of 16 on the RTX 4070 Ti
- Causal DAGs and adjustment-set checks (`R/dag.R`, #33): the MyceliumSeg
  cross-sectional DAG under Gate C (species-medium aliasing, latent dish/time,
  selection for distinctive edge morphology, imaging/segmentation error,
  duplicate/mislabeled files) and future causal intervention DAGs, with
  `dagitty::adjustmentSets` verifying that no adjustment set identifies a causal
  temperature effect (T→Y), medium effect (M→Y), or physical growth rate (T→k)
  from MyceliumSeg, while the future graphs (D, t observed) do identify T→k

### Fixed

### Changed
- E7: Own-rig intervention experiment (#49)
- Run experiment + publish (OME-Zarr, CC BY 4.0, Physarum template) (#53)
- Pre-register predicted rate ratio from E5 (#52)
- Rig calibration: fixed lighting, interval, px→mm (#51)
- Design: species × medium (e.g. oyster on PDA and MYG), randomized positions, temp logging (#50)
- Define observable growth estimands and a data-capability DAG (#32)
- E3: Measurement pipeline (#27)
- E2: MyceliumSeg metadata triage (go/no-go gate) (#19)
- E0: Setup (#2)
- Build and validate generative models for observable growth dynamics (#36)
- Bounded observation likelihood for measured colony radius (#69)
- Parameter recovery: fit ulam to simulated data (#40)
- Fricker grazer: model longitudinal outcomes + estimate treatment contrasts (#45)
- Map supported model targets across datasets before transport (#67)
- Resolve MyceliumSeg area-fraction precision gap (#66)
- Validate candidate tip-like boundary events at available resolution (#62)
- Hold out species/conditions; predict morphology across area_fraction (#44)
- Prototype a sequence model on MyceliumSeg area-rank pseudo-steps (#42)
- Explore area-rank pseudo-time as a mask-sequence training surrogate (#65)
- Fit MyceliumSeg morphology across area_fraction; predictive checks (#41)
- ASSUMPTION: selection dependence (morphology undocumented; time/frame untestable) (#35)
- Count eligible G. lucidum 15°C images for cross-sectional analysis (#34)
- Write DAG in R/dag.R; check adjustmentSets for T→k (#33)
- Shape metrics (area, r_eq, perimeter, roughness, density) → data/measurements.csv (#30)
- Extract shape metrics (area, r_eq, perimeter, roughness, density) → data/measurements.csv (#30)
- Inventory time-resolved image sequences and frame provenance (#61)
- Define frame-level observable growth estimands and units (#60)
- Measurement-error model: bias + SD of area/radius on test set (#31)
- Inference on unlabeled images → masks (#29)
- ASSUMPTION: MyceliumSeg dish ID + timestamp are recoverable (#20)
- Record Gate C for MyceliumSeg identifiability (#26)
- Establish a validated SegFormer baseline for MyceliumSeg measurement (#28)
- Record SegFormer evidence and the measurement handoff (#59)
- Run the agreed SegFormer baseline protocol on the available GPU (#58)
- Run a GPU SegFormer smoke pipeline (#57)
- Expose a CUDA-capable MyceliumSeg training environment (#56)
- Update the growth modeling plan for observable MyceliumSeg dynamics (#55)
- Calibrate px → mm using 90-mm dish rim (#25)
- Recover dish tracks (inoculum position, lid marks, embedding clustering) (#23)
- Convert docs/pdfs to Markdown and extract MyceliumSeg modeling implications (#63)
- Recover dish tracks (inoculum position, lid marks, embedding clustering) (#23)
- Check whether per-image stage / sclerotium colour labels ship in the release (#24)
- Check whether sequential file numbering follows time within dish (thumbnail strips) (#22)
- EXIF scan (DateTimeOriginal etc.) on labeled + probe shard (#21)
- Add Python stack to the shared Nix flake (#6)
- Add R stack to the shared Nix flake (#5)
- E1: Data acquisition (#9)
- Download remaining MyceliumSeg unlabeled shards (GL1, GL3-7, GS_PO_TS) (#13)
- Download Physarum scanner set (Zenodo 22868368) as publishing template (#18)
- Download Aureobasidium time-lapse (Zenodo 14001310) (#17)
- Download Fricker network dataset (Zenodo 5725751, 3.2 GB) (#15)
- Download Podospora movies (Figshare 22666138, 9897056, 10626968) (#16)
- Download Fricker/Boddy grazer series (Zenodo 14931953, 10.2 GB) (#14)
- Download MyceliumSeg probe shard unlabeled-GL2 (6.2 GB) (#12)
- Download MyceliumSeg labeled-GL (1.5 GB) (#11)
- Download MyceliumSeg labeled-MYG_PDA_TEMP + labeled-GS_PO_TS (235 MB) (#10)
- Create project layout + data manifest (#4)
- Check free disk (~80 GB: MyceliumSeg 56.5 + Fricker 13.4 + Podospora/other) (#3)
