# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
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
