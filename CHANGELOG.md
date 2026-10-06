# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Expose a CUDA-capable MyceliumSeg training environment (#56): `devShells.gpu`
  with CUDA torch 2.5.1+cu124 and mmcv/mmsegmentation, host `libcuda.so.1`
  preloaded in the shell hook only, evaluation safe without a GPU, CPU default
  shell retained
- Run a GPU SegFormer smoke pipeline (#57): prepare/train/evaluate on the
  RTX 4070 Ti in ~1 min, evidence in `evidence.json` (device, driver, versions,
  batch size, peak memory, checkpoint hash, metrics); fixed mmcv 1.3.0 passing
  int device ids to torch 2.x `_get_stream` via the compatibility shim
- Reproduce the SegFormer baseline: `scripts/segformer_repro.py` (prepare / config / train / evaluate / evidence),
  the `scripts/segformer_shim/` compatibility shim the pinned benchmark needs on this stack,
  and the results write-up in `notes/segformer_baseline.md` (#28). The `evidence`
  subcommand writes `evidence.json` (device, driver, CUDA/torch/mmcv versions, batch
  size, peak GPU memory, checkpoint SHA-256, metrics, data hashes) for the GPU
  baseline run (`--tag gpu`), which trains SegFormer-B0 for the published 50,000
  iterations at the published effective batch of 16 on the RTX 4070 Ti


### Fixed

### Changed
- Record Gate C for MyceliumSeg identifiability (#26)
- Establish a validated SegFormer baseline for MyceliumSeg measurement (#28)
- Record SegFormer evidence and the measurement handoff (#59)
- Run the agreed SegFormer baseline protocol on the available GPU (#58)
- Run a GPU SegFormer smoke pipeline (#57)
- Expose a CUDA-capable MyceliumSeg training environment (#56)
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
