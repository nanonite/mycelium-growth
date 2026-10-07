# Dataset inventory

Checked 2026-10-02 by opening the downloaded files directly, without extracting them.
All files passed md5 against the repository checksums (see `manifest.csv`).
Licence is CC BY 4.0 for every item.

## Usability for growth modeling

The question for each dataset: can we follow **one colony over time**?

| Dataset | Organism / scale | Colonies × time | Time info | Colony ID | Ready-made measurements | Verdict |
|---|---|---|---|---|---|---|
| **Fricker grazers** (Zenodo 14931953) | 4 cord-forming basidiomycetes, wood block → soil tray, macroscopic | 777 images: 4 spp × 7–9 treatments × 5 reps × 4–7 days | ✅ day in filename + **EXIF capture time** (Nikon COOLPIX P80, 2009) | ✅ treatment + rep in filename | ✅ per-image network table (115 traits) incl. `mycelial_area`, `colony_perimeter`, `length_total`, `micron_per_pixel` | **Best longitudinal set.** Randomized grazer treatment |
| **Podospora thallus** (Figshare 22666138 + 22140692 raw) | *P. anserina*, hyphal-resolution microscopy | 6 colonies (3 × M0, 3 × M2) × 69–85 frames | ⚠ frame index only; interval ~18 min per paper, not stored in files | ✅ one colony per movie | ❌ raw frames (~7000² px); needs segmentation | **Good for the growth-form + nutrient test**, n = 3 per medium |
| **MyceliumSeg** (Zenodo 15224240) | 4 spp on agar, petri dish, macroscopic | 20,176 images, all on disk (IDs 00000001–00020176, contiguous) | ❌ **no EXIF in any of the 20,176** (full scan, #21); zip mtimes are export sessions (9 dates, files seconds apart) | ❌ shuffled sequential numbering | ✅ 567 masks; colony area via segmentation | **Cross-sectional only** unless dishes can be re-identified (see below) |
| Podospora network 2019 (Figshare 9897056, 10626968) | *P. anserina*, microscopy | 1 colony × 82 frames (downsized AVI) + an 11-frame TIFF | ⚠ frame labels only | ✅ | ❌ | Small; supplementary |
| Fricker networks (Zenodo 5725751) | 11 spp (Zygo/Asco/Basidio), macroscopic | 33 images = 11 spp × 3 reps, **one time point each** | single day (t08–t25) | ✅ | ⚠ `.mat` parameter files only | Snapshot only. Usable as a between-species prior on network traits, not on growth rate |
| Aureobasidium (Zenodo 14001310) | — | — | — | — | — | **No data**: the record holds only a conference book of abstracts |
| Physarum (Zenodo 22868368) | slime mould (not a fungus), one camera over 6 dishes | 1 of 24 dishes on disk (001-dish0, 1,696 frames); full set 51,036 dish frames, 133 GB | ✅ per-frame `acquisition/timestamp_utc_ns`, 120–300 s interval | ✅ | ❌ camera pixels only; dish-circle registration + QC flags | Template for publishing own rig data (OME-Zarr v3 layout) |

## Details

### Fricker / Boddy grazer series — `raw/fricker_grazers/`
- `Hf.zip` 215 images, `Pi.zip` 253, `Pv.zip` 171 (+4 in `rejected/`), `Rb.zip` 138. JPG 3648×2736 (572) or
  2048×1536 (205, all Pi); the rejected thumbnails are 600×450.
- **4 byte-identical pairs (#61): `Pi_ctrl2_d1_{1..4}` ≡ `Pi_ctrl2_d2_{1..4}`** — the "day 2" images are copies
  of day 1 (same md5, same EXIF second), so de-duplicate before any per-day differencing.
- Filename `<sp>_[DD2_]<trt>_<D|d><day>_<rep>`. Treatments: `ctrl1`, `ctrl2`, `coll` (collembola), `mill` (millipede),
  `mite`, `nema` (nematode), `wl1`/`wl2` (woodlice), `enchy` (enchytraeids, Pi only).
- Days imaged: Hf 1–16 (some to 24) · Pi up to 38 · Pv 1–16 · Rb 1–8. Five replicates per cell.
- Per image: `images/*_fungus_coded.pdf`, `parameters/*.mat`, `results/*-results.xlsx` with sheets
  Graph, Nodes, Edges, Polygons, robustness sheets and Parameters. Per species: `results/summary_Traits_table_AZB.RDS`
  (one row per image × 115 traits); other `.RDS` files need R to read.
- EXIF exposure and ISO vary between shots (ISO 64–326) → add an imaging-condition node to the measurement model.
- Authors' code: `external/FungalGrazers` (R: download → convert → 4 models → figures). Ecology Letters paper.

### Podospora — `raw/podospora_*`
- `podospora_thallus/M{0,2}_{1,2,3}.avi`: MJPEG, ~7100×6990 px, 69–85 frames each (85/85/81/74/70/69; header
  counts match the stored chunks in all six, #61).
- `podospora_m0_raw/M0_1.tif`: ImageJ stack of M0_1, 4.3 GB uint8, header `images=85` — **but only pages 1–15
  carry image data; pages 16–85 are zero-filled** (#61, found by content hashing). The IFD chain is broken
  (the next-IFD pointer lands in zeros), so `tifffile` sees 2 pages; read the 15 content pages arithmetically:
  data at offset 3,789, stride 50,131,345, 10,584-byte trailer.
- `podospora_net2019a`: 82-frame 1604×1708 AVI. `podospora_net2019b`: 11-frame RGB TIFF.
- Spatial and temporal calibration is not in the files; take it from Ledoux et al. 2023 / Dikec et al. 2020.

### MyceliumSeg — `raw/myceliumseg/`
- On disk: labeled-GL (457 train + 50 test), labeled-GS_PO_TS (3 × 10), labeled-MYG_PDA_TEMP (3 × 10), unlabeled-GL2
  (2,240 images, 00004988–00007227). All 11 archives now on disk (56.5 GB): GL1 00000508–04987, GL2–GL7 2,240 each, GS_PO_TS 00018488–20176 (1,689).
- **All 20,176 images scanned for metadata (`data/triage/exif_scan.csv`, #21): zero EXIF, zero XMP, zero IPTC, zero JPEG
  comments, zero vendor tags.** Every image has the identical bare-JFIF header (623 bytes, no APP1 at all), so
  the re-encode stripped the metadata. Zip mtimes = export sessions across all 11 archives: 9 calendar dates
  total, copy-speed bursts (up to 6.6 img/s), id↔mtime uncorrelated (forward steps ≈ 50%), packing machine at
  UTC+8, and `com.apple.quarantine` xattrs dating the packaging to a Baidu Netdisk download on 2025-10-08.
- 4608×3456 JPG, **but 1,728 of the 19,609 unlabeled images are 1382×1037** (GL1–GL7 only), interleaved through
  the ID range — px→mm calibration (#25) must handle both.
- File number vs time: 51% forward steps (shuffled).
- Visual: no overlays. Each dish has an orange rim dot at a varying angle, plus occasional pen marks.
  The dot is real and measurable (#23), but **it does not fingerprint dishes**: the angle fills all
  720 half-degree bins, no angle is shared by more than 37 of the 16,513 detections (0.22%), and
  the release behaves as one dish per image. Rim-embedding clustering finds no structure either —
  of 203,525,400 pairs every one above 0.99 is a byte-identical duplicate. Colony sizes span the
  whole range, so cross-sectional size-by-condition is possible.
- **133 / 20,176 images are byte-identical duplicates** (#23): the `PO` block is a
  copy of `MYG`, the `PDA` block a copy of `labeled-GL`, and 29 of the 30 `labeled-GS_PO_TS`
  images reappear in `unlabeled-GS_PO_TS` — plus 22 pairs that are the same frame under two
  different ids *inside* `unlabeled-GS_PO_TS.zip`. Folder-derived species/medium labels are
  therefore wrong for 20 images (#24) — de-duplicate before any train/test split, within an
  archive as well as across them (#28/#29). `TEMP15` is clean.
- Apparent dish size differs per condition block (r/W 0.302 GL/MYG/PDA/TEMP15, 0.332 PO/TS, 0.224 GS),
  so px→mm must be calibrated per block, not globally (#25).
- Metadata may still exist with the authors (user is following up).

## Implication for the plan

Phase-1 gate (#26) is **C** (decided — `notes/gate.md`) for MyceliumSeg: no recoverable time.
The plan's fallback anchor, **Fricker grazers**, has everything MyceliumSeg lacks:
- real capture times
- replicate identity
- a randomized treatment
- precomputed colony area and perimeter per image

That makes it the natural first anchor for the longitudinal growth model. MyceliumSeg then serves cross-sectional and
segmentation work, and *Podospora* serves as the growth-form / nutrient transport test.

The per-dataset provenance audit that verifies every claim above (#61) is
`notes/provenance.md`, with its machine-readable capability matrix in
`data/triage/sequence_inventory.csv`.
