# Triage: acquisition-time metadata in MyceliumSeg (#21)

**Question.** Does any of the 20,176 MyceliumSeg images carry a capture timestamp, camera
identity, or other acquisition signal — in EXIF/XMP/IPTC, in JPEG comments, or in the zip
entry mtimes?

**Answer. No.** Zero images out of 20,176 carry any acquisition-time signal. Every image has
an identical bare-JFIF header with no metadata segment at all. The zip mtimes are
**export/copy timestamps**, confirmed across all 11 archives (this issue's earlier comments
only covered 3 archives + a 40-image sample of a fourth).

| Signal | Result |
|---|---|
| EXIF (`DateTimeOriginal`, `DateTimeDigitized`, `DateTime`, `OffsetTime*`, `SubSecTime*`) | **0 / 20,176** |
| EXIF `Make` / `Model` / `Software` (incl. `FPheno2000`) | **0 / 20,176** |
| EXIF `MakerNote` / vendor (unknown-tag) IFDs | **0 / 20,176** |
| XMP packets (`xmp:CreateDate`, `tiff:Make`, vendor namespaces) | **0 / 20,176** |
| IPTC IIM (APP13 / `8BIM` 0x0404) | **0 / 20,176** |
| JPEG `COM` comments | **0 / 20,176** |
| Vendor-keyword sweep over all decoded strings (Fuji/Nikon/Canon/Phenomatics/ImageMagick/libjpeg/…) | **0 / 20,176** |
| Raw-byte deep scan of the first 1 MiB of every image (undeclared metadata blocks) | **0 / 20,176** |
| PNG mask text chunks (`tEXt`/`iTXt`/`zTXt`/`tIME`/`eXIf`), 567 masks | **0 / 567** |
| Zip entry mtime | present, but **export sessions** — see below |

Deliverables: `data/triage/exif_scan.csv` (one row per image, 53 columns) and
`data/triage/exif_scan_summary.csv` (one row per archive). Code: `scripts/triage/exif_scan.py`
(driver) + `scripts/triage/exif_parse.py` (parsers), tests in `tests/test_exif_parse.py`.
Zips are streamed with `zipfile`; nothing is extracted to disk. A full re-scan takes ~10 s on 11
cores, because only the first ~623 bytes of each image are read.

---

## 1. Every image has the same metadata-free header

All 20,176 images share one marker signature and one JFIF block:

```
SOI | APP0 "JFIF" 1.01 (density 1×1, units=0 → aspect ratio) | DQT×2 | SOF0 | DHT×4 | SOS
```

There is **no APP1 at all** — no Exif segment, no XMP packet — and no COM segment. Header size
is 623 bytes for every file. Since APP1 is the only legal carrier of EXIF/XMP in JPEG, the
absence is conclusive rather than merely "no tag matched".

Supporting detail:

- `prec=8`, `3` components, baseline SOF0 — a plain re-encode, no progressive/lossless oddity.
- JFIF density is `units=0, 1×1`, i.e. **no DPI and no aspect hint**: the scaling information a
  camera or scanner would write is gone too.
- `reached_sos=1` and `truncated=0` for all 20,176 rows, so the parse consumed the complete
  header rather than stopping early (0 parse errors).
- **1,728 images are 1382×1037, not 4608×3456** — see §5; they carry no metadata either.

**Interpretation.** The images were re-encoded at some point after capture (a resize/recompress
step, most likely for the benchmark release), and that step dropped every metadata segment. The
camera in the paper (FPheno2000) leaves no trace in the pixels' containers.

## 2. Zip entry mtimes are export sessions, not capture times — confirmed across all 11 archives

The zip central directory stores two timestamps per entry: the DOS field (local time, 2 s
resolution, no timezone) and an **extended timestamp in the extra field** (`NTFS` 0x000a for the
GL archives, `UT` 0x5455 for the GS/PO/TS and labeled archives) giving the same instant in **UTC
at 1-second or sub-second resolution**. `exif_scan.py` parses both
(`parse_zip_extra`) and compares them.

The extended stamps are decisive, and they line up as follows:

| Archive | Images | Stored mtimes (UTC) | Distinct stamps | Median gap | Sessions (gap > 60 s) |
|---|---|---|---|---|---|
| `labeled-GL.zip` | 507 | 2025-03-27 → 2025-10-12 | 507 | 13 s | 62 |
| `labeled-GS_PO_TS.zip` | 30 | 2025-10-08 01:54–01:57 | 7 | 0 s | 2 |
| `labeled-MYG_PDA_TEMP.zip` | 30 | 2025-10-08 01:54 → 2025-10-13 00:49 | 13 | 0 s | 3 |
| `unlabeled-GL1.zip` | 4,480 | 2024-09-20 → 2024-12-17 | 3,720 | 3 s | 22 |
| `unlabeled-GL2.zip` … `GL7.zip` | 2,240 each | 2024-09-20 → 2024-12-17 | ~2,020 each | 5–6 s | 40–44 |
| `unlabeled-GS_PO_TS.zip` | 1,689 | 2025-10-11 03:38–03:42 | 256 | 0 s | **1** |

Evidence that these are copy operations rather than an imaging cadence:

1. **Throughput is a file-copy rate, not a shutter rate.** Grouping all 20,176 images by
   calendar date gives 9 "sessions" with these rates:

   | Date (UTC) | Images | Span | Rate |
   |---|---|---|---|
   | 2024-09-20 | 1,710 | 10.4 min | 2.7 img/s |
   | 2024-11-14 | 9,427 | 9.4 h | 0.28 img/s |
   | 2024-12-17 | 6,783 | 3.6 h | 0.52 img/s |
   | 2025-03-27 | 60 | 40.5 min | 0.02 img/s |
   | 2025-03-28 | 397 | 8.7 h | 0.01 img/s |
   | 2025-10-08 | 50 | 2.8 min | 0.30 img/s |
   | 2025-10-11 | 1,689 | 4.2 min | **6.6 img/s** |
   | 2025-10-12 | 50 | 9 s | **5.6 img/s** |
   | 2025-10-13 | 10 | 47 s | 0.21 img/s |

   6.6 images/second is disk-to-disk copy speed for 2–4 MB JPEGs; the 2024-11-14 session even
   has a median gap of **1 second** between consecutive 2–4 MB images.

2. **The whole set lives on 9 calendar dates.** An incubation time series over weeks of growth
   (days 0→30 per PLAN.md Phase 1) cannot compress into 9 days with sub-minute spacing.

3. **File numbering is anti-correlated with mtime.** Pearson correlation between image id and
   mtime is ≈ 0 within every unlabeled shard (e.g. `unlabeled-GL1` +0.008, `GL2` +0.011, `GS_PO_TS`
   −0.020); forward-step fraction is 0.49–0.51, i.e. a coin flip. Numbers were shuffled
   independently of the copy order.

4. **The shards interleave.** Within 2024-09-20 the archive order is GL7, GL6, GL1, GL3, GL1,
   GL1, GL7, … — a single export job wrote all 7 shards concurrently/interleaved. That is one
   copy session, not 7 imaging sessions.

5. **The packing machine ran at UTC+8** (`local_utc_offset_h = 8.0` for all 20,176 rows), i.e.
   China Standard Time, and the uid/gid in the extra fields is **502** for the labeled archives
   (a Linux uid) and **4** (root) for `unlabeled-GS_PO_TS.zip`. Different archivers, different
   accounts: a curation/packaging pipeline.

6. **`__MACOSX` quarantine xattrs say "downloaded via Baidu Netdisk".** 55 of the 117
   `AppleDouble` sidecars carry `com.apple.quarantine = q/0081;68e5cb7f;百度网盘`; the other 62
   shadow `.DS_Store` files and folders rather than images. All 55 carry the identical hex-epoch
   `68e5cb7f` = **2025-10-08 02:25:03 UTC**, and all 50 that shadow a JPG are attached to those
   image rows. A Baidu Netdisk download is a file *transfer* event, so this dates the packaging,
   not the imaging. (The agent name is stored as literal `\xNN` escapes; `decode_escaped_utf8`
   reassembles it.)

7. **Zip archive comments are empty** on all 11 archives, and the 7 `.DS_Store` blobs are
   Finder view-state only (md5-counted in the summary, no per-file dates).

## 3. Per-archive verdict

**Acquisition-time signal exists in no archive.** In every case below the only time information
in the release is the zip mtime, and §2 shows that is export time.

| Archive | IDs | Images | EXIF | XMP | IPTC | COM | Vendor tags | Mask text | Time signal | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `labeled-GL.zip` | 1–507 | 507 | 0 | 0 | 0 | 0 | 0 | 0 (507 masks) | zip mtimes only (2025-03-27/28 trainset, 2025-10-12 testset) | **none** |
| `labeled-GS_PO_TS.zip` | 18428–18457 | 30 | 0 | 0 | 0 | 0 | 0 | 0 (30 masks) | zip mtimes + quarantine xattr (2025-10-08) | **none** |
| `labeled-MYG_PDA_TEMP.zip` | 18458–18487 | 30 | 0 | 0 | 0 | 0 | 0 | 0 (30 masks) | zip mtimes + quarantine xattr (2025-10-08) | **none** |
| `unlabeled-GL1.zip` | 508–4987 | 4,480 | 0 | 0 | 0 | 0 | 0 | — | zip mtimes only (3 dates) | **none** |
| `unlabeled-GL2.zip` | 4988–7227 | 2,240 | 0 | 0 | 0 | 0 | 0 | — | zip mtimes only (3 dates) | **none** |
| `unlabeled-GL3.zip` | 7228–9467 | 2,240 | 0 | 0 | 0 | 0 | 0 | — | zip mtimes only (3 dates) | **none** |
| `unlabeled-GL4.zip` | 9468–11707 | 2,240 | 0 | 0 | 0 | 0 | 0 | — | zip mtimes only (3 dates) | **none** |
| `unlabeled-GL5.zip` | 11708–13947 | 2,240 | 0 | 0 | 0 | 0 | 0 | — | zip mtimes only (3 dates) | **none** |
| `unlabeled-GL6.zip` | 13948–16187 | 2,240 | 0 | 0 | 0 | 0 | 0 | — | zip mtimes only (3 dates) | **none** |
| `unlabeled-GL7.zip` | 16188–18427 | 2,240 | 0 | 0 | 0 | 0 | 0 | — | zip mtimes only (3 dates) | **none** |
| `unlabeled-GS_PO_TS.zip` | 18488–20176 | 1,689 | 0 | 0 | 0 | 0 | 0 | — | zip mtimes only (2025-10-11, one 4.2-min session) | **none** |
| **Total** | 1–20176, complete | **20,176** | **0** | **0** | **0** | **0** | **0** | **0** | — | **no signal anywhere** |

ID coverage was verified: the 20,176 rows cover `00000001`–`00020176` exactly once, with no gaps
and no duplicates.

## 4. Answer to the specific zip-mtime question

> *"Zip mtimes previously looked like export sessions; confirm or refute across all archives."*

**Confirmed, and strengthened by the extended UTC timestamps.** Across all 11 archives the
stamps fall on 9 calendar dates, cluster into bursts at file-copy throughput (up to 6.6 img/s),
and are uncorrelated with file numbering. Two independent mechanisms date the packaging: the
`com.apple.quarantine` xattrs (Baidu Netdisk, single instant 2025-10-08 02:25:03 UTC) and the
uid/gid + UTC+8 offset in the zip extra fields. None of this is consistent with images being
written by a camera over a weeks-long incubation.

## 5. Incidental findings worth carrying forward

- **1,728 of 19,609 unlabeled images are 1382×1037, not 4608×3456** (422 / 188 / 232 / 214 / 231 /
  218 / 223 across GL1–GL7; none in the labeled or GS_PO_TS archives). They are interleaved
  through the ID range (median gap 8 ids), not clustered. 1,710 of them were written during the
  2024-09-20 export, and 18 during 2024-11-14. Any px→mm calibration (#25) or segmentation
  inference (#28) must handle two resolutions — a fixed dish rim will not be 90 mm in both.
  This is the same resize step that stripped the metadata.
- **The train/test split of `labeled-GL` is dated**: 457 trainset images were exported
  2025-03-27/28, the 50 testset images on 2025-10-12. Consistent with a benchmark assembled in
  stages, and independent confirmation that these stamps are release-engineering dates.
- **The labeled species folders were copied in reverse order**: within `labeled-GS_PO_TS` the
  whole `TS` block lands at 01:54:58–01:55:00, `PO` at 01:56:38, `GS` at 01:57:02–01:57:08, so
  id↔mtime correlates at −0.893. A folder walk, not an acquisition order.

## 6. What this means for the Phase-1 gate (#26)

The `EXIF DateTimeOriginal present?` box in PLAN.md Phase 1 is now definitively **no** for
all 20,176 images — not "not found in a sample". Combined with the shuffled numbering (#22) this
keeps the gate leaning **C (neither dish nor time recoverable)**, and with it the recommendation
to anchor on the Fricker/Boddy grazer series, which does ship real capture times.

Remaining Phase-1 routes to time, none of which this issue touches:

- #23 recover dish tracks from the images themselves (orange rim dot angle, lid marks) — the only
  in-data route to grouping, and it still gives no ordering.
- Author-supplied metadata (PLAN.md "do now": email Yuan et al.) — the only route to true
  acquisition times.

## Reproducing

```bash
.venv/bin/python scripts/triage/exif_scan.py --jobs 11   # ~10 s; writes both CSVs
.venv/bin/python -m pytest tests -q                      # 50 tests
```

Flags: `--limit N` (images per archive, for smoke tests), `--session-gap S`, `--header-budget`,
`--no-deep-scan`, `--vendor-keyword` (repeatable).
