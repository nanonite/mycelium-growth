# Triage: Per-Image Metadata Labels in MyceliumSeg (#24)

**Question.** Do per-image metadata labels (growth stage: 4 classes; sclerotium colour: 5 classes; species; medium; temperature) ship in the release?

**Answer.** **Growth stage and sclerotium colour labels are NOT available per-image anywhere in the release.** Species, medium, and temperature can be inferred for most images from folder structure + paper Table 1, but NOT for 1,689 images in `unlabeled-GS_PO_TS.zip`.

---

## Summary of Findings

| Label | Classes | Available per-image? | Where |
|-------|---------|----------------------|-------|
| **Species** | 4 (GL, GS, PO, TS) | ✅ 18,487 / 20,176 | Folder structure (labeled archives) + Paper Table 1 |
| **Medium** | 2 (MYG, PDA) | ✅ 18,487 / 20,176 | Folder structure (labeled-MYG_PDA_TEMP) + Paper Table 1 |
| **Temperature** | 2 (15°C, 25°C) | ✅ 20,176 / 20,176 | Folder structure (TEMP15) + Paper Table 1 (all others 25°C) |
| **Growth stage** | 4 | ❌ **NOT available** | Paper Table 2 gives aggregate counts only |
| **Sclerotium colour** | 5 | ❌ **NOT available** | Paper Table 3 gives aggregate counts only (for 2,204 images in stage 1) |

---

## Source-by-Source Evidence

### 1. Archive Folder Structure (11 zips, inspected via `zipfile`, no extraction)

| Archive | Images | Folder Structure | Implied Labels |
|---------|--------|------------------|----------------|
| `labeled-GL.zip` | 507 | `labeled-GL/{trainset,testset}/{image,mask}/` | Species=GL (all) |
| `labeled-GS_PO_TS.zip` | 30 | `labeled-GS_PO_TS/{GS,PO,TS}/{image,mask}/` | Species=GS/PO/TS (10 each) |
| `labeled-MYG_PDA_TEMP.zip` | 30 | `{MYG,PDA,TEMP15}/{image,mask}/` | Medium=MYG/PDA, Temp=15°C (TEMP15) |
| `unlabeled-GL{1..7}.zip` | 17,920 | `unlabeled_data{N}/` (flat) | Species=GL (all) |
| `unlabeled-GS_PO_TS.zip` | 1,689 | Flat (no subfolders) | Species=mixed (GS/PO/TS), **no per-image breakdown** |

**No CSV/JSON/TXT/XLSX metadata files exist in any archive.**

### 2. Paper (doi:10.1038/s41597-025-06265-1)

#### Table 1 — Culture Conditions (species-level, not per-image)
| Species | Medium | Temperature |
|---------|--------|-------------|
| *G. lucidum* (GL) | MYG | 15°C, 25°C |
| *G. sinense* (GS) | MYG | 25°C |
| *P. ostreatus* (PO) | PDA | 25°C |
| *Trametes* spp. (TS) | PDA | 25°C |

#### Table 2 — Growth Stage Frequencies (aggregate only)
| Growth Stage | Count | % |
|--------------|-------|---|
| Sclerotium activation & germination | 2,204 | 10.92% |
| Primary hyphal expansion | 2,652 | 13.14% |
| Hyphal network construction | 10,894 | 53.99% |
| Mycelial maturation transition | 4,426 | 21.94% |
| **Total** | **20,176** | **100%** |

**Not provided per-image.**

#### Table 3 — Sclerotium Colour Frequencies (aggregate only, n=2,204)
| Colour | Count | % |
|--------|-------|---|
| Yellow (pure) | 1,122 | 50.91% |
| Blended (yellow-black) | 342 | 15.52% |
| Grey | 317 | 14.38% |
| Brown | 303 | 13.75% |
| Black | 120 | 5.44% |
| **Total** | **2,204** | **100%** |

**Not provided per-image.** Only applies to the 2,204 images in the first growth stage.

### 3. Zenodo Record 15224240
- Lists 11 files with MD5 checksums
- No additional metadata fields per file
- Description only mentions "precise edge segmentation annotations"

### 4. Benchmark Repo (github.com/yuanqianguang/MyceliumSeg-benchmark)
- Dataset class `EPA_Mycelium_Dataset` only defines binary classes: `('background', 'mycelium')`
- No growth stage, sclerotium colour, species, medium, or temperature labels in any config or dataset code
- Only semantic segmentation (mask vs background)

---

## Per-Image Label Assignments (Derived)

Written to `data/triage/known_labels.csv` (columns: `id, species_if_known, medium_if_known, temp_if_known, source`):

- **18,487 images** have species, medium, temperature assigned
  - 507 from `labeled-GL` (GL, MYG, 25°C)
  - 30 from `labeled-GS_PO_TS` (10 GS/MYG/25°C, 10 PO/PDA/25°C, 10 TS/PDA/25°C)
  - 30 from `labeled-MYG_PDA_TEMP` (10 GL/MYG/25°C, 10 TS/PDA/25°C, 10 GL/MYG/15°C)
  - 17,920 from `unlabeled-GL{1..7}` (GL, MYG, 25°C)
- **1,689 images** from `unlabeled-GS_PO_TS` have **only temperature (25°C)**; species and medium unknown per-image (mixed GS/PO/TS on MYG/PDA)
- **0 images** have growth stage or sclerotium colour labels

### Species Breakdown (assigned)
| Species | Count |
|---------|-------|
| GL | 18,447 |
| GS | 10 |
| PO | 10 |
| TS | 20 |
| **Unknown** | **1,689** |

### Temperature Breakdown
| Temperature | Count |
|-------------|-------|
| 15°C | 10 |
| 25°C | 20,166 |

---

## What Is Missing for Growth Modeling

| Needed for Phase 1+ | Status |
|---------------------|--------|
| Per-image growth stage (4 classes) | ❌ Not in release; only aggregate Table 2 |
| Per-image sclerotium colour (5 classes) | ❌ Not in release; only aggregate Table 3 |
| Per-image species for unlabeled-GS_PO_TS (1,689 images) | ❌ Mixed folder; no breakdown |
| Per-image medium for unlabeled-GS_PO_TS | ❌ Mixed MYG/PDA; no breakdown |
| Per-image dish ID / timestamp | ❌ Confirmed absent (#21, #22) |

---

## Deliverables

- `scripts/triage/label_triage.py` — derivation script
- `data/triage/known_labels.csv` — per-image assignments (20,176 rows)
- `notes/triage_labels.md` — this note
- `tests/test_label_triage.py` — unit tests for helper functions

---

## Reproducing

```bash
.venv/bin/python scripts/triage/label_triage.py
.venv/bin/python -m pytest tests/test_label_triage.py -v
```