# Eligible *G. lucidum* 15 °C image count for cross-sectional morphology (#34)

**Date:** 2026-10-07
**Status:** audited — **10 unique, eligible images** at 15 °C; 25 °C GL
context count 18,437 (18,001 calibrated)
**Evidence:** #21–#26 plus this audit — `scripts/triage/audit_15c_images.py`
→ `data/triage/audit_15c.json` (machine-readable) and
`tests/test_audit_15c_images.py` (25 tests)
**Scope:** Gate C (#26) binding — image-count and descriptive-feasibility
audit only. No dish-level power, no causal temperature effect, no physical
growth rate. Images are not independent dishes.

---

## 1. Verdict

| Quantity | Value |
|---|---|
| **Eligible *G. lucidum* 15 °C images** | **10** |
| Source archive | `labeled-MYG_PDA_TEMP.zip` |
| Source folder | `TEMP15/image/` |
| Image IDs | `00018478`–`00018487` |
| 25 °C GL context (unique) | 18,437 |
| 25 °C GL context (calibrated, `fit_status=ok`) | 18,001 |

The previously reported 10 candidates (#26 §3) are **confirmed**: all 10 are
unique (distinct md5), correctly labelled (GL/MYG/15 °C), calibrated
(`fit_status=ok`), present in the measurement table, and free of
cross-archive collisions and label conflicts.

---

## 2. Exclusion accounting

The audit applies the #60 inclusion rule (§ 2) to the 10 TEMP15 candidates.
Every exclusion category is counted; none remove any of the 10.

| Exclusion category | Rule | N excluded | Evidence |
|---|---|---|---|
| **Duplicates** | One representative per md5 group (§ 7) | **0** | All 10 md5s distinct; no TEMP15 md5 appears in any other image across all 11 archives |
| **Cross-archive id collision** | Same image id in two archives | **0** | IDs `00018478`–`00018487` appear only in `labeled-MYG_PDA_TEMP.zip` |
| **Label conflict** | md5 group whose copies carry different non-blank labels | **0** | All 10 have `label_conflict=False` in `measurements.csv` |
| **Label correctness** | Species/medium/temp from `known_labels.csv` | **0** | All 10 are GL/MYG/15C per folder + paper Table 1 |
| **Off-scale** | `fit_status != ok` in `image_scale.csv` | **0** | TEMP15 block: 10 images, 0 off-scale; all 10 individually `ok` |
| **Missing measurement** | Not in `measurements.csv` | **0** | All 10 present with full morphology row |
| **Cross-temperature duplicate** | 15 °C md5 ≡ 25 °C md5 | **0** | No md5 overlap between TEMP15 and the 18,437 GL 25 °C images |

**Eligible = 10 − 0 = 10.**

---

## 3. Verification against existing artifacts

The count is cross-checked against four independent committed artifacts:

| Artifact | What it confirms | Result |
|---|---|---|
| `data/triage/known_labels.csv` (#24) | 10 images with species=GL, temp=15C | ✓ 10 |
| `data/triage/dish_tracks.csv` (#23) | 10 distinct md5s; no duplicate groups | ✓ 10 distinct |
| `data/triage/image_scale.csv` (#25) | 10 images, all `fit_status=ok`, px/mm = 6.0444 | ✓ 10 ok |
| `data/measurements.csv` (#30) | 10 rows, all `label_conflict=False`, full morphology | ✓ 10 rows |

The TEMP15 calibration block (`data/triage/calibration.csv`) reports
px/mm = 6.0444 with 0.0% spread and 0 off-scale — the same constant as the
GL 25 °C block, confirming the 15 °C images were captured under the same
camera setup as the main GL series.

---

## 4. 25 °C GL context count

For context, the validated 25 °C GL count is:

| Quantity | Value | Source |
|---|---|---|
| GL 25 °C images by label | 18,437 | `known_labels.csv` (507 labeled-GL + 17,920 unlabeled-GL + 10 MYG folder) |
| GL 25 °C distinct md5 | 18,437 | `dish_tracks.csv` — no duplicates within GL 25 °C |
| GL 25 °C off-scale | 436 | `image_scale.csv` (435 GL block + 1 MYG block) |
| GL 25 °C calibrated (`ok`) | **18,001** | 18,437 − 436 |

These figures are the **release inventory**, not the number of images with
measurements. At this audit commit, `data/measurements.csv` contains 68 GL
25 °C morphology rows (50 test hold-outs, 10 condition hold-outs, and 8
unlabeled stride samples) and 10 GL 15 °C rows. The archive ratio above must
not be read as the measured comparison sample size or as evidence of
independent biological replication.

---

## 5. Feasibility and sample-size limitations

**Descriptive image-level comparison: feasible on measured rows.** A
cross-sectional summary can use the 10 measured GL 15 °C rows and 68 measured
GL 25 °C rows — area, area fraction, equivalent radius, perimeter, roughness,
and calibrated mm², stratified by condition, with the #31 measurement-error
budget attached. This compares the available image rows only; selection and
measurement coverage differ by source set, so the 18,437-image archive count
is not the denominator for this summary.

**Not feasible (Gate C, #26):**

- **Causal temperature effect** — not identified. There is no dish identity,
  no replication count, and no time axis. The 10 images are not 10
  independent dishes; they are 10 images of unknown dish structure.
- **Physical growth rate** — not identifiable. No capture interval exists
  (0/20,176 EXIF; zip mtimes are export sessions; numbering is shuffled).
- **Dish-level power** — not computable. Images are not independent dishes,
  so no dish-level sample size exists to power.
- **Growth curve** — prohibited. No frame order, no time, no dish.

**Sample-size limitation.** n = 10 images at 15 °C with no dish-level
replication. Any descriptive summary describes these 10 images only; it
cannot support inference to a dish population, a treatment effect, or a
growth rate. The 10 images were "selected for distinctive edge morphology"
(#60 § 1), so they are not even a random sample of 15 °C plates.

---

## 6. Reproducing

```bash
.venv/bin/python scripts/triage/audit_15c_images.py --write   # audit + JSON
.venv/bin/python -m pytest tests/test_audit_15c_images.py -q   # 25 tests
.venv/bin/python -m pytest tests -q                            # full suite
ruff check scripts/triage/audit_15c_images.py                  # lint
```

Outputs: `data/triage/audit_15c.json` (machine-readable audit result).

Surrounding record: `notes/gate.md` (Gate C, #26), `notes/estimands.md`
(#60, measurement contract), `notes/provenance.md` (#61, capability
matrix), `notes/measurements.md` (#30, measurement table),
`notes/triage_labels.md` (#24, label triage), `notes/triage_calibration.md`
(#25, px→mm).
