# Assumption register: #20 falsified; #35 selection statuses; #37 growth-form invariance assessed

**Date:** 2026-10-06
**Status:** **#20 falsified**; **#35 closed with selection evidence unresolved**;
**#37 assessed without accepting invariance**. The #20 issue was left open for
operator close after review; its `falsified` close was not performed by the worker.
**Evidence:** #21 (EXIF), #22 (numbering), #23 (dish tracks + duplicates), #24 (labels),
#25 (px→mm), #26 (Gate C — `notes/gate.md`); `chainlink cascade 20`
**Propagates to:** #34, #41, #42, #44, #52 — a `falsification` comment and a
`needs-reassessment` label on each (the non-closing half of `chainlink falsify`, run by
hand because the operator owns the close)

**Verdict.** Assumption **#20** — that MyceliumSeg dish identity *and* capture time are
recoverable from the released files — is **falsified**. There is no in-release route to
either. The downstream consequence is fixed and recorded here so no later task can
quietly reintroduce a temporal growth curve:

> **MyceliumSeg supports segmentation, cross-sectional morphology (including
> morphology modeled by area-coverage level), and validated frame-indexed
> measurements. It does not identify temporal growth rates or causal effects.**

The full decision and its evidence live in `notes/gate.md` (#26); this note is the
assumption-node record that propagates that decision to the five tasks built on it.

---

## 1. The assumption, and why it fails

| Half of #20 | Test | Result | Source |
|---|---|---|---|
| Capture time recoverable | EXIF/XMP/IPTC/COM/vendor tags over all images; zip mtimes; file numbering vs time | **No.** 0 / 20,176 images carry any acquisition signal; zip mtimes are export sessions; numbering→mtime forward steps 0.489–0.547 | #21, #22 |
| Dish identity recoverable | rim-dot / mark fingerprinting, inoculum position, rim embeddings | **No.** rim dot fills 720/720 angle bins (largest shared angle 0.22%); all >0.99 embedding pairs are byte copies; colony size vs file order ≈ 50% | #23 |
| Labels compensate | per-image stage / sclerotium colour | **No.** 0 images carry them; 1,689 images have no species/medium | #24 |

Two facts follow, and they are the whole falsification: **nothing assigns an image to a
physical dish, and nothing dates it.** A frame index is an index, not a time coordinate.

## 2. Downstream consequence

* **Permitted:** segmentation; cross-sectional morphology (area fraction, equivalent
  radius, perimeter, roughness, calibrated mm²) by species and known condition; and
  cross-sectional models of other image features by area-coverage level; and
  frame-indexed change / boundary-event counts **only** where frame order is
  independently validated, reported in per-frame units.
* **Prohibited:** physical growth-rate models (mm/day, lag τ, saturation), causal
  temperature or medium effects, per-dish varying effects, and calling any MyceliumSeg
  sequence a temporal growth curve. Time-resolved and intervention claims move to the
  Fricker/Boddy grazer series, the *Podospora* movies, or the controlled rig
  (`notes/gate.md` § 5).

This is the same consequence Gate C recorded; #20 being falsified makes it binding on the
tasks below, not only on future work.

## 3. Falsification cascade

`chainlink cascade 20` reports five downstream issues. Each was built on #20 and each
needs reassessment; the specific re-scope follows from the consequence above.

| Issue | What it assumed from #20 | Required re-scope |
|---|---|---|
| **#34** Count *G. lucidum* 15 °C dishes/images (power for primary estimand) | a dish/time-resolved primary estimand exists to power | no rate estimand → no power count; at most a cross-sectional 15 °C association, already only 10 images (#26 §3) |
| **#41** Fit real MyceliumSeg data by area-coverage level | dish identity for per-dish effects and a time axis for the fit | no dish/time → no per-dish or temporal model. Fit cross-sectional morphology against area fraction, plus the conditional frame-level estimand where order holds |
| **#42** Prototype a sequence model on area-rank pseudo-steps | originally, a time-parameterised growth curve to compare | user re-scoped the MyceliumSeg task to an explicitly synthetic `t*` model; the AR(3) prototype lost to persistence in all five features (#65, `notes/pseudo_time_model.md`). This is training-pipeline exploration only. Real temporal alternatives still require linked sequences and actual intervals on another dataset or the rig |
| **#44** Hold out GS/PO/TS within MyceliumSeg; predict with species partial pooling | species partial pooling over growth parameters | keep as cross-sectional transport of image-level morphology across area-coverage levels; drop growth-parameter prediction |
| **#52** Pre-register predicted rate ratio from E5 | a MyceliumSeg rate ratio to pre-register | a rate ratio needs time and dish identity; derive it from the controlled-rig design (and the Fricker/Boddy pilot), not from MyceliumSeg |

Further dependents (not in the one-hop cascade but implied): #42 → #37, #41 → #45–#48,
#52 → #53. They inherit the same constraint through their parents.

## 4. What is unchanged

Gate C already keeps MyceliumSeg as the anchor dataset for segmentation,
measurement-error calibration, and cross-sectional morphology, including models
indexed by area coverage. Falsifying #20 removes the
*longitudinal* reading only; the anchor roles, the px→mm calibration (#25), the
measurement contract (`notes/estimands.md`) and the provenance inventory
(`notes/provenance.md`) stand as written. If the authors later supply dish identity and a
capture interval, this register is amended and the tasks in § 3 are unblocked — not before.

---

## 5. Assumption #35 — image selection and "distinctive edge morphology" (#35)

**Date:** 2026-10-07 (corrected 2026-10-08 after review)
**Status:** issue #35 is **closed as unresolvable from the current source and
release**. Its two evidence statuses remain: morphology-based selection is
**undocumented** (the paper describes the release, not an inclusion rule) and
per-image dependence on latent time/frame is **untestable** under Gate C. Closing
the tracking issue records the limit of available evidence; it does not establish
selection independence or resolve generalization beyond the selected release.
**Evidence:** the paper's Methods / Data collection section (Yuan et al. 2025,
doi:10.1038/s41597-025-06265-1), quoted verbatim:

> We collected 20,176 mycelium images with distinctive edge morphology.

That sentence describes the **collected release**. The same section documents
process-level, periodic fixed-position acquisition:

> … we periodically placed mycelial petri dishes at a fixed position for image
> acquisition.

It does **not** document an image-selection inclusion rule based on edge
morphology. The public [Zenodo data record](https://zenodo.org/records/15224240)
(version 1.0) lists ten data ZIP archives and no separate per-image metadata
file; the local archive inventory likewise finds no per-image dish, capture-time,
acquisition-order or growth-stage link. This supports the absence of public
linkage, but does not reveal why any image was included. Local record: PLAN.md
§0; `notes/estimands.md` §1; `notes/gate.md` §1, §3; `R/dag.R`; machine-readable
mirror `scripts/measurement_contract.py` (`SELECTION_BIAS`, #35)
**Propagates to:** #41, #42 (assumption relations) — both already re-scoped by
the #20 cascade (§ 3 above); #35 adds the selection caveat to that re-scope

**Verdict.** Assumption **#35** — that image selection for distinctive edge
morphology is independent of latent growth features and frame/time — cannot be
settled from the source or the release, and the two halves are recorded with the
status the evidence supports:

| Half of #35 | Status | Why |
|---|---|---|
| Selection is independent of **observed morphology** | **Undocumented** | The paper states it *collected* images "with distinctive edge morphology"; it documents no edge-morphology inclusion rule. There is no candidate / included / excluded image frame and no in-release evidence that selection read the measured morphology, so the release cannot be shown to over-represent the estimands' own features and the **direction of any morphology bias is unknown**. The `Z → Sel ← I` collider in `R/dag.R` is a *hypothesis to test*, not an established mechanism. |
| Selection is independent of **latent time/frame** | **Untestable** | Gate C: no dish identity, no capture timestamps, no ordered sequences (0/20,176 EXIF; zip mtimes are export sessions; numbering shuffled — forward steps 0.489–0.547, `notes/gate.md` §1). The paper's periodic fixed-position acquisition is **process-level context**, not a per-image dish/time link, so no in-release evidence can identify whether selection depended on latent growth features or frame/time. **No evidence for temporal independence is invented here**: the honest status is *untestable*, not *supported*. |

**What a future test would require.** Selection dependence cannot be tested from
this release. It needs (i) the **documented inclusion rule**; (ii) the
**candidate / included / excluded image frame** — which images were eligible and
which were kept or dropped; and (iii) **linked per-image dish/time or growth-stage
metadata** to relate inclusion to a latent time or stage axis. Until the authors
supply those, both halves stay as recorded: undocumented and untestable.

**Supported, and bounded.** What the release *can* support is cross-sectional
morphology **of the selected release**: associational, within-release, by species,
known condition, and area coverage (`notes/estimands.md` §1, §3). Because the selection
mechanism is undocumented, no selection model is identifiable and no claim about
the magnitude *or direction* of a morphology bias may be made. Cross-sectional
interpretation is therefore bounded to the selected release: never a plate
population, a species population, a treatment effect or a temporal growth curve.

The segmentation deliverable is still direct and independent: train an image-to-mask
model on the released annotated pairs so its output marks the mycelium region inside
the dish. The training target is the mask; dish/time metadata and a growth estimand
are not inputs to that task, and species labels are not model inputs. This fit does
not establish species-specific performance. Calculate `area_fraction` afterward from predicted mask
area and dish area, then use it as a coverage coordinate for separate models of other
image features. This sequence supports segmentation and cross-sectional analysis of
the selected release without asserting that one image follows another over time.
The minimum dataset deliverable is the trained and held-out-evaluated segmenter;
area-fraction analysis is optional downstream work. #35's unresolved selection
evidence limits claims beyond the selected release but does not block this deliverable.

**Gate C prohibition preserved.** Nothing in the #35 record licenses a causal
temperature or medium effect or a physical growth-rate claim: those stay
prohibited under Gate C (`notes/gate.md` §3–§4) regardless of how the selection
numbers look. The unresolved selection assumption does not block segmentation
training or image-level modeling by area coverage; those describe this selected
release. The #35 record adds no temporal estimand.

**Consequence for #41/#42.** Both are re-scoped from the original #20 assumptions
(§ 3): #41 is cross-sectional morphology by area coverage, and #42 is an evaluated
area-rank sequence-model prototype with synthetic `t*`. Neither provides per-dish
effects or a time-parameterised growth model on MyceliumSeg. Cross-sectional models may
use area fraction as a coverage coordinate to study other morphology features. #35
adds the selection caveat: any morphology distribution either task reports describes
the selected release; any chronology inferred from file order or area rank would be subject to
the same two unresolved statuses — undocumented morphology selection and
untestable time/frame dependence — and may not be presented as established.

**If the authors supply the documented inclusion rule, the candidate / included
/ excluded image frame, and linked dish/time or stage metadata**, the
undocumented half can be assessed and the untestable half becomes testable, and
this register is amended. Those inputs are necessary to test selection
dependence; they do not by themselves identify a selection model, which would
additionally need the functional form relating selection to the latent features.
Not before.

---

## 6. Assumption #37 — lag / extension / saturation invariance across datasets

**Status:** issue #37 is **closed as assessed, with the invariant form not
supported by current evidence**. This does not establish that all taxa have
different mechanisms. It records that the available datasets measure different
outcomes and do not provide a matched test of a universal lag, extension rate,
and plateau.

**Evidence:** 5-fold complete-series holdouts on Fricker/Boddy found the
lag-extension-plateau model reduced dimensionless equivalent-radius RMSE by
only 0.5% versus linear extension while worsening MAE; its fitted plateau was
at the end of the observed D8 window. Within-movie holdouts on *Podospora*
favored exponential ridge occupancy, while the onset-extension-plateau form
predicted the held-out tail worse. Fricker D1 and *Podospora* first-frame times
are not biological germination origins, and neither dataset observes a
validated full plateau. MyceliumSeg remains excluded from temporal fitting by
Gate C because dish identity and chronology are absent.

The complete model definitions, metrics, source links, and limits are in
`notes/growth_form_invariance.md`; model outputs are under
`data/derived/growth_form_invariance/`. Revisit invariance only with a common
calibrated observable, known biological start, replicate identity, and data
that span onset through saturation.

---

## Reproducing

```bash
chainlink cascade 20                     # the five downstream issues
.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); \
    import measurement_contract as m; print(m.validate_contract())"   # []
.venv/bin/python -m pytest tests/test_measurement_contract.py -q      # contract + #35 guards
.venv/bin/python -m pytest tests -q      # full suite
```

Surrounding record: `notes/gate.md` (Gate C, #26), `notes/provenance.md` (#61),
`notes/estimands.md` (#60), PLAN.md Phase 1 (identifiability gate).
