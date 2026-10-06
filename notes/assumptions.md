# Assumption register: #20 falsified — dish identity and capture time are not recoverable

**Date:** 2026-10-06
**Status:** recorded — **#20 falsified**. The issue is left open; the operator closes it
after review, so the `falsified` close is not performed by the worker.
**Evidence:** #21 (EXIF), #22 (numbering), #23 (dish tracks + duplicates), #24 (labels),
#25 (px→mm), #26 (Gate C — `notes/gate.md`); `chainlink cascade 20`
**Propagates to:** #34, #41, #42, #44, #52 — a `falsification` comment and a
`needs-reassessment` label on each (the non-closing half of `chainlink falsify`, run by
hand because the operator owns the close)

**Verdict.** Assumption **#20** — that MyceliumSeg dish identity *and* capture time are
recoverable from the released files — is **falsified**. There is no in-release route to
either. The downstream consequence is fixed and recorded here so no later task can
quietly reintroduce a growth curve:

> **MyceliumSeg is limited to segmentation, cross-sectional morphology, and
> frame-indexed measurements — nothing else.**

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
  frame-indexed change / boundary-event counts **only** where frame order is
  independently validated, reported in per-frame units.
* **Prohibited:** physical growth-rate models (mm/day, lag τ, saturation), causal
  temperature or medium effects, per-dish varying effects, and calling any MyceliumSeg
  sequence a growth curve. Time-resolved and intervention claims move to the
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
| **#41** Fit real MyceliumSeg data; diagnostics + posterior predictive **per dish** | dish identity for per-dish effects and a time axis for the fit | no dish → no per-dish random effects; no time → no growth model. Replace with the cross-sectional morphology model and the conditional frame-level estimand where order holds |
| **#42** Compare alternatives (logistic, Gompertz, ODE) with PSIS/WAIC | a time-parameterised growth curve to compare | all three are time models and are not identifiable on MyceliumSeg; move model comparison to Fricker/Boddy (days) or the rig |
| **#44** Hold out GS/PO/TS within MyceliumSeg; predict with species partial pooling | species partial pooling over growth parameters | keep only as cross-sectional transport of image-level morphology; drop growth-parameter prediction |
| **#52** Pre-register predicted rate ratio from E5 | a MyceliumSeg rate ratio to pre-register | a rate ratio needs time and dish identity; derive it from the controlled-rig design (and the Fricker/Boddy pilot), not from MyceliumSeg |

Further dependents (not in the one-hop cascade but implied): #42 → #37, #41 → #45–#48,
#52 → #53. They inherit the same constraint through their parents.

## 4. What is unchanged

Gate C already keeps MyceliumSeg as the anchor dataset for segmentation,
measurement-error calibration, and cross-sectional morphology. Falsifying #20 removes the
*longitudinal* reading only; the anchor roles, the px→mm calibration (#25), the
measurement contract (`notes/estimands.md`) and the provenance inventory
(`notes/provenance.md`) stand as written. If the authors later supply dish identity and a
capture interval, this register is amended and the tasks in § 3 are unblocked — not before.

---

## Reproducing

```bash
chainlink cascade 20                     # the five downstream issues
.venv/bin/python -m pytest tests -q      # full suite
```

Surrounding record: `notes/gate.md` (Gate C, #26), `notes/provenance.md` (#61),
`notes/estimands.md` (#60), PLAN.md Phase 1 (identifiability gate).
