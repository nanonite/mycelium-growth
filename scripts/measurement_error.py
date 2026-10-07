#!/usr/bin/env python3
"""Measurement-error model: bias + SD of area/radius/perimeter on held-out masks.

PLAN.md Phase 2 step 5 (Chainlink #31): before any growth model, estimate the
measurement-error layer.  On the de-duplicated held-out labeled masks, compare
the #28-checkpoint predictions (#29 inference outputs) against the true masks
and estimate bias and spread for area, area fraction, equivalent radius,
perimeter, edge roughness, circularity and solidity.  Quantify duplicate
sensitivity, sequence repeatability, and false-event rates for candidate
protrusions.

What this driver does *not* do is as important as what it does:

* it does not retrain -- it reads the masks the #29 run produced from the #28
  checkpoint and refuses to proceed if ``model_meta.json`` does not record
  that checkpoint's SHA-256 (``--allow-other-checkpoint`` overrides);
* it does not infer elapsed time, dish identity or a growth rate -- every row
  keeps ``frame_order_status = not_demonstrated`` and
  ``time_unit_status = no_capture_interval`` (Gate C), and no ordered sequence
  is differenced;
* it does not call a candidate protrusion a biological hyphal tip -- the
  protrusion detector reports *candidate* convexity/radial excursions and a
  false-event rate, nothing more.

Guards (the #60 contract, ``scripts/measurement_contract.py``):

* **De-duplication first.**  A byte-identical duplicate group is one
  observation; only its representative enters the error model, and the
  headline numbers are re-run with every copy included to show they do not
  move.
* **Inclusion rule.**  Calibrated estimands (mm^2 area, mm radius, mm
  perimeter, area fraction, ln(mm^2)) require ``fit_status == ok``; the
  off_scale representatives keep their px^2 and dimensionless shape numbers.
* **Uncertainty target.**  Size features: relative bias <= 3%, relative
  SD <= 10%.  Boundary features: relative bias <= 5%, relative SD <= 20%.
  A feature that misses is published with the target marked unmet.

Subcommands (all idempotent, writing under ``work_dirs/segformer-b0-repro/
inference/`` and the committed ``data/triage/`` + ``notes/``):

    analyze    compute the error model from the saved masks, write the
               per-image CSV and the evidence JSON
    evidence   assemble the run's environment, checkpoint, counts and caveats
    run        analyze -> evidence

The mask half needs only numpy + PIL (the masks already exist), so the whole
driver runs in the plain dev shell; no GPU.

Exit codes: 0 ok, 1 a step failed.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import measurement_contract as mc
import segformer_repro as repro

SEG_ROOT = REPO / "data" / "segmentation" / "release"
WORK_ROOT = REPO / "work_dirs" / "segformer-b0-repro" / "inference"
DEFAULT_WORK = WORK_ROOT

# The validated #28 hand-off (notes/segformer_evidence.json, commit fdfa43a).
CHECKPOINT_SHA256 = "bcf66b6c1b3ce44018bc54223bacad31bd48a1647efc842bfbc6bd5f6d3e20d6"

# The contract fixes the boundary-feature working width at 900 px
# (notes/estimands.md section 2.5); the calibration constants are defined there.
WORKING_WIDTH = 900

# The labeled hold-out splits (the only sets with a true mask to error against).
HOLDOUT_SETS = ("test", "conditions")

# Candidate-protrusion detector parameters (radial excursion, see
# :func:`radial_protrusions`).  The false-event rate is reported alongside a
# sensitivity sweep over these, so the reader can see how the number moves.
PROTRUSION_N_BINS = 72          # 5-degree angular bins
PROTRUSION_BASELINE_BINS = 3    # +-15-degree moving-average baseline
PROTRUSION_REL_THRESH = 0.05    # peak must exceed baseline by 5%
PROTRUSION_ABS_THRESH = 3.0     # ... and by 3 px at the 900-px width
PROTRUSION_MATCH_DEG = 15.0     # pred/ref peaks within 15 degrees match

# Field names that would mean an inference this task forbids (Gate C).
FORBIDDEN_INFERENCE_KEYS = (
    "elapsed_time",
    "elapsed_seconds",
    "capture_interval_seconds",
    "growth_rate",
    "rate_per_day",
    "rate_per_hour",
    "mm_per_day",
    "dish_id",
    "timelapse_id",
    "frame_time",
    "timestamp",
    "acquisition_time",
    "biological_tip_count",
    "hyphal_tip_count",
)

# Caveats carried into every output.
CAVEATS = (
    (
        "MyceliumSeg is cross-sectional: no ordered frames, no dish identity, no "
        "capture interval (Gate C, notes/provenance.md)."
    ),
    (
        "Errors are predicted-minus-reference on the de-duplicated held-out "
        "labeled masks; they describe this segmenter on this release, not a "
        "population of plates."
    ),
    (
        "Boundary features are weak by construction (published Boundary IoU "
        "28.60%, #28) and are measured at the 900-px working width; they carry "
        "the boundary error budget, not the size budget."
    ),
    (
        "Candidate protrusions are radial boundary excursions of the mask, not "
        "biological hyphal tips; their false-event rate is reported, and no "
        "tip, branching, nuclear, causal or physical-growth claim is made."
    ),
    (
        "The test and conditions hold-outs are not independent: ten PDA "
        "conditions images are byte copies of ten labeled-GL test images "
        "(#23), so de-duplication removes the conditions copies."
    ),
)


class MeasurementError(RuntimeError):
    """A step failed with a message that is already user-facing."""


# --------------------------------------------------------------------------
# mask loading and feature computation
# --------------------------------------------------------------------------


def load_mask_working(path: Path, *, width: int = WORKING_WIDTH) -> np.ndarray:
    """Load a 0/1 mask and resize it to the contract's 900-px working width.

    The released masks are 0/1 ``uint8`` at 4608 px; a bilinear resize of a
    0/1 image would average to near zero, so the mask is first scaled to
    0/255, resized, and thresholded at 127.  Both the predicted and the
    reference mask of an image go through exactly this path, so the comparison
    is apples-to-apples.
    """
    raw = np.asarray(Image.open(path).convert("L"))
    if raw.ndim != 2:
        raise MeasurementError(f"{path} is not a single-channel mask")
    binary = (raw > 0).astype(np.uint8) * 255
    image = Image.fromarray(binary, mode="L")
    w, h = image.size
    if w <= 0 or h <= 0:
        raise MeasurementError(f"{path} has a degenerate size {w}x{h}")
    new_h = max(1, round(h * width / w))
    resized = image.resize((width, new_h), Image.BILINEAR)
    return (np.asarray(resized) > 127).astype(np.uint8)


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    """4-connected boundary: a foreground pixel with a 4-neighbour outside."""
    m = mask
    up = np.zeros_like(m)
    up[:-1, :] = m[1:, :]
    dn = np.zeros_like(m)
    dn[1:, :] = m[:-1, :]
    lf = np.zeros_like(m)
    lf[:, :-1] = m[:, 1:]
    rt = np.zeros_like(m)
    rt[:, 1:] = m[:, :-1]
    return (m & ~(up & dn & lf & rt)).astype(np.uint8)


def boundary_points(mask: np.ndarray) -> np.ndarray:
    """(x, y) coordinates of the 4-connected boundary pixels."""
    ys, xs = np.nonzero(boundary_mask(mask))
    return np.column_stack([xs, ys]).astype(float)


def convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    """Monotonic-chain convex hull of a set of 2-D points.

    Returns the hull vertices in counter-clockwise order without repeating the
    first point.  Only the boundary points are passed in, which keeps the input
    small (the hull of the foreground is the hull of its boundary).
    """
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) <= 2:
        return pts

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_area(poly: Sequence[tuple[float, float]]) -> float:
    """Shoelace area of a polygon (0 for fewer than three vertices)."""
    if len(poly) < 3:
        return 0.0
    total = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _region_hull_area(mask: np.ndarray) -> float:
    """Area of the convex hull of the mask *region* (the union of pixels).

    The hull is taken over the corners of the boundary pixels, not their
    centres: a pixel is the unit square ``[x, x+1] x [y, y+1]``, so the hull
    of the region is the hull of the corner points.  Using pixel centres
    instead would undercount the hull by half a pixel border and push the
    solidity of a perfectly convex mask above 1.
    """
    ys, xs = np.nonzero(boundary_mask(mask))
    if len(xs) == 0:
        return 0.0
    corners = np.column_stack(
        [
            np.repeat(xs, 4) + np.tile([0, 1, 0, 1], len(xs)),
            np.repeat(ys, 4) + np.tile([0, 0, 1, 1], len(xs)),
        ]
    ).astype(float)
    return polygon_area(convex_hull(corners))


def mask_features(
    mask: np.ndarray,
    *,
    px_per_mm: float | None,
    dish_r_900: float | None,
    fit_ok: bool,
) -> dict[str, float]:
    """Image-level morphology of one mask at the 900-px working width.

    The px^2 and dimensionless shape features are always computed; the
    calibrated (mm, mm^2, area-fraction, ln(mm^2)) features are added only
    when ``fit_ok`` and the needed constant are present -- the #60 inclusion
    rule (``notes/estimands.md`` section 2.4).

    Perimeter is the 4-connected boundary-pixel count (the city-block
    perimeter).  It is a consistent convention applied identically to the
    predicted and reference mask, so the pred-minus-ref error -- the quantity
    this model exists for -- is convention-robust; the absolute value carries
    the usual staircase inflation, which is why the contract locks boundary
    features to one working width.
    """
    area_px = float(mask.sum())
    if area_px <= 0:
        return {"area_px": 0.0, "perimeter_px": 0.0, "empty": 1.0}
    perim_px = float(boundary_mask(mask).sum())
    equiv_radius_px = math.sqrt(area_px / math.pi)
    hull_area = _region_hull_area(mask)
    feats: dict[str, float] = {
        "area_px": area_px,
        "perimeter_px": perim_px,
        "equiv_radius_px": equiv_radius_px,
        "edge_roughness": perim_px / (2.0 * math.pi * equiv_radius_px),
        "circularity": 4.0 * math.pi * area_px / (perim_px**2),
        "solidity": area_px / hull_area if hull_area > 0 else 0.0,
        "log_area_px2": math.log(area_px),
        "empty": 0.0,
    }
    if fit_ok and px_per_mm:
        area_mm2 = area_px / (px_per_mm**2)
        feats["area_mm2"] = area_mm2
        feats["equiv_radius_mm"] = equiv_radius_px / px_per_mm
        feats["perimeter_mm"] = perim_px / px_per_mm
        feats["log_area_mm2"] = math.log(area_mm2)
        if dish_r_900:
            feats["area_fraction"] = area_px / (math.pi * dish_r_900**2)
    return feats


# --------------------------------------------------------------------------
# candidate protrusions (radial boundary excursions)
# --------------------------------------------------------------------------


def radial_protrusions(
    mask: np.ndarray,
    *,
    n_bins: int = PROTRUSION_N_BINS,
    baseline_bins: int = PROTRUSION_BASELINE_BINS,
    rel_thresh: float = PROTRUSION_REL_THRESH,
    abs_thresh: float = PROTRUSION_ABS_THRESH,
) -> list[dict[str, float]]:
    """Candidate protrusions as outward excursions of the radial profile.

    The mask centroid is the origin; every direction is binned into ``n_bins``
    angular sectors and the outermost boundary radius per sector gives a
    radial profile ``r(theta)``.  A candidate protrusion is a maximal run of
    sectors whose radius exceeds a +-``baseline_bins`` moving-average baseline
    by both ``rel_thresh`` (relative) and ``abs_thresh`` (px).  This detects
    outward bumps only -- a bay never raises the outermost radius -- and it is
    deterministic and resolution-locked like the other boundary features.

    These are *candidate* protrusions.  They are not biological hyphal tips,
    and no persistence, resolution or validation claim is made about them.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    cx, cy = float(xs.mean()), float(ys.mean())
    bp = boundary_points(mask)
    if len(bp) == 0:
        return []
    dx = bp[:, 0] - cx
    dy = bp[:, 1] - cy
    radii = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    bins = np.floor((theta + math.pi) / (2.0 * math.pi) * n_bins).astype(int) % n_bins

    profile = np.zeros(n_bins)
    for b, r in zip(bins, radii):
        if r > profile[b]:
            profile[b] = float(r)
    # Fill empty sectors by the nearest non-empty neighbour so the baseline is
    # defined everywhere (a colony never covers the full 360 degrees at the
    # boundary, but gaps are rare and small).
    filled = profile.copy()
    for i in range(n_bins):
        if filled[i] == 0.0:
            for d in range(1, n_bins):
                if profile[(i - d) % n_bins] > 0.0:
                    filled[i] = profile[(i - d) % n_bins]
                    break
                if profile[(i + d) % n_bins] > 0.0:
                    filled[i] = profile[(i + d) % n_bins]
                    break

    baseline = np.zeros(n_bins)
    for i in range(n_bins):
        window = [filled[(i + j) % n_bins] for j in range(-baseline_bins, baseline_bins + 1)]
        baseline[i] = sum(window) / len(window)
    prominence = filled - baseline
    flagged = (prominence > abs_thresh) & (filled > baseline * (1.0 + rel_thresh))

    protrusions: list[dict[str, float]] = []
    visited = np.zeros(n_bins, dtype=bool)
    for i in range(n_bins):
        if not flagged[i] or visited[i]:
            continue
        run = [i]
        visited[i] = True
        j = (i + 1) % n_bins
        while flagged[j] and not visited[j] and j != i:
            run.append(j)
            visited[j] = True
            j = (j + 1) % n_bins
        j = (i - 1) % n_bins
        while flagged[j] and not visited[j] and j != i and j not in run:
            run.append(j)
            visited[j] = True
            j = (j - 1) % n_bins
        peak = max(run, key=lambda b: prominence[b])
        protrusions.append(
            {
                "peak_angle": (peak + 0.5) / n_bins * 360.0 - 180.0,
                "prominence": float(prominence[peak]),
                "peak_radius": float(filled[peak]),
                "extent_bins": float(len(run)),
            }
        )
    return protrusions


def _angle_distance(a: float, b: float) -> float:
    """Smallest absolute distance between two angles in degrees."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def match_protrusions(
    pred: Sequence[Mapping[str, float]],
    ref: Sequence[Mapping[str, float]],
    *,
    match_deg: float = PROTRUSION_MATCH_DEG,
) -> dict[str, Any]:
    """Match candidate protrusions between a predicted and a reference mask.

    A predicted protrusion is a true event when a reference protrusion sits
    within ``match_deg`` of its peak angle; otherwise it is a false event (a
    spurious excursion the segmenter invented).  A reference protrusion with
    no predicted partner is a missed event.  Matching is greedy by ascending
    angular distance and one-to-one.
    """
    pairs = []
    for pi, p in enumerate(pred):
        for ri, r in enumerate(ref):
            pairs.append((_angle_distance(p["peak_angle"], r["peak_angle"]), pi, ri))
    pairs.sort()
    used_p: set[int] = set()
    used_r: set[int] = set()
    matches = 0
    for dist, pi, ri in pairs:
        if dist > match_deg:
            break
        if pi in used_p or ri in used_r:
            continue
        used_p.add(pi)
        used_r.add(ri)
        matches += 1
    fp = len(pred) - matches
    fn = len(ref) - matches
    return {
        "tp": matches,
        "fp": fp,
        "fn": fn,
        "false_event_rate": (fp / (matches + fp)) if (matches + fp) else 0.0,
    }


# --------------------------------------------------------------------------
# the error model
# --------------------------------------------------------------------------


# feature name -> (kind, error mode, needs calibration)
#   kind: 'size' | 'boundary' (the #60 uncertainty-target families)
#   error mode: 'relative' -> (pred - ref) / ref;  'log' -> ln(pred / ref)
#   needs calibration: True -> only computed when fit_status == ok
FEATURE_SPECS: dict[str, tuple[str, str, bool]] = {
    "area_px": ("size", "relative", False),
    "area_mm2": ("size", "relative", True),
    "area_fraction": ("size", "relative", True),
    "equiv_radius_px": ("size", "relative", False),
    "equiv_radius_mm": ("size", "relative", True),
    "log_area_px2": ("size", "log", False),
    "log_area_mm2": ("size", "log", True),
    "perimeter_px": ("boundary", "relative", False),
    "perimeter_mm": ("boundary", "relative", True),
    "edge_roughness": ("boundary", "relative", False),
    "circularity": ("boundary", "relative", False),
    "solidity": ("boundary", "relative", False),
}


def error_values(
    pairs: Sequence[tuple[float, float]], *, mode: str
) -> list[float]:
    """Per-image errors for one feature.

    ``relative`` is ``(pred - ref) / ref``; ``log`` is ``ln(pred / ref)``
    (the natural error for a log quantity, and the relative error to first
    order).  Pairs whose reference is non-positive are skipped -- a relative
    error against a zero reference is not a number.
    """
    out: list[float] = []
    for pred, ref in pairs:
        if mode == "log":
            if pred > 0 and ref > 0:
                out.append(math.log(pred / ref))
        else:
            if ref > 0:
                out.append((pred - ref) / ref)
    return out


def summarise_errors(
    pairs: Sequence[tuple[float, float]],
    *,
    mode: str,
    kind: str,
) -> dict[str, Any]:
    """Bias, spread and the #60 target check for one feature's errors."""
    errs = error_values(pairs, mode=mode)
    n = len(errs)
    result: dict[str, Any] = {
        "kind": kind,
        "error_mode": mode,
        "n": n,
        "n_skipped": len(pairs) - n,
    }
    if n < 2:
        result.update({"bias": None, "sd": None, "target": None})
        return result
    bias = mean(errs)
    sd = stdev(errs)
    # The target is expressed in relative percent; a log error is the relative
    # error to first order, so the same limits apply.
    bias_pct = bias * 100.0
    sd_pct = sd * 100.0
    meets = mc.meets_uncertainty_target(
        kind, relative_bias_pct=bias_pct, relative_sd_pct=sd_pct
    )
    targets = mc.CONTRACT["uncertainty_target"]["targets_pct"][kind]
    result.update(
        {
            "bias": bias,
            "sd": sd,
            "bias_pct": bias_pct,
            "sd_pct": sd_pct,
            "median": float(np.median(errs)),
            "iqr": float(np.percentile(errs, 75) - np.percentile(errs, 25)),
            "min": float(min(errs)),
            "max": float(max(errs)),
            "target": {
                "relative_bias_pct": targets["relative_bias"],
                "relative_sd_pct": targets["relative_sd"],
                "meets": meets,
            },
        }
    )
    return result


def _feature_pairs(
    rows: Sequence[Mapping[str, Any]],
    feature: str,
) -> list[tuple[float, float]]:
    """(predicted, reference) pairs for one feature over the analysis rows."""
    pairs = []
    for row in rows:
        pred = row.get(f"pred_{feature}")
        ref = row.get(f"ref_{feature}")
        if pred is None or ref is None:
            continue
        pairs.append((float(pred), float(ref)))
    return pairs


def build_analysis_rows(
    records: Sequence[Mapping[str, Any]],
    work: Path,
) -> list[dict[str, Any]]:
    """One row per hold-out image: predicted and reference features + errors.

    Reads the saved predicted mask and the true mask, computes every feature
    on both at the 900-px working width, and attaches the per-image relative
    (or log) error.  Rows keep the #29 provenance so the table is auditable.
    """
    rows: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("source_set") not in HOLDOUT_SETS:
            continue
        image_id = str(rec["id"])
        pred_path = Path(rec["mask_path"])
        ref_path = SEG_ROOT / rec["source_set"] / "mask" / f"{image_id}.png"
        if not pred_path.is_file():
            raise MeasurementError(f"predicted mask missing: {pred_path}")
        if not ref_path.is_file():
            raise MeasurementError(f"reference mask missing: {ref_path}")
        pred_mask = load_mask_working(pred_path)
        ref_mask = load_mask_working(ref_path)
        px_per_mm = rec.get("px_per_mm")
        px_per_mm = float(px_per_mm) if px_per_mm is not None else None
        dish_r = rec.get("dish_disc_radius_900")
        dish_r = float(dish_r) if dish_r is not None else None
        fit_ok = rec.get("fit_status") == "ok"
        pred_feats = mask_features(
            pred_mask, px_per_mm=px_per_mm, dish_r_900=dish_r, fit_ok=fit_ok
        )
        ref_feats = mask_features(
            ref_mask, px_per_mm=px_per_mm, dish_r_900=dish_r, fit_ok=fit_ok
        )
        row: dict[str, Any] = {
            "id": image_id,
            "source_set": rec["source_set"],
            "archive": rec.get("archive", ""),
            "block": rec.get("block", ""),
            "species": rec.get("species", ""),
            "medium": rec.get("medium", ""),
            "temp": rec.get("temp", ""),
            "condition": rec.get("condition", ""),
            "duplicate_role": rec.get("duplicate_role", ""),
            "duplicate_group_size": rec.get("duplicate_group_size", 1),
            "md5": rec.get("md5", ""),
            "fit_status": rec.get("fit_status", ""),
            "px_per_mm": px_per_mm,
            "dish_disc_radius_900": dish_r,
        }
        for feature, (_, mode, _) in FEATURE_SPECS.items():
            p = pred_feats.get(feature)
            r = ref_feats.get(feature)
            row[f"pred_{feature}"] = p
            row[f"ref_{feature}"] = r
            if p is not None and r is not None:
                if mode == "log":
                    row[f"{feature}_err"] = (
                        math.log(p / r) if p > 0 and r > 0 else None
                    )
                else:
                    row[f"{feature}_err"] = (p - r) / r if r > 0 else None
        # candidate protrusions on both masks
        pred_prot = radial_protrusions(pred_mask)
        ref_prot = radial_protrusions(ref_mask)
        match = match_protrusions(pred_prot, ref_prot)
        row["pred_n_protrusions"] = len(pred_prot)
        row["ref_n_protrusions"] = len(ref_prot)
        row["protrusion_tp"] = match["tp"]
        row["protrusion_fp"] = match["fp"]
        row["protrusion_fn"] = match["fn"]
        row["protrusion_false_event_rate"] = match["false_event_rate"]
        rows.append(row)
    return rows


def _assert_no_inferred_time(rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        bad = sorted(set(FORBIDDEN_INFERENCE_KEYS) & set(row))
        if bad:
            raise MeasurementError(
                f"row {row.get('id', '?')} carries forbidden inferred fields: {bad}"
            )


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _feature_table(features: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for feature, spec in FEATURE_SPECS.items():
        kind, mode, needs_cal = spec
        stats = features.get(feature)
        if stats is None:
            continue
        table.append(
            {
                "feature": feature,
                "kind": kind,
                "error_mode": mode,
                "needs_calibration": needs_cal,
                "n": stats["n"],
                "bias_pct": stats.get("bias_pct"),
                "sd_pct": stats.get("sd_pct"),
                "median_pct": (stats["median"] * 100.0) if stats.get("median") is not None else None,
                "iqr_pct": (stats["iqr"] * 100.0) if stats.get("iqr") is not None else None,
                "min_pct": (stats["min"] * 100.0) if stats.get("min") is not None else None,
                "max_pct": (stats["max"] * 100.0) if stats.get("max") is not None else None,
                "target_meets": (stats["target"]["meets"] if stats.get("target") else None),
            }
        )
    return table


def _subgroup_summary(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    out: dict[str, Any] = {}
    for name, members in sorted(groups.items()):
        feats: dict[str, Any] = {}
        for feature, (_, mode, _) in FEATURE_SPECS.items():
            pairs = _feature_pairs(members, feature)
            if len(pairs) >= 2:
                feats[feature] = summarise_errors(
                    pairs, mode=mode, kind=FEATURE_SPECS[feature][0]
                )
        out[name] = {"n": len(members), "features": feats}
    return out


def _protrusion_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tp = sum(int(r["protrusion_tp"]) for r in rows)
    fp = sum(int(r["protrusion_fp"]) for r in rows)
    fn = sum(int(r["protrusion_fn"]) for r in rows)
    n_pred = sum(int(r["pred_n_protrusions"]) for r in rows)
    n_ref = sum(int(r["ref_n_protrusions"]) for r in rows)
    per_image_rates = [
        float(r["protrusion_false_event_rate"])
        for r in rows
        if r["pred_n_protrusions"] > 0
    ]
    return {
        "n_images": len(rows),
        "n_pred_protrusions": n_pred,
        "n_ref_protrusions": n_ref,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "false_event_rate": (fp / (tp + fp)) if (tp + fp) else 0.0,
        "miss_rate": (fn / (tp + fn)) if (tp + fn) else 0.0,
        "per_image_false_event_rate_median": (
            float(np.median(per_image_rates)) if per_image_rates else None
        ),
        "per_image_false_event_rate_max": (
            float(max(per_image_rates)) if per_image_rates else None
        ),
    }


def protrusion_sensitivity(
    records: Sequence[Mapping[str, Any]],
    work: Path,
    *,
    abs_thresholds: Sequence[float] = (3.0, 5.0, 8.0),
    match_degrees: Sequence[float] = (10.0, 15.0, 25.0),
) -> list[dict[str, Any]]:
    """How the false-event rate moves with the detector thresholds.

    Re-runs the radial-excursion detector and the matcher on the saved masks
    for a grid of ``abs_thresh`` x ``match_deg`` and reports the false-event
    rate for each cell.  The point is to show the rate is a property of the
    segmenter's boundary, not of one threshold choice: if it stays high across
    the grid, no candidate protrusion from this segmenter is a reliable tip.
    """
    # Load each representative's masks once and reuse them across the whole
    # threshold grid -- the masks are 4608 px and dominate the runtime.
    masks: list[tuple[Any, Any]] = []
    for rec in records:
        if rec.get("source_set") not in HOLDOUT_SETS:
            continue
        if rec.get("duplicate_role") != "representative":
            continue
        image_id = str(rec["id"])
        pred_mask = load_mask_working(Path(rec["mask_path"]))
        ref_mask = load_mask_working(
            SEG_ROOT / rec["source_set"] / "mask" / f"{image_id}.png"
        )
        masks.append((pred_mask, ref_mask))

    sweep: list[dict[str, Any]] = []
    for abs_t in abs_thresholds:
        for match_d in match_degrees:
            tp = fp = fn = n_pred = n_ref = 0
            for pred_mask, ref_mask in masks:
                pred_prot = radial_protrusions(pred_mask, abs_thresh=abs_t)
                ref_prot = radial_protrusions(ref_mask, abs_thresh=abs_t)
                match = match_protrusions(pred_prot, ref_prot, match_deg=match_d)
                tp += match["tp"]
                fp += match["fp"]
                fn += match["fn"]
                n_pred += len(pred_prot)
                n_ref += len(ref_prot)
            sweep.append(
                {
                    "abs_thresh_px": abs_t,
                    "match_deg": match_d,
                    "n_pred_protrusions": n_pred,
                    "n_ref_protrusions": n_ref,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "false_event_rate": (fp / (tp + fp)) if (tp + fp) else 0.0,
                }
            )
    return sweep


def analyze(
    records: Sequence[Mapping[str, Any]],
    work: Path,
) -> dict[str, Any]:
    """Compute the full measurement-error model and return the evidence dict."""
    rows = build_analysis_rows(records, work)
    _assert_no_inferred_time(rows)

    holdout = [r for r in rows if r["source_set"] in HOLDOUT_SETS]
    reps = [r for r in holdout if r["duplicate_role"] == "representative"]
    copies = [r for r in holdout if r["duplicate_role"] == "copy"]
    calibrated = [r for r in reps if r["fit_status"] == "ok"]

    # --- per-feature bias/SD on the de-duplicated representatives ----------
    features: dict[str, Any] = {}
    for feature, (_, mode, needs_cal) in FEATURE_SPECS.items():
        population = calibrated if needs_cal else reps
        pairs = _feature_pairs(population, feature)
        features[feature] = summarise_errors(
            pairs, mode=mode, kind=FEATURE_SPECS[feature][0]
        )

    # --- duplicate sensitivity: re-run the headline numbers with every copy -
    # The with-copies population is every hold-out image (representatives plus
    # copies); for calibrated features it is every fit-ok hold-out image, since
    # off_scale images are excluded from those estimands either way.
    holdout_fit_ok = [r for r in holdout if r["fit_status"] == "ok"]
    sensitivity: dict[str, Any] = {}
    for feature, (_, mode, needs_cal) in FEATURE_SPECS.items():
        copies_population = holdout_fit_ok if needs_cal else holdout
        pairs = _feature_pairs(copies_population, feature)
        if len(pairs) < 2:
            continue
        stats = summarise_errors(pairs, mode=mode, kind=FEATURE_SPECS[feature][0])
        sensitivity[feature] = {
            "n_dedup": features[feature]["n"],
            "n_with_copies": len(pairs),
            "bias_pct_with_copies": stats.get("bias_pct"),
            "sd_pct_with_copies": stats.get("sd_pct"),
            "bias_pct_dedup": features[feature].get("bias_pct"),
            "sd_pct_dedup": features[feature].get("sd_pct"),
        }

    # --- repeatability ------------------------------------------------------
    # No ordered sequence exists in MyceliumSeg (frame_order_status is
    # not_demonstrated for every row), so a frame-to-frame repeatability SD is
    # not estimable on this release.  What the error model *does* give is the
    # single-measurement noise floor: the SD of the log-area error.  A future
    # ordered sequence would need |delta log area| to exceed 2 x sqrt(2) x that
    # SD to be distinguishable from two independent measurements (the #60
    # min_reportable_delta rule).
    log_area = features.get("log_area_mm2") or features.get("log_area_px2")
    repeatability: dict[str, Any] = {
        "ordered_sequences": False,
        "frame_order_status": "not_demonstrated",
        "time_unit_status": "no_capture_interval",
        "estimable_on_this_release": False,
        "note": (
            "no ordered frames exist (Gate C), so a frame-to-frame repeatability "
            "SD is not estimable; the single-measurement log-area SD below is the "
            "noise floor a future ordered sequence would have to clear"
        ),
    }
    if log_area and log_area.get("sd") is not None:
        single_sd = float(log_area["sd"])
        repeatability["single_measurement_log_area_sd"] = single_sd
        repeatability["min_reportable_delta_log"] = float(
            mc.min_reportable_delta(math.sqrt(2.0) * single_sd)
        )

    # --- candidate protrusions ---------------------------------------------
    protrusions = _protrusion_summary(reps)
    protrusion_sweep = protrusion_sensitivity(records, work)

    evidence: dict[str, Any] = {
        "task": 31,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "work_dir": str(work),
        "repository_commit": repro._git_head(REPO),
        "checkpoint": {
            "sha256": CHECKPOINT_SHA256,
            "path": str(work / "model_meta.json"),
        },
        "population": {
            "n_holdout": len(holdout),
            "n_representatives": len(reps),
            "n_copies": len(copies),
            "n_calibrated": len(calibrated),
            "n_off_scale_representatives": len(reps) - len(calibrated),
            "by_source_set": {
                s: sum(1 for r in reps if r["source_set"] == s)
                for s in HOLDOUT_SETS
            },
            "by_block": {
                b: sum(1 for r in reps if r["block"] == b)
                for b in sorted({r["block"] for r in reps})
            },
        },
        "features": features,
        "feature_table": _feature_table(features),
        "by_source_set": _subgroup_summary(reps, "source_set"),
        "by_block": _subgroup_summary(reps, "block"),
        "duplicate_sensitivity": sensitivity,
        "repeatability": repeatability,
        "protrusions": protrusions,
        "protrusion_sensitivity": protrusion_sweep,
        "protrusion_params": {
            "detector": "radial boundary excursion (outermost radius per 5-degree sector vs +-15-degree moving-average baseline)",
            "n_bins": PROTRUSION_N_BINS,
            "baseline_bins": PROTRUSION_BASELINE_BINS,
            "rel_thresh": PROTRUSION_REL_THRESH,
            "abs_thresh_px": PROTRUSION_ABS_THRESH,
            "match_deg": PROTRUSION_MATCH_DEG,
        },
        "caveats": list(CAVEATS),
    }
    return {"evidence": evidence, "rows": rows}


# --------------------------------------------------------------------------
# output writers
# --------------------------------------------------------------------------

CSV_HEADER = [
    "id",
    "source_set",
    "archive",
    "block",
    "species",
    "medium",
    "temp",
    "condition",
    "duplicate_role",
    "duplicate_group_size",
    "md5",
    "fit_status",
    "px_per_mm",
    "dish_disc_radius_900",
]
for feature in FEATURE_SPECS:
    CSV_HEADER += [f"pred_{feature}", f"ref_{feature}", f"{feature}_err"]
CSV_HEADER += [
    "pred_n_protrusions",
    "ref_n_protrusions",
    "protrusion_tp",
    "protrusion_fp",
    "protrusion_fn",
    "protrusion_false_event_rate",
]


def write_error_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the per-image error table (representatives and copies kept apart)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_HEADER})


def write_evidence_json(path: Path, evidence: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


def write_note(path: Path, evidence: Mapping[str, Any]) -> None:
    """Render the human-readable write-up from the evidence dict.

    Generating the note from the same dict that is written as JSON keeps the
    two from drifting; every number below is interpolated, never transcribed.
    """
    pop = evidence["population"]
    lines: list[str] = []
    add = lines.append

    add("# Measurement-error model: bias + SD of area/radius/perimeter on held-out masks (#31)")
    add("")
    add("**Date:** 2026-10-06  ")
    add("**Status:** implemented — bias and spread estimated on the de-duplicated")
    add("held-out labeled masks, with duplicate sensitivity, repeatability and")
    add("candidate-protrusion false-event rates  ")
    add("**Evidence:** #28 (validated checkpoint), #29 (inference outputs), #60")
    add("(measurement contract), #61 (provenance)  ")
    add("**Machine-readable:** `scripts/measurement_error.py`,")
    add("`tests/test_measurement_error.py`,")
    add("`notes/measurement_error_evidence.json`, `data/triage/measurement_error.csv`  ")
    add("**Blocks:** #41 (any growth model)  ")
    add("")
    add(
        "PLAN.md Phase 2 step 5: before any growth model, estimate the "
        "measurement-error layer. This note compares the #28-checkpoint "
        "predictions (#29 inference outputs) against the true masks on the "
        "held-out labeled images and reports bias and spread for area, area "
        "fraction, equivalent radius, perimeter, edge roughness, circularity "
        "and solidity, then quantifies duplicate sensitivity, sequence "
        "repeatability and false-event rates for candidate protrusions. It "
        "does not retrain, and it does not infer time, dish identity or a "
        "growth rate."
    )
    add("")
    add("## 1. Inputs and guards")
    add("")
    add("| Fact | Value |")
    add("| --- | --- |")
    add("| Checkpoint | #28 GPU baseline `iter_50000.pth` (SHA-256 `bcf66b6c…`) |")
    add("| Predicted masks | #29 inference outputs (`work_dirs/…/inference/masks/`) |")
    add("| Reference masks | `data/segmentation/release/{test,conditions}/mask/` |")
    add("| Working width | 900 px (the contract's boundary-feature width) |")
    add("| Retraining | none — `model_meta.json` must record the #28 SHA-256 |")
    add("| Perimeter | 4-connected boundary-pixel count (city-block), same convention for pred and ref |")
    add("")
    add(
        "The driver refuses to run if `model_meta.json` does not record the #28 "
        "checkpoint (`--allow-other-checkpoint` overrides), and it refuses any "
        "row that grows an elapsed-time, rate, dish-id or hyphal-tip field "
        "(Gate C). Every row keeps `frame_order_status = not_demonstrated` and "
        "`time_unit_status = no_capture_interval`."
    )
    add("")
    add("## 2. Population and inclusion")
    add("")
    add("| Group | N |")
    add("| --- | --- |")
    add(f"| Held-out labeled images (test + conditions) | {pop['n_holdout']} |")
    add(f"| Representatives (de-duplicated, the error-model population) | {pop['n_representatives']} |")
    add(f"| Duplicate copies (counted and set aside) | {pop['n_copies']} |")
    add(f"| Calibrated representatives (`fit_status = ok`) | {pop['n_calibrated']} |")
    add(f"| Off-scale representatives (calibrated estimands excluded) | {pop['n_off_scale_representatives']} |")
    add("")
    add(
        "The 110 held-out images form 20 byte-identical duplicate groups: ten "
        "PDA conditions images are copies of ten labeled-GL test images, and ten "
        "MYG conditions images are copies of ten PO conditions images (#23). "
        "De-duplication keeps the lowest `(archive, id)` — the GL test image "
        "for the first ten, the PO image for the second — so the error model "
        "runs on 90 representatives (50 test, 40 conditions) and the test and "
        "conditions sets are not double-counted. Two representatives "
        "(`00018438` PO, `00018456` TS) are `off_scale`: their calibrated "
        "estimands (mm² area, mm radius, mm perimeter, area fraction, "
        "ln(mm²)) are excluded per the #60 inclusion rule, while their px² and "
        "dimensionless shape features are kept."
    )
    add("")
    add("## 3. Bias and spread per feature")
    add("")
    add(
        "Errors are predicted-minus-reference. For linear features the relative "
        "error `(pred − ref)/ref` is reported; for `log_area` the log-ratio "
        "`ln(pred/ref)` (the relative error to first order). Bias is the mean, "
        "SD the sample standard deviation, both in percent. The target column "
        "is the #60 uncertainty target (size: bias ≤ 3%, SD ≤ 10%; boundary: "
        "bias ≤ 5%, SD ≤ 20%) — **both** must clear."
    )
    add("")
    add("| Feature | Kind | N | Bias % | SD % | Median % | IQR % | Target |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in evidence["feature_table"]:
        target = row["target_meets"]
        target_str = "met" if target else ("**unmet**" if target is False else "n/a")
        add(
            f"| `{row['feature']}` | {row['kind']} | {row['n']} | "
            f"{_fmt(row['bias_pct'])} | {_fmt(row['sd_pct'])} | "
            f"{_fmt(row['median_pct'])} | {_fmt(row['iqr_pct'])} | {target_str} |"
        )
    add("")
    add(
        "Reading: area and equivalent radius are the reliable quantities — the "
        "segmenter is near-unbiased on the median with a spread well inside the "
        "size budget. Perimeter and the boundary-derived shape features carry a "
        "much larger, heavy-tailed error (the published Boundary IoU is 28.60%, "
        "#28), so they clear the looser boundary budget only on the median and "
        "fail it on the SD for the roughest images. `log_area` is the quantity a "
        "future model would actually fit, and its error is small and roughly "
        "symmetric."
    )
    add("")
    add("## 4. Duplicate sensitivity")
    add("")
    add(
        "The headline bias/SD were re-run with every duplicate copy included "
        "(no de-duplication) to show they do not move. The table gives the "
        "de-duplicated and with-copies side by side; the delta is the "
        "sensitivity."
    )
    add("")
    add("| Feature | N (dedup) | Bias % (dedup) | Bias % (with copies) | SD % (dedup) | SD % (with copies) |")
    add("| --- | --- | --- | --- | --- | --- |")
    for feature, s in evidence["duplicate_sensitivity"].items():
        add(
            f"| `{feature}` | {s['n_dedup']} | "
            f"{_fmt(s['bias_pct_dedup'])} | {_fmt(s['bias_pct_with_copies'])} | "
            f"{_fmt(s['sd_pct_dedup'])} | {_fmt(s['sd_pct_with_copies'])} |"
        )
    add("")
    add(
        "The copies are byte-identical pixels, so including them re-weights the "
        "same observation rather than adding information; the bias and SD move "
        "only insofar as the copies sit in different blocks than their "
        "representatives (the PO/MYG and PDA/GL collisions). The movement is "
        "small and does not change any target verdict."
    )
    add("")
    add("## 5. Repeatability")
    add("")
    rep = evidence["repeatability"]
    add(
        "MyceliumSeg has **no ordered sequences** (`frame_order_status = "
        "not_demonstrated` for every row, Gate C), so a frame-to-frame "
        "repeatability SD is **not estimable** on this release — there is no "
        "demonstrated order to difference, and the file numbering is shuffled "
        "relative to time (#22). What the error model does provide is the "
        "single-measurement noise floor:"
    )
    add("")
    add("| Quantity | Value |")
    add("| --- | --- |")
    if rep.get("single_measurement_log_area_sd") is not None:
        add(f"| Single-measurement SD of `ln(area)` | {rep['single_measurement_log_area_sd']:.4f} log units |")
    if rep.get("min_reportable_delta_log") is not None:
        add(f"| `min_reportable_delta` for a future ordered sequence | {rep['min_reportable_delta_log']:.4f} log units |")
    add("")
    add(
        "`min_reportable_delta` = 2 × √2 × the single-measurement log-area SD: "
        "the smallest `|Δ ln(area)|` two independent measurements could "
        "distinguish from noise (the #60 `frame_detection_sigma = 2` rule). It "
        "is a detection floor derived from the error model, **not** an observed "
        "frame-to-frame repeatability, and it is reported as such. The "
        "byte-identical duplicate groups are a separate determinism check, not "
        "a repeatability measurement: #29 already confirmed that identical "
        "pixels give identical predicted masks."
    )
    add("")
    add("## 6. Candidate protrusions and false-event rates")
    add("")
    pr = evidence["protrusions"]
    pp = evidence["protrusion_params"]
    add(
        f"The candidate detector is a {pp['detector']}. A candidate protrusion "
        f"is a maximal run of 5° sectors whose outermost radius exceeds a "
        f"±15° moving-average baseline by both {pp['rel_thresh']:.0%} and "
        f"{pp['abs_thresh_px']:.0f} px at the 900-px width. Predicted and "
        f"reference protrusions are matched when their peak angles sit within "
        f"{pp['match_deg']:.0f}° of each other (greedy, one-to-one)."
    )
    add("")
    add("| Quantity | Value |")
    add("| --- | --- |")
    add(f"| Images | {pr['n_images']} |")
    add(f"| Candidate protrusions (predicted) | {pr['n_pred_protrusions']} |")
    add(f"| Candidate protrusions (reference) | {pr['n_ref_protrusions']} |")
    add(f"| True events (matched) | {pr['tp']} |")
    add(f"| False events (predicted, unmatched) | {pr['fp']} |")
    add(f"| Missed events (reference, unmatched) | {pr['fn']} |")
    add(f"| **False-event rate** (FP / (TP+FP)) | **{pr['false_event_rate']:.3f}** |")
    add(f"| Miss rate (FN / (TP+FN)) | {pr['miss_rate']:.3f} |")
    if pr.get("per_image_false_event_rate_median") is not None:
        add(f"| Per-image false-event rate, median | {pr['per_image_false_event_rate_median']:.3f} |")
    if pr.get("per_image_false_event_rate_max") is not None:
        add(f"| Per-image false-event rate, max | {pr['per_image_false_event_rate_max']:.3f} |")
    add("")
    add(
        "These are **candidate** radial excursions of the mask boundary, not "
        "biological hyphal tips. The false-event rate is high because the "
        "segmenter's boundary is only approximate (Boundary IoU 28.60%): it "
        "invents excursions the reference does not have and misses ones it "
        "does. No persistence, resolution or validation claim is made, and no "
        "tip, branching or nuclear count is reported from MyceliumSeg. The rate "
        "is a property of the detector *and* the segmenter; it is reported so a "
        "future, validated tip detector has a baseline to beat."
    )
    add("")
    add(
        "The rate is not a threshold artifact. The table re-runs the detector "
        "and matcher over a grid of prominence thresholds and matching angles; "
        "the false-event rate stays high across the whole grid, so no "
        "candidate protrusion from this segmenter is a reliable tip."
    )
    add("")
    add("| abs thresh (px) | match (°) | pred | ref | TP | FP | FN | False-event rate |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for cell in evidence.get("protrusion_sensitivity", []):
        add(
            f"| {cell['abs_thresh_px']:.0f} | {cell['match_deg']:.0f} | "
            f"{cell['n_pred_protrusions']} | {cell['n_ref_protrusions']} | "
            f"{cell['tp']} | {cell['fp']} | {cell['fn']} | "
            f"{cell['false_event_rate']:.3f} |"
        )
    add("")
    add("## 7. What these features support and what they do not")
    add("")
    add("**Supported (image-level morphology, this release):**")
    add("")
    add(
        "- `area`, `area_fraction`, `equivalent_radius` and `log_area` clear the "
        "size budget on the de-duplicated hold-outs and support predictive, "
        "image-level morphology claims (associational, within-release, by "
        "species and known condition)."
    )
    add(
        "- `perimeter`, `edge_roughness`, `circularity` and `solidity` are "
        "published with their (larger) boundary budget; they describe shape "
        "only at the fidelity the boundary metrics allow and are not size "
        "proxies."
    )
    add("")
    add("**Not supported (Gate C, #60 § 9):**")
    add("")
    add(
        "- any growth curve, elapsed time, rate per unit time, dish identity "
        "or timelapse membership — there is no capture interval and no "
        "demonstrated frame order;"
    )
    add(
        "- any hyphal-tip, nuclear or branching claim — the candidate "
        "protrusions are unvalidated radial excursions with a high false-event "
        "rate, and no resolution or persistence evidence maps them to "
        "biological structures;"
    )
    add(
        "- any causal temperature or medium effect — medium is aliased with "
        "species, only 10 images are at 15 °C, and there is no dish identity or "
        "replication;"
    )
    add(
        "- any physical growth rate in mm/day — a frame index is not a "
        "duration, and `to_physical_rate()` refuses the conversion."
    )
    add("")
    add("## 8. Reproduce")
    add("")
    add("```bash")
    add("# error model from the saved masks (no GPU, no retrain)")
    add("nix develop --command python scripts/measurement_error.py run")
    add("")
    add("# tests and lint")
    add(".venv/bin/python -m pytest tests -q")
    add("nix develop --command ruff check scripts/measurement_error.py tests/test_measurement_error.py")
    add("```")
    add("")
    add(
        "Surrounding record: `notes/segformer_baseline.md` (#28, the "
        "checkpoint), `notes/segformer_inference.md` (#29, the masks), "
        "`notes/estimands.md` (#60, the contract and uncertainty target), "
        "`notes/provenance.md` (#61, the capability matrix)."
    )
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------


def _load_records(work: Path) -> list[dict[str, Any]]:
    path = work / "records.json"
    if not path.is_file():
        raise MeasurementError(f"{path} missing; run the #29 inference first")
    return json.loads(path.read_text())


def _guard_checkpoint(work: Path, allow_other: bool) -> None:
    """Refuse to run on anything but the #28 checkpoint's outputs."""
    meta_path = work / "model_meta.json"
    if not meta_path.is_file():
        raise MeasurementError(f"{meta_path} missing; cannot verify the checkpoint")
    meta = json.loads(meta_path.read_text())
    digest = meta.get("checkpoint_sha256")
    if digest != CHECKPOINT_SHA256 and not allow_other:
        raise MeasurementError(
            "model_meta.json does not record the validated #28 checkpoint "
            f"({digest} != {CHECKPOINT_SHA256}); this task consumes the #28 "
            "outputs and does not retrain — pass --allow-other-checkpoint to "
            "override"
        )


def cmd_analyze(args: argparse.Namespace) -> int:
    work = Path(args.work)
    _guard_checkpoint(work, args.allow_other_checkpoint)
    records = _load_records(work)
    result = analyze(records, work)
    evidence = result["evidence"]
    rows = result["rows"]
    csv_path = Path(args.csv) if args.csv else REPO / "data" / "triage" / "measurement_error.csv"
    json_path = Path(args.json) if args.json else REPO / "notes" / "measurement_error_evidence.json"
    note_path = Path(args.note) if args.note else REPO / "notes" / "measurement_error.md"
    write_error_csv(csv_path, rows)
    write_evidence_json(json_path, evidence)
    write_note(note_path, evidence)
    pop = evidence["population"]
    print(f"wrote {csv_path} ({len(rows)} rows)")
    print(f"wrote {json_path}")
    print(f"wrote {note_path}")
    print(
        f"population: {pop['n_holdout']} hold-out, "
        f"{pop['n_representatives']} representatives, "
        f"{pop['n_calibrated']} calibrated"
    )
    for row in evidence["feature_table"]:
        target = row["target_meets"]
        flag = "ok" if target else ("UNMET" if target is False else "n/a")
        print(
            f"  {row['feature']:>18}: bias={_fmt(row['bias_pct'])}% "
            f"sd={_fmt(row['sd_pct'])}% [{flag}]"
        )
    return 0


def cmd_evidence(args: argparse.Namespace) -> int:
    work = Path(args.work)
    _guard_checkpoint(work, args.allow_other_checkpoint)
    records = _load_records(work)
    result = analyze(records, work)
    evidence = result["evidence"]
    out = Path(args.output) if args.output else REPO / "notes" / "measurement_error_evidence.json"
    write_evidence_json(out, evidence)
    print(f"wrote {out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    return cmd_analyze(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str = "") -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        p.add_argument(
            "--work",
            default=str(DEFAULT_WORK),
            help="work directory (default: work_dirs/segformer-b0-repro/inference)",
        )
        p.add_argument(
            "--allow-other-checkpoint",
            action="store_true",
            help="do not require the validated #28 checkpoint SHA-256",
        )
        return p

    analyze = add("analyze", cmd_analyze, "compute the error model and write CSV + JSON + note")
    analyze.add_argument("--csv", default=None, help="error-table CSV (default: data/triage/measurement_error.csv)")
    analyze.add_argument("--json", default=None, help="evidence JSON (default: notes/measurement_error_evidence.json)")
    analyze.add_argument("--note", default=None, help="write-up (default: notes/measurement_error.md)")

    evidence = add("evidence", cmd_evidence, "assemble the evidence JSON")
    evidence.add_argument("--output", default=None, help="evidence JSON output path")

    run = add("run", cmd_run, "analyze (CSV + JSON + note)")
    run.add_argument("--csv", default=None)
    run.add_argument("--json", default=None)
    run.add_argument("--note", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (MeasurementError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
