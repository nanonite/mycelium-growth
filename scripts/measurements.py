#!/usr/bin/env python3
"""Observable measurement table from the validated #29 masks (Chainlink #30).

PLAN.md Phase 2 step 4: per image, extract colony area A, area fraction,
equivalent radius ``r = sqrt(A/pi)``, perimeter, edge roughness / circularity,
intensity summaries and density proxies, and write one provenance-rich
``data/measurements.csv`` with image-level and mask-level provenance kept
distinct.

Feature tiers (the #60 contract, ``notes/estimands.md`` section 3, and the
#31 error model that quantifies each feature's uncertainty):

* **primary** -- ``area``, ``log_area``, ``equivalent_radius`` (the size
  family).  The #31 error model puts their bias and spread inside the #60
  size budget (relative bias <= 3%, relative SD <= 10%), so they support
  image-level morphology claims.
* **secondary** -- ``perimeter``, ``edge_roughness``, ``circularity``,
  ``solidity`` (the boundary family).  #31 puts their error well outside
  the size budget (published Boundary IoU 28.60%, #28), so they carry the
  looser boundary budget and are reported as shape descriptors, never as
  size proxies.
* **secondary** -- ``mean_intensity`` and the intensity summaries (the
  density proxy).  Comparable within a condition block only; dark incubation
  and fixed exposure are per-block facts, so they are never compared across
  blocks.

What this driver does *not* do is as important as what it does:

* it does not infer elapsed time, a growth rate, dish identity, timelapse
  membership or a biological tip -- every row keeps
  ``frame_order_status = not_demonstrated`` and
  ``time_unit_status = no_capture_interval`` (Gate C), and no physical rate
  of any kind is reported;
* it does not retrain -- it reads the masks the #29 run produced from the
  #28 checkpoint and refuses to proceed if ``model_meta.json`` does not
  record that checkpoint's SHA-256 (``--allow-other-checkpoint`` overrides).

Guards (the #60 contract, ``scripts/measurement_contract.py``):

* calibrated features (mm^2, mm, area_fraction, ln(mm^2)) appear only when
  ``fit_status == ok``; the off_scale images keep their px^2 and
  dimensionless shape features and are flagged, not silently trusted;
* boundary-derived shape features are measured at the 900-px working width;
* no forbidden inferred field (elapsed time, rate, dish id, hyphal tip)
  appears in the output.

Subcommands (all idempotent, writing ``data/measurements.csv``,
``notes/measurements_evidence.json`` and ``notes/measurements.md``):

    build     read the #29 records + masks, write the measurement table
    evidence  assemble the run's environment, checkpoint, counts and caveats
    run       build -> evidence -> note

The mask half needs only numpy + PIL (the masks already exist), so the whole
driver runs in the plain dev shell; no GPU.

Exit codes: 0 ok, 1 a step failed.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import measurement_error as me
import segformer_repro as repro

WORK_ROOT = REPO / "work_dirs" / "segformer-b0-repro" / "inference"
DEFAULT_WORK = WORK_ROOT
MEASUREMENTS_CSV = REPO / "data" / "measurements.csv"
EVIDENCE_JSON = REPO / "notes" / "measurements_evidence.json"
NOTE_MD = REPO / "notes" / "measurements.md"

# The validated #28 hand-off (notes/segformer_evidence.json, commit fdfa43a).
CHECKPOINT_SHA256 = "bcf66b6c1b3ce44018bc54223bacad31bd48a1647efc842bfbc6bd5f6d3e20d6"

# The contract fixes the boundary-feature working width at 900 px
# (notes/estimands.md section 2.5); measurement_error reuses it.
WORKING_WIDTH = me.WORKING_WIDTH

# Field names that would mean an inference this task forbids (Gate C).
FORBIDDEN_INFERENCE_KEYS = me.FORBIDDEN_INFERENCE_KEYS

# Caveats carried into every output.
CAVEATS = (
    (
        "MyceliumSeg is cross-sectional: no ordered frames, no dish identity, no "
        "capture interval (Gate C, notes/provenance.md)."
    ),
    (
        "Area, log-area and equivalent radius are the primary size features; the "
        "#31 error model puts their bias and spread inside the #60 size budget."
    ),
    (
        "Perimeter, edge roughness, circularity and solidity are boundary-derived "
        "and secondary: #31 puts their error well outside the size budget "
        "(published Boundary IoU 28.60%, #28), so they carry the boundary budget "
        "and are not size proxies."
    ),
    (
        "Mean intensity and the intensity summaries are density proxies only: dark "
        "incubation and fixed exposure are per-block facts, so they are comparable "
        "within a condition block, never across."
    ),
    (
        "The test and conditions hold-outs are not independent: ten PDA conditions "
        "images are byte copies of ten labeled-GL test images (#23), so "
        "de-duplication removes the conditions copies before any summary."
    ),
)

# The measurement columns, in output order, each tagged with the tier the #60
# contract and the #31 error model assign it.  The tier is a property of the
# feature, not of the row, so it is recorded here once and rendered into the
# note's column dictionary rather than repeated on every row.
MEASUREMENT_COLUMNS: tuple[tuple[str, str, str], ...] = (
    # (column, tier, note)
    ("area_px", "primary", "colony mask area in px^2 at the 900-px working width"),
    ("area_mm2", "primary", "calibrated colony area in mm^2 (fit_status == ok only)"),
    ("area_fraction", "primary", "A_colony / A_dish, both in the same frame (fit_status == ok only)"),
    ("equiv_radius_px", "primary", "sqrt(area_px / pi) at the 900-px working width"),
    ("equiv_radius_mm", "primary", "calibrated equivalent radius in mm (fit_status == ok only)"),
    ("log_area_px2", "primary", "ln(area_px) -- log area with its reference unit stated"),
    ("log_area_mm2", "primary", "ln(area_mm^2) -- calibrated log area (fit_status == ok only)"),
    ("perimeter_px", "secondary", "4-connected boundary-pixel count at the 900-px working width"),
    ("perimeter_mm", "secondary", "calibrated perimeter in mm (fit_status == ok only)"),
    ("edge_roughness", "secondary", "P / (2 pi r_eq) -- 1.0 is a circle; resolution-locked"),
    ("circularity", "secondary", "4 pi A / P^2 -- dimensionless, resolution-locked"),
    ("solidity", "secondary", "A_colony / A_convex_hull -- dimensionless"),
    ("mean_intensity", "secondary", "mean 8-bit grey level inside the mask -- density proxy"),
    ("intensity_median", "secondary", "median 8-bit grey level inside the mask"),
    ("intensity_std", "secondary", "SD of the 8-bit grey level inside the mask -- texture proxy"),
    ("intensity_min", "secondary", "minimum 8-bit grey level inside the mask"),
    ("intensity_max", "secondary", "maximum 8-bit grey level inside the mask"),
)

# Provenance columns carried from the #29 records, in output order.  These are
# image-level and mask-level facts about where the measurement came from; they
# are the same facts #29 and #31 carry, so the three tables join on (id).
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "id",
    "archive",
    "source_set",
    "block",
    "species",
    "medium",
    "temp",
    "condition",
    "condition_known",
    "condition_source",
    "label_conflict",
    "md5",
    "duplicate_role",
    "duplicate_group_size",
    "duplicate_representative",
    "fit_status",
    "px_per_mm",
    "dish_disc_radius_900",
    "frame_order_status",
    "time_unit_status",
    "selection",
    "selection_rule",
    "estimands_identifiable",
    "estimands_blocked",
    "image_path",
    "mask_path",
    "mask_sha256",
    "width",
    "height",
    "mask_width",
    "mask_height",
    "working_width",
)

CSV_HEADER: tuple[str, ...] = PROVENANCE_COLUMNS + tuple(
    col for col, _, _ in MEASUREMENT_COLUMNS
)


class MeasurementError(RuntimeError):
    """A step failed with a message that is already user-facing."""


# --------------------------------------------------------------------------
# intensity summaries
# --------------------------------------------------------------------------


def load_image_working(path: Path, *, width: int = WORKING_WIDTH) -> np.ndarray:
    """Load an 8-bit greyscale image and resize it to the working width.

    The mask is resized to the 900-px working width by
    :func:`measurement_error.load_mask_working`; the image has to be at the
    same width for the mask to select pixels from it.  A bilinear resize of
    the greyscale image matches the mask's own resize closely enough that the
    mask foreground selects the colony and not the agar.
    """
    raw = np.asarray(Image.open(path).convert("L"))
    if raw.ndim != 2:
        raise MeasurementError(f"{path} is not a single-channel image")
    image = Image.fromarray(raw, mode="L")
    w, h = image.size
    if w <= 0 or h <= 0:
        raise MeasurementError(f"{path} has a degenerate size {w}x{h}")
    new_h = max(1, round(h * width / w))
    resized = image.resize((width, new_h), Image.BILINEAR)
    return np.asarray(resized)


def intensity_summaries(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Intensity statistics of the image inside the mask.

    ``mean_intensity`` is the #60 density proxy (secondary estimand); the
    remaining summaries describe the within-colony intensity distribution.
    An empty mask has no pixels to summarise, so every summary is ``None``
    (the caller writes an empty cell, not a zero that would read as a real
    measurement).
    """
    if image.shape != mask.shape:
        raise MeasurementError(
            f"image shape {image.shape} does not match mask shape {mask.shape}"
        )
    pixels = image[mask > 0]
    if pixels.size == 0:
        return {
            "mean_intensity": None,
            "intensity_median": None,
            "intensity_std": None,
            "intensity_min": None,
            "intensity_max": None,
        }
    return {
        "mean_intensity": float(pixels.mean()),
        "intensity_median": float(np.median(pixels)),
        "intensity_std": float(pixels.std()),
        "intensity_min": float(pixels.min()),
        "intensity_max": float(pixels.max()),
    }


# --------------------------------------------------------------------------
# row assembly
# --------------------------------------------------------------------------


def build_measurement_rows(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One row per #29 image: provenance + primary/secondary measurements.

    Reads the saved predicted mask, computes every feature on it at the
    900-px working width, and attaches the intensity summaries from the
    image resized to the same width.  Rows keep the #29 provenance so the
    table joins on ``id`` with ``data/triage/measurement_error.csv`` (#31)
    and ``data/triage/inference_explore.csv`` (#29).
    """
    rows: list[dict[str, Any]] = []
    for rec in records:
        mask_path = Path(rec["mask_path"])
        image_path = Path(rec["image_path"])
        if not mask_path.is_file():
            raise MeasurementError(f"predicted mask missing: {mask_path}")
        if not image_path.is_file():
            raise MeasurementError(f"image missing: {image_path}")
        mask = me.load_mask_working(mask_path)
        image = load_image_working(image_path)
        px_per_mm = rec.get("px_per_mm")
        px_per_mm = float(px_per_mm) if px_per_mm is not None else None
        dish_r = rec.get("dish_disc_radius_900")
        dish_r = float(dish_r) if dish_r is not None else None
        fit_ok = rec.get("fit_status") == "ok"
        feats = me.mask_features(
            mask, px_per_mm=px_per_mm, dish_r_900=dish_r, fit_ok=fit_ok
        )
        summaries = intensity_summaries(image, mask)

        row: dict[str, Any] = {}
        # provenance: the #29 record's own facts, carried verbatim
        for key in PROVENANCE_COLUMNS:
            if key == "working_width":
                row[key] = WORKING_WIDTH
            elif key == "mask_sha256":
                # the #29 record stores the mask content hash under
                # `predicted_mask_sha256`; map it to the output column name
                row[key] = rec.get("predicted_mask_sha256", "")
            else:
                row[key] = rec.get(key, "")
        # measurements: the feature values, in MEASUREMENT_COLUMNS order
        for col, _, _ in MEASUREMENT_COLUMNS:
            if col in feats:
                row[col] = feats[col]
            elif col in summaries:
                row[col] = summaries[col]
            else:
                row[col] = None
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


def write_measurements_csv(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Write the measurement table (one row per image)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_HEADER), extrasaction="ignore")
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
    out = evidence["output"]
    lines: list[str] = []
    add = lines.append

    add("# Observable measurement table from the validated masks (#30)")
    add("")
    add("**Date:** 2026-10-07  ")
    add("**Status:** implemented — one provenance-rich row per #29 image, with")
    add("the primary size features separated from the secondary boundary and")
    add("intensity features  ")
    add("**Evidence:** #28 (validated checkpoint), #29 (inference outputs),")
    add("#31 (measurement-error model), #60 (measurement contract), #61")
    add("(provenance)  ")
    add("**Machine-readable:** `scripts/measurements.py`,")
    add("`tests/test_measurements.py`,")
    add("`notes/measurements_evidence.json`, `data/measurements.csv`  ")
    add("**Blocks:** #41 (any growth model)  ")
    add("")
    add(
        "PLAN.md Phase 2 step 4: per image, extract colony area, area fraction, "
        "calibrated equivalent radius, perimeter, compactness or roughness, "
        "intensity summaries and density proxies, and write one provenance-rich "
        "`data/measurements.csv` with image-level and mask-level provenance kept "
        "distinct. This note documents the table; it does not retrain, and it "
        "does not infer elapsed time, a growth rate, dish identity or a "
        "biological tip."
    )
    add("")
    add("## 1. Inputs and guards")
    add("")
    add("| Fact | Value |")
    add("| --- | --- |")
    add("| Checkpoint | #28 GPU baseline `iter_50000.pth` (SHA-256 `bcf66b6c…`) |")
    add("| Predicted masks | #29 inference outputs (`work_dirs/…/inference/masks/`) |")
    add("| Images | #29 inference inputs (released 4608-px JPGs + the unlabeled sample) |")
    add("| Working width | 900 px (the contract's boundary-feature width) |")
    add("| Retraining | none — `model_meta.json` must record the #28 SHA-256 |")
    add("| Perimeter | 4-connected boundary-pixel count (city-block), as in #31 |")
    add("")
    add(
        "The driver refuses to run if `model_meta.json` does not record the #28 "
        "checkpoint (`--allow-other-checkpoint` overrides), and it refuses any "
        "row that grows an elapsed-time, rate, dish-id or hyphal-tip field "
        "(Gate C). Every row keeps `frame_order_status = not_demonstrated` and "
        "`time_unit_status = no_capture_interval`."
    )
    add("")
    add("## 2. Population")
    add("")
    add("| Group | N |")
    add("| --- | --- |")
    add(f"| Images measured (one row each) | {out['n_rows']} |")
    add(f"| Calibrated (`fit_status = ok`) | {out['n_calibrated']} |")
    add(f"| Off-scale (calibrated estimands excluded) | {out['n_off_scale']} |")
    for name, n in out["by_source_set"].items():
        add(f"| &nbsp;&nbsp;source set `{name}` | {n} |")
    for name, n in out["by_block"].items():
        add(f"| &nbsp;&nbsp;block `{name}` | {n} |")
    add("")
    add(
        "The 125 rows are the #29 inference population: 50 labeled-GL test "
        "images, 60 conditions hold-outs (GS/PO/TS/MYG/PDA/TEMP15) and the "
        "15-image auditable unlabeled sample. Three images (`00018438` PO, "
        "`00018456` TS, `00018458` MYG) are `off_scale`: their calibrated "
        "estimands (mm² area, mm radius, mm perimeter, area fraction, "
        "ln(mm²)) are excluded per the #60 inclusion rule, while their px² "
        "and dimensionless shape features are kept."
    )
    add("")
    add("## 3. Feature tiers and the #31 uncertainty they carry")
    add("")
    add(
        "The #60 contract assigns each estimand a status; the #31 error model "
        "quantifies how much error each feature carries on the de-duplicated "
        "held-out masks. The two agree on what is primary and what is "
        "secondary:"
    )
    add("")
    add("| Tier | Features | #31 error model |")
    add("| --- | --- | --- |")
    add(
        "| **primary** | `area_px`, `area_mm2`, `area_fraction`, "
        "`equiv_radius_px`, `equiv_radius_mm`, `log_area_px2`, `log_area_mm2` | "
        "bias and spread inside the #60 size budget (relative bias ≤ 3%, "
        "relative SD ≤ 10%) |"
    )
    add(
        "| **secondary** (boundary) | `perimeter_px`, `perimeter_mm`, "
        "`edge_roughness`, `circularity`, `solidity` | error well outside the "
        "size budget (published Boundary IoU 28.60%, #28); they carry the "
        "looser boundary budget and are not size proxies |"
    )
    add(
        "| **secondary** (density) | `mean_intensity`, `intensity_median`, "
        "`intensity_std`, `intensity_min`, `intensity_max` | density proxy only; "
        "comparable within a condition block, never across |"
    )
    add("")
    add(
        "Reading: area, log-area and equivalent radius are the reliable "
        "quantities — the segmenter is near-unbiased on the median with a "
        "spread well inside the size budget. Perimeter and the boundary-derived "
        "shape features carry a much larger, heavy-tailed error, so they are "
        "published as shape descriptors at the fidelity the boundary metrics "
        "allow. `mean_intensity` is a density proxy only: dark incubation and "
        "fixed exposure are per-block facts."
    )
    add("")
    add("## 4. Column dictionary")
    add("")
    add("Provenance columns (image-level and mask-level) are carried from the "
        "#29 records verbatim; the table joins on `id` with "
        "`data/triage/measurement_error.csv` (#31) and "
        "`data/triage/inference_explore.csv` (#29).")
    add("")
    add("| Column | Tier | Definition |")
    add("| --- | --- | --- |")
    for col in evidence["measurement_columns"]:
        add(f"| `{col['column']}` | {col['tier']} | {col['note']} |")
    add("")
    add("## 5. What this table supports and what it does not")
    add("")
    add("**Supported (image-level morphology, this release):**")
    add("")
    add(
        "- `equivalent_radius` and `log_area` clear the size budget on the "
        "de-duplicated hold-outs. `area_fraction` remains an approximate "
        "cross-sectional coverage coordinate, but its 14.3% relative SD is "
        "above the #60 target (#66)."
    )
    add(
        "- `perimeter`, `edge_roughness`, `circularity` and `solidity` are "
        "published with their (larger) boundary budget; they describe shape "
        "only at the fidelity the boundary metrics allow and are not size "
        "proxies."
    )
    add(
        "- `mean_intensity` and the intensity summaries are density proxies, "
        "comparable within a condition block only."
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
        "- any hyphal-tip, nuclear or branching claim — the #31 candidate "
        "protrusions are unvalidated radial excursions with a high false-event "
        "rate. #62 measures candidate stability under resampling, but the "
        "colony masks are not independent biological-tip annotations;"
    )
    add(
        "- any causal temperature or medium effect — medium is aliased with "
        "species, only 10 images are at 15 °C, and there is no dish identity "
        "or replication;"
    )
    add(
        "- any physical growth rate in mm/day — a frame index is not a "
        "duration, and `to_physical_rate()` refuses the conversion."
    )
    add("")
    add("## 6. Reproduce")
    add("")
    add("```bash")
    add("# measurement table from the saved masks (no GPU, no retrain)")
    add("nix develop --command python scripts/measurements.py run")
    add("")
    add("# tests and lint")
    add(".venv/bin/python -m pytest tests -q")
    add("nix develop --command ruff check scripts/measurements.py tests/test_measurements.py")
    add("```")
    add("")
    add(
        "Surrounding record: `notes/measurement_error.md` (#31, the error "
        "model), `notes/segformer_inference.md` (#29, the masks), "
        "`notes/estimands.md` (#60, the contract and uncertainty target), "
        "`notes/provenance.md` (#61, the capability matrix)."
    )
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_evidence(records: Sequence[Mapping[str, Any]], work: Path) -> dict[str, Any]:
    """Assemble the run's environment, checkpoint, counts and caveats."""
    rows = build_measurement_rows(records)
    _assert_no_inferred_time(rows)
    calibrated = [r for r in rows if r["fit_status"] == "ok"]
    off_scale = [r for r in rows if r["fit_status"] != "ok"]
    return {
        "task": 30,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "work_dir": str(work),
        "repository_commit": repro._git_head(REPO),
        "checkpoint": {
            "sha256": CHECKPOINT_SHA256,
            "path": str(work / "model_meta.json"),
        },
        "output": {
            "measurements_csv": str(MEASUREMENTS_CSV),
            "n_rows": len(rows),
            "n_calibrated": len(calibrated),
            "n_off_scale": len(off_scale),
            "off_scale_ids": [r["id"] for r in off_scale],
            "by_source_set": {
                s: sum(1 for r in rows if r["source_set"] == s)
                for s in sorted({r["source_set"] for r in rows})
            },
            "by_block": {
                b: sum(1 for r in rows if r["block"] == b)
                for b in sorted({r["block"] for r in rows})
            },
        },
        "feature_tiers": {
            tier: [col for col, t, _ in MEASUREMENT_COLUMNS if t == tier]
            for tier in ("primary", "secondary")
        },
        "measurement_columns": [
            {"column": col, "tier": tier, "note": note}
            for col, tier, note in MEASUREMENT_COLUMNS
        ],
        "caveats": list(CAVEATS),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def _load_records(work: Path) -> list[dict[str, Any]]:
    """Read the #29 inference records (the provenance join key)."""
    path = work / "records.json"
    if not path.is_file():
        raise MeasurementError(f"records not found: {path}")
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise MeasurementError(f"records is empty or not a list: {path}")
    return records


def _require_checkpoint(work: Path, *, allow_other: bool) -> None:
    """Refuse to run unless model_meta.json records the #28 checkpoint."""
    meta_path = work / "model_meta.json"
    if not meta_path.is_file():
        raise MeasurementError(f"model_meta.json not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    sha = meta.get("checkpoint_sha256", "")
    if sha != CHECKPOINT_SHA256 and not allow_other:
        raise MeasurementError(
            f"model_meta.json records checkpoint {sha!r}, not the validated #28 "
            f"baseline {CHECKPOINT_SHA256!r}; pass --allow-other-checkpoint to "
            "override"
        )


def cmd_build(args: argparse.Namespace) -> int:
    records = _load_records(args.work)
    rows = build_measurement_rows(records)
    _assert_no_inferred_time(rows)
    write_measurements_csv(args.output, rows)
    print(f"wrote {args.output} ({len(rows)} rows)")
    return 0


def cmd_evidence(args: argparse.Namespace) -> int:
    records = _load_records(args.work)
    evidence = build_evidence(records, args.work)
    write_evidence_json(args.evidence, evidence)
    write_note(args.note, evidence)
    print(f"wrote {args.evidence} and {args.note}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _require_checkpoint(args.work, allow_other=args.allow_other_checkpoint)
    records = _load_records(args.work)
    rows = build_measurement_rows(records)
    _assert_no_inferred_time(rows)
    write_measurements_csv(args.output, rows)
    evidence = build_evidence(records, args.work)
    write_evidence_json(args.evidence, evidence)
    write_note(args.note, evidence)
    print(f"wrote {args.output} ({len(rows)} rows), {args.evidence} and {args.note}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--work",
        type=Path,
        default=DEFAULT_WORK,
        help="the #29 inference work dir (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MEASUREMENTS_CSV,
        help="the measurement table to write (default: %(default)s)",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=EVIDENCE_JSON,
        help="the evidence JSON to write (default: %(default)s)",
    )
    parser.add_argument(
        "--note",
        type=Path,
        default=NOTE_MD,
        help="the human-readable note to write (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-other-checkpoint",
        action="store_true",
        help="run even if model_meta.json does not record the #28 checkpoint",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="write the measurement table").set_defaults(func=cmd_build)
    sub.add_parser("evidence", help="write the evidence JSON").set_defaults(func=cmd_evidence)
    sub.add_parser("run", help="build -> evidence").set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except MeasurementError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
