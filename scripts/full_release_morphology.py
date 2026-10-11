#!/usr/bin/env python3
"""Extract shape features from the full-release predicted masks.

This is an image-level feature table for Chainlink #41 and #65. It does not
read or infer chronology; area-rank coordinates are added by the modeling
step and remain synthetic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import measurement_error as me

DEFAULT_ROOT = REPO / "work_dirs/segformer-b0-repro/full-release-all-labeled-20261009"
SOURCE_NAME = "area_fraction_unique_sorted.csv"
JSONL_NAME = "morphology_features.jsonl"
CSV_NAME = "morphology_features.csv"
STATE_NAME = "morphology_features_state.json"
FLUSH_EVERY = 100

OUTPUT_FIELDS = (
    "record_key", "id", "archive", "entry", "source_set", "selection",
    "selection_rule", "md5", "species", "medium", "temp", "condition",
    "condition_known", "condition_source", "block", "scale_group",
    "fit_status", "area_fraction_qc", "area_exclusion",
    "predicted_area_fraction", "predicted_area_px", "prediction_status",
    "prediction_error", "prediction_fallback", "duplicate_role",
    "duplicate_group_size", "duplicate_representative", "label_conflict",
    "frame_order_status", "time_unit_status", "width", "height",
    "prediction_width", "prediction_height", "mask_path",
    "predicted_mask_sha256", "mask_area_px", "mask_perimeter_px",
    "mask_equiv_radius_px", "log_mask_area_px2", "edge_roughness",
    "circularity", "solidity", "bbox_fill", "bbox_aspect_ratio",
    "eccentricity", "empty_mask",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_key(row: dict[str, str]) -> str:
    return f"{row['archive']}::{row['entry']}"


def bbox_features(mask: np.ndarray) -> dict[str, float | None]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"bbox_fill": None, "bbox_aspect_ratio": None, "eccentricity": None}
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    if len(xs) < 2:
        eccentricity = 0.0
    else:
        coords = np.column_stack((xs, ys)).astype(np.float64)
        eigenvalues = np.linalg.eigvalsh(np.cov(coords, rowvar=False))
        major = max(float(eigenvalues[-1]), 0.0)
        minor = max(float(eigenvalues[0]), 0.0)
        eccentricity = math.sqrt(max(0.0, 1.0 - minor / major)) if major > 0 else 0.0
    return {
        "bbox_fill": float(len(xs) / (width * height)),
        "bbox_aspect_ratio": float(width / height),
        "eccentricity": eccentricity,
    }


def extract_record(row: dict[str, str], root: Path) -> dict[str, Any]:
    mask_path = root / row["mask_path"]
    if not mask_path.is_file():
        raise FileNotFoundError(f"predicted mask missing: {mask_path}")
    mask = me.load_mask_working(mask_path)
    base = me.mask_features(mask, px_per_mm=None, dish_r_900=None, fit_ok=False)
    result: dict[str, Any] = {
        "record_key": row_key(row),
        **{key: row.get(key, "") for key in OUTPUT_FIELDS if key in row},
        "mask_area_px": base["area_px"],
        "mask_perimeter_px": base["perimeter_px"],
        "mask_equiv_radius_px": base.get("equiv_radius_px"),
        "log_mask_area_px2": base.get("log_area_px2"),
        "edge_roughness": base.get("edge_roughness"),
        "circularity": base.get("circularity"),
        "solidity": base.get("solidity"),
        "empty_mask": bool(base.get("empty", 0.0)),
    }
    result.update(bbox_features(mask))
    return result


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    good_lines: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    contents = path.read_text()
    lines = contents.splitlines()
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        good_lines.append(line)
        rows[row["record_key"]] = row
    if len(good_lines) != len(lines) or (contents and not contents.endswith("\n")):
        path.write_text("".join(line + "\n" for line in good_lines))
    return rows


def append_jsonl(row: dict[str, Any], handle: Any) -> None:
    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])
    tmp.replace(path)


def run(root: Path) -> None:
    source = root / SOURCE_NAME
    cache_path = root / JSONL_NAME
    state_path = root / STATE_NAME
    if not source.is_file():
        raise FileNotFoundError(f"sorted input missing: {source}")
    inference_state = json.loads((root / "state.json").read_text())
    if inference_state.get("status") != "complete":
        raise RuntimeError("full-release inference state is not complete")
    with source.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    expected = {row_key(row) for row in source_rows}
    source_by_key = {row_key(row): row for row in source_rows}
    if len(expected) != len(source_rows):
        raise RuntimeError("deduplicated input contains repeated archive/member keys")
    completed = load_cache(cache_path)
    extra = set(completed) - expected
    if extra:
        raise RuntimeError(f"feature cache has {len(extra)} rows absent from source table")
    pending = [row for row in source_rows if row_key(row) not in completed]
    print(f"morphology features: {len(completed)}/{len(source_rows)} cached; {len(pending)} to process")
    with cache_path.open("a") as handle:
        for index, row in enumerate(pending, start=1):
            features = extract_record(row, root)
            completed[features["record_key"]] = features
            append_jsonl(features, handle)
            if index % FLUSH_EVERY == 0 or index == len(pending):
                handle.flush()
                os.fsync(handle.fileno())
                print(f"morphology features: {len(completed)}/{len(source_rows)}")
    for key, features in completed.items():
        source_row = source_by_key[key]
        for field in OUTPUT_FIELDS:
            if field in source_row:
                features[field] = source_row[field]
    if set(completed) != expected:
        raise RuntimeError("feature extraction did not cover every deduplicated record")
    write_csv(root / CSV_NAME, completed)
    state = {
        "status": "complete",
        "source_csv": SOURCE_NAME,
        "source_csv_sha256": sha256_file(source),
        "segmentation_checkpoint_sha256": inference_state.get("checkpoint_sha256", ""),
        "rows": len(completed),
        "mask_feature_width": me.WORKING_WIDTH,
        "perimeter_definition": "4-connected foreground boundary-pixel count",
        "features": list(OUTPUT_FIELDS),
        "limitations": [
            "Shape features are image-level measurements from predicted masks.",
            "Boundary features inherit the segmenter's boundary uncertainty.",
            "No image order, specimen identity, or capture time is inferred.",
            "Area-rank pseudo-time is added only in the downstream exploratory model.",
        ],
    }
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(f"wrote {root / CSV_NAME}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=("run",))
    parser.add_argument("--work", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    run(args.work.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
