#!/usr/bin/env python3
"""Series-held-out check of lag → extension → saturation for Fricker (#37).

The response is baseline-relative equivalent radius from mycelial area. This is
a dimensionless image-derived coordinate, not calibrated front speed. The first
linked archive image is the time origin, not a verified germination/inoculation
time. Rates vary by species × grazer cell while candidate curve shape is shared.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/derived/fricker_grazers/network_series.csv"
OUTPUT = ROOT / "data/derived/growth_form_invariance/fricker_triphasic_cv.json"
SPECIES = {"Hf", "Pi", "Pv", "Rb"}
TREATMENTS = {"control", "coll", "mill", "nema", "wl1", "wl2"}


def number(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def trajectory_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    baseline: dict[str, float] = {}
    first: dict[str, dict[str, str]] = {}
    for row in source:
        treatment = "control" if row["treatment"].startswith("ctrl") else row["treatment"]
        area = number(row, "summary_mycelial_area")
        if (
            row["species"] in SPECIES and treatment in TREATMENTS
            and row["day_label"] == "1" and row["has_image"].lower() == "true"
            and area is not None and area > 0
        ):
            baseline[row["series_key"]] = area
            first[row["series_key"]] = row

    result: list[dict[str, Any]] = []
    for row in source:
        treatment = "control" if row["treatment"].startswith("ctrl") else row["treatment"]
        key = row["series_key"]
        area = number(row, "summary_mycelial_area")
        elapsed = number(row, "elapsed_days_from_first")
        if (
            row["species"] not in SPECIES or treatment not in TREATMENTS
            or key not in baseline or row["day_label"] not in {"1", "2", "4", "8"}
            or row["has_image"].lower() != "true"
            or row["same_timestamp_as_prior"].lower() == "true"
            or area is None or area <= 0 or elapsed is None
        ):
            continue
        item = {
            "series": key,
            "cell": f"{row['species']}|{treatment}",
            "elapsed_days": elapsed,
            "day_label": int(row["day_label"]),
            # Relative radius increment, dimensionless and anchored to D1 area.
            "y": math.sqrt(area / baseline[key]) - 1.0,
        }
        result.append(item)
    return result


def shape_values(t: np.ndarray, lag: float, saturation: float | None) -> np.ndarray:
    extension = np.maximum(t - lag, 0.0)
    if saturation is not None:
        extension = np.minimum(extension, max(saturation - lag, 0.0))
    return extension


def fit_group_slopes(
    rows: list[dict[str, Any]], shape: Callable[[np.ndarray], np.ndarray]
) -> tuple[dict[str, float], float]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[row["cell"]].append(row)
    global_t = np.asarray([row["elapsed_days"] for row in rows], dtype=float)
    global_y = np.asarray([row["y"] for row in rows], dtype=float)
    global_x = shape(global_t)
    denom = float(global_x @ global_x)
    global_slope = max(0.0, float(global_x @ global_y / denom)) if denom > 0 else 0.0
    slopes: dict[str, float] = {}
    for cell, cell_rows in by_cell.items():
        t = np.asarray([row["elapsed_days"] for row in cell_rows], dtype=float)
        y = np.asarray([row["y"] for row in cell_rows], dtype=float)
        x = shape(t)
        denom = float(x @ x)
        slopes[cell] = max(0.0, float(x @ y / denom)) if denom > 0 else global_slope
    return slopes, global_slope


def train_sse(rows: list[dict[str, Any]], lag: float, saturation: float | None) -> tuple[float, dict[str, float], float]:
    shape = lambda t: shape_values(t, lag, saturation)
    slopes, global_slope = fit_group_slopes(rows, shape)
    residual = np.asarray([
        row["y"] - slopes.get(row["cell"], global_slope) * float(shape(np.asarray([row["elapsed_days"]]))[0])
        for row in rows
    ])
    return float(residual @ residual), slopes, global_slope


def best_shape(rows: list[dict[str, Any]], model: str) -> tuple[float, float | None, dict[str, float], float]:
    if model == "linear_extension":
        sse, slopes, global_slope = train_sse(rows, 0.0, None)
        return 0.0, None, slopes, global_slope
    max_t = max(float(row["elapsed_days"]) for row in rows)
    lag_grid = np.arange(0.0, min(4.0, max_t) + 1e-9, 0.25)
    candidates: list[tuple[float, float, float | None, dict[str, float], float]] = []
    if model == "lag_extension":
        for lag in lag_grid:
            sse, slopes, global_slope = train_sse(rows, float(lag), None)
            candidates.append((sse, float(lag), None, slopes, global_slope))
    else:
        sat_grid = np.arange(0.5, max_t + 1e-9, 0.25)
        sat_grid = np.append(sat_grid, np.inf)
        for lag in lag_grid:
            for saturation in sat_grid:
                if saturation <= lag:
                    continue
                sat = None if math.isinf(float(saturation)) else float(saturation)
                sse, slopes, global_slope = train_sse(rows, float(lag), sat)
                candidates.append((sse, float(lag), sat, slopes, global_slope))
    best = min(candidates, key=lambda item: item[0])
    return best[1], best[2], best[3], best[4]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = trajectory_rows(args.input)
    y = np.asarray([row["y"] for row in rows], dtype=float)
    groups = np.asarray([row["series"] for row in rows])
    x_rows = np.arange(len(rows))
    splitter = GroupKFold(n_splits=5)
    model_names = ("no_growth", "linear_extension", "lag_extension", "lag_extension_plateau")
    observed_by_model: dict[str, list[float]] = {name: [] for name in model_names}
    predicted_by_model: dict[str, list[float]] = {name: [] for name in model_names}
    fold_results: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x_rows, y, groups), start=1):
        train_rows = [rows[int(i)] for i in train_idx]
        test_rows = [rows[int(i)] for i in test_idx]
        fold_item: dict[str, Any] = {
            "fold": fold,
            "train_series": len({row["series"] for row in train_rows}),
            "test_series": len({row["series"] for row in test_rows}),
            "models": {},
        }
        for model in model_names:
            if model == "no_growth":
                lag, saturation, slopes, global_slope = 0.0, None, {}, 0.0
            else:
                lag, saturation, slopes, global_slope = best_shape(train_rows, model)
            if model == "linear_extension":
                lag, saturation, slopes, global_slope = best_shape(train_rows, model)
            elif model == "no_growth":
                pass
            shape = lambda times, lag=lag, saturation=saturation: shape_values(times, lag, saturation)
            pred = np.asarray([
                slopes.get(row["cell"], global_slope) * float(shape(np.asarray([row["elapsed_days"]]))[0])
                for row in test_rows
            ], dtype=float)
            actual = np.asarray([row["y"] for row in test_rows], dtype=float)
            observed_by_model[model].extend(actual.tolist())
            predicted_by_model[model].extend(pred.tolist())
            fold_item["models"][model] = {
                "lag_days_since_first_image": lag,
                "saturation_day_since_first_image": saturation,
                "rmse_fractional_equivalent_radius": float(np.sqrt(np.mean((actual - pred) ** 2))),
                "mae_fractional_equivalent_radius": float(np.mean(np.abs(actual - pred))),
            }
        fold_results.append(fold_item)

    aggregate: dict[str, Any] = {}
    for model in model_names:
        actual = np.asarray(observed_by_model[model], dtype=float)
        predicted = np.asarray(predicted_by_model[model], dtype=float)
        aggregate[model] = {
            "n_predictions": len(actual),
            "grouped_cv_rmse_fractional_equivalent_radius": float(np.sqrt(np.mean((actual - predicted) ** 2))),
            "grouped_cv_mae_fractional_equivalent_radius": float(np.mean(np.abs(actual - predicted))),
        }

    payload = {
        "dataset": "Fricker/Boddy fungal grazer series",
        "outcome": "sqrt(mycelial area / D1 baseline area) - 1; dimensionless equivalent-radius change",
        "n_series": len({row["series"] for row in rows}),
        "n_observations_including_baseline": len(rows),
        "folds": 5,
        "validation": "GroupKFold by complete replicate series; shape shared across data, extension rate varies by species × grazer cell",
        "candidate_forms": {
            "no_growth": "zero change from the D1 baseline",
            "linear_extension": "constant equivalent-radius extension from D1; no lag or saturation",
            "lag_extension": "shared fitted lag since first linked image, then constant extension; no saturation",
            "lag_extension_plateau": "shared fitted lag, constant extension, then a plateau; saturation can be beyond observed horizon",
        },
        "limitations": [
            "D1 is an archive-relative origin; due the documented paper/archive interval discrepancy it is not asserted to be time since inoculation or biological growth onset.",
            "The series end at the D8 label, so a plateau later than the observed window cannot be identified.",
            "Equivalent radius is a mathematical transform of workbook area, not a calibrated radial-front measurement; these wood/soil colonies are not assumed circular.",
            "Treatment allocation is undocumented; this is a predictive shape check, not causal treatment inference.",
        ],
        "aggregate": aggregate,
        "folds_detail": fold_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
