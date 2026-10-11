#!/usr/bin/env python3
"""Held-out check of a movie-relative onset → extension → plateau form for #37.

This evaluates a candidate shape on the image-derived Podospora ridge-occupancy
proxy. Its onset is relative to each AVI's first frame, not biological germination.
The square-root area is only an equivalent size coordinate; the script does not
measure literal hyphal-front displacement or a biological growth lag.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/derived/podospora_growth_model/frame_measurements.csv"
DEFAULT_OUTPUT = ROOT / "data/derived/growth_form_invariance/podospora_triphasic_cv.json"
STEP_HOURS = 0.3
WINDOW_HOURS = 20.0
BASELINE_FRAMES = 5
TRAIN_FRACTION = 0.70


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def robust_baseline(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return center, center + max(3.0 * 1.4826 * mad, 50.0)


def line_radius_fit(t: np.ndarray, radius: np.ndarray, cap_time: float | None) -> tuple[float, float]:
    x = t if cap_time is None else np.minimum(t, cap_time)
    intercept, slope = fit_line(x, radius)
    if slope < 0.0:
        return max(float(radius.mean()), 1.0), 0.0
    return max(intercept, 1.0), slope


def best_plateau_fit(t: np.ndarray, radius: np.ndarray, full_window_max_h: float) -> tuple[float, float, float]:
    """Train a radial-linear curve with a fixed plateau time, selected by train SSE."""
    train_end = float(t.max())
    # Include plateau times through the held-out horizon. Times after train_end
    # have identical in-sample predictions; the first minimum favors the
    # conservative earliest plateau and exposes that extrapolation in holdout.
    grid = np.linspace(0.0, full_window_max_h, 401)
    best: tuple[float, float, float, float] | None = None
    for cap_time in grid:
        intercept, slope = line_radius_fit(t, radius, float(cap_time))
        prediction = intercept + slope * np.minimum(t, cap_time)
        sse = float(np.sum((radius - prediction) ** 2))
        candidate = (sse, float(cap_time), intercept, slope)
        if best is None or candidate[0] < best[0] - 1e-12:
            best = candidate
    assert best is not None
    return best[1], best[2], best[3]


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def analyze_movie(movie: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    rows.sort(key=lambda row: int(row["frame_index"]))
    common = [row for row in rows if float(row["time_from_first_h"]) <= WINDOW_HOURS]
    area = np.asarray([float(row["mask_area_px_resampled"]) for row in common])
    baseline, cutoff = robust_baseline(area[:BASELINE_FRAMES])
    above = area > cutoff
    onset = next(
        (index for index in range(max(0, len(above) - 2)) if bool(np.all(above[index:index + 3]))),
        None,
    )
    if onset is None:
        return {"movie": movie, "status": "no detected onset"}

    # Post-onset signal and time are the same primary q=.99 measurement used in #47.
    active = common[onset:]
    t_all = np.asarray([float(row["time_from_first_h"]) - onset * STEP_HOURS for row in active])
    signal = np.asarray([max(float(row["mask_area_px_resampled"]) - baseline, 1.0) for row in active])
    log_area = np.log(signal)
    radius = np.sqrt(signal)
    split = max(4, int(len(active) * TRAIN_FRACTION))
    if len(active) - split < 2:
        return {"movie": movie, "status": "too few holdout frames"}
    train_t, test_t = t_all[:split], t_all[split:]
    train_log, test_log = log_area[:split], log_area[split:]
    train_radius = radius[:split]

    # Exponential and power-law occupancy baselines are included from #47.
    exp_a, exp_b = fit_line(train_t, train_log)
    exp_pred = exp_a + exp_b * test_t
    power_x = np.log(train_t + STEP_HOURS)
    power_a, power_b = fit_line(power_x, train_log)
    power_pred = power_a + power_b * np.log(test_t + STEP_HOURS)

    # Constant extension is linear growth of equivalent radius; the triphasic
    # model uses the threshold-derived onset as its movie-relative lag, then a
    # linear radius trajectory capped at an estimated plateau time.
    line_a, line_b = line_radius_fit(train_t, train_radius, None)
    line_pred = np.log(np.maximum(line_a + line_b * test_t, 1.0) ** 2)
    plateau_h, plateau_a, plateau_b = best_plateau_fit(train_t, train_radius, float(t_all.max()))
    plateau_radius = plateau_a + plateau_b * np.minimum(test_t, plateau_h)
    plateau_pred = np.log(np.maximum(plateau_radius, 1.0) ** 2)

    return {
        "movie": movie,
        "condition": movie[:2],
        "status": "ok",
        "frames_in_shared_window": len(common),
        "onset_frame": onset,
        "observed_signal_onset_after_first_frame_h": round(onset * STEP_HOURS, 3),
        "active_frames": len(active),
        "train_frames": split,
        "holdout_frames": len(active) - split,
        "holdout_rmse_log_signal": {
            "exponential_occupancy": rmse(test_log, exp_pred),
            "power_law_occupancy": rmse(test_log, power_pred),
            "constant_equivalent_radius_extension": rmse(test_log, line_pred),
            "onset_extension_plateau": rmse(test_log, plateau_pred),
        },
        "onset_extension_plateau": {
            "plateau_time_after_onset_h": round(plateau_h, 3),
            "radius_intercept_px_equivalent": round(plateau_a, 3),
            "radius_slope_px_equivalent_per_h": round(plateau_b, 3),
            "plateau_selected_before_training_end": bool(plateau_h <= float(train_t.max()) + 1e-9),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.input.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in source_rows:
        grouped.setdefault(row["movie"], []).append(row)
    movie_results = [analyze_movie(movie, rows) for movie, rows in sorted(grouped.items())]
    successful = [row for row in movie_results if row["status"] == "ok"]
    model_names = list(successful[0]["holdout_rmse_log_signal"]) if successful else []
    aggregate = {
        name: {
            "n_movies": len(successful),
            "mean_holdout_rmse_log_signal": float(np.mean([
                row["holdout_rmse_log_signal"][name] for row in successful
            ])),
            "movies_best_or_tied": sum(
                row["holdout_rmse_log_signal"][name]
                <= min(row["holdout_rmse_log_signal"].values()) + 1e-12
                for row in successful
            ),
        }
        for name in model_names
    }
    payload = {
        "dataset": "Podospora M0/M2 six-movie dataset",
        "outcome": "q=.99 baseline-subtracted ridge-occupancy area; models scored on held-out log signal",
        "shared_window_h": WINDOW_HOURS,
        "time_origin": "first frame per movie; onset is a robust signal threshold and not biological germination",
        "holdout": "within each movie, final 30% of post-onset observations; earlier 70% used to fit",
        "triphasic_definition": "threshold-derived movie-relative onset, then linear growth in square-root area (equivalent-size proxy), capped at a fitted plateau time",
        "caveats": [
            "The equivalent-radius coordinate is derived from candidate ridge occupancy and is not literal hyphal-front displacement.",
            "Saturation time beyond the observed window is not identifiable; the holdout only tests plateau predictions within the observed tail.",
        ],
        "aggregate": aggregate,
        "movies": movie_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
