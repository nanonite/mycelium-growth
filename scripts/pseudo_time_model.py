#!/usr/bin/env python3
"""Prototype a synthetic area-rank sequence model from predicted masks (#65)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO / "work_dirs/segformer-b0-repro/full-release-all-labeled-20261009"
INPUT_NAME = "morphology_features.csv"
TRAJECTORY_NAME = "pseudo_time_trajectory.csv"
FORECAST_NAME = "pseudo_time_forecasts.csv"
SUMMARY_NAME = "pseudo_time_model_summary.csv"
EVIDENCE_NAME = "pseudo_time_model_evidence.json"
NOTE_NAME = "pseudo_time_model_note.md"
MODEL_NAME = "pseudo_time_ar3_surrogate.joblib"
TARGETS = ("circularity", "solidity", "edge_roughness", "bbox_fill", "eccentricity")
MAIN_CONDITION = "GL/MYG/25C"
N_BINS = 100
TRAIN_FRACTION = 0.70
N_LAGS = 3


def numeric(row: dict[str, str], name: str) -> float | None:
    value = row.get(name, "")
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if row.get("area_fraction_qc") == "ok"
        and row.get("empty_mask", "").lower() == "false"
        and row.get("source_set") == "unlabeled"
        and row.get("species") == "GL"
        and row.get("condition") == MAIN_CONDITION
        and numeric(row, "predicted_area_fraction") is not None
    ]
    return sorted(selected, key=lambda row: (float(row["predicted_area_fraction"]), row["record_key"]))


def make_trajectory(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if len(rows) < N_BINS * 5:
        raise RuntimeError(f"not enough images for stable quantile bins: {len(rows)}")
    indices = np.array_split(np.arange(len(rows)), N_BINS)
    trajectory: list[dict[str, Any]] = []
    for step, group in enumerate(indices):
        group_rows = [rows[index] for index in group]
        record: dict[str, Any] = {
            "pseudo_step": step,
            "t_star": (step + 0.5) / N_BINS,
            "coverage_fraction_median": float(np.median([
                float(row["predicted_area_fraction"]) for row in group_rows
            ])),
            "image_count": len(group_rows),
        }
        for target in TARGETS:
            values = [numeric(row, target) for row in group_rows]
            finite = [value for value in values if value is not None]
            record[f"{target}_median"] = float(np.median(finite)) if finite else None
        trajectory.append(record)
    for index, row in enumerate(trajectory):
        for target in TARGETS:
            name = f"{target}_median"
            current = row[name]
            previous = trajectory[index - 1][name] if index > 0 else None
            row[f"delta_{target}"] = (
                float(current - previous)
                if current is not None and previous is not None else None
            )
    return trajectory


def one_step_features(trajectory: list[dict[str, Any]], target: str, endpoint: int) -> list[float]:
    return [
        float(trajectory[endpoint]["t_star"]),
        *[
            float(trajectory[endpoint - lag][f"{target}_median"])
            for lag in range(1, N_LAGS + 1)
        ],
    ]


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def fit_target(trajectory: list[dict[str, Any]], target: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = np.asarray([float(row[f"{target}_median"]) for row in trajectory], dtype=float)
    t = np.asarray([float(row["t_star"]) for row in trajectory], dtype=float).reshape(-1, 1)
    p = np.asarray([float(row["coverage_fraction_median"]) for row in trajectory], dtype=float)
    split = int(len(y) * TRAIN_FRACTION)
    train_endpoints = np.arange(N_LAGS + 1, split)
    test_endpoints = np.arange(split, len(y))
    if len(train_endpoints) < 20 or len(test_endpoints) < 10:
        raise RuntimeError("pseudo-time split is too small for the planned comparison")
    ar_x = np.asarray([one_step_features(trajectory, target, j) for j in train_endpoints])
    ar_y = y[train_endpoints] - y[train_endpoints - 1]
    ar_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ar_model.fit(ar_x, ar_y)
    trend_model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), Ridge(alpha=1.0))
    trend_model.fit(t[:split], y[:split])
    predictions: list[dict[str, Any]] = []
    actual_test = y[test_endpoints]
    actual_delta = actual_test - y[test_endpoints - 1]
    pred_persist = y[test_endpoints - 1]
    pred_trend = trend_model.predict(t[test_endpoints])
    # Multi-step baselines: repeat the last observed training value for
    # persistence, and evaluate the coverage-only curve directly at each
    # future pseudo-step (it has no feedback state to recurse).
    pred_persist_recursive = np.full(len(test_endpoints), y[split - 1], dtype=float)
    pred_trend_recursive = pred_trend.copy()
    test_x = np.asarray([one_step_features(trajectory, target, j) for j in test_endpoints])
    pred_ar_delta = ar_model.predict(test_x)
    pred_ar = pred_persist + pred_ar_delta
    recursive_history = list(y[:split])
    pred_recursive: list[float] = []
    for endpoint in test_endpoints:
        recursive_x = [[
            float(t[endpoint, 0]),
            *[float(recursive_history[-lag]) for lag in range(1, N_LAGS + 1)],
        ]]
        predicted_delta = float(ar_model.predict(np.asarray(recursive_x))[0])
        predicted_value = recursive_history[-1] + predicted_delta
        pred_recursive.append(predicted_value)
        recursive_history.append(predicted_value)
    model_predictions = {
        "persistence": pred_persist,
        "pseudo_time_cubic_trend": pred_trend,
        "pseudo_time_ar3": pred_ar,
        "persistence_recursive": pred_persist_recursive,
        "pseudo_time_cubic_trend_recursive": pred_trend_recursive,
        "pseudo_time_ar3_recursive": np.asarray(pred_recursive),
    }
    summary: dict[str, Any] = {
        "target": target,
        "n_bins": len(y),
        "train_bins": len(train_endpoints),
        "test_bins": len(test_endpoints),
        "train_end_t_star": float(t[split - 1, 0]),
        "test_start_coverage_fraction": float(p[split]),
        "forecast_type": "rolling one-step and recursive holdout on synthetic pseudo-bins",
        "modeled_quantity": "change in median mask feature across one synthetic coverage step",
    }
    for name, prediction in model_predictions.items():
        measured = error_metrics(actual_test, prediction)
        summary[f"{name}_rmse"] = measured["rmse"]
        summary[f"{name}_mae"] = measured["mae"]
    delta_baseline_rmse = error_metrics(actual_delta, np.zeros_like(actual_delta))["rmse"]
    delta_model_rmse = error_metrics(actual_delta, pred_ar_delta)["rmse"]
    summary["zero_change_delta_rmse"] = delta_baseline_rmse
    summary["ar3_delta_rmse"] = delta_model_rmse
    summary["ar3_delta_improvement_vs_zero"] = (
        float(1.0 - delta_model_rmse / delta_baseline_rmse) if delta_baseline_rmse else 0.0
    )
    summary["ar_rmse_improvement_vs_persistence"] = float(
        1.0 - summary["pseudo_time_ar3_rmse"] / summary["persistence_rmse"]
    ) if summary["persistence_rmse"] else 0.0
    summary["ar_rmse_improvement_vs_trend"] = float(
        1.0 - summary["pseudo_time_ar3_rmse"] / summary["pseudo_time_cubic_trend_rmse"]
    ) if summary["pseudo_time_cubic_trend_rmse"] else 0.0
    summary["recursive_ar3_rmse_improvement_vs_persistence"] = float(
        1.0 - summary["pseudo_time_ar3_recursive_rmse"] / summary["persistence_recursive_rmse"]
    ) if summary["persistence_recursive_rmse"] else 0.0
    summary["recursive_ar3_rmse_improvement_vs_trend"] = float(
        1.0 - summary["pseudo_time_ar3_recursive_rmse"]
        / summary["pseudo_time_cubic_trend_recursive_rmse"]
    ) if summary["pseudo_time_cubic_trend_recursive_rmse"] else 0.0
    for position, endpoint in enumerate(test_endpoints):
        record: dict[str, Any] = {
            "target": target,
            "pseudo_step": int(endpoint),
            "t_star": float(t[endpoint, 0]),
            "coverage_fraction_median": float(p[endpoint]),
            "observed": float(actual_test[position]),
            "observed_delta": float(actual_delta[position]),
            "predicted_delta_zero_baseline": 0.0,
            "predicted_delta_ar3": float(pred_ar_delta[position]),
        }
        record.update({f"predicted_{name}": float(values[position]) for name, values in model_predictions.items()})
        predictions.append(record)
    return summary, predictions


def fit_full_surrogate_models(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for target in TARGETS:
        y = np.asarray([float(row[f"{target}_median"]) for row in trajectory], dtype=float)
        endpoints = np.arange(N_LAGS + 1, len(y))
        x = np.asarray([one_step_features(trajectory, target, j) for j in endpoints])
        delta = y[endpoints] - y[endpoints - 1]
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x, delta)
        models[f"delta_{target}"] = model
    return models


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_note(path: Path, evidence: dict[str, Any]) -> None:
    lines = [
        "# Area-rank pseudo-time mask model (#65)",
        "",
        "This is a sequence-learning experiment on cross-sectional masks, not a biological time series.",
        "The synthetic coordinate `t_star` is the area-coverage quantile from 0 to 1 within GL/MYG/25C.",
        "Each of 100 pseudo-steps is the median shape of a coverage-ordered bin of unrelated release images.",
        "No mask pixels are differenced directly: specimen identity and image registration are unavailable.",
        "",
        f"Unlabeled input images: {evidence['image_count']}; pseudo-bins: {evidence['pseudo_bins']}; "
        f"train/test bins: {evidence['train_bins']}/{evidence['test_bins']}.",
        "The AR(3) predicts a change in the next bin's median shape feature from the previous three bins and t_star.",
        "Rolling AR forecasts use previous observed pseudo-bins; recursive AR forecasts feed their own prior predictions.",
        "The recursive persistence baseline repeats the last training-bin value. The cubic t* model is a direct multi-step baseline evaluated over the same holdout, without feedback.",
        "",
        "| Mask feature | Persistence rolling RMSE | Cubic t* RMSE | AR(3) rolling RMSE | Persistence recursive RMSE | Cubic t* direct RMSE | AR(3) recursive RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evidence["targets"]:
        lines.append(
            f"| `{row['target']}` | {row['persistence_rmse']:.5g} | "
            f"{row['pseudo_time_cubic_trend_rmse']:.5g} | {row['pseudo_time_ar3_rmse']:.5g} | "
            f"{row['persistence_recursive_rmse']:.5g} | "
            f"{row['pseudo_time_cubic_trend_recursive_rmse']:.5g} | "
            f"{row['pseudo_time_ar3_recursive_rmse']:.5g} |"
        )
    lines.extend([
        "",
        "Compare rolling AR(3) with rolling persistence and the direct t* trend; compare recursive AR(3) with repeated-value persistence and the same direct trend over the full holdout.",
        "A better score means only that adjacent coverage bins have learnable shape-feature changes in this selected release.",
        "It does not estimate elapsed time, a per-frame change, same-dish growth, a rate, or causal dynamics.",
        "The experiment tests whether this ordering can prototype sequence-model features and validation code before using verified time-lapse data.",
        "",
        "Artifacts: `pseudo_time_trajectory.csv`, `pseudo_time_forecasts.csv`, `pseudo_time_model_summary.csv`, and `pseudo_time_ar3_surrogate.joblib`.",
    ])
    path.write_text("\n".join(lines) + "\n")


def run(root: Path) -> None:
    rows = read_rows(root / INPUT_NAME)
    trajectory = make_trajectory(rows)
    summaries: list[dict[str, Any]] = []
    forecasts: list[dict[str, Any]] = []
    for target in TARGETS:
        summary, predictions = fit_target(trajectory, target)
        summaries.append(summary)
        forecasts.extend(predictions)
    models = fit_full_surrogate_models(trajectory)
    inference_state = json.loads((root / "state.json").read_text())
    joblib.dump({
        "models": models,
        "model_card": {
            "scope": "synthetic area-rank surrogate in GL/MYG/25C",
            "input": ["t_star", "previous 3 shape-feature medians"],
            "targets": [f"delta_{target}" for target in TARGETS],
            "pseudo_steps": len(trajectory),
            "training_observations_per_target": len(trajectory) - N_LAGS - 1,
            "segmentation_checkpoint_sha256": inference_state.get("checkpoint_sha256", ""),
            "interpretation": "Training prototype only; not elapsed time or biological growth.",
        },
    }, root / MODEL_NAME)
    write_csv(root / TRAJECTORY_NAME, trajectory)
    write_csv(root / FORECAST_NAME, forecasts)
    write_csv(root / SUMMARY_NAME, summaries)
    evidence = {
        "status": "complete",
        "scope": "exploratory synthetic-index sequence model",
        "stratum": MAIN_CONDITION,
        "segmentation_checkpoint_sha256": inference_state.get("checkpoint_sha256", ""),
        "image_count": len(rows),
        "source_set": "unlabeled",
        "pseudo_bins": len(trajectory),
        "train_bins": int(len(trajectory) * TRAIN_FRACTION) - N_LAGS - 1,
        "test_bins": len(trajectory) - int(len(trajectory) * TRAIN_FRACTION),
        "synthetic_coordinate": "t_star = area-coverage rank quantile; unitless, not elapsed time",
        "binning": "100 equal-count bins ordered by predicted_area_fraction; median features per bin",
        "targets": summaries,
        "trained_model_file": MODEL_NAME,
        "limitations": [
            "Consecutive pseudo-steps contain different, unlinked specimens.",
            "A mask-shape delta across adjacent bins is a cross-sectional coverage contrast, not a temporal increment.",
            "One-step evaluation supplies previous observed pseudo-bins; it does not validate biological forecasting.",
            "No chronology, same-dish trajectory, physical rate, or causal effect is inferred.",
        ],
    }
    (root / EVIDENCE_NAME).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    write_note(root / NOTE_NAME, evidence)
    write_note(REPO / "notes/pseudo_time_model.md", evidence)
    print(f"wrote {root / NOTE_NAME}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=("run",))
    parser.add_argument("--work", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    run(args.work.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
