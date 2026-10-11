#!/usr/bin/env python3
"""Fit image-level morphology models across MyceliumSeg area coverage (#41)."""

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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO / "work_dirs/segformer-b0-repro/full-release-all-labeled-20261009"
INPUT_NAME = "morphology_features.csv"
CV_NAME = "coverage_model_cv.csv"
CONDITION_NAME = "coverage_model_condition_holdout.csv"
PREDICTION_NAME = "coverage_model_oof_predictions.csv"
SENSITIVITY_NAME = "coverage_model_measurement_error_sensitivity.csv"
EVIDENCE_NAME = "coverage_model_evidence.json"
NOTE_NAME = "coverage_model_note.md"
MODEL_NAME = "coverage_model_final.joblib"
TARGETS = ("circularity", "solidity", "edge_roughness", "bbox_fill", "eccentricity")
ERROR_TARGETS = ("circularity", "solidity", "edge_roughness")
ERROR_DRAWS = 30
MODEL_STRATA = (
    {
        "name": "known_GL_MYG_25C_scale1",
        "species": "GL",
        "condition": "GL/MYG/25C",
        "block": "GL",
        "scale_group": "1",
        "source_set": "unlabeled",
    },
    {
        "name": "unknown_species_25C_flat_scale1",
        "species": "",
        "condition": "25C",
        "block": "GS_PO_TS_flat",
        "scale_group": "1",
        "source_set": "unlabeled",
    },
)


def numeric(row: dict[str, str], name: str) -> float | None:
    value = row.get(name, "")
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def read_features(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def eligible(row: dict[str, str]) -> bool:
    area = numeric(row, "predicted_area_fraction")
    return (
        row.get("area_fraction_qc") == "ok"
        and row.get("empty_mask", "").lower() == "false"
        and area is not None
        and 0.0 < area < 1.0
    )


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else 0.0,
    }


def make_model(train_n: int):
    knots = min(8, max(3, train_n // 100))
    return make_pipeline(
        SplineTransformer(n_knots=knots, degree=3, include_bias=False),
        Ridge(alpha=1.0),
    )


def coverage_cv(rows: list[dict[str, str]], target: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = [(numeric(row, "predicted_area_fraction"), numeric(row, target), row) for row in rows]
    data = [(x, y, row) for x, y, row in data if x is not None and y is not None]
    data.sort(key=lambda item: (item[0], item[2].get("record_key", "")))
    x = np.asarray([item[0] for item in data], dtype=float).reshape(-1, 1)
    y = np.asarray([item[1] for item in data], dtype=float)
    group_id = np.empty(len(data), dtype=int)
    for group, indices in enumerate(np.array_split(np.arange(len(data)), 10)):
        group_id[indices] = group
    prediction = np.full(len(data), np.nan)
    baseline = np.full(len(data), np.nan)
    for train, test in GroupKFold(n_splits=5).split(x, y, group_id):
        model = make_model(len(train))
        model.fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
        baseline[test] = float(np.median(y[train]))
    pred_metrics = metrics(y, prediction)
    base_metrics = metrics(y, baseline)
    summary = {
        "target": target,
        "n": len(data),
        "coverage_min": float(x.min()),
        "coverage_max": float(x.max()),
        "spline_rmse": pred_metrics["rmse"],
        "spline_mae": pred_metrics["mae"],
        "spline_r2": pred_metrics["r2"],
        "constant_baseline_rmse": base_metrics["rmse"],
        "constant_baseline_mae": base_metrics["mae"],
        "rmse_improvement_vs_constant": float(1.0 - pred_metrics["rmse"] / base_metrics["rmse"])
        if base_metrics["rmse"] > 0 else 0.0,
    }
    predictions = [
        {
            "record_key": item[2].get("record_key", ""),
            "target": target,
            "area_fraction": item[0],
            "observed": item[1],
            "oof_prediction": float(prediction[index]),
            "constant_baseline": float(baseline[index]),
        }
        for index, item in enumerate(data)
    ]
    return summary, predictions


def condition_holdouts(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    training = rows_for_stratum(rows, MODEL_STRATA[0])
    model_data: dict[str, tuple[Any, np.ndarray, float]] = {}
    for target in ERROR_TARGETS:
        pairs = [
            (numeric(row, "predicted_area_fraction"), numeric(row, target))
            for row in training
        ]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        x_train = np.asarray([pair[0] for pair in pairs]).reshape(-1, 1)
        y_train = np.asarray([pair[1] for pair in pairs])
        model = make_model(len(pairs))
        model.fit(x_train, y_train)
        model_data[target] = (model, x_train, float(np.median(y_train)))
    with (REPO / "data/triage/measurement_error.csv").open(newline="") as handle:
        holdout_rows = [
            row for row in csv.DictReader(handle)
            if row.get("duplicate_role") == "representative"
            and row.get("fit_status") == "ok"
            and row.get("species")
        ]
    conditions = sorted({row.get("condition", "") for row in holdout_rows if row.get("condition")})
    results: list[dict[str, Any]] = []
    for condition in conditions:
        test_rows = [row for row in holdout_rows if row.get("condition") == condition]
        for target in ERROR_TARGETS:
            test = [
                (numeric(row, "pred_area_fraction"), numeric(row, f"ref_{target}"), row)
                for row in test_rows
            ]
            test = [(x, y, row) for x, y, row in test if x is not None and y is not None]
            if not test:
                continue
            model, x_train, baseline_median = model_data[target]
            x_test = np.asarray([pair[0] for pair in test]).reshape(-1, 1)
            y_test = np.asarray([pair[1] for pair in test])
            predicted = model.predict(x_test)
            base = np.full(len(test), baseline_median)
            result = metrics(y_test, predicted)
            baseline = metrics(y_test, base)
            results.append({
                "heldout_condition": condition,
                "target": target,
                "n_train": len(x_train),
                "n_test": len(test),
                "test_target_source": "reference features from held-out masks (#31)",
                "test_area_source": "independent #28 predicted area_fraction",
                "spline_rmse_vs_reference": result["rmse"],
                "spline_mae_vs_reference": result["mae"],
                "spline_r2_vs_reference": result["r2"],
                "train_median_baseline_rmse_vs_reference": baseline["rmse"],
                "area_range_extrapolation": bool(
                    x_test.min() < x_train.min() or x_test.max() > x_train.max()
                ),
            })
    return results


def read_gl_errors(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("block") == "GL" and row.get("duplicate_role") == "representative"
        ]
    columns = {"area": "area_fraction_err"}
    columns.update({target: f"{target}_err" for target in ERROR_TARGETS})
    complete = [
        row for row in rows
        if all(
            numeric(row, column) is not None and numeric(row, column) > -0.95
            for column in columns.values()
        )
    ]
    return {
        key: np.asarray([float(row[column]) for row in complete], dtype=float)
        for key, column in columns.items()
    }


def measurement_error_sensitivity(
    rows: list[dict[str, str]], target: str, error_samples: dict[str, np.ndarray]
) -> dict[str, Any]:
    if "area" not in error_samples or target not in error_samples:
        raise RuntimeError(f"missing held-out error samples for {target}")
    rng = np.random.default_rng(41)
    improvements: list[float] = []
    spline_rmses: list[float] = []
    baseline_rmses: list[float] = []
    original_x = np.asarray([float(row["predicted_area_fraction"]) for row in rows])
    original_y = np.asarray([float(row[target]) for row in rows])
    for _ in range(ERROR_DRAWS):
        sampled = rng.integers(0, len(error_samples["area"]), size=len(rows))
        area_error = error_samples["area"][sampled]
        feature_error = error_samples[target][sampled]
        corrected_x = np.clip(original_x / (1.0 + area_error), 1e-8, 0.999999)
        corrected_y = original_y / (1.0 + feature_error)
        perturbed = [dict(row) for row in rows]
        for index, row in enumerate(perturbed):
            row["predicted_area_fraction"] = str(float(corrected_x[index]))
            row[target] = str(float(corrected_y[index]))
        summary, _ = coverage_cv(perturbed, target)
        spline = float(summary["spline_rmse"])
        baseline = float(summary["constant_baseline_rmse"])
        spline_rmses.append(spline)
        baseline_rmses.append(baseline)
        improvements.append(1.0 - spline / baseline if baseline else 0.0)
    return {
        "target": target,
        "draws": ERROR_DRAWS,
        "source": "empirical GL residuals from held-out #28 measurement-error rows",
        "improvement_median": float(np.median(improvements)),
        "improvement_p05": float(np.quantile(improvements, 0.05)),
        "improvement_p95": float(np.quantile(improvements, 0.95)),
        "spline_rmse_median": float(np.median(spline_rmses)),
        "constant_rmse_median": float(np.median(baseline_rmses)),
        "transfer_caveat": "Residuals are from #28, not independently measured errors for the all-labeled production checkpoint.",
    }


def fit_final_models(rows: list[dict[str, str]]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for target in TARGETS:
        pairs = [
            (numeric(row, "predicted_area_fraction"), numeric(row, target))
            for row in rows
        ]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        x = np.asarray([pair[0] for pair in pairs], dtype=float).reshape(-1, 1)
        y = np.asarray([pair[1] for pair in pairs], dtype=float)
        model = make_model(len(pairs))
        model.fit(x, y)
        models[target] = model
    return models


def rows_for_stratum(rows: list[dict[str, str]], stratum: dict[str, str]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if eligible(row)
        and all(row.get(key, "") == value for key, value in stratum.items() if key != "name")
    ]


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
        "# MyceliumSeg morphology across area coverage (#41)",
        "",
        "This is an exploratory cross-sectional model of predicted-mask shape by area coverage.",
        "It uses no time, dish, or frame-order assumption. All results describe the selected release.",
        "Boundary features come from predicted masks and inherit the segmenter's boundary error.",
        "",
        f"Models cover {len(evidence['strata'])} large strata, from "
        f"{evidence['all_feature_rows']} deduplicated feature rows.",
        "",
        "| Stratum | Target | N | Spline RMSE | Constant RMSE | RMSE improvement | Spline R² |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in evidence["coverage_cv"]:
        lines.append(
            f"| `{row['stratum']}` | `{row['target']}` | {row['n']} | {row['spline_rmse']:.5g} | "
            f"{row['constant_baseline_rmse']:.5g} | "
            f"{100*row['rmse_improvement_vs_constant']:.1f}% | {row['spline_r2']:.3f} |"
        )
    lines.extend([
        "",
        "Approximate measurement-error sensitivity (30 draws resampling joint GL relative errors from #31):",
        "",
        "| Target | Median RMSE improvement | 5th–95th percentile |",
        "|---|---:|---:|",
    ])
    for row in evidence["error_sensitivity"]:
        lines.append(
            f"| `{row['target']}` | {100*row['improvement_median']:.1f}% | "
            f"{100*row['improvement_p05']:.1f}% to {100*row['improvement_p95']:.1f}% |"
        )
    lines.extend([
        "",
        "Validation holds out contiguous area-coverage quantile groups. The constant baseline is the training-fold median.",
        "The sensitivity draw transfers held-out #28 residuals to the all-labeled model; it is approximate, not a production-checkpoint validation interval.",
        "Training includes only unlabeled images so the all-labeled segmenter is not evaluated on its own mask-training images. Condition-holdout scores use #31 reference-mask features and #28 predicted area on 90 deduplicated labeled images; groups of about ten are stress checks, not strong transport claims.",
        "The model does not use area fraction as its own response. It predicts shape descriptors from coverage.",
        "",
        "Artifacts: `morphology_features.csv`, `coverage_model_cv.csv`, `coverage_model_condition_holdout.csv`, `coverage_model_oof_predictions.csv`, and `coverage_model_final.joblib` in this work directory.",
    ])
    path.write_text("\n".join(lines) + "\n")


def run(root: Path) -> None:
    rows = read_features(root / INPUT_NAME)
    strata_rows = {stratum["name"]: rows_for_stratum(rows, stratum) for stratum in MODEL_STRATA}
    for name, selected in strata_rows.items():
        if len(selected) < 1000:
            raise RuntimeError(f"stratum too small for the planned coverage model: {name} n={len(selected)}")
    summaries: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    final_models: dict[str, dict[str, Any]] = {}
    for stratum_name, selected in strata_rows.items():
        final_models[stratum_name] = fit_final_models(selected)
        for target in TARGETS:
            summary, target_predictions = coverage_cv(selected, target)
            summary["stratum"] = stratum_name
            for prediction in target_predictions:
                prediction["stratum"] = stratum_name
            summaries.append(summary)
            predictions.extend(target_predictions)
    holdouts = condition_holdouts(rows)
    error_samples = read_gl_errors(REPO / "data/triage/measurement_error.csv")
    sensitivity = [
        measurement_error_sensitivity(strata_rows[MODEL_STRATA[0]["name"]], target, error_samples)
        for target in ERROR_TARGETS
    ]
    inference_state = json.loads((root / "state.json").read_text())
    model_card = {
        "scope": "separate selected-release coverage strata",
        "input": "predicted_area_fraction",
        "targets": list(TARGETS),
        "training_rows_by_stratum": {name: len(selected) for name, selected in strata_rows.items()},
        "strata": list(MODEL_STRATA),
        "segmentation_checkpoint_sha256": inference_state.get("checkpoint_sha256", ""),
        "interpretation": "Cross-sectional shape prediction by coverage; not chronology or growth rate. Unknown species are modeled separately.",
        "validation_artifact": CV_NAME,
    }
    joblib.dump({"models": final_models, "model_card": model_card}, root / MODEL_NAME)
    evidence = {
        "status": "complete",
        "scope": "selected-release cross-sectional morphology",
        "time_claim": "none; area coverage is not chronology",
        "all_feature_rows": len(rows),
        "strata": model_card["training_rows_by_stratum"],
        "segmentation_checkpoint_sha256": model_card["segmentation_checkpoint_sha256"],
        "final_model_file": MODEL_NAME,
        "validation": "5-fold GroupKFold over 10 contiguous area-coverage rank groups",
        "coverage_cv": summaries,
        "error_sensitivity": sensitivity,
        "condition_holdout_rows": len(holdouts),
        "condition_holdout_caveat": "Held-out targets are reference-mask features and area predictors are #28 outputs; groups of about 9-10 are stress checks, not strong transport claims.",
        "model_card": model_card,
    }
    (root / EVIDENCE_NAME).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    write_csv(root / CV_NAME, summaries)
    write_csv(root / CONDITION_NAME, holdouts)
    write_csv(root / PREDICTION_NAME, predictions)
    write_csv(root / SENSITIVITY_NAME, sensitivity)
    write_note(root / NOTE_NAME, evidence)
    write_note(REPO / "notes/coverage_model.md", evidence)
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
