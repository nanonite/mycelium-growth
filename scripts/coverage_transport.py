#!/usr/bin/env python3
"""Evaluate cross-condition/species transport of shape-by-coverage models (#44).

Models are fit on unlabeled GL/MYG/25C images and scored against independent
reference-mask features from the #31 held-out set. Reported species labels for
the label-conflict subset are retained as diagnostics and are never treated as
clean species holdouts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import measurement_error as me
from full_release_morphology import bbox_features

DEFAULT_ROOT = REPO / "work_dirs/segformer-b0-repro/full-release-all-labeled-20261009"
FEATURES_NAME = "morphology_features.csv"
ERROR_NAME = "data/triage/measurement_error.csv"
SUMMARY_NAME = "coverage_transport_holdouts.csv"
PREDICTION_NAME = "coverage_transport_predictions.csv"
EVIDENCE_NAME = "coverage_transport_evidence.json"
MODEL_NAME = "coverage_transport_models.joblib"
NOTE_NAME = "coverage_transport_note.md"
TARGETS = ("circularity", "solidity", "edge_roughness", "bbox_fill", "eccentricity")
ERROR_TARGETS = ("circularity", "solidity", "edge_roughness")
TRAIN_STRATUM = {
    "source_set": "unlabeled",
    "species": "GL",
    "condition": "GL/MYG/25C",
    "block": "GL",
    "scale_group": "1",
}


def numeric(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def eligible_feature(row: dict[str, str]) -> bool:
    area = numeric(row, "predicted_area_fraction")
    return (
        row.get("area_fraction_qc") == "ok"
        and row.get("empty_mask", "").lower() == "false"
        and area is not None
        and 0.0 < area < 1.0
    )


def fit_model(x: np.ndarray, y: np.ndarray):
    knots = min(8, max(3, len(x) // 100))
    model = make_pipeline(
        SplineTransformer(n_knots=knots, degree=3, include_bias=False),
        Ridge(alpha=1.0),
    )
    model.fit(x.reshape(-1, 1), y)
    return model


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else 0.0,
    }


def reference_targets(row: dict[str, str], root: Path) -> dict[str, float]:
    targets = {target: numeric(row, f"ref_{target}") for target in ERROR_TARGETS}
    mask_path = root / "data/segmentation/release" / row["source_set"] / "mask" / f"{row['id']}.png"
    if not mask_path.is_file():
        raise FileNotFoundError(f"reference mask missing for #31 row {row['id']}: {mask_path}")
    mask = me.load_mask_working(mask_path)
    targets.update(bbox_features(mask))
    if any(value is None or not math.isfinite(float(value)) for value in targets.values()):
        raise RuntimeError(f"incomplete reference features for #31 row {row['id']}")
    return {key: float(value) for key, value in targets.items()}


def group_name(row: dict[str, Any]) -> str:
    status = "conflict" if row["label_conflict"] else "clean"
    return f"{row['condition']}|label_{status}"


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
        "# Held-out transport of morphology across area coverage (#44)",
        "",
        "This is cross-sectional prediction within the selected release. It does not estimate chronology, dish-level growth, or physical rates.",
        "",
        f"Training: {evidence['training_rows']} unlabeled GL/MYG/25C/scale1 images. ",
        f"Reference holdout: {evidence['reference_holdout_rows']} deduplicated, calibrated #31 images; ",
        f"{evidence['label_conflict_rows']} have conflicting archive labels.",
        "The model learns mask-shape features from the all-labeled checkpoint predictions; test targets come from human reference masks.",
        "Primary test area inputs are independent #28 predicted area_fraction values. Reference-mask area_fraction is included as an input sensitivity check.",
        "",
        "| Held-out condition / label status | Target | N | Area input | Model RMSE | Train-median RMSE | RMSE change vs median | R² | Area extrapolation |",
        "|---|---|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for row in evidence["holdout_scores"]:
        lines.append(
            f"| `{row['group']}` | `{row['target']}` | {row['n_test']} | {row['area_input_source']} | "
            f"{row['model_rmse']:.5g} | {row['train_median_rmse']:.5g} | "
            f"{100 * row['rmse_improvement_vs_train_median']:.1f}% | {row['model_r2']:.3f} | "
            f"{'yes' if row['area_range_extrapolation'] else 'no'} |"
        )
    lines.extend([
        "",
        "Clean GL/MYG/15C rows are the only uncontested held-out condition group. GL/MYG/25C reference rows are a same-condition independent-image control, not condition transport.",
        "GS, PO, and TS archive-labeled rows are reported separately as label-conflict diagnostics; they cannot support clean species-specific transport claims. The unlabeled flat GS/PO/TS block is excluded from species transport because its species are unknown and it has no reference masks.",
        "Duplicate copies were excluded before scoring; only representatives remain. Off-scale rows, empty masks, invalid area fractions, and unresolved label-conflict rows are not pooled with clean groups. All test groups are tiny, so per-group scores are exploratory stress checks.",
        "Training area/features come from the all-labeled production checkpoint; #31 test areas come from the independent #28 checkpoint. The two area-input sources differ, and predicted-mask shape targets inherit segmentation error. Reference-area scoring is a sensitivity check, not the deployment score.",
        "All conclusions describe the selected release. The paper's image inclusion rule is undocumented, so morphology-selection bias is unresolved.",
        "",
        "Artifacts: `coverage_transport_holdouts.csv`, `coverage_transport_predictions.csv`, `coverage_transport_models.joblib`, and `coverage_transport_evidence.json` in the full-release work directory.",
    ])
    path.write_text("\n".join(lines) + "\n")


def run(root: Path) -> None:
    features = read_csv(root / FEATURES_NAME)
    training = [
        row for row in features
        if eligible_feature(row)
        and all(row.get(key, "") == value for key, value in TRAIN_STRATUM.items())
    ]
    if len(training) < 1000:
        raise RuntimeError(f"too few training rows for the fixed stratum: n={len(training)}")
    feature_by_id = {
        row["id"]: row for row in features
        if row.get("source_set") == "labeled" and row.get("duplicate_role") == "representative"
    }

    error_rows = read_csv(REPO / ERROR_NAME)
    representative_rows = [row for row in error_rows if row.get("duplicate_role") == "representative"]
    eligible_rows: list[dict[str, Any]] = []
    excluded = {"nonrepresentative": len(error_rows) - len(representative_rows), "off_scale_or_invalid": 0}
    for row in representative_rows:
        area = numeric(row, "pred_area_fraction")
        reference_area = numeric(row, "ref_area_fraction")
        meta = feature_by_id.get(row.get("id", ""))
        if row.get("fit_status") != "ok" or area is None or reference_area is None or not (0 < area < 1 and 0 < reference_area < 1) or not meta:
            excluded["off_scale_or_invalid"] += 1
            continue
        targets = reference_targets(row, REPO)
        eligible_rows.append({
            "id": row["id"],
            "condition": row.get("condition", "unknown"),
            "species_label": row.get("species", ""),
            "medium_label": row.get("medium", ""),
            "temp_label": row.get("temp", ""),
            "source_set": row.get("source_set", ""),
            "label_conflict": meta.get("label_conflict", "False") == "True",
            "duplicate_role": row.get("duplicate_role", ""),
            "duplicate_group_size": row.get("duplicate_group_size", ""),
            "scale_group": meta.get("scale_group", ""),
            "pred_area_fraction": area,
            "ref_area_fraction": reference_area,
            "targets": targets,
        })

    if len(eligible_rows) < 30:
        raise RuntimeError(f"too few calibrated reference rows for grouped holdout: n={len(eligible_rows)}")
    train_area = np.asarray([float(row["predicted_area_fraction"]) for row in training])
    train_area_min, train_area_max = float(train_area.min()), float(train_area.max())
    models: dict[str, Any] = {}
    train_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for target in TARGETS:
        pairs = [(numeric(row, "predicted_area_fraction"), numeric(row, target)) for row in training]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        x = np.asarray([pair[0] for pair in pairs], dtype=float)
        y = np.asarray([pair[1] for pair in pairs], dtype=float)
        model = fit_model(x, y)
        models[target] = model
        train_data[target] = (x, y, float(np.median(y)))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_rows:
        grouped[group_name(row)].append(row)

    summaries: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        for target in TARGETS:
            model = models[target]
            _, _, baseline_value = train_data[target]
            for x_source in ("pred_area_fraction", "ref_area_fraction"):
                x_test = np.asarray([float(row[x_source]) for row in group_rows], dtype=float)
                y_test = np.asarray([float(row["targets"][target]) for row in group_rows], dtype=float)
                predicted = model.predict(x_test.reshape(-1, 1))
                baseline_pred = np.full(len(y_test), baseline_value, dtype=float)
                model_metrics = score(y_test, predicted)
                baseline_metrics = score(y_test, baseline_pred)
                summaries.append({
                    "group": group,
                    "condition_label": group_rows[0]["condition"],
                    "species_label": group_rows[0]["species_label"],
                    "label_conflict": group_rows[0]["label_conflict"],
                    "target": target,
                    "n_test": len(group_rows),
                    "area_input_source": "#28_predicted_area_fraction" if x_source == "pred_area_fraction" else "reference_mask_area_fraction_sensitivity",
                    "model_rmse": model_metrics["rmse"],
                    "model_mae": model_metrics["mae"],
                    "model_r2": model_metrics["r2"],
                    "train_median_rmse": baseline_metrics["rmse"],
                    "train_median_mae": baseline_metrics["mae"],
                    "rmse_improvement_vs_train_median": 1.0 - model_metrics["rmse"] / baseline_metrics["rmse"] if baseline_metrics["rmse"] else 0.0,
                    "test_area_min": float(x_test.min()),
                    "test_area_max": float(x_test.max()),
                    "train_area_min": train_area_min,
                    "train_area_max": train_area_max,
                    "area_range_extrapolation": bool(x_test.min() < train_area_min or x_test.max() > train_area_max),
                    "target_source": "reference mask",
                })
                for row, x_value, actual, prediction, baseline in zip(
                    group_rows, x_test, y_test, predicted, baseline_pred
                ):
                    predictions.append({
                        "id": row["id"],
                        "condition_label": row["condition"],
                        "species_label": row["species_label"],
                        "label_conflict": row["label_conflict"],
                        "scale_group": row["scale_group"],
                        "target": target,
                        "area_input_source": "#28_predicted_area_fraction" if x_source == "pred_area_fraction" else "reference_mask_area_fraction_sensitivity",
                        "area_fraction": float(x_value),
                        "observed_reference": float(actual),
                        "model_prediction": float(prediction),
                        "train_median_baseline": float(baseline),
                    })

    inference_state = json.loads((root / "state.json").read_text())
    evidence = {
        "status": "complete",
        "issue": 44,
        "scope": "cross-sectional transport across held-out condition/species labels",
        "training_stratum": TRAIN_STRATUM,
        "training_rows": len(training),
        "training_area_range": [train_area_min, train_area_max],
        "targets": list(TARGETS),
        "measurement_error_input_rows": len(error_rows),
        "reference_holdout_rows": len(eligible_rows),
        "clean_reference_holdout_rows": sum(not row["label_conflict"] for row in eligible_rows),
        "label_conflict_rows": sum(row["label_conflict"] for row in eligible_rows),
        "excluded_reference_rows": excluded,
        "holdout_group_counts": {group: len(group_rows) for group, group_rows in sorted(grouped.items())},
        "test_target_source": "human reference masks from held-out #31 set",
        "primary_test_area_source": "independent #28 checkpoint predicted area_fraction",
        "area_input_sensitivity": "reference-mask area_fraction; oracle measurement sensitivity only, not deployment score",
        "segmentation_checkpoint_sha256": inference_state.get("checkpoint_sha256", ""),
        "holdout_scores": summaries,
        "limitations": [
            "The clean held-out condition sample is GL/MYG/15C (n=10); same-condition GL/MYG/25C rows are an independent-image control, not condition transport.",
            "The GS, PO, and TS holdout labels all conflict across duplicate image records; their scores are diagnostics, not clean species-specific transport claims.",
            "Groups contain only 9-10 images and are stress checks, not precise transport estimates.",
            "The unknown-species unlabeled GS/PO/TS flat stratum has no reference masks and cannot be used as a labeled species holdout.",
            "Training targets derive from all-labeled checkpoint predictions while test targets derive from reference masks; test area inputs come from #28, so there is checkpoint/source shift.",
            "All results describe the selected release; the image inclusion rule remains undocumented.",
        ],
    }
    model_card = {
        "scope": evidence["scope"],
        "training_stratum": TRAIN_STRATUM,
        "targets": list(TARGETS),
        "segmentation_checkpoint_sha256": evidence["segmentation_checkpoint_sha256"],
        "interpretation": "Cross-sectional image-level transport only; no temporal or species mechanism claim.",
    }
    joblib.dump({"models": models, "model_card": model_card}, root / MODEL_NAME)
    write_csv(root / SUMMARY_NAME, summaries)
    write_csv(root / PREDICTION_NAME, predictions)
    (root / EVIDENCE_NAME).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    write_note(root / NOTE_NAME, evidence)
    write_note(REPO / "notes/coverage_transport.md", evidence)
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
