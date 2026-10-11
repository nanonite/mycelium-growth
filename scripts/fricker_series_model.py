#!/usr/bin/env python3
"""Fit a series-aware, dataset-specific trajectory model to the Fricker archive.

The primary response is within-series log change from the linked D1 workbook.
Species-by-grazer slopes are estimated on EXIF elapsed days. Validation holds
out whole replicate series, and uncertainty resamples whole series within each
species/grazer group. The archive day labels provide a sensitivity analysis.

Run from any directory with:
    python scripts/fricker_series_model.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/derived/fricker_grazers/network_series.csv"
OUT = ROOT / "data/derived/fricker_grazers"
RNG_SEED = 68045
BOOTSTRAPS = 5000
SPECIES = ("Hf", "Pi", "Pv", "Rb")
CORE_TREATMENTS = ("control", "coll", "mill", "nema", "wl1", "wl2")
OUTCOMES = {
    "mycelial_area": "summary_mycelial_area",
    "total_length": "summary_length_total",
    "volume": "summary_volume",
    "tip_count": "summary_ntips",
    "mean_width": "summary_width_mean",
    "edge_count": "summary_num_edges",
    "loop_count": "summary_cyclomatic",
}


def number(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_records() -> list[dict[str, Any]]:
    with INPUT.open(newline="") as handle:
        records: list[dict[str, Any]] = list(csv.DictReader(handle))
    for row in records:
        row["day_label"] = int(row["day_label"])
        row["species"] = row["species"]
        row["treatment_model"] = (
            "control" if row["treatment"].startswith("ctrl") else row["treatment"]
        )
        row["elapsed_days"] = number(row, "elapsed_days_from_first")
        row["has_image_bool"] = row["has_image"].lower() == "true"
        row["same_timestamp_bool"] = row["same_timestamp_as_prior"].lower() == "true"
    return records


def baseline_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    baseline: dict[str, dict[str, Any]] = {}
    for row in records:
        if (
            row["treatment_model"] in CORE_TREATMENTS + ("mite", "enchy")
            and row["day_label"] == 1
            and row["has_image_bool"]
            and row["capture_datetime"]
        ):
            baseline[row["series_key"]] = row
    return baseline


def trajectory_rows(
    records: list[dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in records:
        if (
            row["treatment_model"] not in CORE_TREATMENTS
            or row["species"] not in SPECIES
            or row["series_key"] not in baselines
            or row["day_label"] not in (1, 2, 4, 8)
            or not row["has_image_bool"]
            or not row["capture_datetime"]
            or row["same_timestamp_bool"]
        ):
            continue
        y = number(row, metric)
        y0 = number(baselines[row["series_key"]], metric)
        if y is None or y0 is None or y <= 0 or y0 <= 0:
            continue
        item = dict(row)
        item["delta_log"] = math.log(y) - math.log(y0)
        item["cell"] = f"{row['species']}|{row['treatment_model']}"
        item["label_elapsed"] = float(row["day_label"] - 1)
        result.append(item)
    return result


def design(rows: list[dict[str, Any]], cells: list[str], model: str) -> np.ndarray:
    cell_index = {cell: index for index, cell in enumerate(cells)}
    if model == "day_factor":
        days = (2, 4, 8)
        x = np.zeros((len(rows), len(cells) * len(days)), dtype=float)
        day_index = {day: index for index, day in enumerate(days)}
        for i, row in enumerate(rows):
            if row["day_label"] == 1:
                continue
            col = cell_index[row["cell"]] * len(days) + day_index[row["day_label"]]
            x[i, col] = 1.0
        return x

    x = np.zeros((len(rows), len(cells)), dtype=float)
    time_key = "label_elapsed" if model == "label_linear" else "elapsed_days"
    for i, row in enumerate(rows):
        x[i, cell_index[row["cell"]]] = float(row[time_key] or 0.0)
    return x


def grouped_cv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # D1 is the observed reference point (delta=0), not a predictive test row.
    scored = [row for row in rows if row["day_label"] != 1]
    cells = [f"{species}|{treatment}" for species in SPECIES for treatment in CORE_TREATMENTS]
    y = np.asarray([row["delta_log"] for row in scored], dtype=float)
    groups = np.asarray([row["series_key"] for row in scored])
    splitter = GroupKFold(n_splits=5)
    results: list[dict[str, Any]] = []
    for model in ("elapsed_linear", "label_linear", "day_factor"):
        x = design(scored, cells, model)
        observed: list[float] = []
        predicted: list[float] = []
        for train, test in splitter.split(x, y, groups):
            beta, *_ = np.linalg.lstsq(x[train], y[train], rcond=None)
            predicted.extend((x[test] @ beta).tolist())
            observed.extend(y[test].tolist())
        residual = np.asarray(observed) - np.asarray(predicted)
        results.append(
            {
                "model": model,
                "n_postbaseline_observations": len(scored),
                "n_series": len(set(groups)),
                "grouped_cv_rmse_log_change": float(np.sqrt(np.mean(residual**2))),
                "grouped_cv_mae_log_change": float(np.mean(np.abs(residual))),
            }
        )
    return results


def bootstrap_cell_slopes(rows: list[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    by_cell_series: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["day_label"] != 1:
            by_cell_series[row["cell"]][row["series_key"]].append(row)

    output: list[dict[str, Any]] = []
    for species in SPECIES:
        for treatment in CORE_TREATMENTS:
            cell = f"{species}|{treatment}"
            series_rows = by_cell_series.get(cell, {})
            series = list(series_rows)
            if not series:
                continue

            def estimate(sample: list[str]) -> float:
                tx: list[float] = []
                dy: list[float] = []
                for key in sample:
                    for row in series_rows[key]:
                        tx.append(float(row["elapsed_days"] or 0.0))
                        dy.append(float(row["delta_log"]))
                t = np.asarray(tx, dtype=float)
                d = np.asarray(dy, dtype=float)
                denom = float(np.dot(t, t))
                return float(np.dot(t, d) / denom) if denom > 0 else math.nan

            beta = estimate(series)
            boots = np.empty(BOOTSTRAPS, dtype=float)
            for b in range(BOOTSTRAPS):
                sample_indices = rng.integers(0, len(series), size=len(series))
                boots[b] = estimate([series[i] for i in sample_indices])
            low, high = np.quantile(boots[np.isfinite(boots)], (0.025, 0.975))
            observed_days = [float(r["elapsed_days"]) for s in series for r in series_rows[s]]
            output.append(
                {
                    "species": species,
                    "grazer_group": treatment,
                    "n_replicate_series": len(series),
                    "n_postbaseline_observations": sum(len(series_rows[s]) for s in series),
                    "log_relative_change_per_exif_day": beta,
                    "relative_change_percent_per_exif_day": 100.0 * math.expm1(beta),
                    "rate_ci_low_percent_per_exif_day": 100.0 * math.expm1(float(low)),
                    "rate_ci_high_percent_per_exif_day": 100.0 * math.expm1(float(high)),
                    "median_postbaseline_elapsed_days": float(np.median(observed_days)),
                }
            )
    return output


def endpoint_rows(
    records: list[dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    metric: str,
) -> dict[tuple[str, str], list[dict[str, float | str]]]:
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if (
            row["species"] in SPECIES
            and row["treatment_model"] in CORE_TREATMENTS + ("mite", "enchy")
            and row["has_image_bool"]
            and row["capture_datetime"]
            and row["day_label"] in (1, 8)
            and not row["same_timestamp_bool"]
        ):
            by_series[row["series_key"]].append(row)

    grouped: dict[tuple[str, str], list[dict[str, float | str]]] = defaultdict(list)
    for series_key, series in by_series.items():
        if series_key not in baselines:
            continue
        baseline = baselines[series_key]
        d8 = next((row for row in series if row["day_label"] == 8), None)
        if d8 is None:
            continue
        y0, y8 = number(baseline, metric), number(d8, metric)
        if y0 is None or y8 is None or y0 <= 0 or y8 <= 0:
            continue
        grouped[(d8["species"], d8["treatment_model"])].append(
            {
                "series_key": series_key,
                "delta_log": math.log(y8) - math.log(y0),
                "elapsed_days": float(d8["elapsed_days"] or 0.0),
            }
        )
    return grouped


def baseline_adjusted_contrasts(
    records: list[dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    metric_name: str,
    metric: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    grouped = endpoint_rows(records, baselines, metric)
    output: list[dict[str, Any]] = []
    for treatment in ("coll", "mill", "nema", "wl1", "wl2", "mite", "enchy"):
        samples: list[tuple[list[float], list[float]]] = []
        for species in SPECIES:
            treated = [float(r["delta_log"]) for r in grouped.get((species, treatment), [])]
            control = [float(r["delta_log"]) for r in grouped.get((species, "control"), [])]
            if treated and control:
                samples.append((treated, control))
        if not samples:
            continue
        species_effects = [float(np.mean(t)) - float(np.mean(c)) for t, c in samples]
        estimate = float(np.mean(species_effects))
        boot = np.zeros(BOOTSTRAPS, dtype=float)
        for treated, control in samples:
            bt = rng.choice(treated, size=(BOOTSTRAPS, len(treated)), replace=True).mean(axis=1)
            bc = rng.choice(control, size=(BOOTSTRAPS, len(control)), replace=True).mean(axis=1)
            boot += (bt - bc) / len(samples)
        coverage = len(samples)
        elapsed = [
            float(row["elapsed_days"])
            for species in SPECIES
            for group in (treatment, "control")
            for row in grouped.get((species, group), [])
        ]
        output.append(
            {
                "outcome": metric_name,
                "grazer_group": treatment,
                "species_coverage_n": coverage,
                "species_coverage": ",".join(
                    species for species in SPECIES
                    if grouped.get((species, treatment)) and grouped.get((species, "control"))
                ),
                "n_treated_series": sum(len(grouped.get((s, treatment), [])) for s in SPECIES),
                "n_control_series": sum(len(grouped.get((s, "control"), [])) for s in SPECIES),
                "median_exif_days_to_d8_label": float(np.median(elapsed)),
                "delta_log_d8_change_vs_control": estimate,
                "geometric_ratio_of_baseline_adjusted_change": math.exp(estimate),
                "ratio_ci_low": math.exp(float(np.quantile(boot, 0.025))),
                "ratio_ci_high": math.exp(float(np.quantile(boot, 0.975))),
                "interpretation": "descriptive association; treatment allocation not documented",
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def day2_start_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if (
            row["treatment_model"] in CORE_TREATMENTS
            and row["species"] in SPECIES
            and row.get("first_observed_day") == "2"
            and row["has_image_bool"]
            and row["capture_datetime"]
            and row["day_label"] in (2, 8)
            and not row["same_timestamp_bool"]
        ):
            by_series[row["series_key"]].append(row)

    output: list[dict[str, Any]] = []
    for series_key, series in sorted(by_series.items()):
        d2 = next((row for row in series if row["day_label"] == 2), None)
        d8 = next((row for row in series if row["day_label"] == 8), None)
        item: dict[str, Any] = {
            "series_key": series_key,
            "species": series[0]["species"],
            "grazer_group": series[0]["treatment_model"],
            "has_d8_label": bool(d8),
            "elapsed_days_d2_to_d8": (
                float(d8["elapsed_days"] - d2["elapsed_days"])
                if d2 and d8 and d8["elapsed_days"] is not None and d2["elapsed_days"] is not None
                else ""
            ),
            "note": "D2 is the first linked image; this is not a D1-baseline-adjusted trajectory",
        }
        for outcome, metric in OUTCOMES.items():
            y2 = number(d2, metric) if d2 else None
            y8 = number(d8, metric) if d8 else None
            item[f"d2_to_d8_log_change_{outcome}"] = (
                math.log(y8) - math.log(y2)
                if y2 is not None and y8 is not None and y2 > 0 and y8 > 0
                else ""
            )
        output.append(item)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = read_records()
    baselines = baseline_map(records)
    rng = np.random.default_rng(RNG_SEED)

    cv_all: list[dict[str, Any]] = []
    slopes_all: list[dict[str, Any]] = []
    contrasts_all: list[dict[str, Any]] = []
    metric_counts: dict[str, dict[str, int]] = {}
    for outcome, metric in OUTCOMES.items():
        rows = trajectory_rows(records, baselines, metric)
        post = [r for r in rows if r["day_label"] != 1]
        if len({r["series_key"] for r in post}) < 5:
            continue
        for row in grouped_cv(rows):
            cv_all.append({"outcome": outcome, **row})
        for row in bootstrap_cell_slopes(rows, rng):
            slopes_all.append({"outcome": outcome, **row})
        for row in baseline_adjusted_contrasts(records, baselines, outcome, metric, rng):
            contrasts_all.append(row)
        metric_counts[outcome] = {
            "trajectory_observations_including_baseline": len(rows),
            "baseline_linked_series": len({r["series_key"] for r in rows}),
            "postbaseline_observations_scored": len(post),
        }

    write_csv(OUT / "series_trajectory_cv.csv", cv_all)
    write_csv(OUT / "series_growth_rates.csv", slopes_all)
    write_csv(OUT / "baseline_adjusted_d8_contrasts.csv", contrasts_all)

    primary_series = {
        row["series_key"]
        for row in records
        if row["treatment_model"] in CORE_TREATMENTS and row["species"] in SPECIES
    }
    starts_d2 = {
        row["series_key"]
        for row in records
        if row["series_key"] in primary_series and row["day_label"] == 2
        and row.get("first_observed_day") == "2"
    }
    duplicate_pairs = [
        row["file_stem"]
        for row in records
        if row["treatment"] == "ctrl2" and row["species"] == "Pi"
        and row["day_label"] == 2 and row["same_timestamp_bool"]
    ]
    write_csv(OUT / "day2_start_series.csv", day2_start_rows(records))
    summary = {
        "model": "species-by-grazer within-series log-relative trajectory",
        "model_equation": "log(Y_it / Y_i_D1) = beta_species,grazer * EXIF_elapsed_days_it + error_it",
        "interpretation": "dataset-specific descriptive/predictive rates; not causal treatment effects",
        "input": str(INPUT.relative_to(ROOT)),
        "source_records": len(records),
        "baseline_linked_series_core_treatments": sum(
            row["treatment_model"] in CORE_TREATMENTS
            for key, row in baselines.items()
        ),
        "baseline_linked_series_all_archive_treatments": len(baselines),
        "core_series_with_any_records": len(primary_series),
        "day2_start_series_without_d1_baseline": len(starts_d2),
        "same_session_pi_ctrl2_day2_rows_excluded": len(duplicate_pairs),
        "same_session_pi_ctrl2_files": duplicate_pairs,
        "bootstrap_resamples_per_group": BOOTSTRAPS,
        "bootstrap_seed": RNG_SEED,
        "outcomes": metric_counts,
        "outputs": [
            "data/derived/fricker_grazers/series_trajectory_cv.csv",
            "data/derived/fricker_grazers/series_growth_rates.csv",
            "data/derived/fricker_grazers/baseline_adjusted_d8_contrasts.csv",
            "data/derived/fricker_grazers/day2_start_series.csv",
        ],
        "treatment_allocation_review": "The primary paper reports weekly random tray repositioning; it does not document randomized treatment allocation. Contrasts remain associational.",
    }
    (OUT / "series_model_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
