#!/usr/bin/env python3
"""Summarize held-out equivalent-radius error for Chainlink issue #69.

This is a calibration audit for the MyceliumSeg #28 checkpoint on the #31
held-out labeled masks. It does not fit a temporal model, choose a likelihood
for another dataset, or treat these errors as transportable calibration.

Run from the repository root:
    python scripts/bounded_radius_likelihood.py
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "triage" / "measurement_error.csv"
OUTPUT = ROOT / "data" / "derived" / "bounded_radius_likelihood"
MAX_RADIUS_MM = 45.0


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else math.nan


def summarize(name: str, pairs: list[tuple[float, float]]) -> dict[str, object]:
    reference = [row[0] for row in pairs]
    predicted = [row[1] for row in pairs]
    error_mm = [p - r for r, p in pairs]
    relative_error = [e / r for e, r in zip(error_mm, reference)]
    log_ratio = [math.log(p / r) for r, p in pairs]
    logit_error = [
        math.log((p / MAX_RADIUS_MM) / (1 - p / MAX_RADIUS_MM))
        - math.log((r / MAX_RADIUS_MM) / (1 - r / MAX_RADIUS_MM))
        for r, p in pairs
    ]
    return {
        "stratum": name,
        "n": len(pairs),
        "reference_radius_min_mm": min(reference),
        "reference_radius_max_mm": max(reference),
        "reference_radius_mean_mm": statistics.mean(reference),
        "absolute_error_bias_mm": statistics.mean(error_mm),
        "absolute_error_sd_mm": sample_sd(error_mm),
        "relative_error_bias": statistics.mean(relative_error),
        "relative_error_sd": sample_sd(relative_error),
        "log_ratio_bias": statistics.mean(log_ratio),
        "log_ratio_sd": sample_sd(log_ratio),
        "logit_radius_error_sd": sample_sd(logit_error),
    }


def load_pairs() -> list[tuple[float, float]]:
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing #31 measurement-error table: {INPUT}")
    pairs = []
    with INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["duplicate_role"] != "representative" or row["fit_status"] != "ok":
                continue
            reference = float(row["ref_equiv_radius_mm"])
            predicted = float(row["pred_equiv_radius_mm"])
            if not (0 < reference < MAX_RADIUS_MM and 0 < predicted < MAX_RADIUS_MM):
                raise ValueError(
                    "This interior-error audit expects finite radii strictly between 0 and 45 mm."
                )
            pairs.append((reference, predicted))
    if not pairs:
        raise ValueError("No eligible representative rows found in the #31 table.")
    return pairs


def main() -> None:
    pairs = sorted(load_pairs(), key=lambda pair: pair[0])
    rows = [summarize("all", pairs)]
    group_size = len(pairs) // 4
    for index in range(4):
        start = index * group_size
        end = (index + 1) * group_size if index < 3 else len(pairs)
        rows.append(summarize(f"reference_radius_quartile_{index + 1}", pairs[start:end]))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT / "segmentation_error_summary.csv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {result_path.relative_to(ROOT)}")
    for row in rows:
        print(
            f"{row['stratum']}: n={row['n']}, "
            f"radius={row['reference_radius_min_mm']:.2f}–"
            f"{row['reference_radius_max_mm']:.2f} mm, "
            f"absolute SD={row['absolute_error_sd_mm']:.3f} mm, "
            f"relative SD={100 * row['relative_error_sd']:.2f}%, "
            f"log-ratio SD={row['log_ratio_sd']:.4f}"
        )


if __name__ == "__main__":
    main()
