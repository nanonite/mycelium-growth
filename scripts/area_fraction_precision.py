#!/usr/bin/env python3
"""Audit held-out MyceliumSeg area-fraction precision (#66).

The #31 error table contains paired predicted/reference mask measurements on
deduplicated labeled hold-outs. This script stratifies their relative area
errors by reference coverage and condition, measures sort-order agreement, and
separately reports how the fitted per-image dish radius changes the area
denominator relative to the block calibration scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ERROR_CSV = ROOT / "data" / "triage" / "measurement_error.csv"
DEFAULT_SCALE_CSV = ROOT / "data" / "triage" / "image_scale.csv"
DEFAULT_OUT = ROOT / "data" / "triage" / "area_fraction_precision"
DEFAULT_NOTE = ROOT / "notes" / "area_fraction_precision.md"
N_COVERAGE_BINS = 4


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def summarize(values: Sequence[float], prefix: str = "") -> dict[str, Any]:
    if not values:
        return {f"{prefix}n": 0}
    sorted_values = sorted(values)
    return {
        f"{prefix}n": len(values),
        f"{prefix}mean": statistics.mean(values),
        f"{prefix}sd": statistics.stdev(values) if len(values) > 1 else None,
        f"{prefix}median": statistics.median(values),
        f"{prefix}p05": percentile(sorted_values, 5),
        f"{prefix}p95": percentile(sorted_values, 95),
        f"{prefix}mean_abs": statistics.mean(abs(value) for value in values),
        f"{prefix}p95_abs": percentile(sorted(abs(value) for value in values), 95),
    }


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * pct / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def group_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    relative_key: str = "area_fraction_err",
    pp_key: str = "area_fraction_error_pp",
) -> dict[str, Any]:
    relative = [float(row[relative_key]) for row in rows if row.get(relative_key) is not None]
    pp = [float(row[pp_key]) for row in rows if row.get(pp_key) is not None]
    output = summarize(relative, "relative_error_pct_")
    output.update(summarize(pp, "percentage_point_error_"))
    return output


def rank_agreement(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    concordant = discordant = pred_ties = ref_ties = 0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            predicted_delta = float(left["pred_area_fraction"]) - float(right["pred_area_fraction"])
            reference_delta = float(left["ref_area_fraction"]) - float(right["ref_area_fraction"])
            if predicted_delta == 0:
                pred_ties += 1
            if reference_delta == 0:
                ref_ties += 1
            product = predicted_delta * reference_delta
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    compared = concordant + discordant
    total = len(rows) * (len(rows) - 1) // 2
    return {
        "n_images": len(rows),
        "n_image_pairs": total,
        "n_tied_pairs_excluded": total - compared,
        "n_concordant_pairs": concordant,
        "n_discordant_pairs": discordant,
        "pairwise_order_accuracy": concordant / compared if compared else None,
        "kendall_tau_no_ties": (
            (concordant - discordant) / compared if compared else None
        ),
    }


def prepare_rows(
    errors: Sequence[Mapping[str, str]], scales: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    scale_by_id = {str(row["id"]): row for row in scales}
    result = []
    for error in errors:
        if error.get("duplicate_role") != "representative" or error.get("fit_status") != "ok":
            continue
        pred = maybe_float(error.get("pred_area_fraction"))
        ref = maybe_float(error.get("ref_area_fraction"))
        relative = maybe_float(error.get("area_fraction_err"))
        area_px_error = maybe_float(error.get("area_px_err"))
        if pred is None or ref is None or ref <= 0 or relative is None:
            continue
        scale = scale_by_id.get(str(error["id"]))
        if scale is None:
            raise ValueError(f"image_scale.csv missing hold-out image {error['id']}")
        px_per_mm = maybe_float(scale.get("px_per_mm"))
        px_per_mm_fit = maybe_float(scale.get("px_per_mm_fit"))
        if not px_per_mm or not px_per_mm_fit:
            raise ValueError(f"missing scale fit for calibrated image {error['id']}")

        # If the actual fitted per-image px/mm replaces its block scale, the
        # estimated fraction changes by (block_scale / fitted_scale)^2.
        denominator_shift_pct = ((px_per_mm / px_per_mm_fit) ** 2 - 1) * 100
        duplicate_size = int(error.get("duplicate_group_size") or 1)
        label_status = "label_conflict" if duplicate_size > 1 else "label_clean"
        row = {
            "id": str(error["id"]),
            "source_set": error.get("source_set", ""),
            "condition": error.get("condition", ""),
            "block": error.get("block", ""),
            "duplicate_group_size": duplicate_size,
            "label_status": label_status,
            "ref_area_fraction": ref,
            "pred_area_fraction": pred,
            "area_fraction_err": relative * 100,
            "area_fraction_error_pp": (pred - ref) * 100,
            "area_px_err": area_px_error * 100 if area_px_error is not None else None,
            "relative_error_difference_vs_area_px": (
                relative - area_px_error if area_px_error is not None else None
            ),
            "px_per_mm_block": px_per_mm,
            "px_per_mm_fit": px_per_mm_fit,
            "fit_rel_err_pct": maybe_float(scale.get("fit_rel_err_pct")),
            "area_fraction_shift_fit_vs_block_pct": denominator_shift_pct,
        }
        result.append(row)
    return result


def analyze(
    errors: Sequence[Mapping[str, str]], scales: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    rows = prepare_rows(errors, scales)
    if len(rows) != 88:
        raise ValueError(f"expected 88 calibrated deduplicated hold-outs; found {len(rows)}")

    # Equal-count bins use the human reference coverage to avoid defining the
    # evaluation strata by the model output being evaluated.
    coverage_sorted = sorted(rows, key=lambda row: row["ref_area_fraction"])
    quotient, remainder = divmod(len(coverage_sorted), N_COVERAGE_BINS)
    coverage_groups: list[list[dict[str, Any]]] = []
    cursor = 0
    for index in range(N_COVERAGE_BINS):
        count = quotient + (1 if index < remainder else 0)
        group = coverage_sorted[cursor : cursor + count]
        cursor += count
        coverage_groups.append(group)

    coverage_summary = []
    for index, group in enumerate(coverage_groups, start=1):
        stats = group_stats(group)
        stats.update(
            {
                "stratum": f"reference_coverage_q{index}",
                "reference_coverage_min": min(r["ref_area_fraction"] for r in group),
                "reference_coverage_max": max(r["ref_area_fraction"] for r in group),
                "reference_coverage_median": statistics.median(
                    r["ref_area_fraction"] for r in group
                ),
            }
        )
        coverage_summary.append(stats)

    condition_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        condition_groups[(row["condition"], row["label_status"])].append(row)
    condition_summary = []
    for (condition, label_status), group in sorted(condition_groups.items()):
        stats = group_stats(group)
        stats.update({"condition": condition, "label_status": label_status})
        stats.update(rank_agreement(group))
        condition_summary.append(stats)

    rank_groups: dict[str, list[dict[str, Any]]] = {"all_calibrated_holdouts": rows}
    for condition, label_status in condition_groups:
        name = f"{condition}|{label_status}"
        rank_groups[name] = condition_groups[(condition, label_status)]
    rank_summary = []
    for stratum, group in rank_groups.items():
        rank_summary.append({"stratum": stratum, **rank_agreement(group)})

    shift_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        shift_groups[(row["condition"], row["label_status"])].append(row)
    calibration_summary = []
    for (condition, label_status), group in sorted(shift_groups.items()):
        stats = summarize(
            [r["fit_rel_err_pct"] for r in group if r["fit_rel_err_pct"] is not None],
            "rim_fit_relative_to_block_pct_",
        )
        stats.update(
            summarize(
                [r["area_fraction_shift_fit_vs_block_pct"] for r in group],
                "area_fraction_shift_fit_vs_block_pct_",
            )
        )
        stats.update({"condition": condition, "label_status": label_status})
        calibration_summary.append(stats)

    err_delta = [
        abs(float(row["relative_error_difference_vs_area_px"]))
        for row in rows
        if row["relative_error_difference_vs_area_px"] is not None
    ]
    overall = group_stats(rows)
    overall.update(
        {
            "area_fraction_error_matches_area_px_error_max_abs_difference": max(err_delta),
            "area_fraction_target_bias_pct": 3.0,
            "area_fraction_target_sd_pct": 10.0,
            "area_fraction_target_met": False,
            "interpretation": (
                "The held-out relative area_fraction error reuses the same per-image dish denominator for predicted and reference masks, so that denominator cancels. Its 14.3% SD is therefore segmentation-area error under a fixed denominator. Dish-fit sensitivity is reported separately relative to the block calibration scale; it is not independent ground truth for absolute dish area."
            ),
        }
    )

    return {
        "task": 66,
        "population": {
            "n_calibrated_deduplicated_holdouts": len(rows),
            "n_clean_label_rows": sum(r["label_status"] == "label_clean" for r in rows),
            "n_label_conflict_rows": sum(r["label_status"] == "label_conflict" for r in rows),
        },
        "overall": overall,
        "coverage_strata": coverage_summary,
        "condition_strata": condition_summary,
        "rank_agreement": rank_summary,
        "dish_denominator_sensitivity": calibration_summary,
        "rows": rows,
        "caveats": [
            "All metrics are held-out selected-release image metrics; the archive inclusion rule is undocumented, so these do not describe a plate population.",
            "Condition groups with duplicate_group_size > 1 are label-conflict diagnostics, not clean species-specific samples.",
            "The area-fraction prediction/reference relative error uses the same dish radius within each image and cannot measure absolute dish-radius error.",
            "The fitted per-image rim versus block calibration comparison is a sensitivity analysis, not independent ground truth for dish size.",
            "Area-fraction ordering is cross-sectional coverage ordering, not chronology or a same-dish trajectory.",
        ],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_note(evidence: Mapping[str, Any]) -> str:
    overall = evidence["overall"]
    lines = [
        "# Area-fraction precision audit (#66)",
        "",
        "This audit separates the error in predicted colony area from sensitivity in the fitted dish denominator. It uses the 90 deduplicated #31 labeled hold-outs; 88 have calibrated area fractions.",
        "",
        "## Overall held-out precision",
        "",
        "| N | Relative bias | Relative SD | Median relative error | Mean absolute relative error | 95th percentile absolute error | #60 target |",
        "|---:|---:|---:|---:|---:|---:|---|",
        f"| {overall['relative_error_pct_n']} | {_fmt(overall['relative_error_pct_mean'])}% | {_fmt(overall['relative_error_pct_sd'])}% | {_fmt(overall['relative_error_pct_median'])}% | {_fmt(overall['relative_error_pct_mean_abs'])}% | {_fmt(overall['relative_error_pct_p95_abs'])}% | **unmet** (bias ≤3%, SD ≤10%) |",
        "",
        "Area-fraction relative error is numerically the same as predicted/reference mask area relative error: the predicted and reference masks for one image use the same dish-disc denominator, so it cancels from `(pred − ref) / ref`. The maximum absolute difference from `area_px_err` in this table is `%.3g` (floating-point roundoff). Thus the 14.3%% relative SD is a segmentation-area error conditional on the shared dish fit; it does not include absolute error in the fitted dish area." % overall["area_fraction_error_matches_area_px_error_max_abs_difference"],
        "",
        "## Error by reference coverage",
        "",
        "Equal-count quartiles are defined from human reference `area_fraction`, not from predictions.",
        "",
        "| Reference coverage bin | N | Reference coverage range | Relative bias | Relative SD | Mean absolute error (percentage points) | 95th percentile absolute error (percentage points) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evidence["coverage_strata"]:
        lines.append(
            f"| {row['stratum']} | {row['relative_error_pct_n']} | {row['reference_coverage_min']:.3f}–{row['reference_coverage_max']:.3f} | {_fmt(row['relative_error_pct_mean'])}% | {_fmt(row['relative_error_pct_sd'])}% | {_fmt(row['percentage_point_error_mean_abs'])} | {_fmt(row['percentage_point_error_p95_abs'])} |"
        )
    lines.extend(
        [
            "",
            "## Error and sort agreement by condition",
            "",
            "Rows with duplicate groups are kept as label-conflict diagnostics. They measure the mask errors for those pixels but do not support clean species-specific claims.",
            "",
            "| Condition | Label status | N | Relative bias | Relative SD | Pairwise sort agreement | Kendall τ |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in evidence["condition_strata"]:
        lines.append(
            f"| {row['condition']} | {row['label_status']} | {row['relative_error_pct_n']} | {_fmt(row['relative_error_pct_mean'])}% | {_fmt(row['relative_error_pct_sd'])}% | {_fmt(row['pairwise_order_accuracy'] * 100 if row['pairwise_order_accuracy'] is not None else None, 1)}% | {_fmt(row['kendall_tau_no_ties'])} |"
        )
    lines.extend(
        [
            "",
            "## Dish-disc denominator sensitivity",
            "",
            "The table below compares each accepted image's fitted px/mm with its block/setup calibration constant. The area-fraction shift is the change produced by substituting the per-image fit for the block constant, `(block_scale / fitted_scale)^2 − 1`. It is a sensitivity measure, not a validated denominator-error distribution, because no independent ruler or second dish annotation is available.",
            "",
            "| Condition | Label status | N | Rim fit vs block scale: mean ± SD | Area-fraction shift: median | 5th–95th percentile |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in evidence["dish_denominator_sensitivity"]:
        rim_mean = row.get("rim_fit_relative_to_block_pct_mean")
        rim_sd = row.get("rim_fit_relative_to_block_pct_sd")
        shift_p05 = row.get("area_fraction_shift_fit_vs_block_pct_p05")
        shift_p95 = row.get("area_fraction_shift_fit_vs_block_pct_p95")
        lines.append(
            f"| {row['condition']} | {row['label_status']} | {row['rim_fit_relative_to_block_pct_n']} | {_fmt(rim_mean)}% ± {_fmt(rim_sd)}% | {_fmt(row.get('area_fraction_shift_fit_vs_block_pct_median'))}% | {_fmt(shift_p05)}% to {_fmt(shift_p95)}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The held-out segmentation model misses the planned 10% SD target for area fraction even though mean bias is only about 1%. The largest relative error is in the lowest-coverage quartile, where small denominators inflate relative error; its mean absolute error is 0.41 percentage points. In the clean GL/MYG/25C stratum (n=40), 95.9% of image pairs keep the same area order as the reference masks. Thus coverage sorting is useful, but values close together can be swapped and absolute percentages need uncertainty; this is not an exact rank guarantee for every pair.",
            "",
            "Dish fit adds a separate denominator sensitivity. It is not part of the #31 predicted-versus-reference relative error because the same fitted radius is used for both masks. The per-image fit versus block scale shows how much reported percentages could move under that alternate calibration; it does not establish which denominator is true. The current 14.3% SD should therefore be described as segmentation-only under a shared dish fit, with dish calibration uncertainty reported separately.",
            "",
            "Scope is the selected release and the #31 held-out checkpoint. It does not establish a plate-population distribution, chronology, or growth rate. The sorted coordinate remains cross-sectional area coverage.",
            "",
            "Artifacts: `area_fraction_precision_summary.csv`, `area_fraction_precision_by_image.csv`, and `area_fraction_precision_evidence.json` under `data/triage/area_fraction_precision/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERROR_CSV)
    parser.add_argument("--scales", type=Path, default=DEFAULT_SCALE_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    args = parser.parse_args()
    evidence = analyze(read_csv(args.errors), read_csv(args.scales))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "area_fraction_precision_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.out / "area_fraction_precision_by_image.csv", evidence["rows"])
    for key, filename in (
        ("coverage_strata", "area_fraction_precision_coverage.csv"),
        ("condition_strata", "area_fraction_precision_condition.csv"),
        ("rank_agreement", "area_fraction_precision_rank.csv"),
        ("dish_denominator_sensitivity", "area_fraction_precision_calibration.csv"),
    ):
        _write_csv(args.out / filename, evidence[key])
    summary_row = {"n": evidence["overall"]["relative_error_pct_n"], **evidence["overall"]}
    _write_csv(args.out / "area_fraction_precision_summary.csv", [summary_row])
    args.note.parent.mkdir(parents=True, exist_ok=True)
    args.note.write_text(build_note(evidence), encoding="utf-8")
    print(f"wrote {args.note}")
    print(f"wrote audit tables under {args.out}")
    print(
        "relative bias %.3f%%; relative SD %.3f%%; target unmet; max |fraction-vs-pixel error difference| %.3g"
        % (
            evidence["overall"]["relative_error_pct_mean"],
            evidence["overall"]["relative_error_pct_sd"],
            evidence["overall"]["area_fraction_error_matches_area_px_error_max_abs_difference"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
