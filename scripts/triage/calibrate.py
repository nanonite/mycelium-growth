#!/usr/bin/env python3
"""Calibrate px -> mm per condition block from the 90-mm dish rim (Chainlink #25).

PLAN.md's Phase 1 checklist needs *mask area in px^2 -> mm^2*, and #23 showed
that one global constant is wrong by up to ~36%: the same 90-mm dish is fitted
at r/W = 0.302, 0.331 or 0.222 depending on the condition block.  This driver
turns the per-image rim fits that #23 published into the constant the
measurement pipeline should actually use.

For every condition block it takes the fitted rim scales, splits them into the
camera setups they contain (``dish_scale.scale_modes``), and reports each
setup's median -- a median, because ~3% of the individual fits lock onto the
colony edge or an inner reflection instead of the rim.  Every image also gets
one row saying which constant applies to it and whether its own fit agrees.

Outputs
    data/triage/calibration.csv    one row per block and scale mode
    data/triage/image_scale.csv    one row per image: block, group, px/mm

Usage
    .venv/bin/python scripts/triage/calibrate.py
    .venv/bin/python scripts/triage/calibrate.py --tracks ... --out ...
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

try:  # executed as a script: this directory is already on sys.path
    from dish_scale import (
        FIT_TOLERANCE,
        REFERENCE_WIDTH,
        STATUS_OFF_SCALE,
        STATUS_OK,
        ScaleSummary,
        assign_scales,
        block_modes,
        condition_block,
        scale_summary,
    )
except ImportError:  # imported as scripts.triage.calibrate
    from .dish_scale import (  # type: ignore[no-redef]
        FIT_TOLERANCE,
        REFERENCE_WIDTH,
        STATUS_OFF_SCALE,
        STATUS_OK,
        ScaleSummary,
        assign_scales,
        block_modes,
        condition_block,
        scale_summary,
    )

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRACKS = ROOT / "data" / "triage" / "dish_tracks.csv"
DEFAULT_OUT = ROOT / "data" / "triage" / "calibration.csv"
DEFAULT_PER_IMAGE = ROOT / "data" / "triage" / "image_scale.csv"

CALIBRATION_FIELDS = [
    "block",
    "scale_group",
    "dominant",
    "n_images",
    "n_off_scale",
    "r_over_w_median",
    "r_over_w_iqr",
    "px_per_mm",
    "px_per_mm_p25",
    "px_per_mm_p75",
    "spread_pct",
    "global_err_pct",
    "reference_width",
]

IMAGE_FIELDS = [
    "id",
    "archive",
    "block",
    "scale_group",
    "px_per_mm",
    "px_per_mm_fit",
    "fit_rel_err_pct",
    "fit_status",
]


def read_tracks(path: Path) -> tuple[list[dict[str, str]], int]:
    """Load the per-image rim fits, counting the rows that carry no fit."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    usable = [r for r in rows if r.get("dish_r_over_w") and r.get("pixels_per_mm")]
    return usable, len(rows) - len(usable)


def calibrate(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict], float]:
    """Build the calibration table and the per-image table.

    Returns ``(calibration_rows, image_rows, global_px_per_mm)``.  The global
    constant is the median over the whole release; it exists only so every
    calibration row can say how wrong that number would have been here.
    """
    per_block: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        per_block[condition_block(row["archive"], row["entry"])].append(row)
    global_px_per_mm = statistics.median(float(r["pixels_per_mm"]) for r in rows)

    calibration_rows: list[dict] = []
    image_rows: list[dict] = []
    for block in sorted(per_block):
        block_rows = per_block[block]
        scales = [float(r["dish_r_over_w"]) for r in block_rows]
        fits = [float(r["pixels_per_mm"]) for r in block_rows]
        modes = block_modes(scales)
        assignments = assign_scales(scales, modes)
        summaries = _group_summaries(scales, fits, assignments, len(modes))
        statuses = _fit_statuses(fits, assignments, summaries)
        calibration_rows.extend(
            _calibration_rows(block, assignments, summaries, statuses, global_px_per_mm)
        )
        image_rows.extend(
            _image_rows(block, block_rows, fits, assignments, summaries, statuses)
        )
    return calibration_rows, image_rows, global_px_per_mm


def _group_summaries(
    scales: list[float],
    fits: list[float],
    assignments: list[tuple[int, str]],
    n_groups: int,
) -> list[ScaleSummary]:
    """Summarise each group over the images whose own fit agrees with it.

    Off-scale images are assigned to a group but do not get a vote on its
    constant -- that is the point of flagging them.  ``sample`` guards the
    degenerate case where a group has no agreeing image at all.
    """
    summaries: list[ScaleSummary] = []
    for group in range(n_groups):
        members = [i for i, (g, _) in enumerate(assignments) if g == group]
        sample = [i for i in members if assignments[i][1] == STATUS_OK] or members
        summaries.append(
            scale_summary([scales[i] for i in sample], [fits[i] for i in sample])
        )
    return summaries


def _fit_statuses(
    fits: list[float],
    assignments: list[tuple[int, str]],
    summaries: list[ScaleSummary],
) -> list[str]:
    """Decide each image's reported status from its own fit vs its constant.

    The group assignment only picks *which* constant applies; whether the image
    is allowed to use its own number is decided here, against the number the
    calibration actually publishes, so the two columns of ``image_scale.csv``
    can never contradict each other.
    """
    return [
        (
            STATUS_OK
            if abs(fit / summaries[group].px_per_mm - 1.0) <= FIT_TOLERANCE
            else STATUS_OFF_SCALE
        )
        for fit, (group, _) in zip(fits, assignments)
    ]


def _calibration_rows(
    block: str,
    assignments: list[tuple[int, str]],
    summaries: list[ScaleSummary],
    statuses: list[str],
    global_px_per_mm: float,
) -> list[dict]:
    rows = []
    for index, summary in enumerate(summaries):
        members = sum(1 for group, _ in assignments if group == index)
        agreeing = sum(
            1
            for i, (group, _) in enumerate(assignments)
            if group == index and statuses[i] == STATUS_OK
        )
        rows.append(
            {
                "block": block,
                "scale_group": index + 1,
                "dominant": int(index == 0),
                "n_images": members,
                "n_off_scale": members - agreeing,
                **summary.as_row(),
                "global_err_pct": round(
                    (global_px_per_mm / summary.px_per_mm - 1) * 100, 2
                ),
                "reference_width": REFERENCE_WIDTH,
            }
        )
    return rows


def _image_rows(
    block: str,
    block_rows: list[dict[str, str]],
    fits: list[float],
    assignments: list[tuple[int, str]],
    summaries: list[ScaleSummary],
    statuses: list[str],
) -> list[dict]:
    """One row per image: the constant to use with it, and how well it agrees."""
    rows = []
    for row, fit, (group, _), status in zip(block_rows, fits, assignments, statuses):
        constant = summaries[group].px_per_mm
        rows.append(
            {
                "id": row["id"],
                "archive": row["archive"],
                "block": block,
                "scale_group": group + 1,
                "px_per_mm": constant,
                "px_per_mm_fit": round(fit, 4),
                "fit_rel_err_pct": round((fit / constant - 1.0) * 100.0, 2),
                "fit_status": status,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_report(
    calibration_rows: list[dict], image_rows: list[dict], skipped: int
) -> None:
    print(
        f"{'block':<16}{'grp':>4}{'dom':>5}{'images':>8}{'off':>6}"
        f"{'px/mm':>9}{'IQR%':>7}{'global constant error':>24}"
    )
    for row in calibration_rows:
        print(
            f"{row['block']:<16}{row['scale_group']:>4}{row['dominant']:>5}"
            f"{row['n_images']:>8}{row['n_off_scale']:>6}"
            f"{row['px_per_mm']:>9.4f}{row['spread_pct']:>7.2f}"
            f"{row['global_err_pct']:>+23.2f}%"
        )
    worst = max(abs(row["global_err_pct"]) for row in calibration_rows)
    off = sum(1 for row in image_rows if row["fit_status"] != STATUS_OK)
    blocks = {row["block"] for row in calibration_rows}
    print(
        f"\nblocks: {len(blocks)}, scale modes: {len(calibration_rows)}, "
        f"images: {len(image_rows)}"
    )
    print(f"worst error of one global constant: {worst:.1f}%")
    print(
        f"images whose own rim fit disagrees with their constant by "
        f">{FIT_TOLERANCE * 100:.0f}%: {off}"
    )
    if skipped:
        print(f"skipped {skipped} rows without a rim fit", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks",
        type=Path,
        default=DEFAULT_TRACKS,
        help="per-image rim fits written by dish_tracks.py",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="per-block calibration CSV"
    )
    parser.add_argument(
        "--per-image", type=Path, default=DEFAULT_PER_IMAGE, help="per-image scale CSV"
    )
    args = parser.parse_args(argv)

    if not args.tracks.exists():
        parser.error(f"--tracks needs {args.tracks} to exist")
    rows, skipped = read_tracks(args.tracks)
    if not rows:
        parser.error(f"no usable rim fits in {args.tracks}")

    calibration_rows, image_rows, global_px_per_mm = calibrate(rows)
    write_csv(args.out, calibration_rows, CALIBRATION_FIELDS)
    write_csv(args.per_image, image_rows, IMAGE_FIELDS)

    print(
        f"global constant (right for one block only): "
        f"{global_px_per_mm:.4f} px/mm at {REFERENCE_WIDTH} px wide\n"
    )
    print_report(calibration_rows, image_rows, skipped)
    print(f"\nwrote {args.out} and {args.per_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
