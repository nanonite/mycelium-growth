#!/usr/bin/env python3
"""Measure scale stability of candidate colony-boundary protrusions (#62).

This analyzes candidate radial excursions on the 90 deduplicated human-labeled
hold-outs.  It resamples each existing predicted/reference mask to a fixed set
of widths and reports (a) predicted/reference boundary disagreement at each
width and (b) whether candidates found in the native-width mask survive at
coarser scales.  This is algorithmic scale stability, not temporal persistence
or validation of biological hyphal tips.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import measurement_error as me

DEFAULT_WORK = ROOT / "work_dirs" / "segformer-b0-repro" / "inference"
DEFAULT_OUTPUT = (
    ROOT / "work_dirs" / "segformer-b0-repro" / "tip-event-resolution"
)
DEFAULT_NOTE = ROOT / "notes" / "tip_event_validation.md"
WIDTHS = (300, 450, 900, 1800, 4608)
NATIVE_WIDTH = 4608
NATIVE_MATCH_DEG = 5.01  # one 5-degree detector bin, including quantization edge
REFERENCE_MATCH_DEG = me.PROTRUSION_MATCH_DEG


class ValidationError(RuntimeError):
    """An expected input or data-integrity failure."""


def load_at_width(path: Path, width: int) -> np.ndarray:
    """Resize binary mask exactly as #31 does, at a requested image width."""
    if not path.is_file():
        raise ValidationError(f"mask missing: {path}")
    raw = np.asarray(Image.open(path).convert("L"))
    binary = (raw > 0).astype(np.uint8) * 255
    image = Image.fromarray(binary, mode="L")
    source_w, source_h = image.size
    if source_w <= 0 or source_h <= 0:
        raise ValidationError(f"degenerate mask dimensions: {path}")
    height = max(1, round(source_h * width / source_w))
    if width != source_w:
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return (np.asarray(image) > 127).astype(np.uint8)


def match_event_indices(
    left: Sequence[Mapping[str, float]],
    right: Sequence[Mapping[str, float]],
    *,
    tolerance_deg: float,
) -> list[tuple[int, int]]:
    """Greedy one-to-one event matches by circular peak-angle distance."""
    pairs = []
    for li, l_event in enumerate(left):
        for ri, r_event in enumerate(right):
            distance = me._angle_distance(
                float(l_event["peak_angle"]), float(r_event["peak_angle"])
            )
            pairs.append((distance, li, ri))
    pairs.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[int, int]] = []
    for distance, li, ri in pairs:
        if distance > tolerance_deg:
            break
        if li in used_left or ri in used_right:
            continue
        used_left.add(li)
        used_right.add(ri)
        matches.append((li, ri))
    return matches


def _detect_at_width(path: Path, width: int) -> list[dict[str, float]]:
    mask = load_at_width(path, width)
    # Keep the prominence cutoff physically constant relative to the #31
    # 900-px working width, rather than using 3 px at every output width.
    return me.radial_protrusions(
        mask, abs_thresh=me.PROTRUSION_ABS_THRESH * width / me.WORKING_WIDTH
    )


def _provenance_and_pairs(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs = []
    for record in records:
        if record.get("source_set") not in me.HOLDOUT_SETS:
            continue
        if record.get("duplicate_role") != "representative":
            continue
        image_id = str(record["id"])
        source_set = str(record["source_set"])
        ref_path = me.SEG_ROOT / source_set / "mask" / f"{image_id}.png"
        pred_path = Path(str(record["mask_path"]))
        if not ref_path.is_file() or not pred_path.is_file():
            raise ValidationError(
                f"paired masks missing for {source_set}/{image_id}: "
                f"pred={pred_path.is_file()} ref={ref_path.is_file()}"
            )
        with Image.open(ref_path) as im:
            ref_width = im.width
        with Image.open(pred_path) as im:
            pred_width = im.width
        if ref_width != pred_width or ref_width != NATIVE_WIDTH:
            raise ValidationError(
                f"expected paired {NATIVE_WIDTH}px masks for {image_id}; "
                f"reference={ref_width}, predicted={pred_width}"
            )
        pairs.append(
            {
                "id": image_id,
                "source_set": source_set,
                "block": record.get("block", ""),
                "condition": record.get("condition", ""),
                "fit_status": record.get("fit_status", ""),
                "dish_disc_radius_900": record.get("dish_disc_radius_900"),
                "pred_path": pred_path,
                "ref_path": ref_path,
            }
        )
    return pairs


def analyze(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    images = _provenance_and_pairs(records)
    if not images:
        raise ValidationError("no deduplicated labeled hold-out mask pairs found")

    metrics: list[dict[str, Any]] = []
    persistence_rows: list[dict[str, Any]] = []
    native_event_counts: dict[str, int] = {"predicted": 0, "reference": 0}
    matched_native_counts: dict[str, dict[int, int]] = {
        "predicted": {width: 0 for width in WIDTHS},
        "reference": {width: 0 for width in WIDTHS},
    }
    all_pred_false = {width: 0 for width in WIDTHS}
    all_pred_tp = {width: 0 for width in WIDTHS}
    all_ref_missed = {width: 0 for width in WIDTHS}
    all_pred_events = {width: 0 for width in WIDTHS}
    all_ref_events = {width: 0 for width in WIDTHS}
    image_false_rates: dict[int, list[float]] = {width: [] for width in WIDTHS}

    for image in images:
        per_width: dict[int, dict[str, list[dict[str, float]]]] = {}
        for width in WIDTHS:
            pred_events = _detect_at_width(image["pred_path"], width)
            ref_events = _detect_at_width(image["ref_path"], width)
            per_width[width] = {"predicted": pred_events, "reference": ref_events}
            match = me.match_protrusions(
                pred_events, ref_events, match_deg=REFERENCE_MATCH_DEG
            )
            all_pred_events[width] += len(pred_events)
            all_ref_events[width] += len(ref_events)
            all_pred_tp[width] += int(match["tp"])
            all_pred_false[width] += int(match["fp"])
            all_ref_missed[width] += int(match["fn"])
            if pred_events:
                image_false_rates[width].append(float(match["false_event_rate"]))

        for kind in ("predicted", "reference"):
            native_events = per_width[NATIVE_WIDTH][kind]
            native_event_counts[kind] += len(native_events)
            width_matches: dict[int, dict[int, int]] = {}
            for width in WIDTHS:
                lower_or_equal_events = per_width[width][kind]
                matches = match_event_indices(
                    native_events,
                    lower_or_equal_events,
                    tolerance_deg=NATIVE_MATCH_DEG,
                )
                width_matches[width] = {native_i: target_i for native_i, target_i in matches}
                matched_native_counts[kind][width] += len(matches)
            for native_i, event in enumerate(native_events):
                present_widths = [
                    width for width in WIDTHS if native_i in width_matches[width]
                ]
                persistence_rows.append(
                    {
                        "id": image["id"],
                        "source_set": image["source_set"],
                        "event_mask": kind,
                        "native_event_index": native_i,
                        "native_peak_angle_deg": round(
                            float(event["peak_angle"]), 4
                        ),
                        "native_prominence_px": round(
                            float(event["prominence"]), 4
                        ),
                        "n_widths_detected": len(present_widths),
                        "width_fraction": len(present_widths) / len(WIDTHS),
                        "detected_widths_px": ";".join(map(str, present_widths)),
                    }
                )

        for width in WIDTHS:
            match = me.match_protrusions(
                per_width[width]["predicted"],
                per_width[width]["reference"],
                match_deg=REFERENCE_MATCH_DEG,
            )
            n_pred = len(per_width[width]["predicted"])
            n_ref = len(per_width[width]["reference"])
            metrics.append(
                {
                    "id": image["id"],
                    "source_set": image["source_set"],
                    "block": image["block"],
                    "resolution_width_px": width,
                    "resolution_height_px": round(3456 * width / NATIVE_WIDTH),
                    "abs_prominence_cutoff_px": round(
                        me.PROTRUSION_ABS_THRESH * width / me.WORKING_WIDTH, 4
                    ),
                    "predicted_candidates": n_pred,
                    "reference_candidates": n_ref,
                    "matched_candidates": int(match["tp"]),
                    "mask_disagreement_fp": int(match["fp"]),
                    "mask_disagreement_fn": int(match["fn"]),
                    "mask_disagreement_false_fraction": match["false_event_rate"],
                    "prediction_native_event_retention": (
                        len(
                            match_event_indices(
                                per_width[NATIVE_WIDTH]["predicted"],
                                per_width[width]["predicted"],
                                tolerance_deg=NATIVE_MATCH_DEG,
                            )
                        )
                        / len(per_width[NATIVE_WIDTH]["predicted"])
                        if per_width[NATIVE_WIDTH]["predicted"]
                        else None
                    ),
                    "reference_native_event_retention": (
                        len(
                            match_event_indices(
                                per_width[NATIVE_WIDTH]["reference"],
                                per_width[width]["reference"],
                                tolerance_deg=NATIVE_MATCH_DEG,
                            )
                        )
                        / len(per_width[NATIVE_WIDTH]["reference"])
                        if per_width[NATIVE_WIDTH]["reference"]
                        else None
                    ),
                }
            )

    resolution_summary: list[dict[str, Any]] = []
    for width in WIDTHS:
        tp = all_pred_tp[width]
        fp = all_pred_false[width]
        fn = all_ref_missed[width]
        resolution_summary.append(
            {
                "width_px": width,
                "height_px": round(3456 * width / NATIVE_WIDTH),
                "abs_prominence_cutoff_px": round(
                    me.PROTRUSION_ABS_THRESH * width / me.WORKING_WIDTH, 4
                ),
                "predicted_candidates": all_pred_events[width],
                "reference_candidates": all_ref_events[width],
                "matched_candidates": tp,
                "mask_disagreement_fp": fp,
                "mask_disagreement_fn": fn,
                "mask_disagreement_false_fraction": fp / (tp + fp) if tp + fp else None,
                "reference_event_recall": tp / (tp + fn) if tp + fn else None,
                "median_per_image_mask_disagreement_false_fraction": (
                    statistics.median(image_false_rates[width])
                    if image_false_rates[width]
                    else None
                ),
                "predicted_native_event_retention": (
                    matched_native_counts["predicted"][width]
                    / native_event_counts["predicted"]
                    if native_event_counts["predicted"]
                    else None
                ),
                "reference_native_event_retention": (
                    matched_native_counts["reference"][width]
                    / native_event_counts["reference"]
                    if native_event_counts["reference"]
                    else None
                ),
            }
        )

    radius_values = []
    for image in images:
        radius = image.get("dish_disc_radius_900")
        if image.get("fit_status") == "ok" and radius is not None:
            radius = float(radius)
            if radius > 0:
                radius_native = radius * NATIVE_WIDTH / me.WORKING_WIDTH
                radius_values.append(90_000.0 / (2 * radius_native))

    return {
        "task": 62,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "population": {
            "n_deduplicated_labeled_holdout_pairs": len(images),
            "n_reference_mask_events_native": native_event_counts["reference"],
            "n_predicted_mask_events_native": native_event_counts["predicted"],
        },
        "protocol": {
            "widths_px": list(WIDTHS),
            "native_width_px": NATIVE_WIDTH,
            "event_detector": "#31 radial protrusion detector",
            "relative_prominence_threshold": me.PROTRUSION_REL_THRESH,
            "absolute_prominence_threshold_at_900px": me.PROTRUSION_ABS_THRESH,
            "absolute_threshold_scaled_with_width": True,
            "predicted_reference_match_angle_deg": REFERENCE_MATCH_DEG,
            "native_scale_event_match_angle_deg": NATIVE_MATCH_DEG,
            "scale_persistence_definition": (
                "fraction of five resampled widths where a native-width event "
                "has a one-to-one event match within one 5-degree detector bin"
            ),
            "nominal_native_sampling_um_per_px": (
                {
                    "median": statistics.median(radius_values),
                    "p10": float(np.percentile(radius_values, 10)),
                    "p90": float(np.percentile(radius_values, 90)),
                    "n_calibrated_images": len(radius_values),
                }
                if radius_values
                else None
            ),
            "temporal_persistence_estimable": False,
            "biological_tip_validation_possible_from_these_masks": False,
        },
        "resolution_summary": resolution_summary,
        "image_resolution_metrics": metrics,
        "event_persistence": persistence_rows,
        "interpretation": [
            "Cross-resolution survival describes algorithmic stability under resampling of the same image; it is not temporal persistence.",
            "Predicted/reference false-event and miss counts measure mask-boundary disagreement only, not false or missed biological tips.",
            "The available labels are colony masks, not independently annotated hyphal tips; no width in this analysis supports calling an event a biological tip.",
            "Downsampling the same source images does not establish the optical resolution or recover detail absent from the released images.",
        ],
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_note(evidence: Mapping[str, Any]) -> str:
    population = evidence["population"]
    protocol = evidence["protocol"]
    lines = [
        "# Candidate boundary-event resolution and scale stability (#62)",
        "",
        "This analysis measures how radial boundary candidates change when the same labeled hold-out masks are resampled. It does not infer time or validate biological hyphal tips.",
        "",
        f"Population: {population['n_deduplicated_labeled_holdout_pairs']} de-duplicated image pairs, with {population['n_reference_mask_events_native']} reference-mask and {population['n_predicted_mask_events_native']} predicted-mask candidates at native width.",
        "",
        "## Method",
        "",
        f"Each predicted/reference pair was resampled to widths {', '.join(map(str, protocol['widths_px']))} px using the same bilinear resize and 127 threshold used by #31. The detector kept its 5% relative prominence threshold; its 3 px absolute threshold at 900 px was scaled with width to preserve the same nominal image-space cutoff. Candidate masks were matched within 15° at each width.",
        "",
        "Scale persistence is the fraction of the five widths where a candidate detected in the native 4608 px mask has a one-to-one candidate match within one 5° detector bin. This is algorithmic stability under resampling of a single image; it is not persistence over time.",
        "",
        "## Results",
        "",
        "| Width (px) | Predicted candidates | Reference candidates | Matched | Mask disagreement FP | Mask disagreement FN | FP fraction | Reference recall | Predicted native event retention | Reference native event retention |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evidence["resolution_summary"]:
        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        lines.append(
            f"| {row['width_px']} | {row['predicted_candidates']} | {row['reference_candidates']} | {row['matched_candidates']} | {row['mask_disagreement_fp']} | {row['mask_disagreement_fn']} | {fmt(row['mask_disagreement_false_fraction'])} | {fmt(row['reference_event_recall'])} | {fmt(row['predicted_native_event_retention'])} | {fmt(row['reference_native_event_retention'])} |"
        )
    persistence = evidence["event_persistence"]
    persistence_summaries = []
    for kind in ("predicted", "reference"):
        events = [row for row in persistence if row["event_mask"] == kind]
        all_scale = sum(row["n_widths_detected"] == len(WIDTHS) for row in events)
        persistence_summaries.append(
            f"{all_scale}/{len(events)} {kind} native candidates ({all_scale / len(events):.1%}) "
            "were detected at all five widths"
            if events
            else f"no {kind} native candidates were detected"
        )
    sampling = protocol.get("nominal_native_sampling_um_per_px")
    lines.extend(
        [
            "",
            f"At native width, the calibrated 90 mm dish fit implies a nominal image sampling pitch of {sampling['median']:.1f} µm/px (10th–90th percentile {sampling['p10']:.1f}–{sampling['p90']:.1f}, n={sampling['n_calibrated_images']}). This is geometric sampling derived from the dish, not a measured optical resolution.",
            "",
            "## Interpretation",
            "",
            "Across all five widths, " + "; ".join(persistence_summaries) + ".",
            "",
            "The resampling sweep quantifies whether a geometric boundary candidate is stable at a chosen working width and whether the predicted mask agrees with the human colony mask. The reference mask is not an independent hyphal-tip annotation, so a matched candidate is not evidence of a real hyphal tip.",
            "",
            "No resolution in this dataset can be designated as the point where an event may be called a biological hyphal tip: the labels describe colony extent, and there is no independent microscopic annotation or temporal sequence. MyceliumSeg supports candidate boundary morphology only. The available-resolution question therefore remains biologically unresolvable from this release; #62 is complete as a bounded mask/resampling validation, with tip claims explicitly excluded.",
            "",
            "The 900 px results are the closest comparison with the earlier #31 false-event estimate. Differences arise from applying the detector at multiple scales while preserving its nominal absolute cutoff; these false and missed events are disagreements with the reference mask, not biological event error rates.",
            "",
            "Artifacts: `tip_event_resolution.csv`, `tip_event_persistence.csv`, and `tip_event_resolution_evidence.json` in `work_dirs/segformer-b0-repro/tip-event-resolution/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run",), nargs="?", default="run")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    args = parser.parse_args(argv)
    try:
        records = me._load_records(args.work)
        evidence = analyze(records)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "tip_event_resolution_evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_csv(args.output / "tip_event_resolution.csv", evidence["resolution_summary"])
        write_csv(
            args.output / "tip_event_image_metrics.csv",
            evidence["image_resolution_metrics"],
        )
        write_csv(args.output / "tip_event_persistence.csv", evidence["event_persistence"])
        args.note.parent.mkdir(parents=True, exist_ok=True)
        args.note.write_text(build_note(evidence), encoding="utf-8")
        print(f"wrote {args.note}")
        print(f"wrote resolution summary and event-level persistence under {args.output}")
        for row in evidence["resolution_summary"]:
            fpr = row["mask_disagreement_false_fraction"]
            recall = row["reference_event_recall"]
            retention = row["predicted_native_event_retention"]
            print(
                f"{row['width_px']:>4}px: pred={row['predicted_candidates']} "
                f"ref={row['reference_candidates']} fp_fraction="
                f"{fpr if fpr is not None else float('nan'):.3f} "
                f"ref_recall={recall if recall is not None else float('nan'):.3f} "
                f"native_retention={retention if retention is not None else float('nan'):.3f}"
            )
        return 0
    except (ValidationError, me.MeasurementError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
