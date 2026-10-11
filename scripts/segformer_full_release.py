#!/usr/bin/env python3
"""Infer masks and area coverage for every image in the MyceliumSeg release.

Streams images from the released ZIP archives, writes one predicted 0/1 mask per
image, and exports a provenance-rich table sorted by per-image dish coverage.
The sort is cross-sectional area coverage, never elapsed time or growth order.

Example (use the all-labeled fit; retain #28 as the independent benchmark):

    nix develop .#gpu --command env PYTHONPATH=scripts/segformer_shim \\
        python scripts/segformer_full_release.py run \\
        --config work_dirs/segformer-b0-repro/full-all-labeled-all-unique-20261008/config.py \\
        --checkpoint work_dirs/segformer-b0-repro/full-all-labeled-all-unique-20261008/train/iter_50000.pth
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import segformer_inference as inference
import segformer_repro as repro

DEFAULT_CONFIG = (
    REPO
    / "work_dirs/segformer-b0-repro/full-all-labeled-all-unique-20261008/config.py"
)
DEFAULT_CHECKPOINT = (
    REPO
    / "work_dirs/segformer-b0-repro/full-all-labeled-all-unique-20261008/train/iter_50000.pth"
)
DEFAULT_WORK = (
    REPO / "work_dirs/segformer-b0-repro/full-release-all-labeled-20261009"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
CSV_FIELDS = [
    "coverage_rank",
    "id",
    "archive",
    "entry",
    "source_set",
    "selection",
    "width",
    "height",
    "prediction_width",
    "prediction_height",
    "predicted_area_fraction",
    "predicted_area_px",
    "predicted_area_mm2",
    "predicted_equiv_radius_px",
    "predicted_equiv_radius_mm",
    "predicted_log_area_px2",
    "predicted_log_area_mm2",
    "predicted_empty_mask",
    "area_fraction_qc",
    "area_exclusion",
    "prediction_status",
    "prediction_error",
    "prediction_fallback",
    "mask_path",
    "predicted_mask_sha256",
    "md5",
    "duplicate_role",
    "duplicate_group_size",
    "duplicate_representative",
    "label_conflict",
    "species",
    "medium",
    "temp",
    "condition",
    "condition_known",
    "condition_source",
    "block",
    "scale_group",
    "px_per_mm",
    "fit_status",
    "frame_order_status",
    "time_unit_status",
]


def archive_inventory(provenance: dict[str, inference.Provenance]) -> dict[str, list[inference.Provenance]]:
    grouped: dict[str, list[inference.Provenance]] = defaultdict(list)
    for item in provenance.values():
        grouped[item.archive].append(item)
    return {name: sorted(items, key=lambda item: item.entry) for name, items in sorted(grouped.items())}


def released_archive_manifest(archives: list[str]) -> dict[str, dict[str, Any]]:
    manifest_path = REPO / "data/manifest.csv"
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {
        row["file"]: row
        for row in rows
        if row.get("dataset") == "myceliumseg" and row.get("file") in archives
    }
    out: dict[str, dict[str, Any]] = {}
    for archive in archives:
        path = inference.RAW_ARCHIVES / archive
        record = by_name.get(archive)
        if record is None or not path.is_file():
            raise RuntimeError(f"archive or manifest record missing: {archive}")
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"archive size differs from data/manifest.csv: {archive}")
        out[archive] = {
            "bytes": int(record["bytes"]),
            "md5": record["md5"],
            "path": str(path.relative_to(REPO)),
            "provenance_records": 0,
        }
    return out


def initial_state(args: argparse.Namespace, archive_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    config = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not config.is_file():
        raise RuntimeError(f"config not found: {config}")
    if not checkpoint.is_file():
        raise RuntimeError(f"checkpoint not found: {checkpoint}")
    return {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "running",
        "config": str(config),
        "config_sha256": repro.sha256_file(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": repro.sha256_file(checkpoint),
        "expected_images": args.limit,
        "source_archives": archive_meta,
        "outputs": {
            "records_jsonl": "records.jsonl",
            "sorted_csv": "area_fraction_sorted.csv",
            "unique_sorted_csv": "area_fraction_unique_sorted.csv",
            "masks": "masks/<archive>/<image_id>.png",
        },
        "limitations": [
            "The production checkpoint was fitted on all 547 unique labeled pairs; "
            "its predictions on those images are not held-out evaluation.",
            "The #28 benchmark remains the independent held-out reference.",
            "area_fraction is an image-level coverage coordinate, not time or growth rate.",
            "Masks with off-scale or unreadable dish geometry retain pixel area but have "
            "null calibrated area_fraction.",
        ],
    }


def read_existing(work: Path, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    state_path = work / "state.json"
    records_path = work / "records.jsonl"
    if not work.exists():
        work.mkdir(parents=True)
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        records_path.touch()
        return {}
    if not state_path.is_file() or not records_path.is_file():
        if any(work.iterdir()):
            raise RuntimeError(
                f"{work} exists without resumable state; choose a new --work directory"
            )
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        records_path.touch()
        return {}
    prior = json.loads(state_path.read_text())
    for field in ("schema_version", "config_sha256", "checkpoint_sha256", "expected_images", "source_archives"):
        if prior.get(field) != state.get(field):
            raise RuntimeError(f"existing run has a different {field}; choose a new --work directory")
    rows: dict[str, dict[str, Any]] = {}
    content = records_path.read_bytes()
    lines = content.splitlines(keepends=True)
    offset = 0
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_number == len(lines):
                # A host interruption may leave only the final JSONL record partial.
                with records_path.open("r+b") as handle:
                    handle.truncate(offset)
                break
            raise RuntimeError(f"invalid JSONL at {records_path}:{line_number}") from exc
        key = row.get("record_key")
        if key:
            rows[str(key)] = row
        offset += len(line)
    if lines and not lines[-1].endswith(b"\n") and offset == len(content):
        with records_path.open("ab") as handle:
            handle.write(b"\n")
    return rows


def mask_relative_path(archive: str, image_id: str) -> Path:
    return Path("masks") / Path(archive).stem / f"{image_id}.png"


def base_record(item: inference.Provenance) -> dict[str, Any]:
    row = item.as_dict()
    row.update(
        {
            "record_key": f"{item.archive}::{item.entry}",
            "entry": item.entry,
            "source_set": "labeled" if item.archive.startswith("labeled-") else "unlabeled",
            "selection": "full_release",
            "selection_rule": "every JPEG image member in the released archive",
            "image_path": f"{item.archive}::{item.entry}",
            "prediction_status": "pending",
            "prediction_error": "",
            "prediction_fallback": "",
            "mask_path": "",
            "predicted_mask_sha256": "",
        }
    )
    return row


def error_measurements() -> dict[str, Any]:
    return {
        "predicted_area_px": None,
        "predicted_empty_mask": None,
        "predicted_equiv_radius_px": None,
        "predicted_log_area_px2": None,
        "predicted_area_mm2": None,
        "predicted_equiv_radius_mm": None,
        "predicted_log_area_mm2": None,
        "predicted_area_fraction": None,
        "area_exclusion": "prediction_error",
    }


def write_mask(mask: np.ndarray, path: Path) -> str:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.png")
    Image.fromarray(mask.astype(np.uint8), mode="L").save(temp)
    temp.replace(path)
    return repro.sha256_file(path)


def predict_member(model: Any, data: bytes) -> tuple[np.ndarray, int, int]:
    import cv2
    import mmseg.apis.inference as mmseg_inference

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("cv2 could not decode this JPEG member")
    result = mmseg_inference.inference_segmentor(model, image)
    if not result:
        raise RuntimeError("segmenter returned no prediction")
    return np.asarray(result[0]).astype(np.uint8), int(image.shape[1]), int(image.shape[0])


def is_cuda_oom(exc: BaseException) -> bool:
    """Return whether a prediction failed because CUDA ran out of memory."""
    try:
        import torch
    except ImportError:
        return "cuda out of memory" in str(exc).lower()
    oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return (bool(oom_type) and isinstance(exc, oom_type)) or (
        "cuda out of memory" in str(exc).lower()
    )


def persist_row(handle, row: dict[str, Any], *, sync: bool = False) -> None:
    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    if sync:
        os.fsync(handle.fileno())


def validate_archive_members(
    archive: str,
    items: list[inference.Provenance],
) -> tuple[zipfile.ZipFile, dict[str, zipfile.ZipInfo]]:
    path = inference.RAW_ARCHIVES / archive
    zf = zipfile.ZipFile(path)
    image_infos = {
        info.filename: info
        for info in zf.infolist()
        if not info.is_dir()
        and Path(info.filename).suffix.lower() in IMAGE_SUFFIXES
        # Some released archives contain macOS resource-fork sidecars ending
        # in .jpg. They are not source images and have no provenance records.
        and "__MACOSX" not in Path(info.filename).parts
        and not Path(info.filename).name.startswith("._")
    }
    expected = {item.entry for item in items}
    missing = expected - image_infos.keys()
    unexpected = image_infos.keys() - expected
    if missing or unexpected:
        zf.close()
        raise RuntimeError(
            f"{archive} image inventory mismatch: missing={len(missing)}, "
            f"unmapped={len(unexpected)}"
        )
    return zf, image_infos


def export_sorted(rows: list[dict[str, Any]], output: Path) -> tuple[int, int]:
    measurable = [row for row in rows if row.get("predicted_area_fraction") is not None]
    excluded = [row for row in rows if row.get("predicted_area_fraction") is None]
    measurable.sort(
        key=lambda row: (
            float(row["predicted_area_fraction"]),
            str(row.get("archive", "")),
            str(row.get("id", "")),
        )
    )
    excluded.sort(key=lambda row: (str(row.get("archive", "")), str(row.get("id", ""))))
    for rank, row in enumerate(measurable, start=1):
        row["coverage_rank"] = rank
    for row in excluded:
        row["coverage_rank"] = ""
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(measurable + excluded)
    return len(measurable), len(excluded)


def run(args: argparse.Namespace) -> int:
    provenance = inference.load_provenance()
    grouped = archive_inventory(provenance)
    archive_meta = released_archive_manifest(list(grouped))
    for archive, items in grouped.items():
        archive_meta[archive]["provenance_records"] = len(items)
    limit = args.limit
    ordered_items = [(archive, item) for archive, items in grouped.items() for item in items]
    expected = min(limit, len(ordered_items)) if limit is not None else len(ordered_items)
    args.expected_images = expected
    state = initial_state(args, archive_meta)
    state["expected_images"] = expected
    work = Path(args.work).resolve()
    records = read_existing(work, state)

    try:
        model, model_meta = inference.load_model(
            Path(args.config), Path(args.checkpoint), args.device
        )
    except inference.InferenceError as exc:
        raise RuntimeError(str(exc)) from exc
    if model_meta["checkpoint_sha256"] != state["checkpoint_sha256"]:
        raise RuntimeError("loaded checkpoint hash differs from run state")
    state["model"] = model_meta
    state["expected_images"] = expected
    (work / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    start = time.monotonic()
    processed = len(records)
    if expected != len(provenance) and args.limit is None:
        raise RuntimeError(f"expected all {len(provenance)} images, got {expected}")
    print(
        f"full-release inference: {len(provenance)} provenance records; "
        f"this run will process {expected}; existing resumable rows={len(records)}"
    )
    records_path = work / "records.jsonl"
    cpu_model = None
    with records_path.open("a") as handle:
        current_archive = None
        zf = None
        infos: dict[str, zipfile.ZipInfo] = {}
        try:
            for index, (archive, item) in enumerate(ordered_items[:expected], start=1):
                key = f"{archive}::{item.entry}"
                prior = records.get(key)
                if prior and prior.get("prediction_status") == "ok":
                    prior_mask = work / str(prior.get("mask_path", ""))
                    if prior_mask.is_file():
                        continue
                if archive != current_archive:
                    if zf is not None:
                        zf.close()
                    zf, infos = validate_archive_members(archive, grouped[archive])
                    current_archive = archive
                row = base_record(item)
                output_mask = mask_relative_path(archive, item.id)
                try:
                    assert zf is not None
                    image_bytes = zf.read(infos[item.entry])
                    try:
                        mask, image_width, image_height = predict_member(model, image_bytes)
                    except Exception as prediction_exc:
                        if not is_cuda_oom(prediction_exc) or not str(args.device).startswith("cuda"):
                            raise
                        import torch

                        torch.cuda.empty_cache()
                        try:
                            if cpu_model is None:
                                cpu_model, cpu_meta = inference.load_model(
                                    Path(args.config), Path(args.checkpoint), "cpu"
                                )
                                if cpu_meta["checkpoint_sha256"] != state["checkpoint_sha256"]:
                                    raise RuntimeError("CPU fallback checkpoint hash differs from run state")
                            mask, image_width, image_height = predict_member(cpu_model, image_bytes)
                            row["prediction_fallback"] = "cpu_after_cuda_oom"
                        except Exception as fallback_exc:
                            raise RuntimeError(
                                f"CUDA prediction failed ({prediction_exc}); "
                                f"CPU fallback failed ({fallback_exc})"
                            ) from fallback_exc
                    expected_ratio = item.width / item.height if item.height else 0
                    image_ratio = image_width / image_height if image_height else 0
                    if not math.isclose(expected_ratio, image_ratio, rel_tol=0, abs_tol=1e-3):
                        raise RuntimeError(
                            f"decoded image aspect ratio {image_ratio:.6f} differs from "
                            f"provenance working-frame ratio {expected_ratio:.6f}"
                        )
                    if mask.shape != (image_height, image_width):
                        raise RuntimeError(
                            f"mask shape {mask.shape} != decoded image shape "
                            f"{(image_height, image_width)}"
                        )
                    mask_sha = write_mask(mask, work / output_mask)
                    area_px = int(np.count_nonzero(mask))
                    row.update(
                        inference.derive_measurements(
                            area_px, item, frame_width=int(mask.shape[1])
                        )
                    )
                    row["prediction_status"] = "ok"
                    row["prediction_width"] = image_width
                    row["prediction_height"] = image_height
                    row["mask_path"] = output_mask.as_posix()
                    row["predicted_mask_sha256"] = mask_sha
                    fraction = row["predicted_area_fraction"]
                    row["area_fraction_qc"] = (
                        "no_calibrated_dish_geometry"
                        if fraction is None
                        else "greater_than_one"
                        if fraction > 1
                        else "ok"
                    )
                except Exception as exc:  # retain a row for every source image
                    row.update(error_measurements())
                    row["prediction_status"] = "error"
                    row["prediction_error"] = f"{type(exc).__name__}: {exc}"
                    row["prediction_fallback"] = ""
                    row["area_fraction_qc"] = "prediction_error"
                inference.assert_no_inferred_time([row])
                persist_row(handle, row, sync=index % 100 == 0 or index == expected)
                records[key] = row
                processed += 1
                if index % args.log_interval == 0 or index == expected:
                    elapsed = max(time.monotonic() - start, 1e-6)
                    rate = processed / elapsed
                    remaining = max(expected - len(records), 0) / rate
                    print(
                        f"[{index}/{expected}] {archive}::{item.id}; "
                        f"{rate:.2f} images/s, ETA {remaining / 3600:.2f} h, "
                        f"status={row['prediction_status']}"
                    )
        finally:
            if zf is not None:
                zf.close()

    final_rows = list(records.values())
    final_rows.sort(key=lambda row: row["record_key"])
    for row in final_rows:
        fraction = row.get("predicted_area_fraction")
        if row.get("prediction_status") == "ok":
            row["area_fraction_qc"] = (
                "no_calibrated_dish_geometry"
                if fraction is None
                else "greater_than_one"
                if float(fraction) > 1
                else "ok"
            )
        else:
            row["area_fraction_qc"] = "prediction_error"
    errors = [row for row in final_rows if row.get("prediction_status") != "ok"]
    if len(final_rows) != expected:
        raise RuntimeError(f"incomplete records: {len(final_rows)} / {expected}")
    records_temp = records_path.with_name("records.jsonl.tmp")
    with records_temp.open("w") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    records_temp.replace(records_path)
    if errors:
        state["status"] = "completed_with_errors"
    else:
        state["status"] = "complete"
    state["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    valid_count, excluded_count = export_sorted(
        final_rows, work / "area_fraction_sorted.csv"
    )
    unique_rows = [row for row in final_rows if row.get("duplicate_role") == "representative"]
    unique_valid_count, unique_excluded_count = export_sorted(
        unique_rows, work / "area_fraction_unique_sorted.csv"
    )
    fractions = [
        float(row["predicted_area_fraction"])
        for row in unique_rows
        if row.get("predicted_area_fraction") is not None
    ]
    status_counts = Counter(row.get("prediction_status", "") for row in final_rows)
    state["results"] = {
        "records": len(final_rows),
        "predictions_ok": status_counts.get("ok", 0),
        "prediction_errors": status_counts.get("error", 0),
        "area_fraction_present": valid_count,
        "area_fraction_excluded": excluded_count,
        "unique_image_representatives": len(unique_rows),
        "unique_area_fraction_present": unique_valid_count,
        "unique_area_fraction_excluded": unique_excluded_count,
        "empty_masks": sum(bool(row.get("predicted_empty_mask")) for row in final_rows),
        "duplicate_records": sum(
            1 for row in final_rows if row.get("duplicate_role") == "copy"
        ),
        "unique_area_fraction_over_one": sum(
            1 for row in unique_rows if row.get("area_fraction_qc") == "greater_than_one"
        ),
        "area_fraction_min": min(fractions) if fractions else None,
        "area_fraction_median": float(np.median(fractions)) if fractions else None,
        "area_fraction_max": max(fractions) if fractions else None,
    }
    state["prediction_errors_sample"] = [
        {"record_key": row.get("record_key"), "error": row.get("prediction_error")}
        for row in errors[:25]
    ]
    (work / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps(state["results"], indent=2, sort_keys=True))
    print(f"sorted table: {work / 'area_fraction_sorted.csv'}")
    print(f"deduplicated coverage table: {work / 'area_fraction_unique_sorted.csv'}")
    print(f"evidence: {work / 'state.json'}")
    return 2 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", default="run", choices=("run",))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work", default=str(DEFAULT_WORK))
    parser.add_argument("--limit", type=int, default=None, help="process first N images (pipeline smoke)")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (RuntimeError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
