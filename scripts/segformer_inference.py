#!/usr/bin/env python3
"""Inference on labeled hold-outs and an auditable unlabeled sample (Chainlink #29).

Phase 2 step 3 of [PLAN.md](../PLAN.md): after the measurement contract (#60),
the provenance specification (#61) and the validated SegFormer-B0 baseline
(#28), run the checkpoint from #28 over the labeled hold-outs and a small,
deterministic, fully auditable sample of the unlabeled release, save the
predicted masks, and attach the provenance each row needs.

What this driver does *not* do is as important as what it does:

* it does not retrain -- it loads ``iter_50000.pth`` from the #28 GPU baseline
  and refuses to proceed if that checkpoint's SHA-256 is not the one #28
  recorded (``--allow-other-checkpoint`` overrides, for a future model);
* it does not infer elapsed time, dish identity or timelapse membership from
  the file numbering -- ``frame_order_status`` is ``not_demonstrated`` and
  ``time_unit_status`` is ``no_capture_interval`` for every MyceliumSeg row
  (``notes/provenance.md`` § 4.1, Gate C);
* it does not turn a boundary event into a biological tip.

Subcommands (all idempotent, all writing under ``work_dirs/segformer-b0-repro/
inference/``):

    sample     extract the deterministic unlabeled sample from the raw zips
    predict    run inference on a set and write per-image records + masks
    explore    the exploratory smoke test: colony area / area fraction sorted
               inside provenance-safe archive x block strata
    evidence   assemble the run's environment, checkpoint, counts and caveats
    run        sample -> predict (all sets) -> explore -> evidence

The pure-python half (provenance join, deterministic sample, area/area-fraction
arithmetic, the exploratory sort and its guardrails) is tested in
``tests/test_segformer_inference.py`` without torch.  The model half needs the
GPU shell and the compatibility shim:

    nix develop .#gpu --command env PYTHONPATH=scripts/segformer_shim \\
        python scripts/segformer_inference.py run

Exit codes: 0 ok, 1 a step failed.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import sys
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import measurement_contract as mc
import segformer_repro as repro

BENCHMARK = REPO / "external" / "MyceliumSeg-benchmark"
SEG_ROOT = REPO / "data" / "segmentation" / "release"
TRIAGE = REPO / "data" / "triage"
RAW_ARCHIVES = REPO / "data" / "raw" / "myceliumseg"
WORK_ROOT = REPO / "work_dirs" / "segformer-b0-repro" / "inference"
DEFAULT_WORK = WORK_ROOT
DEFAULT_CONFIG = REPO / "work_dirs" / "segformer-b0-repro" / "full-gpu" / "config.py"
DEFAULT_CHECKPOINT = (
    REPO / "work_dirs" / "segformer-b0-repro" / "full-gpu" / "train" / "iter_50000.pth"
)

# The validated #28 hand-off (notes/segformer_evidence.json, commit fdfa43a).
CHECKPOINT_SHA256 = "bcf66b6c1b3ce44018bc54223bacad31bd48a1647efc842bfbc6bd5f6d3e20d6"
CONFIG_SHA256 = "e3d3e6f6ed0fcdaa8eea3852d23167b41962cb442a1a0aae30a91a7780f4e57e"

# The three labeled splits, as extracted by scripts/segformer_repro.py prepare.
LABELED_SETS: dict[str, dict[str, Any]] = {
    "test": {
        "image_dir": SEG_ROOT / "test" / "image",
        "mask_dir": SEG_ROOT / "test" / "mask",
        "expected": 50,
        "role": "GL testset, the paper's Table 6 hold-out",
    },
    "conditions": {
        "image_dir": SEG_ROOT / "conditions" / "image",
        "mask_dir": SEG_ROOT / "conditions" / "mask",
        "expected": 60,
        "role": "GS/PO/TS/MYG/PDA/TEMP15 robustness hold-outs",
    },
}
HOLDOUT_ORDER = ("test", "conditions")

# The auditable unlabeled sample: a fixed stride through two archives plus two
# deliberate byte-copy probes, so the duplicate-status column is exercised
# against a labeled hold-out rather than only within the unlabeled release.
SAMPLE_STRIDE: dict[str, dict[str, int]] = {
    "unlabeled-GL2.zip": {"stride": 280, "limit": 8},
    "unlabeled-GS_PO_TS.zip": {"stride": 420, "limit": 5},
}
# 00019773 is a byte copy of labeled conditions image 00018428, 00019301 of
# 00018429 (data/triage/dish_tracks.csv md5 groups, #23).  They make the
# cross-archive duplicate path observable in the smoke test.
DUPLICATE_PROBES = ("00019773", "00019301")

# Sampling-frame and reporting caveats carried into every output.
CAVEATS = (
    (
        "MyceliumSeg is cross-sectional: no ordered frames, no dish identity, no "
        "capture interval (Gate C, notes/provenance.md)."
    ),
    (
        "A frame index is not a duration and the file numbering is shuffled: no "
        "elapsed time, growth rate, timelapse membership or dish identity may be "
        "read from the sorted area order."
    ),
    (
        "Area ordering inside an archive/block stratum is an exploratory "
        "inspection only; coherent bands may be acquisition, selection or "
        "condition artefacts, not biological growth."
    ),
    (
        "Boundary features are weak by construction (published Boundary IoU "
        "28.60%, #28): no boundary event here is a biological hyphal tip."
    ),
    (
        "The labeled hold-out comparison is a smoke sanity check of the masks; the "
        "bias/SD measurement-error model is #31, not this task."
    ),
)

# Field names that would mean an inference this task forbids.  The guard is a
# hard error, not a warning: a record that grows one of these is a bug.
FORBIDDEN_INFERENCE_KEYS = (
    "elapsed_time",
    "elapsed_seconds",
    "capture_interval_seconds",
    "growth_rate",
    "rate_per_day",
    "rate_per_hour",
    "mm_per_day",
    "dish_id",
    "timelapse_id",
    "frame_time",
    "timestamp",
    "acquisition_time",
    "biological_tip_count",
    "hyphal_tip_count",
)
# Substrings that may not appear in a committed exploration CSV column.
FORBIDDEN_COLUMN_SUBSTRINGS = (
    "elapsed",
    "timestamp",
    "growth_rate",
    "per_day",
    "per_hour",
    "dish_id",
    "timelapse",
    "hyphal_tip",
)

EXPLORE_CSV = REPO / "data" / "triage" / "inference_explore.csv"
BAND_GAP_FACTOR = 4.0


class InferenceError(RuntimeError):
    """A step failed with a message that is already user-facing."""


# --------------------------------------------------------------------------
# provenance: one row per release image
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """Everything known about one released image, without inferring anything."""

    id: str
    archive: str
    entry: str
    width: int
    height: int
    block: str
    scale_group: str
    px_per_mm: float | None
    fit_status: str
    md5: str
    dish_r_900: float | None
    species: str
    medium: str
    temp: str
    condition_source: str
    duplicate_role: str
    duplicate_group_size: int
    duplicate_representative: str
    label_conflict: bool
    ordered_frames: str
    capture_interval: str
    estimands_identifiable: str
    estimands_blocked: str

    @property
    def condition_known(self) -> bool:
        return bool(self.species or self.medium or self.temp)

    @property
    def condition(self) -> str:
        parts = [p for p in (self.species, self.medium, self.temp) if p]
        return "/".join(parts) if parts else "unknown"

    @property
    def frame_order_status(self) -> str:
        # MyceliumSeg's ordered_frames column is "no"; anything else would need
        # an independent order demonstration before it could be true.
        return "not_demonstrated"

    @property
    def time_unit_status(self) -> str:
        return "no_capture_interval"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "archive": self.archive,
            "entry": self.entry,
            "width": self.width,
            "height": self.height,
            "block": self.block,
            "scale_group": self.scale_group,
            "px_per_mm": self.px_per_mm,
            "fit_status": self.fit_status,
            "md5": self.md5,
            "dish_disc_radius_900": self.dish_r_900,
            "species": self.species,
            "medium": self.medium,
            "temp": self.temp,
            "condition": self.condition,
            "condition_source": self.condition_source,
            "condition_known": self.condition_known,
            "duplicate_role": self.duplicate_role,
            "duplicate_group_size": self.duplicate_group_size,
            "duplicate_representative": self.duplicate_representative,
            "label_conflict": self.label_conflict,
            "frame_order_status": self.frame_order_status,
            "time_unit_status": self.time_unit_status,
            "ordered_frames": self.ordered_frames,
            "capture_interval": self.capture_interval,
            "estimands_identifiable": self.estimands_identifiable,
            "estimands_blocked": self.estimands_blocked,
        }


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_provenance(
    tracks: Iterable[Mapping[str, Any]],
    scales: Iterable[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
    capability: Mapping[str, Any],
) -> dict[str, Provenance]:
    """Join the committed triage tables into one provenance row per image id.

    ``tracks`` is ``data/triage/dish_tracks.csv`` (archive, entry, geometry,
    md5), ``scales`` is ``image_scale.csv`` (block, px/mm, fit_status),
    ``labels`` is ``known_labels.csv`` and ``capability`` is the ``myceliumseg``
    row of ``sequence_inventory.csv``.  Duplicate roles follow the #60 rule:
    the lowest ``(archive, id)`` of an md5 group is the representative.
    """
    scale_by_id = {str(row["id"]): row for row in scales}
    label_by_id = {str(row["id"]): row for row in labels}

    tracks = list(tracks)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in tracks:
        groups[str(row.get("md5", ""))].append(row)

    representative: dict[str, str] = {}
    size: dict[str, int] = {}
    for digest, members in groups.items():
        if not digest:
            continue
        ranked = sorted(
            members, key=lambda row: (str(row.get("archive", "")), str(row.get("id", "")))
        )
        keep = str(ranked[0]["id"])
        for member in members:
            representative[str(member["id"])] = keep
            size[str(member["id"])] = len(members)

    # A label conflict is an md5 group whose copies disagree on a non-blank
    # label (#23/#24): condition follows the md5 map, never the folder name.
    conflict_ids: set[str] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        seen: set[tuple[str, str, str]] = set()
        for member in members:
            label = label_by_id.get(str(member["id"]), {})
            seen.add(
                (
                    str(label.get("species_if_known", "")),
                    str(label.get("medium_if_known", "")),
                    str(label.get("temp_if_known", "")),
                )
            )
        non_blank = {item for item in seen if any(item)}
        if len(non_blank) > 1:
            conflict_ids.update(str(member["id"]) for member in members)

    ordered = str(capability.get("ordered_frames", "no"))
    interval = str(capability.get("capture_interval", "none"))
    identifiable = str(capability.get("estimands_identifiable", ""))
    blocked = str(capability.get("estimands_blocked", ""))

    provenance: dict[str, Provenance] = {}
    for row in tracks:
        image_id = str(row["id"])
        scale = scale_by_id.get(image_id, {})
        label = label_by_id.get(image_id, {})
        digest = str(row.get("md5", ""))
        provenance[image_id] = Provenance(
            id=image_id,
            archive=str(row.get("archive", "")),
            entry=str(row.get("entry", "")),
            width=_as_int(row.get("width")),
            height=_as_int(row.get("height")),
            block=str(scale.get("block", "")),
            scale_group=str(scale.get("scale_group", "")),
            px_per_mm=_as_float(scale.get("px_per_mm")),
            fit_status=str(scale.get("fit_status", "unknown")),
            md5=digest,
            dish_r_900=_as_float(row.get("dish_r")),
            species=str(label.get("species_if_known", "")),
            medium=str(label.get("medium_if_known", "")),
            temp=str(label.get("temp_if_known", "")),
            condition_source=str(label.get("source", "")),
            duplicate_role=("representative" if representative.get(image_id, image_id) == image_id else "copy"),
            duplicate_group_size=size.get(image_id, 1),
            duplicate_representative=representative.get(image_id, image_id),
            label_conflict=image_id in conflict_ids,
            ordered_frames=ordered,
            capture_interval=interval,
            estimands_identifiable=identifiable,
            estimands_blocked=blocked,
        )
    return provenance


def load_provenance(triage: Path = TRIAGE) -> dict[str, Provenance]:
    """Read the committed triage tables and build the provenance map."""

    def read(name: str) -> list[dict[str, str]]:
        with (triage / name).open(newline="") as handle:
            return list(csv.DictReader(handle))

    capability = next(
        row
        for row in read("sequence_inventory.csv")
        if row.get("dataset") == "myceliumseg"
    )
    return build_provenance(
        read("dish_tracks.csv"),
        read("image_scale.csv"),
        read("known_labels.csv"),
        capability,
    )


# --------------------------------------------------------------------------
# deterministic sample and measurement arithmetic
# --------------------------------------------------------------------------


def select_unlabeled_sample(
    provenance: Mapping[str, Provenance],
    *,
    stride_spec: Mapping[str, Mapping[str, int]] = SAMPLE_STRIDE,
    probes: Sequence[str] = DUPLICATE_PROBES,
) -> list[dict[str, Any]]:
    """The auditable unlabeled sample: a fixed stride plus duplicate probes.

    The stride is deterministic and recorded per row, so the sample can be
    regenerated byte-for-byte from the committed triage tables without the raw
    zips.  Each item carries the exact ``(archive, entry)`` it came from.
    """
    by_archive: dict[str, list[Provenance]] = defaultdict(list)
    for item in provenance.values():
        by_archive[item.archive].append(item)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for archive, spec in stride_spec.items():
        members = sorted(by_archive.get(archive, []), key=lambda item: item.id)
        stride = max(1, int(spec.get("stride", 1)))
        limit = int(spec.get("limit", 0)) or len(members)
        for index in range(0, len(members), stride):
            if len([s for s in selected if s["archive"] == archive and s["selection"] == "stride"]) >= limit:
                break
            item = members[index]
            if item.id in seen:
                continue
            seen.add(item.id)
            selected.append(
                {
                    "id": item.id,
                    "archive": item.archive,
                    "entry": item.entry,
                    "selection": "stride",
                    "selection_rule": f"every {stride}th image of {archive}, first {limit}",
                }
            )
    for probe in probes:
        item = provenance.get(probe)
        if item is None or item.id in seen:
            continue
        seen.add(item.id)
        selected.append(
            {
                "id": item.id,
                "archive": item.archive,
                "entry": item.entry,
                "selection": "duplicate_probe",
                "selection_rule": "byte-copy probe for the duplicate-status column",
            }
        )
    return selected


def safe_log_area(area: float, reference_unit: str) -> float | None:
    """``mc.log_area`` guarded for an empty mask (area 0 is not loggable)."""
    if area is None or area <= 0:
        return None
    return mc.log_area(float(area), reference_unit=reference_unit)


def derive_measurements(
    area_px: int, prov: Provenance, *, frame_width: int
) -> dict[str, Any]:
    """Image-level morphology from one mask area, under the #60 contract.

    ``frame_width`` is the width of the frame the area was counted in (the
    predicted mask's own width), because ``dish_tracks.csv`` and
    ``image_scale.csv`` publish their geometry at the 900-px working width: the
    full-frame disc radius is ``dish_r_900 * frame_width / 900`` and the px/mm
    constant rescales the same way (``notes/estimands.md`` § 6).

    ``area_fraction`` needs a readable dish disc and ``area_mm2`` needs the
    px/mm constant, so an ``off_scale`` image keeps its px^2 numbers and reports
    the calibrated ones as null -- exactly the inclusion rule of
    ``notes/estimands.md`` § 2.4.
    """
    out: dict[str, Any] = {
        "predicted_area_px": int(area_px),
        "predicted_empty_mask": int(area_px) == 0,
        "predicted_equiv_radius_px": (
            math.sqrt(area_px / math.pi) if area_px > 0 else 0.0
        ),
        "predicted_log_area_px2": safe_log_area(area_px, "px^2"),
        "predicted_area_mm2": None,
        "predicted_equiv_radius_mm": None,
        "predicted_log_area_mm2": None,
        "predicted_area_fraction": None,
        "area_exclusion": "",
    }
    scale_ok = prov.fit_status == "ok"
    if prov.px_per_mm and frame_width > 0:
        px_per_mm_full = prov.px_per_mm * frame_width / 900.0
        area_mm2 = area_px / (px_per_mm_full**2)
        out["predicted_area_mm2"] = area_mm2
        out["predicted_equiv_radius_mm"] = math.sqrt(area_mm2 / math.pi)
        out["predicted_log_area_mm2"] = safe_log_area(area_mm2, "mm^2")
    if scale_ok and prov.dish_r_900 and frame_width > 0:
        dish_r_px = prov.dish_r_900 * frame_width / 900.0
        dish_area_px = math.pi * dish_r_px**2
        if dish_area_px > 0:
            out["predicted_area_fraction"] = area_px / dish_area_px
    else:
        reasons = []
        if not scale_ok:
            reasons.append(f"fit_status={prov.fit_status}")
        if not prov.dish_r_900:
            reasons.append("no_dish_disc")
        out["area_exclusion"] = ";".join(reasons)
    return out


def assert_no_inferred_time(rows: Iterable[Mapping[str, Any]]) -> None:
    """Refuse a record that smuggles in an elapsed time, rate or dish id."""
    for row in rows:
        bad = sorted(set(FORBIDDEN_INFERENCE_KEYS) & set(row))
        if bad:
            raise InferenceError(
                f"record {row.get('id', '?')} carries forbidden inferred fields: {bad}"
            )


# --------------------------------------------------------------------------
# the exploratory smoke test
# --------------------------------------------------------------------------


def _flag_record(row: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    if row.get("duplicate_role") == "copy":
        flags.append("duplicate_copy")
    if row.get("duplicate_role") == "representative" and row.get("duplicate_group_size", 1) > 1:
        flags.append("duplicate_representative")
    if row.get("label_conflict"):
        flags.append("label_conflict")
    if row.get("fit_status") != "ok":
        flags.append("off_scale")
    if row.get("predicted_empty_mask"):
        flags.append("empty_mask")
    if row.get("predicted_over_reference_area") is not None:
        ratio = row["predicted_over_reference_area"]
        if ratio < 0.5 or ratio > 2.0:
            flags.append("area_disagrees_with_reference")
    return flags


def _detect_bands(values: Sequence[float], *, gap_factor: float) -> list[int]:
    """Assign a band index to each sorted value by gap analysis.

    A break sits where the gap between neighbours exceeds ``gap_factor`` times
    the median *positive* gap.  With fewer than two positive gaps no break is
    claimed -- the honest answer for a tiny sample is "one band".  This is a
    descriptive sort, not a clustering model.
    """
    if len(values) < 2:
        return [0] * len(values)
    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    positive = [gap for gap in gaps if gap > 0]
    if len(positive) < 2:
        return [0] * len(values)
    threshold = gap_factor * median(positive)
    bands = [0]
    current = 0
    for gap in gaps:
        if gap > threshold and gap > 0:
            current += 1
        bands.append(current)
    return bands


def _id_runs(ids: Sequence[str]) -> list[list[str]]:
    """Maximal runs of consecutive integer release ids (an index artefact)."""
    numbers = sorted({int(i) for i in ids if str(i).isdigit()})
    runs: list[list[str]] = []
    for number in numbers:
        if runs and number == int(runs[-1][-1]) + 1:
            runs[-1].append(f"{number:08d}")
        else:
            runs.append([f"{number:08d}"])
    return [run for run in runs if len(run) > 1]


def explore_records(
    records: Sequence[Mapping[str, Any]],
    *,
    gap_factor: float = BAND_GAP_FACTOR,
) -> dict[str, Any]:
    """Sort by area fraction inside provenance-safe ``(archive, block)`` strata.

    Returns blocks with their sorted rows, gap-detected bands, duplicate groups
    and consecutive-id runs.  Every grouping is labelled exploratory and the
    output carries the caveats above; nothing here may be read as time.
    """
    assert_no_inferred_time(records)
    blocks: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        blocks[(str(row.get("archive", "")), str(row.get("block", "")))].append(row)

    block_reports: list[dict[str, Any]] = []
    for (archive, block), rows in sorted(blocks.items()):
        scored = [row for row in rows if row.get("predicted_area_fraction") is not None]
        excluded = [row for row in rows if row.get("predicted_area_fraction") is None]
        scored.sort(key=lambda row: (row["predicted_area_fraction"], str(row["id"])))
        values = [float(row["predicted_area_fraction"]) for row in scored]
        bands = _detect_bands(values, gap_factor=gap_factor)
        sorted_rows = []
        for rank, (row, band) in enumerate(zip(scored, bands), start=1):
            sorted_rows.append(
                {
                    "id": row["id"],
                    "predicted_area_fraction": row["predicted_area_fraction"],
                    "predicted_area_mm2": row.get("predicted_area_mm2"),
                    "duplicate_role": row.get("duplicate_role"),
                    "rank": rank,
                    "band": band,
                    "flags": _flag_record(row),
                }
            )
        band_summary = []
        for band in sorted(set(bands)):
            members = [values[i] for i, b in enumerate(bands) if b == band]
            band_summary.append(
                {
                    "band": band,
                    "n": len(members),
                    "area_fraction_min": min(members),
                    "area_fraction_max": max(members),
                }
            )
        block_reports.append(
            {
                "archive": archive,
                "block": block,
                "n": len(rows),
                "n_scored": len(scored),
                "n_excluded": len(excluded),
                "excluded": [
                    {
                        "id": row["id"],
                        "area_exclusion": row.get("area_exclusion"),
                        "flags": _flag_record(row),
                    }
                    for row in excluded
                ],
                "n_bands": len(band_summary),
                "bands": band_summary,
                "sorted": sorted_rows,
                "id_runs": _id_runs([str(row["id"]) for row in rows]),
            }
        )

    by_md5: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_md5[str(row.get("md5", ""))].append(row)
    duplicate_groups = []
    for digest, members in sorted(by_md5.items()):
        if not digest or len(members) < 2:
            continue
        mask_hashes = {str(member.get("predicted_mask_sha256", "")) for member in members}
        duplicate_groups.append(
            {
                "md5": digest,
                "ids": sorted(str(member["id"]) for member in members),
                "archives": sorted({str(member.get("archive", "")) for member in members}),
                "predicted_mask_sha256_identical": len(mask_hashes) == 1,
            }
        )

    return {
        "gap_factor": gap_factor,
        "n_records": len(records),
        "blocks": block_reports,
        "duplicate_groups": duplicate_groups,
        "caveats": list(CAVEATS),
    }


def explore_csv_header() -> list[str]:
    return [
        "source_set",
        "selection",
        "archive",
        "id",
        "block",
        "scale_group",
        "species",
        "medium",
        "temp",
        "condition",
        "condition_known",
        "condition_source",
        "label_conflict",
        "md5",
        "duplicate_role",
        "duplicate_group_size",
        "duplicate_representative",
        "frame_order_status",
        "time_unit_status",
        "ordered_frames",
        "capture_interval",
        "estimands_identifiable",
        "estimands_blocked",
        "fit_status",
        "px_per_mm",
        "dish_disc_radius_900",
        "predicted_area_px",
        "predicted_area_mm2",
        "predicted_area_fraction",
        "predicted_equiv_radius_px",
        "predicted_equiv_radius_mm",
        "predicted_empty_mask",
        "area_exclusion",
        "predicted_mask_sha256",
        "reference_mask_area_px",
        "predicted_over_reference_area",
        "explore_block",
        "explore_rank",
        "explore_band",
        "explore_flags",
    ]


def explore_csv_rows(exploration: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten the exploration into one row per record, sorted within block.

    Rows excluded from the area-fraction estimand (``off_scale``) are kept with
    empty rank/band so the committed table still carries their provenance.
    """
    rows: list[dict[str, Any]] = []
    for block in exploration["blocks"]:
        key = f"{block['archive']}::{block['block']}"
        for entry in block["sorted"]:
            rows.append(
                {
                    "explore_block": key,
                    "explore_rank": entry["rank"],
                    "explore_band": entry["band"],
                    "explore_flags": ";".join(entry["flags"]),
                    **{k: v for k, v in entry.items() if k not in ("rank", "band", "flags")},
                }
            )
        for entry in block["excluded"]:
            rows.append(
                {
                    "explore_block": key,
                    "explore_rank": "",
                    "explore_band": "",
                    "explore_flags": ";".join(entry.get("flags", [])),
                    "id": entry["id"],
                }
            )
    return rows


def write_explore_csv(path: Path, records: Sequence[Mapping[str, Any]], exploration: Mapping[str, Any]) -> None:
    """Write the committed smoke-test table, guarded against forbidden columns."""
    header = explore_csv_header()
    for column in header:
        for forbidden in FORBIDDEN_COLUMN_SUBSTRINGS:
            if forbidden in column:
                raise InferenceError(
                    f"exploration column {column!r} names a forbidden inference "
                    f"({forbidden!r})"
                )
    ranked = {row["id"]: row for row in explore_csv_rows(exploration)}
    by_id = {str(record["id"]): record for record in records}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for image_id, rank_row in ranked.items():
            record = dict(by_id[image_id])
            record.update(rank_row)
            writer.writerow({key: record.get(key) for key in header})


# --------------------------------------------------------------------------
# the model half (needs mmcv/mmseg + the shim)
# --------------------------------------------------------------------------


def load_model(config: Path, checkpoint: Path, device: str) -> tuple[Any, dict[str, Any]]:
    """Build the segmenter from the #28 config and load the #28 checkpoint."""
    if not config.is_file():
        raise InferenceError(f"config not found: {config}")
    if not checkpoint.is_file():
        raise InferenceError(f"checkpoint not found: {checkpoint}")
    try:
        import torch  # noqa: F401
        from mmseg.apis.inference import init_segmentor
    except ImportError as exc:
        raise InferenceError(
            f"mmseg/torch not importable ({exc}); run inside `nix develop .#gpu` "
            "with scripts/segformer_shim on PYTHONPATH"
        ) from exc
    model = init_segmentor(str(config), str(checkpoint), device=device)
    meta = {
        "config": str(config),
        "config_sha256": repro.sha256_file(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": repro.sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "device": str(next(model.parameters()).device),
    }
    return model, meta


def predict_mask(model: Any, image_path: Path) -> Any:
    """One slide-inference pass, the same test pipeline the #28 eval used."""
    import numpy as np
    from mmseg.apis.inference import inference_segmentor

    result = inference_segmentor(model, str(image_path))
    if not result:
        raise InferenceError(f"no prediction for {image_path}")
    return np.asarray(result[0]).astype(np.uint8)


def save_mask(mask: Any, path: Path) -> str:
    """Save a 0/1 uint8 PNG (the labeled-mask convention) and return its hash."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)
    return repro.sha256_file(path)


def _record_base(
    prov: Provenance,
    *,
    source_set: str,
    selection: str,
    selection_rule: str,
) -> dict[str, Any]:
    record = prov.as_dict()
    record.update(
        {
            "source_set": source_set,
            "selection": selection,
            "selection_rule": selection_rule,
        }
    )
    return record


def predict_holdout(
    model: Any,
    set_name: str,
    provenance: Mapping[str, Provenance],
    work: Path,
    *,
    limit: int | None = None,
    reuse: bool = False,
) -> list[dict[str, Any]]:
    """Infer on one labeled hold-out split and compare with its true mask.

    With ``reuse`` set (the caller checks the cached checkpoint hash), a mask
    already on disk is read back instead of re-inferred, so an interrupted run
    resumes instead of restarting.
    """
    spec = LABELED_SETS[set_name]
    image_dir: Path = spec["image_dir"]
    mask_dir: Path = spec["mask_dir"]
    if not image_dir.is_dir():
        raise InferenceError(
            f"{image_dir} missing; run `nix develop --command python "
            "scripts/segformer_repro.py prepare` first"
        )
    items = sorted(image_dir.glob("*.jpg"))
    if limit is not None:
        items = items[:limit]
    records: list[dict[str, Any]] = []
    import numpy as np
    from PIL import Image

    for index, image_path in enumerate(items, start=1):
        image_id = image_path.stem
        prov = provenance.get(image_id)
        if prov is None:
            raise InferenceError(f"{image_id} has no provenance row")
        mask_path = work / "masks" / set_name / f"{image_id}.png"
        cached = reuse and mask_path.is_file()
        if cached:
            mask = np.asarray(Image.open(mask_path)).astype(np.uint8)
            mask_sha = repro.sha256_file(mask_path)
        else:
            mask = predict_mask(model, image_path)
            mask_sha = save_mask(mask, mask_path)
        area_px = int(np.count_nonzero(mask))
        reference_path = mask_dir / f"{image_id}.png"
        reference_area = None
        if reference_path.is_file():
            reference = np.asarray(Image.open(reference_path)) > 0
            reference_area = int(np.count_nonzero(reference))
        record = _record_base(
            prov,
            source_set=set_name,
            selection="labeled_holdout",
            selection_rule=spec["role"],
        )
        record.update(derive_measurements(area_px, prov, frame_width=int(mask.shape[1])))
        record.update(
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "mask_width": int(mask.shape[1]),
                "mask_height": int(mask.shape[0]),
                "predicted_mask_sha256": mask_sha,
                "mask_reused": cached,
                "reference_mask_area_px": reference_area,
                "predicted_over_reference_area": (
                    area_px / reference_area
                    if reference_area
                    else None
                ),
            }
        )
        assert_no_inferred_time([record])
        records.append(record)
        marker = "reused" if cached else "inferred"
        print(f"[{set_name}] {index}/{len(items)} {image_id} area={area_px} ({marker})")
    return records


def materialize_sample(
    provenance: Mapping[str, Provenance],
    work: Path,
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Extract the deterministic unlabeled sample from the raw zips."""
    sample = select_unlabeled_sample(provenance)
    manifest_path = work / "sample" / "sample_manifest.json"
    manifest = {
        "selection": sample,
        "archives": sorted({item["archive"] for item in sample}),
    }
    if manifest_path.is_file() and not force:
        try:
            if json.loads(manifest_path.read_text()) == manifest:
                return sample
        except json.JSONDecodeError:
            pass
    image_dir = work / "sample" / "image"
    image_dir.mkdir(parents=True, exist_ok=True)
    by_archive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sample:
        by_archive[item["archive"]].append(item)
    for archive, items in by_archive.items():
        archive_path = RAW_ARCHIVES / archive
        if not archive_path.is_file():
            raise InferenceError(f"missing raw archive: {archive_path}")
        with zipfile.ZipFile(archive_path) as zf:
            for item in items:
                target = image_dir / f"{item['id']}.jpg"
                with zf.open(item["entry"]) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"materialized {len(sample)} unlabeled images under {image_dir}")
    return sample


def predict_unlabeled(
    model: Any,
    provenance: Mapping[str, Provenance],
    work: Path,
    *,
    limit: int | None = None,
    reuse: bool = False,
) -> list[dict[str, Any]]:
    """Infer on the materialized unlabeled sample."""
    sample = materialize_sample(provenance, work)
    if limit is not None:
        sample = sample[:limit]
    import numpy as np
    from PIL import Image

    records: list[dict[str, Any]] = []
    for index, item in enumerate(sample, start=1):
        image_id = item["id"]
        prov = provenance[image_id]
        image_path = work / "sample" / "image" / f"{image_id}.jpg"
        mask_path = work / "masks" / "unlabeled" / f"{image_id}.png"
        cached = reuse and mask_path.is_file()
        if cached:
            mask = np.asarray(Image.open(mask_path)).astype(np.uint8)
            mask_sha = repro.sha256_file(mask_path)
        else:
            mask = predict_mask(model, image_path)
            mask_sha = save_mask(mask, mask_path)
        area_px = int(np.count_nonzero(mask))
        record = _record_base(
            prov,
            source_set="unlabeled",
            selection=item["selection"],
            selection_rule=item["selection_rule"],
        )
        record.update(derive_measurements(area_px, prov, frame_width=int(mask.shape[1])))
        record.update(
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "mask_width": int(mask.shape[1]),
                "mask_height": int(mask.shape[0]),
                "predicted_mask_sha256": mask_sha,
                "mask_reused": cached,
                "reference_mask_area_px": None,
                "predicted_over_reference_area": None,
            }
        )
        assert_no_inferred_time([record])
        records.append(record)
        marker = "reused" if cached else "inferred"
        print(f"[unlabeled] {index}/{len(sample)} {image_id} area={area_px} ({marker})")
    return records


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------


def collect_evidence(work: Path, model_meta: Mapping[str, Any] | None, records: Sequence[Mapping[str, Any]], exploration: Mapping[str, Any] | None) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record["source_set"])] += 1
    evidence: dict[str, Any] = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "work_dir": str(work),
        "task": 29,
        "repository_commit": repro._git_head(REPO),
        "benchmark_commit": repro._git_head(BENCHMARK),
        "environment": repro.collect_environment(),
        "counts": dict(sorted(counts.items())),
        "checkpoint_guard": {
            "expected_sha256": CHECKPOINT_SHA256,
            "expected_config_sha256": CONFIG_SHA256,
        },
        "caveats": list(CAVEATS),
        "commands": [
            (
                "nix develop .#gpu --command env PYTHONPATH=scripts/segformer_shim "
                "python scripts/segformer_inference.py run"
            ),
            (
                "nix develop .#gpu --command env PYTHONPATH=scripts/segformer_shim "
                "python scripts/segformer_inference.py predict --set all"
            ),
            "nix develop --command python scripts/segformer_inference.py explore",
        ],
    }
    if model_meta is not None:
        evidence["model"] = dict(model_meta)
    if exploration is not None:
        evidence["exploration"] = {
            "gap_factor": exploration["gap_factor"],
            "n_blocks": len(exploration["blocks"]),
            "n_duplicate_groups": len(exploration["duplicate_groups"]),
            "duplicate_groups": exploration["duplicate_groups"],
            "blocks": [
                {
                    "archive": block["archive"],
                    "block": block["block"],
                    "n": block["n"],
                    "n_scored": block["n_scored"],
                    "n_bands": block["n_bands"],
                }
                for block in exploration["blocks"]
            ],
        }
    return evidence


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------


def _work(args: argparse.Namespace) -> Path:
    return Path(args.work)


def _load_records(work: Path) -> list[dict[str, Any]]:
    path = work / "records.json"
    if not path.is_file():
        raise InferenceError(f"{path} missing; run `predict` first")
    return json.loads(path.read_text())


def cmd_sample(args: argparse.Namespace) -> int:
    provenance = load_provenance()
    sample = materialize_sample(provenance, _work(args), force=args.force)
    print(json.dumps(sample, indent=2, sort_keys=True))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    work = _work(args)
    work.mkdir(parents=True, exist_ok=True)
    provenance = load_provenance()
    model, meta = load_model(Path(args.config), Path(args.checkpoint), args.device)
    if not args.allow_other_checkpoint:
        if meta["checkpoint_sha256"] != CHECKPOINT_SHA256:
            raise InferenceError(
                "checkpoint SHA-256 does not match the validated #28 hand-off "
                f"({meta['checkpoint_sha256']} != {CHECKPOINT_SHA256}); pass "
                "--allow-other-checkpoint to use it anyway"
            )
        if meta["config_sha256"] != CONFIG_SHA256:
            print(
                "warning: config SHA-256 differs from the #28 record "
                f"({meta['config_sha256']} != {CONFIG_SHA256})"
            )
    records: list[dict[str, Any]] = []
    reuse = False
    meta_path = work / "model_meta.json"
    if meta_path.is_file():
        try:
            previous = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            previous = {}
        reuse = previous.get("checkpoint_sha256") == meta["checkpoint_sha256"]
    if reuse:
        print("reusing masks whose checkpoint hash matches the cached run")
    if args.set in ("all", *HOLDOUT_ORDER):
        for set_name in HOLDOUT_ORDER:
            if args.set not in ("all", set_name):
                continue
            records.extend(
                predict_holdout(
                    model, set_name, provenance, work, limit=args.limit, reuse=reuse
                )
            )
    if args.set in ("all", "unlabeled"):
        records.extend(
            predict_unlabeled(
                model, provenance, work, limit=args.limit, reuse=reuse
            )
        )
    records_path = work / "records.json"
    records_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"wrote {records_path} ({len(records)} records)")
    return 0


def cmd_explore(args: argparse.Namespace) -> int:
    work = _work(args)
    records = _load_records(work)
    exploration = explore_records(records, gap_factor=args.gap_factor)
    out = work / "exploration.json"
    out.write_text(json.dumps(exploration, indent=2, sort_keys=True) + "\n")
    csv_path = Path(args.csv) if args.csv else EXPLORE_CSV
    write_explore_csv(csv_path, records, exploration)
    print(f"wrote {out}")
    print(f"wrote {csv_path}")
    for block in exploration["blocks"]:
        print(
            f"  {block['archive']}::{block['block']}: n={block['n']} "
            f"scored={block['n_scored']} bands={block['n_bands']}"
        )
    print(f"  duplicate groups: {len(exploration['duplicate_groups'])}")
    return 0


def cmd_evidence(args: argparse.Namespace) -> int:
    work = _work(args)
    records = _load_records(work) if (work / "records.json").is_file() else []
    meta_path = work / "model_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else None
    exploration = None
    exploration_path = work / "exploration.json"
    if exploration_path.is_file():
        exploration = json.loads(exploration_path.read_text())
    evidence = collect_evidence(work, meta, records, exploration)
    out = Path(args.output) if args.output else work / "evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    predict_ns = argparse.Namespace(
        work=args.work,
        set=getattr(args, "set", "all"),
        config=getattr(args, "config", str(DEFAULT_CONFIG)),
        checkpoint=getattr(args, "checkpoint", str(DEFAULT_CHECKPOINT)),
        device=getattr(args, "device", "cuda:0"),
        limit=getattr(args, "limit", None),
        allow_other_checkpoint=getattr(args, "allow_other_checkpoint", False),
    )
    explore_ns = argparse.Namespace(
        work=args.work,
        gap_factor=getattr(args, "gap_factor", BAND_GAP_FACTOR),
        csv=getattr(args, "csv", None),
    )
    evidence_ns = argparse.Namespace(work=args.work, output=getattr(args, "output", None))
    for step, step_ns in (
        (cmd_predict, predict_ns),
        (cmd_explore, explore_ns),
        (cmd_evidence, evidence_ns),
    ):
        code = step(step_ns)
        if code != 0:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str = "") -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        p.add_argument(
            "--work",
            default=str(DEFAULT_WORK),
            help="work directory (default: work_dirs/segformer-b0-repro/inference)",
        )
        return p

    sample = add("sample", cmd_sample, "extract the deterministic unlabeled sample")
    sample.add_argument("--force", action="store_true", help="re-extract even if up to date")

    predict = add("predict", cmd_predict, "run inference and write masks + records")
    predict.add_argument(
        "--set",
        choices=("test", "conditions", "unlabeled", "all"),
        default="all",
    )
    predict.add_argument("--config", default=str(DEFAULT_CONFIG))
    predict.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    predict.add_argument("--device", default="cuda:0")
    predict.add_argument("--limit", type=int, default=None, help="cap images per set (smoke)")
    predict.add_argument(
        "--allow-other-checkpoint",
        action="store_true",
        help="do not require the validated #28 checkpoint SHA-256",
    )

    explore = add("explore", cmd_explore, "exploratory area/area-fraction smoke test")
    explore.add_argument("--gap-factor", type=float, default=BAND_GAP_FACTOR)
    explore.add_argument("--csv", default=None, help="output CSV (default: data/triage/inference_explore.csv)")

    evidence = add("evidence", cmd_evidence, "assemble the run evidence JSON")
    evidence.add_argument("--output", default=None)

    run = add("run", cmd_run, "predict all sets -> explore -> evidence")
    run.add_argument(
        "--set",
        choices=("test", "conditions", "unlabeled", "all"),
        default="all",
    )
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--limit", type=int, default=None, help="cap images per set (smoke)")
    run.add_argument("--allow-other-checkpoint", action="store_true")
    run.add_argument("--gap-factor", type=float, default=BAND_GAP_FACTOR)
    run.add_argument("--csv", default=None)
    run.add_argument("--output", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (InferenceError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
