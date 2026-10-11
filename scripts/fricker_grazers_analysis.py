#!/usr/bin/env python3
"""Build and analyze the image-linked Fricker/Boddy grazer series.

The source ZIPs contain one Fungal Network Analysis workbook per processed
image.  This script reads each workbook's Summary table and Parameters sheet,
links it to the original JPEG when present, and preserves the camera timestamp,
pixel dimensions, and per-image micron-per-pixel calibration.

Run from any directory with:
    python scripts/fricker_grazers_analysis.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import statistics
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fricker_grazers"
OUT = ROOT / "data" / "derived" / "fricker_grazers"
SPECIES = ("Hf", "Pi", "Pv", "Rb")
NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}
EXIF_DATETIME_ORIGINAL = 36867
EXIF_POINTER = 34665


def _cell_column(cell_ref: str) -> str:
    return re.sub(r"\d", "", cell_ref)


def _read_sheet_rows(
    book: zipfile.ZipFile, target: str, shared: list[str]
) -> list[dict[str, str]]:
    target = target.lstrip("/")
    if not target.startswith("xl/"):
        target = "xl/" + target
    root = ET.fromstring(book.read(target))
    rows = root.findall(".//m:sheetData/m:row", NS)
    if len(rows) < 2:
        return []

    def value(cell: ET.Element) -> str:
        kind = cell.attrib.get("t")
        v = cell.find("m:v", NS)
        if kind == "inlineStr":
            return "".join(t.text or "" for t in cell.findall(".//m:t", NS))
        raw = v.text if v is not None else ""
        if kind == "s" and raw:
            return shared[int(raw)]
        return raw

    headers = {
        _cell_column(cell.attrib["r"]): value(cell)
        for cell in rows[0].findall("m:c", NS)
    }
    return [
        {
            headers[_cell_column(cell.attrib["r"])]: value(cell)
            for cell in row.findall("m:c", NS)
            if _cell_column(cell.attrib["r"]) in headers
        }
        for row in rows[1:]
    ]


def _workbook_tables(data: bytes) -> tuple[dict[str, str], dict[str, str]]:
    with zipfile.ZipFile(io.BytesIO(data)) as book:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in book.namelist():
            strings_root = ET.fromstring(book.read("xl/sharedStrings.xml"))
            shared = [
                "".join(t.text or "" for t in si.findall(".//m:t", NS))
                for si in strings_root.findall("m:si", NS)
            ]
        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        rel_root = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        rels = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root}
        targets: dict[str, str] = {}
        for sheet in workbook.findall(".//m:sheet", NS):
            rid = sheet.attrib["{" + NS["r"] + "}id"]
            targets[sheet.attrib["name"]] = rels[rid]
        summary_rows = _read_sheet_rows(book, targets["Summary table"], shared)
        parameter_rows = _read_sheet_rows(book, targets["Parameters"], shared)
        summary = summary_rows[0] if summary_rows else {}
        parameters = parameter_rows[0] if parameter_rows else {}
        return summary, parameters


def _parse_stem(species: str, stem: str) -> dict[str, Any]:
    tokens = stem.split("_")
    if species == "Hf":
        if len(tokens) != 5:
            raise ValueError(f"Unexpected Hf filename: {stem}")
        _, isolate, treatment, day, replicate = tokens
    else:
        if len(tokens) != 4:
            raise ValueError(f"Unexpected {species} filename: {stem}")
        _, treatment, day, replicate = tokens
        isolate = ""
    match = re.fullmatch(r"[dD](\d+)", day)
    if not match:
        raise ValueError(f"Cannot parse day label in {stem}")
    return {
        "species": species,
        "isolate": isolate,
        "treatment": treatment.lower(),
        "day_label": int(match.group(1)),
        "replicate": int(replicate),
        "series_key": "|".join(
            [species, isolate, treatment.lower(), str(int(replicate))]
        ),
    }


def _image_info(data: bytes) -> dict[str, Any]:
    image = Image.open(io.BytesIO(data))
    exif = image.getexif()
    sub_ifd = exif.get_ifd(EXIF_POINTER)
    raw_datetime = sub_ifd.get(EXIF_DATETIME_ORIGINAL) or exif.get(306)
    capture = None
    if raw_datetime:
        try:
            capture = datetime.strptime(str(raw_datetime), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass
    return {
        "width_px": image.width,
        "height_px": image.height,
        "capture_datetime": capture.isoformat(sep=" ") if capture else "",
        "image_md5": hashlib.md5(data).hexdigest(),
    }


def extract_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for species in SPECIES:
        archive_path = RAW / f"{species}.zip"
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            images = {
                Path(name).stem: name
                for name in names
                if name.startswith(species + "/")
                and name.count("/") == 1
                and name.lower().endswith(".jpg")
            }
            result_names = [
                name
                for name in names
                if name.startswith(species + "/results/")
                and name.endswith("-results.xlsx")
            ]
            for result_name in result_names:
                stem = Path(result_name).name[: -len("-results.xlsx")]
                record = _parse_stem(species, stem)
                summary, parameters = _workbook_tables(archive.read(result_name))
                image_name = images.get(stem)
                image_data = archive.read(image_name) if image_name else None
                image_info = _image_info(image_data) if image_data else {}
                record.update(
                    {
                        "file_stem": stem,
                        "workbook_path": result_name,
                        "image_path": image_name or "",
                        "has_image": bool(image_name),
                        "micron_per_pixel": parameters.get("micron_per_pixel", ""),
                        **image_info,
                    }
                )
                for key, value in summary.items():
                    if key.startswith("summary_"):
                        record[key] = value
                records.append(record)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["series_key"]].append(row)
    for series in groups.values():
        series.sort(
            key=lambda x: (
                x["day_label"],
                x.get("capture_datetime", ""),
                x["file_stem"],
            )
        )
        dated = [
            (row, datetime.fromisoformat(row["capture_datetime"]))
            for row in series
            if row.get("capture_datetime")
        ]
        origin = min((stamp for _, stamp in dated), default=None)
        first_observed_day = min(
            (int(row["day_label"]) for row, _ in dated), default=""
        )
        seen: set[str] = set()
        for row, stamp in dated:
            row["elapsed_days_from_first"] = (stamp - origin).total_seconds() / 86400.0
            row["same_timestamp_as_prior"] = stamp.isoformat(sep=" ") in seen
            row["first_observed_day"] = first_observed_day
            seen.add(stamp.isoformat(sep=" "))
        for row in series:
            row.setdefault("elapsed_days_from_first", "")
            row.setdefault("same_timestamp_as_prior", False)
            row.setdefault("first_observed_day", first_observed_day)

    return records


def write_inventory(records: list[dict[str, Any]]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "network_series.csv"
    base_fields = [
        "species",
        "isolate",
        "treatment",
        "replicate",
        "series_key",
        "day_label",
        "first_observed_day",
        "elapsed_days_from_first",
        "capture_datetime",
        "same_timestamp_as_prior",
        "has_image",
        "width_px",
        "height_px",
        "micron_per_pixel",
        "image_md5",
        "file_stem",
        "image_path",
        "workbook_path",
    ]
    metric_fields = sorted({k for row in records for k in row if k.startswith("summary_")})
    fields = base_fields + metric_fields
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return path


def _float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _time_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    intervals: dict[int, list[float]] = defaultdict(list)
    zero_pairs: list[str] = []
    same_day_pairs: list[str] = []
    time_reversals: list[str] = []
    adjacent_transitions = 0
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_series[row["series_key"]].append(row)
    for series in by_series.values():
        dated_series = [row for row in series if row.get("capture_datetime")]
        dated_series.sort(key=lambda x: (x["day_label"], x["file_stem"]))
        for a, b in zip(dated_series, dated_series[1:]):
            adjacent_transitions += 1
            dta = datetime.fromisoformat(a["capture_datetime"])
            dtb = datetime.fromisoformat(b["capture_datetime"])
            observed = (dtb - dta).total_seconds() / 86400.0
            nominal = int(b["day_label"]) - int(a["day_label"])
            if observed == 0:
                zero_pairs.append(f"{a['file_stem']} -> {b['file_stem']}")
            elif observed < 0:
                time_reversals.append(f"{a['file_stem']} -> {b['file_stem']}")
            if nominal == 0:
                same_day_pairs.append(f"{a['file_stem']} -> {b['file_stem']}")
            if nominal > 0:
                intervals[nominal].append(observed)
    return {
        "images_with_linked_capture_time": sum(bool(r.get("capture_datetime")) for r in records),
        "result_workbooks": len(records),
        "adjacent_dated_transitions": adjacent_transitions,
        "missing_jpeg_for_workbook": [r["file_stem"] for r in records if not r.get("has_image")],
        "same_timestamp_transitions": zero_pairs,
        "same_filename_day_transitions": same_day_pairs,
        "capture_time_reversals": time_reversals,
        "intervals_by_filename_day_gap": {
            str(gap): {
                "n": len(values),
                "min_days": min(values),
                "median_days": statistics.median(values),
                "max_days": max(values),
            }
            for gap, values in sorted(intervals.items())
        },
        "resolution_counts": {
            f"{w}x{h}": sum(
                int(r.get("width_px") or 0) == w and int(r.get("height_px") or 0) == h
                for r in records
            )
            for w, h in sorted(
                {
                    (int(r.get("width_px") or 0), int(r.get("height_px") or 0))
                    for r in records
                    if r.get("width_px")
                }
            )
        },
    }


def _feature_dict(
    row: dict[str, Any], transform: str, tau: float = 3.0
) -> dict[str, Any]:
    species = row["species"]
    treatment = row["treatment_model"]
    t = float(row["elapsed_days_from_first"])
    features: dict[str, Any] = {
        "species": species,
        "treatment": treatment,
        "species_treatment": species + "|" + treatment,
        "start_label": str(row.get("first_observed_day", "")),
    }
    if transform == "linear":
        x = t / 8.0
        features["time"] = x
        for s in SPECIES:
            features[f"time_species_{s}"] = x if species == s else 0.0
        for tr in ("control", "coll", "mill", "nema", "wl1", "wl2"):
            features[f"time_treatment_{tr}"] = x if treatment == tr else 0.0
    elif transform == "log1p":
        x = math.log1p(t) / math.log1p(8.0)
        features["time"] = x
        for s in SPECIES:
            features[f"time_species_{s}"] = x if species == s else 0.0
        for tr in ("control", "coll", "mill", "nema", "wl1", "wl2"):
            features[f"time_treatment_{tr}"] = x if treatment == tr else 0.0
    elif transform.startswith("saturating"):
        tau = float(transform.removeprefix("saturating_tau"))
        x = (t / (tau + t)) / (8.0 / (tau + 8.0)) if t > 0 else 0.0
        features["time"] = x
        for s in SPECIES:
            features[f"time_species_{s}"] = x if species == s else 0.0
        for tr in ("control", "coll", "mill", "nema", "wl1", "wl2"):
            features[f"time_treatment_{tr}"] = x if treatment == tr else 0.0
    else:
        day = int(row["day_label"])
        features[f"time_level_{day}"] = 1.0
        for s in SPECIES:
            features[f"time_{s}_{day}"] = 1.0 if species == s else 0.0
        for tr in ("control", "coll", "mill", "nema", "wl1", "wl2"):
            features[f"time_{tr}_{day}"] = 1.0 if treatment == tr else 0.0
    return features


def analyze(records: list[dict[str, Any]], inventory_path: Path) -> dict[str, Any]:
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    accepted_treatments = {"ctrl1", "ctrl2", "coll", "mill", "nema", "wl1", "wl2"}
    endpoint_treatments = accepted_treatments | {"mite", "enchy"}
    baseline_series = {
        row["series_key"]
        for row in records
        if row["treatment"] in accepted_treatments
        and int(row["day_label"]) == 1
        and row.get("has_image")
        and row.get("capture_datetime")
    }
    primary: list[dict[str, Any]] = []
    for source in records:
        row = dict(source)
        if row["treatment"] not in accepted_treatments or int(row["day_label"]) > 8:
            continue
        if not row.get("has_image") or not row.get("capture_datetime"):
            continue
        if row.get("same_timestamp_as_prior"):
            continue
        row["treatment_model"] = "control" if row["treatment"].startswith("ctrl") else row["treatment"]
        primary.append(row)

    metric_keys = {
        "mycelial_area": "summary_mycelial_area",
        "length_total": "summary_length_total",
        "volume": "summary_volume",
        "tip_count": "summary_ntips",
        "width_mean": "summary_width_mean",
        "edge_count": "summary_num_edges",
        "loop_count": "summary_cyclomatic",
    }
    temporal_output = OUT / "temporal_model_comparison.csv"
    temporal_rows: list[dict[str, Any]] = []
    for outcome, key in metric_keys.items():
        data = [
            r
            for r in primary
            if int(r["day_label"]) in (1, 2, 4, 8)
            and (y := _float(r, key)) is not None
            and y > 0
        ]
        if len(data) < 100:
            continue
        groups = np.asarray([r["series_key"] for r in data])
        y = np.log(np.asarray([float(_float(r, key)) for r in data]))
        folds = list(GroupKFold(n_splits=5).split(np.arange(len(data)), y, groups))
        for model_name in (
            "linear",
            "log1p",
            "saturating_tau1",
            "saturating_tau2",
            "saturating_tau4",
            "saturating_tau8",
            "day_factor",
        ):
            feature_rows = [_feature_dict(r, model_name) for r in data]
            vectorizer = DictVectorizer(sparse=False)
            x = vectorizer.fit_transform(feature_rows)
            actual: list[float] = []
            predicted: list[float] = []
            for train, test in folds:
                model = Ridge(alpha=1.0)
                model.fit(x[train], y[train])
                predicted.extend(model.predict(x[test]).tolist())
                actual.extend(y[test].tolist())
            residual = np.asarray(actual) - np.asarray(predicted)
            temporal_rows.append(
                {
                    "outcome": outcome,
                    "model": model_name,
                    "n_observations": len(data),
                    "n_series": len(set(groups)),
                    "grouped_cv_rmse_log": float(np.sqrt(np.mean(residual**2))),
                    "grouped_cv_mae_log": float(np.mean(np.abs(residual))),
                }
            )
    with temporal_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(temporal_rows[0]) if temporal_rows else [])
        if temporal_rows:
            writer.writeheader()
            writer.writerows(temporal_rows)

    treatment_rows: list[dict[str, Any]] = []
    species_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(45)
    n_boot = 10000
    for outcome, key in metric_keys.items():
        endpoints: dict[tuple[str, str], list[float]] = defaultdict(list)
        for source in records:
            if (
                source["treatment"] not in endpoint_treatments
                or int(source["day_label"]) != 8
                or not source.get("has_image")
                or not source.get("capture_datetime")
            ):
                continue
            y = _float(source, key)
            if y is None or y <= 0:
                continue
            treatment_model = (
                "control" if source["treatment"].startswith("ctrl") else source["treatment"]
            )
            endpoints[(source["species"], treatment_model)].append(math.log(y))
        for treatment in ("coll", "mill", "nema", "wl1", "wl2", "mite", "enchy"):
            species_effects: list[float] = []
            species_samples: list[tuple[list[float], list[float]]] = []
            for species in SPECIES:
                treated = endpoints.get((species, treatment), [])
                control = endpoints.get((species, "control"), [])
                if not treated or not control:
                    continue
                effect = statistics.mean(treated) - statistics.mean(control)
                species_effects.append(effect)
                species_samples.append((treated, control))
                species_rows.append(
                    {
                        "outcome": outcome,
                        "species": species,
                        "grazer": treatment,
                        "n_treated": len(treated),
                        "n_control": len(control),
                        "delta_log_endpoint": effect,
                        "geometric_mean_ratio": math.exp(effect),
                    }
                )
            if not species_samples:
                continue
            estimate = statistics.mean(species_effects)
            bootstrap = np.zeros(n_boot, dtype=float)
            for treated, control in species_samples:
                treated_array = np.asarray(treated, dtype=float)
                control_array = np.asarray(control, dtype=float)
                boot_treated = rng.choice(
                    treated_array, size=(n_boot, len(treated_array)), replace=True
                ).mean(axis=1)
                boot_control = rng.choice(
                    control_array, size=(n_boot, len(control_array)), replace=True
                ).mean(axis=1)
                bootstrap += (boot_treated - boot_control) / len(species_samples)
            row = {
                "outcome": outcome,
                "grazer": treatment,
                "n_species": len(species_samples),
                "coverage": (
                    "all four species"
                    if len(species_samples) == len(SPECIES)
                    else "partial species coverage"
                ),
                "delta_log_endpoint": estimate,
                "geometric_mean_ratio": math.exp(estimate),
                "ratio_ci_low": math.exp(float(np.quantile(bootstrap, 0.025))),
                "ratio_ci_high": math.exp(float(np.quantile(bootstrap, 0.975))),
            }
            treatment_rows.append(row)

    def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("")
            return
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_rows(OUT / "treatment_day8_effects.csv", treatment_rows)
    write_rows(OUT / "treatment_day8_effects_by_species.csv", species_rows)
    return {
        "primary_observations_for_temporal_cv": len(primary),
        "primary_series_for_temporal_cv": len({r["series_key"] for r in primary}),
        "series_with_verified_day1_baseline": len(baseline_series),
        "temporal_origin": "first linked image within each series; first filename day retained",
        "temporal_models": str(temporal_output.relative_to(ROOT)),
        "treatment_effect_rows": len(treatment_rows),
        "treatment_effects": str((OUT / "treatment_day8_effects.csv").relative_to(ROOT)),
        "treatment_effects_by_species": str(
            (OUT / "treatment_day8_effects_by_species.csv").relative_to(ROOT)
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = extract_records()
    inventory_path = write_inventory(records)
    timing = _time_audit(records)
    (OUT / "timing_audit.json").write_text(json.dumps(timing, indent=2) + "\n")
    model = analyze(records, inventory_path)
    summary = {
        "inventory": str(inventory_path.relative_to(ROOT)),
        "timing_audit": str((OUT / "timing_audit.json").relative_to(ROOT)),
        "timing": timing,
        "models": model,
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
