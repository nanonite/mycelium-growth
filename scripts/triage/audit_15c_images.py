#!/usr/bin/env python3
"""Audit eligible G. lucidum 15 °C images for cross-sectional morphology (Chainlink #34).

Gate C (#26) established that MyceliumSeg has no dish identity and no capture
time, so this is an image-count and descriptive-feasibility audit only: it
verifies the 10 candidate 15 °C images against the source archive, the
duplicate map, the label triage, the px→mm calibration and the existing
measurement/provenance artifacts, and reports a validated 25 °C GL count for
context.  It does **not** compute dish-level power, estimate causal
temperature effects, or claim physical growth rates — images are not
independent dishes.

Outputs
    data/triage/audit_15c.json    machine-readable audit result
    stdout                        human-readable summary

Usage
    .venv/bin/python scripts/triage/audit_15c_images.py
    .venv/bin/python scripts/triage/audit_15c_images.py --write
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw" / "myceliumseg"
TRIAGE_DIR = ROOT / "data" / "triage"
MEASUREMENTS = ROOT / "data" / "measurements.csv"
KNOWN_LABELS = TRIAGE_DIR / "known_labels.csv"
DISH_TRACKS = TRIAGE_DIR / "dish_tracks.csv"
IMAGE_SCALE = TRIAGE_DIR / "image_scale.csv"
OUT_PATH = TRIAGE_DIR / "audit_15c.json"

TEMP15_ARCHIVE = "labeled-MYG_PDA_TEMP.zip"
TEMP15_FOLDER = "TEMP15"


def load_known_labels() -> dict[str, dict[str, str]]:
    """id -> {species, medium, temp, source} from known_labels.csv."""
    labels: dict[str, dict[str, str]] = {}
    with KNOWN_LABELS.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = {
                "species": row["species_if_known"],
                "medium": row["medium_if_known"],
                "temp": row["temp_if_known"],
                "source": row["source"],
            }
    return labels


def load_md5() -> dict[str, str]:
    """id -> md5 from dish_tracks.csv (all 20,176 images)."""
    md5: dict[str, str] = {}
    with DISH_TRACKS.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            md5[row["id"]] = row["md5"]
    return md5


def load_image_scale() -> dict[str, dict[str, str]]:
    """id -> {block, px_per_mm, fit_status} from image_scale.csv."""
    scale: dict[str, dict[str, str]] = {}
    with IMAGE_SCALE.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scale[row["id"]] = {
                "block": row["block"],
                "px_per_mm": row["px_per_mm"],
                "fit_status": row["fit_status"],
            }
    return scale


def load_measurements() -> dict[str, dict[str, str]]:
    """id -> row from measurements.csv (125 labeled hold-outs)."""
    meas: dict[str, dict[str, str]] = {}
    with MEASUREMENTS.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meas[row["id"]] = row
    return meas


def find_temp15_in_archives() -> list[dict[str, str]]:
    """Find all TEMP15 image entries across all 11 archives.

    Returns a list of {archive, entry, id} for images whose zip path starts
    with TEMP15/.  Also detects any TEMP15 image id that appears in a
    *different* archive (cross-archive id collision).
    """
    import re
    import zipfile

    _id_re = re.compile(r"(\d{4,8})(?=\.[^.]+$)")
    entries: list[dict[str, str]] = []
    for zpath in sorted(RAW_DIR.glob("*.zip")):
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                if "__MACOSX/" in name or name.split("/")[-1].startswith("._"):
                    continue
                if not name.lower().endswith((".jpg", ".jpeg")):
                    continue
                if name.split("/")[0] != TEMP15_FOLDER:
                    continue
                m = _id_re.search(name.rsplit("/", 1)[-1])
                if m:
                    entries.append({
                        "archive": zpath.name,
                        "entry": name,
                        "id": m.group(1),
                    })
    return entries


def find_id_collisions(temp15_ids: set[str]) -> list[dict[str, str]]:
    """Check whether any TEMP15 image id appears in a different archive."""
    import re
    import zipfile

    _id_re = re.compile(r"(\d{4,8})(?=\.[^.]+$)")
    collisions: list[dict[str, str]] = []
    for zpath in sorted(RAW_DIR.glob("*.zip")):
        if zpath.name == TEMP15_ARCHIVE:
            continue
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                if "__MACOSX/" in name or name.split("/")[-1].startswith("._"):
                    continue
                if not name.lower().endswith((".jpg", ".jpeg")):
                    continue
                m = _id_re.search(name.rsplit("/", 1)[-1])
                if m and m.group(1) in temp15_ids:
                    collisions.append({
                        "id": m.group(1),
                        "archive": zpath.name,
                        "entry": name,
                    })
    return collisions


def audit() -> dict[str, object]:
    """Run the full audit and return the result dict."""
    labels = load_known_labels()
    md5 = load_md5()
    scale = load_image_scale()
    meas = load_measurements()

    # ── 1. TEMP15 images in the source archive ──
    temp15_entries = find_temp15_in_archives()
    temp15_ids = sorted(e["id"] for e in temp15_entries)

    # ── 2. Cross-archive id collisions ──
    collisions = find_id_collisions(set(temp15_ids))

    # ── 3. Duplicate check (md5) ──
    temp15_md5s = {i: md5.get(i) for i in temp15_ids}
    md5_groups: dict[str, list[str]] = defaultdict(list)
    for img_id, h in md5.items():
        md5_groups[h].append(img_id)
    duplicate_ids: list[str] = []
    for img_id in temp15_ids:
        h = temp15_md5s.get(img_id)
        if h and len(md5_groups[h]) > 1:
            duplicate_ids.append(img_id)
    n_distinct_md5 = len({h for h in temp15_md5s.values() if h})

    # ── 4. Label verification ──
    label_issues: list[dict[str, str]] = []
    for img_id in temp15_ids:
        lbl = labels.get(img_id, {})
        if lbl.get("species") != "GL":
            label_issues.append({"id": img_id, "issue": f"species={lbl.get('species')}"})
        if lbl.get("medium") != "MYG":
            label_issues.append({"id": img_id, "issue": f"medium={lbl.get('medium')}"})
        if lbl.get("temp") != "15C":
            label_issues.append({"id": img_id, "issue": f"temp={lbl.get('temp')}"})

    # ── 5. Quality / off-scale check ──
    off_scale_ids: list[str] = []
    for img_id in temp15_ids:
        s = scale.get(img_id, {})
        if s.get("fit_status") != "ok":
            off_scale_ids.append(img_id)

    # ── 6. Measurement artifact verification ──
    missing_from_measurements: list[str] = []
    label_conflicts: list[str] = []
    for img_id in temp15_ids:
        row = meas.get(img_id)
        if row is None:
            missing_from_measurements.append(img_id)
        elif row.get("label_conflict", "").lower() == "true":
            label_conflicts.append(img_id)

    # ── 7. Eligibility ──
    eligible_ids = [
        i for i in temp15_ids
        if i not in duplicate_ids
        and i not in off_scale_ids
        and i not in missing_from_measurements
        and i not in label_conflicts
        and not any(iss["id"] == i for iss in label_issues)
    ]

    # ── 8. 25 °C GL context count ──
    gl25_ids = [
        i for i, lbl in labels.items()
        if lbl["species"] == "GL" and lbl["temp"] == "25C"
    ]
    gl25_md5s = {md5[i] for i in gl25_ids if i in md5}
    gl25_off_scale = sum(
        1 for i in gl25_ids if scale.get(i, {}).get("fit_status") != "ok"
    )
    gl25_ok = len(gl25_ids) - gl25_off_scale

    # ── 9. Cross-temperature duplicate check ──
    gl25_md5_set = gl25_md5s
    temp15_md5_set = {h for h in temp15_md5s.values() if h}
    cross_temp_overlap = temp15_md5_set & gl25_md5_set

    return {
        "task": "#34",
        "scope": "image-count and descriptive-feasibility audit (Gate C)",
        "temp15": {
            "source_archive": TEMP15_ARCHIVE,
            "source_folder": TEMP15_FOLDER,
            "n_images": len(temp15_ids),
            "image_ids": temp15_ids,
            "entries": temp15_entries,
            "n_distinct_md5": n_distinct_md5,
            "duplicate_ids": duplicate_ids,
            "cross_archive_collisions": collisions,
            "label_issues": label_issues,
            "off_scale_ids": off_scale_ids,
            "missing_from_measurements": missing_from_measurements,
            "label_conflicts": label_conflicts,
            "n_eligible": len(eligible_ids),
            "eligible_ids": eligible_ids,
        },
        "gl25_context": {
            "n_images": len(gl25_ids),
            "n_distinct_md5": len(gl25_md5s),
            "n_off_scale": gl25_off_scale,
            "n_ok": gl25_ok,
        },
        "cross_temperature_md5_overlap": len(cross_temp_overlap),
        "feasibility": {
            "descriptive_comparison": "feasible",
            "causal_temperature_effect": "not identified (Gate C: no dish identity, no replication, no time axis)",
            "physical_growth_rate": "not identifiable (no capture interval)",
            "dish_level_power": "not computable (images are not independent dishes)",
            "sample_size_limitation": "n=10 images at 15 °C; no dish-level replication; descriptive only",
        },
    }


def print_summary(result: dict[str, object]) -> None:
    """Print a human-readable summary."""
    t = result["temp15"]
    g = result["gl25_context"]
    print("=" * 70)
    print("Chainlink #34: Eligible G. lucidum 15 °C image audit")
    print("=" * 70)
    print(f"\nSource: {t['source_archive']} / {t['source_folder']}/")
    print(f"  Images found:          {t['n_images']}")
    print(f"  Distinct md5:          {t['n_distinct_md5']}")
    print(f"  Duplicate ids:         {len(t['duplicate_ids'])}")
    print(f"  Cross-archive collisions: {len(t['cross_archive_collisions'])}")
    print(f"  Label issues:          {len(t['label_issues'])}")
    print(f"  Off-scale:             {len(t['off_scale_ids'])}")
    print(f"  Missing from measurements: {len(t['missing_from_measurements'])}")
    print(f"  Label conflicts:       {len(t['label_conflicts'])}")
    print("  ──")
    print(f"  ELIGIBLE:              {t['n_eligible']}")
    print("\n25 °C GL context:")
    print(f"  Images:                {g['n_images']}")
    print(f"  Distinct md5:          {g['n_distinct_md5']}")
    print(f"  Off-scale:             {g['n_off_scale']}")
    print(f"  fit_status=ok:         {g['n_ok']}")
    print(f"\nCross-temperature md5 overlap: {result['cross_temperature_md5_overlap']}")
    print(f"\nFeasibility: {result['feasibility']['descriptive_comparison']}")
    print(f"  Causal temperature effect: {result['feasibility']['causal_temperature_effect']}")
    print(f"  Physical growth rate:     {result['feasibility']['physical_growth_rate']}")
    print(f"  Dish-level power:         {result['feasibility']['dish_level_power']}")
    print(f"  Sample-size limitation:   {result['feasibility']['sample_size_limitation']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true",
        help=f"write audit result to {OUT_PATH}",
    )
    args = parser.parse_args()

    result = audit()
    print_summary(result)

    if args.write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
