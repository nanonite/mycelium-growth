#!/usr/bin/env python3
"""Recover per-dish tracks from the MyceliumSeg images (Chainlink #23).

Phase 1 of the plan asks: *can the images be grouped into dish tracks*, using
inoculum position, lid marks and image-embedding clustering?  This script
measures each of those signals for every image and reports whether any of them
actually partitions the release.

For each image it records, from ``scripts/triage/dish_geometry.py``:

* **dish geometry** -- rim centre and radius, found from the outer rim edge.
  The camera is fixed, so this also yields the px -> mm calibration (#25).
* **rim marker dot** -- angle (and radial position) of the saturated yellow dot
  painted on the rim.  This is the most dish-specific feature in the release.
* **colony / inoculum** -- area, equivalent radius, and the offset of the colony
  centroid from the dish centre, i.e. where the inoculum was placed.
* **two rim embeddings** -- a rotation-sensitive angular fingerprint and a
  rotation-invariant one (angular Fourier magnitudes), both detrended against
  the stage's uneven illumination.

Clustering is then tested against a Gaussian null that reproduces the observed
feature covariance, because the shared illumination structure otherwise makes
any two frames look similar.

It also MD5-hashes every image, which turns up exact duplicates across
archives -- a labelling error that matters well beyond this issue.

Outputs
    data/triage/dish_tracks.csv       one row per image
    data/triage/dish_tracks_summary.csv
    data/triage/dish_embeddings.npz    the two embeddings, for reuse
    notes/triage_dish_tracks.json      the machine-readable statistics
    notes/triage_dish_tracks.md        the write-up (edited by hand)

Usage
    .venv/bin/python scripts/triage/dish_tracks.py
    .venv/bin/python scripts/triage/dish_tracks.py --limit 40 --jobs 4   # smoke
    .venv/bin/python scripts/triage/dish_tracks.py --report-only         # no rescan
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

try:  # executed as a script: this directory is already on sys.path
    from dish_geometry import (
        RIM_DOT_N_ANG,
        TARGET_LONG_SIDE,
        Disc,
        colony_metrics,
        correlation_matrix,
        find_disc,
        find_rim_dots,
        forward_step_fraction,
        gaussian_null,
        rim_signature,
        rotation_invariant_signature,
        zscore,
    )
    from exif_parse import ENTRY_IMAGE, classify_entry, image_id_from_entry
except ImportError:  # imported as scripts.triage.dish_tracks
    from .dish_geometry import (  # type: ignore[no-redef]
        RIM_DOT_N_ANG,
        TARGET_LONG_SIDE,
        Disc,
        colony_metrics,
        correlation_matrix,
        find_disc,
        find_rim_dots,
        forward_step_fraction,
        gaussian_null,
        rim_signature,
        rotation_invariant_signature,
        zscore,
    )
    from .exif_parse import (  # type: ignore[no-redef]
        ENTRY_IMAGE,
        classify_entry,
        image_id_from_entry,
    )

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "myceliumseg"
DEFAULT_OUT = ROOT / "data" / "triage" / "dish_tracks.csv"
DEFAULT_SUMMARY = ROOT / "data" / "triage" / "dish_tracks_summary.csv"
DEFAULT_EMBED = ROOT / "data" / "triage" / "dish_embeddings.npz"
DEFAULT_NOTES = ROOT / "notes" / "triage_dish_tracks.md"

CSV_FIELDS = [
    "id", "archive", "entry", "width", "height",
    "dish_cx", "dish_cy", "dish_r", "dish_r_over_w", "edge_sharpness",
    "pixels_per_mm",
    "dot_angle_deg", "dot_saturation", "dot_concentration", "dot_rho", "n_dots",
    "colony_area_frac", "colony_radius_frac",
    "inoculum_offset", "inoculum_angle_deg",
    "agar_background", "agar_noise", "md5",
]


def load_rgb(data: bytes, target: int = TARGET_LONG_SIDE) -> np.ndarray:
    """Decode a JPEG to a small float array without paying full-resolution cost.

    ``Image.draft`` asks libjpeg for a DCT-domain downscale, which is 3-4x
    cheaper than decoding 4608x3456 and then resizing.  It only supports
    halving, so the result is finished off with a box resize.
    """
    image = Image.open(io.BytesIO(data))
    if max(image.size) > target:
        image.draft("RGB", (target, target))
    image = image.convert("RGB")
    if max(image.size) > target:
        scale = target / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BOX,
        )
    return np.asarray(image, dtype=np.float32)


def _blank_row(image_id: str, archive: str, entry: str, digest: str,
               width: int, height: int) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in CSV_FIELDS}
    row.update(id=image_id, archive=archive, entry=entry, md5=digest,
               width=width, height=height)
    return row


def _fmt(value: float, places: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{places}f}"


def scan_image(rgb: np.ndarray) -> dict[str, object]:
    """Geometry, rim-dot and colony measurements for one decoded image."""
    lum = rgb.mean(axis=2)
    disc = find_disc(lum)
    dots = find_rim_dots(rgb, disc)
    colony = colony_metrics(rgb, disc)
    strongest = dots[0] if dots else None
    return {
        "disc": disc,
        "n_dots": len(dots),
        "dot_angle_deg": strongest["angle_deg"] if strongest else float("nan"),
        "dot_saturation": strongest["saturation"] if strongest else float("nan"),
        "dot_concentration": strongest["concentration"] if strongest else float("nan"),
        "dot_rho": strongest["rho"] if strongest else float("nan"),
        **colony,
    }


def scan_archive(task: tuple[Path, int]) -> tuple[str, list[dict[str, object]], dict[str, np.ndarray]]:
    """Measure every image in one archive.

    Returns the CSV rows, the two rim embeddings, and a keyed record of which
    image each embedding row belongs to.
    """
    zip_path, limit = task
    archive = zip_path.name
    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    rot_vectors: list[np.ndarray] = []

    with zipfile.ZipFile(zip_path) as zf:
        entries = [i.filename for i in zf.infolist()
                   if classify_entry(i.filename) == ENTRY_IMAGE]
        entries.sort(key=image_id_from_entry)
        if limit:
            entries = entries[:limit]
        for entry in entries:
            image_id = image_id_from_entry(entry)
            try:
                with zf.open(entry) as handle:
                    payload = handle.read()
            except (KeyError, zipfile.BadZipFile, OSError) as exc:
                row = _blank_row(image_id, archive, entry, "", 0, 0)
                row["error"] = f"read failed: {exc}"
                rows.append(row)
                continue
            digest = hashlib.md5(payload).hexdigest()
            try:
                rgb = load_rgb(payload)
            except Exception as exc:                # noqa: BLE001 - one bad frame must not kill the run
                row = _blank_row(image_id, archive, entry, digest, 0, 0)
                row["error"] = f"decode failed: {exc}"
                rows.append(row)
                continue

            measured = scan_image(rgb)
            disc: Disc = measured["disc"]
            row = _blank_row(image_id, archive, entry, digest, disc.width, disc.height)
            row.update({
                "dish_cx": _fmt(disc.cx, 2),
                "dish_cy": _fmt(disc.cy, 2),
                "dish_r": _fmt(disc.r, 3),
                "dish_r_over_w": _fmt(disc.r_over_w, 5),
                "edge_sharpness": _fmt(disc.edge, 3),
                "pixels_per_mm": _fmt(disc.pixels_per_mm, 4),
                "dot_angle_deg": _fmt(measured["dot_angle_deg"], 2),
                "dot_saturation": _fmt(measured["dot_saturation"], 1),
                "dot_concentration": _fmt(measured["dot_concentration"], 4),
                "dot_rho": _fmt(measured["dot_rho"], 3),
                "n_dots": measured["n_dots"],
                "colony_area_frac": _fmt(measured["area_frac"], 5),
                "colony_radius_frac": _fmt(measured["radius_frac"], 5),
                "inoculum_offset": _fmt(measured["inoculum_offset"], 4),
                "inoculum_angle_deg": _fmt(measured["inoculum_angle_deg"], 2),
                "agar_background": _fmt(measured["agar_background"], 2),
                "agar_noise": _fmt(measured["agar_noise"], 2),
            })
            rows.append(row)
            vectors.append(rim_signature(rgb, disc))
            rot_vectors.append(rotation_invariant_signature(rgb, disc))

    keys = np.array([f"{r['archive']}|{r['entry']}" for r in rows], dtype=object)
    arrays = {
        "keys": keys,
        "rim": np.asarray(vectors, dtype=np.float32) if vectors else np.zeros((0, 2)),
        "rim_rot": np.asarray(rot_vectors, dtype=np.float32) if rot_vectors else np.zeros((0, 2)),
    }
    return archive, rows, arrays


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def duplicate_report(rows: list[dict[str, object]]) -> dict[str, list[list[str]]]:
    """Group byte-identical images by md5.

    Labels a duplicate when it appears under more than one condition folder,
    which is how the mislabelled ``PO`` block surfaced.
    """
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        digest = str(row.get("md5") or "")
        if digest:
            by_hash[digest].append(f"{row['archive']}|{row['entry']}")
    return {h: v for h, v in by_hash.items() if len(v) > 1}


def duplicate_pair_similarities(
    features: np.ndarray,
    keys: np.ndarray,
    md5_by_key: dict[str, str],
) -> list[float]:
    """Similarity of every byte-identical pair of images.

    There are only ~80 such pairs, so they are scored directly instead of
    materialising a membership mask over all 200M pairs.  The point of the list
    is to check *which* of the extreme pairs the clustering table is counting:
    if the pairs above 0.99 are exactly these, the tail of the similarity
    distribution is duplicates, not dish reuse.
    """
    by_hash: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        digest = md5_by_key.get(str(key), "")
        if digest:
            by_hash[digest].append(i)
    similarities: list[float] = []
    for positions in by_hash.values():
        if len(positions) < 2:
            continue
        block = features[positions]
        norms = np.maximum(np.linalg.norm(block, axis=1), 1e-12)
        unit = block / norms[:, None]
        corr = unit @ unit.T
        iu = np.triu_indices(len(positions), 1)
        similarities.extend(float(x) for x in corr[iu])
    return similarities


def clustering_report(rows: list[dict[str, object]], arrays: dict[str, np.ndarray],
                      seed: int = 0) -> dict[str, object]:
    """Compare nearest-neighbour similarity against a covariance-matched null."""
    report: dict[str, object] = {}
    md5_by_key = {f"{r['archive']}|{r['entry']}": str(r.get("md5") or "") for r in rows}
    for name in ("rim", "rim_rot"):
        features = arrays[name]
        if len(features) < 10:
            report[name] = {"n": len(features)}
            continue
        standard = zscore(features)
        corr = correlation_matrix(standard)
        nn = corr.max(axis=1)
        null = gaussian_null(standard, seed=seed)
        iu = np.triu_indices(len(standard), 1)
        pairs = corr[iu]
        duplicate_sims = duplicate_pair_similarities(standard, arrays["keys"], md5_by_key)
        block = {
            "n": len(features),
            "nn_observed": [float(x) for x in np.percentile(nn, [50, 90, 99, 100])],
            "nn_null": [float(x) for x in np.percentile(null, [50, 90, 99, 100])],
            "pairs_total": len(pairs),
            "pairs_gt_0p90": int((pairs > 0.90).sum()),
            "pairs_gt_0p95": int((pairs > 0.95).sum()),
            "pairs_gt_0p99": int((pairs > 0.99).sum()),
            "duplicate_pairs": len(duplicate_sims),
            "duplicate_pairs_gt_0p90": sum(s > 0.90 for s in duplicate_sims),
            "duplicate_pairs_gt_0p95": sum(s > 0.95 for s in duplicate_sims),
            "duplicate_pairs_gt_0p99": sum(s > 0.99 for s in duplicate_sims),
        }
        # what is left once the byte-identical pairs are taken out
        for label in ("0p90", "0p95", "0p99"):
            block[f"non_duplicate_pairs_gt_{label}"] = (
                block[f"pairs_gt_{label}"] - block[f"duplicate_pairs_gt_{label}"]
            )
        if (block["pairs_gt_0p95"] or 0) > 0:
            top = np.argsort(-pairs)[:10]
            keys = arrays["keys"]
            block["top_pairs"] = [
                {
                    "similarity": float(pairs[t]),
                    "a": str(keys[iu[0][t]]),
                    "b": str(keys[iu[1][t]]),
                }
                for t in top
            ]
        report[name] = block
    return report


def disc_consistency(rows: list[dict[str, object]]) -> dict[str, float]:
    """How stable the fitted rim is, per archive.

    A fixed camera should give a near-constant ``r_over_w``.  Large spread means
    the fit is locking onto something other than the rim, and every other
    measurement in that archive inherits the error.
    """
    per_archive: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = str(row.get("dish_r_over_w") or "")
        if value:
            per_archive[str(row["archive"])].append(float(value))
    out: dict[str, float] = {}
    for archive, values in per_archive.items():
        array = np.asarray(values)
        out[f"{archive}:median"] = float(np.median(array))
        out[f"{archive}:spread_iqr"] = float(np.percentile(array, 75) - np.percentile(array, 25))
    return out


def build_summary(rows: list[dict[str, object]], duplicates: dict[str, list[str]],
                  clusters: dict[str, object]) -> list[dict[str, object]]:
    by_archive: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_archive[str(row["archive"])].append(row)

    summary: list[dict[str, object]] = []
    for archive, group in sorted(by_archive.items()):
        radii = [float(r["dish_r_over_w"]) for r in group if r["dish_r_over_w"]]
        dots = [float(r["dot_angle_deg"]) for r in group if r["dot_angle_deg"]]
        areas = [float(r["colony_radius_frac"]) for r in group if r["colony_radius_frac"]]
        offsets = [float(r["inoculum_offset"]) for r in group if r["inoculum_offset"]]
        summary.append({
            "archive": archive,
            "images": len(group),
            "dish_r_over_w_median": round(float(np.median(radii)), 5) if radii else "",
            "dish_r_over_w_iqr": round(float(np.percentile(radii, 75) - np.percentile(radii, 25)), 5) if radii else "",
            "with_dot": len(dots),
            "dot_angles_distinct": len(set(dots)),
            "dot_angle_fwd_steps": round(forward_step_fraction(dots), 4) if dots else "",
            "colony_radius_frac_median": round(float(np.median(areas)), 4) if areas else "",
            "colony_radius_fwd_steps": round(forward_step_fraction(areas), 4) if areas else "",
            "inoculum_offset_median": round(float(np.median(offsets)), 4) if offsets else "",
            "duplicated_images": sum(1 for r in group if r["md5"] in duplicates),
        })
    return summary


def same_angle_pairs(angles: np.ndarray, tol: float) -> tuple[int, float]:
    """Pairs of images whose dot angles are within ``tol`` degrees.

    Returns ``(observed, expected_if_uniform)``.  The observed count includes
    the endpoints (the CSV stores angles to 0.01 deg, so ties are common), the
    expectation is ``C(n, 2) * 2 * tol / 360`` for angles spread uniformly over
    the circle.

    Read as a *quantisation* check, not as a dish test: below the sampling grid
    of ``find_rim_dots`` the pairs are dominated by the grid itself.  The dish
    question is answered by the multiplicity numbers in
    :func:`dot_angle_structure` -- a reused dish shows up as one angle with a
    multiplicity of ``n / dishes``, not as a few thousand slightly-nudged ones.
    """
    ordered = np.sort(angles)
    n = len(ordered)
    wrapped = np.concatenate([ordered, ordered + 360.0])
    idx = np.searchsorted(wrapped, wrapped[:n] + tol, side="right")
    observed = int((idx - np.arange(1, n + 1)).sum())
    expected = n * (n - 1) / 2.0 * (2.0 * tol / 360.0)
    return observed, round(expected, 4)


def dot_angle_structure(rows: list[dict[str, object]], bin_deg: float = 0.5) -> dict[str, object]:
    """Is the rim-dot angle discrete (a few dishes) or spread out (many)?

    Three readings of the same question, from strongest to weakest:

    * **multiplicity** -- if *K* dishes were re-photographed with the dish in a
      fixed orientation, an angle would repeat about ``n / K`` times.  The
      largest repetition actually seen is reported here.
    * **occupancy** -- a small *K* leaves whole arcs of the circle empty; the
      half-degree histogram has ``360 / bin_deg`` cells, and its occupancy and
      chi-square against uniform are both reported.
    * **collision counts** -- pairs of images within 0.05 / 0.10 deg, against
      the uniform expectation.  These are *not* a dish test (see
      :func:`same_angle_pairs`); they are recorded so the fine-scale pile-up is
      on the record rather than open to interpretation.
    """
    angles = [float(r["dot_angle_deg"]) for r in rows if r["dot_angle_deg"]]
    if not angles:
        return {}
    array = np.asarray(angles)
    bins = np.clip((array / bin_deg).astype(int), 0, int(360.0 / bin_deg) - 1)
    observed = np.bincount(bins, minlength=int(360.0 / bin_deg)).astype(float)
    expected = np.full(len(observed), len(array) / len(observed))
    chi2 = float((((observed - expected) ** 2) / expected).sum())

    values, counts = np.unique(array, return_counts=True)
    grid_deg = 360.0 / RIM_DOT_N_ANG
    collisions = {}
    for tol in (0.05, 0.10):
        seen, chance = same_angle_pairs(array, tol)
        collisions[f"{tol:.2f}"] = {"observed": seen, "expected_uniform": chance}

    return {
        "n": len(array),
        "distinct": len(values),
        "max_multiplicity": int(counts.max()),
        "max_multiplicity_share": round(float(counts.max()) / len(array), 5),
        "median_multiplicity": float(np.median(counts)),
        "on_detector_grid_frac": round(
            float(np.mean(np.abs(np.round(array / grid_deg) - array / grid_deg) < 1e-6)), 4
        ),
        "detector_grid_deg": grid_deg,
        "occupied_bins": int((observed > 0).sum()),
        "total_bins": len(observed),
        "chi2_vs_uniform": round(chi2, 1),
        "df": len(observed) - 1,
        "chi2_per_df": round(chi2 / (len(observed) - 1), 4),
        "collisions": collisions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBED)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--limit", type=int, default=0, help="images per archive (smoke test)")
    parser.add_argument("--jobs", type=int, default=11, help="archives scanned in parallel")
    parser.add_argument("--report-only", action="store_true",
                        help="reuse an existing --out and --embeddings instead of rescanning")
    args = parser.parse_args(argv)

    archives = sorted(args.raw_dir.glob("*.zip"))
    if not archives:
        parser.error(f"no *.zip under {args.raw_dir}")

    arrays: dict[str, np.ndarray] = {}
    if args.report_only:
        if not args.out.exists():
            parser.error(f"--report-only needs {args.out} to exist")
        with args.out.open(encoding="utf-8") as handle:
            rows: list[dict[str, object]] = list(csv.DictReader(handle))
        print(f"reusing {len(rows)} rows from {args.out}", flush=True)
        # The clustering block lives in the embeddings, not in the CSV, so
        # rebuild it from the saved arrays -- otherwise a report-only run would
        # silently rewrite the notes with an empty "clusters" section.
        if args.embeddings.exists():
            with np.load(args.embeddings, allow_pickle=True) as bundle:
                arrays = {key: bundle[key] for key in bundle.files}
            print(f"reusing embeddings from {args.embeddings}", flush=True)
        else:
            print(f"warning: {args.embeddings} missing; clustering skipped",
                  file=sys.stderr, flush=True)
    else:
        tasks = [(path, args.limit) for path in archives]
        print(f"scanning {len(archives)} archives "
              f"(limit={args.limit or 'all'}, jobs={args.jobs})", flush=True)
        started = time.time()
        rows = []
        parts: list[dict[str, np.ndarray]] = []
        if args.jobs > 1 and len(archives) > 1:
            with ProcessPoolExecutor(max_workers=min(args.jobs, len(archives))) as pool:
                for archive, archive_rows, part in pool.map(scan_archive, tasks):
                    print(f"  {archive}: {len(archive_rows)} images", flush=True)
                    rows.extend(archive_rows)
                    parts.append(part)
        else:
            for task in tasks:
                archive, archive_rows, part = scan_archive(task)
                print(f"  {archive}: {len(archive_rows)} images", flush=True)
                rows.extend(archive_rows)
                parts.append(part)
        for key in ("keys", "rim", "rim_rot"):
            arrays[key] = np.concatenate([part[key] for part in parts], axis=0)
        rows.sort(key=lambda r: (str(r["archive"]), str(r["id"])))
        print(f"scanned {len(rows)} images in {time.time() - started:.0f}s", flush=True)
        write_csv(args.out, rows)
        args.embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.embeddings, **arrays)
        print(f"wrote {args.out} and {args.embeddings}", flush=True)

    rows.sort(key=lambda r: (str(r["archive"]), str(r["id"])))
    duplicates = duplicate_report(rows)
    clusters = clustering_report(rows, arrays) if "rim" in arrays else {}
    summary = build_summary(rows, duplicates, clusters)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    structure = dot_angle_structure(rows)
    report = {
        "images": len(rows),
        "dot_angles": structure,
        "duplicates": {h: v for h, v in duplicates.items()},
        "clusters": clusters,
        "disc_consistency": disc_consistency(rows),
    }
    payload = json.dumps(report, indent=2, default=str)
    args.notes.with_suffix(".json").write_text(payload, encoding="utf-8")

    print(f"\nimages scanned      : {len(rows)}")
    print(f"exact duplicates    : {len(duplicates)} hashes, "
          f"{sum(len(v) for v in duplicates.values())} images")
    if structure:
        print(f"dot angles          : {structure['n']} detected, "
              f"{structure['distinct']} distinct, chi2/df vs uniform "
              f"{structure['chi2_per_df']}")
    for name, block in clusters.items():
        if "nn_observed" not in block:
            continue
        print(f"{name:8s} nn-corr median obs={block['nn_observed'][0]:.3f} "
              f"null={block['nn_null'][0]:.3f} | pairs>0.95: "
              f"{block['pairs_gt_0p95']}/{block['pairs_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())