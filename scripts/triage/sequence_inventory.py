#!/usr/bin/env python3
"""Inventory time-resolved image sequences and frame provenance (#61).

For every dataset under ``data/raw`` this records the six provenance facts a
sequence claim needs -- ordered frames, specimen/dish identity, capture
interval, resolution, duplicates, condition labels -- and maps each dataset to
the estimand families of ``notes/estimands.md`` that it can identify.

The MyceliumSeg half reuses the stored outputs of #21-#25 instead of rescanning
56 GB; the sequence half reads container headers, EXIF capture times, zarr
metadata and per-frame content hashes directly from the files on disk.
``zipfile-deflate64`` (optional) lets the stdlib zipfile code read the
DEFLATE64-compressed ``Zygomycetous1.zip``; without it those six entries are
recorded as unreadable rather than dropped.

Outputs
    data/triage/sequence_inventory.csv   one row per dataset: the capability matrix
    notes/sequence_inventory.json        per-file evidence behind every row

Usage
    .venv/bin/python scripts/triage/sequence_inventory.py                # full audit
    .venv/bin/python scripts/triage/sequence_inventory.py --no-hashes   # skip content hashing
    .venv/bin/python scripts/triage/sequence_inventory.py --limit 5     # smoke test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

try:  # executed as a script: this directory is already on sys.path
    from exif_parse import parse_jpeg_header
except ImportError:  # imported as scripts.triage.sequence_inventory
    from .exif_parse import parse_jpeg_header  # type: ignore[no-redef]

try:  # optional: Zygomycetous1.zip stores DEFLATE64 (method 9) members
    from zipfile_deflate64 import ZipFile as Deflate64ZipFile

    HAVE_DEFLATE64 = True
except ImportError:  # pragma: no cover - environment without the extra
    Deflate64ZipFile = None  # type: ignore[assignment,misc]
    HAVE_DEFLATE64 = False

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_TRIAGE_DIR = ROOT / "data" / "triage"
DEFAULT_OUT = DEFAULT_TRIAGE_DIR / "sequence_inventory.csv"
DEFAULT_JSON_OUT = ROOT / "notes" / "sequence_inventory.json"

#: One row per dataset; the capability matrix of notes/provenance.md mirrors this.
CSV_FIELDS = [
    "dataset",
    "kind",
    "location",
    "n_units",
    "n_specimens",
    "ordered_frames",
    "order_evidence",
    "specimen_identity",
    "identity_basis",
    "capture_interval",
    "interval_source",
    "resolution",
    "resolution_source",
    "duplicates",
    "duplicate_rule",
    "condition_labels",
    "estimands_identifiable",
    "estimands_blocked",
    "evidence",
]

#: Estimand families (notes/estimands.md); the vocabulary a row may claim.
ESTIMAND_TOKENS = frozenset(
    {"image_morphology", "delta_per_ordered_step", "physical_rate",
     "intervention_effect"}
)

GRAZER_RX = re.compile(
    r"^(?P<sp>[A-Z][a-z]{1,2})_(?:(?P<block>DD\d+)_)?(?P<trt>[a-z]+\d*)_"
    r"[Dd](?P<day>\d+)_(?P<rep>\d+)\.jpg$"
)
PODOSPORA_RX = re.compile(r"^(?P<medium>M\d)_(?P<rep>\d)\.avi$")
NETWORK_SCHEME1_RX = re.compile(
    r"^(?P<taxon>[A-Za-z0-9]+)\((?P<rep>\d+)\)_t(?P<t>\d+)(?:_c)?$"
)
NETWORK_SCHEME2_RX = re.compile(
    r"^(?P<sp>[A-Z][a-z])_(?P<trt>[a-z]+\d*)_d(?P<day>\d+)_(?P<rep>\d+)$"
)
NETWORK_SCHEME3_RX = re.compile(
    r"^(?P<sp>[A-Z][a-z]) (?P<temp>\d+C) day (?P<day>\d+) (?P<rep>\d+)$"
)


# --------------------------------------------------------------------------- #
# pure parsing / statistics helpers
# --------------------------------------------------------------------------- #


def parse_grazer_name(base: str) -> dict[str, object] | None:
    """``Hf_DD2_coll_D1_1.jpg`` -> species, treatment, day, replicate.

    The block (``DD2``) is optional and Hf-only; the day marker is ``D`` in
    some treatments and ``d`` in others -- both mean the imaging day.
    """
    m = GRAZER_RX.match(base)
    if not m:
        return None
    return {
        "species": m.group("sp"),
        "block": m.group("block") or "",
        "treatment": m.group("trt"),
        "day": int(m.group("day")),
        "replicate": int(m.group("rep")),
    }


def parse_network_name(base: str) -> dict[str, object] | None:
    """Three coexisting naming schemes in the Fricker network zips.

    ``C41(1)_t25_c.jpg`` (taxon, replicate, day 25) --
    ``Pv_ctrl1_d8_1.jpg`` (species, treatment, day 8, replicate) --
    ``Pi 18C day 8 001.jpg`` (species, temperature, day 8, replicate).
    The extension is stripped before parsing.  Each image is one colony at one
    time point; ``colony`` keys the group.
    """
    m = NETWORK_SCHEME1_RX.match(base)
    if m:
        taxon, rep = m.group("taxon"), int(m.group("rep"))
        return {"colony": f"{taxon}({rep})", "day": int(m.group("t")),
                "condition": ""}
    m = NETWORK_SCHEME2_RX.match(base)
    if m:
        sp, trt = m.group("sp"), m.group("trt")
        rep = int(m.group("rep"))
        return {"colony": f"{sp}_{trt}_{rep}", "day": int(m.group("day")),
                "condition": trt}
    m = NETWORK_SCHEME3_RX.match(base)
    if m:
        sp, rep = m.group("sp"), int(m.group("rep"))
        return {"colony": f"{sp}_{m.group('temp')}_{rep}",
                "day": int(m.group("day")),
                "condition": m.group("temp")}
    return None


def parse_imagej_description(text: str) -> dict[str, int]:
    """``ImageJ=1.52t\\nimages=85\\n...`` -> ``{"images": 85}``."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        try:
            out[key.strip()] = int(value.strip())
        except ValueError:
            continue
    return out


def imagej_frame_layout(
    filesize: int, data_start: int, strip_bytes: int
) -> tuple[int, int, bool]:
    """Frames and trailer of an ImageJ stack whose IFD chain is broken.

    ImageJ writes one IFD and then ``n`` contiguous uncompressed pages; the
    next-IFD pointer overflows, so the page count is recovered arithmetically:
    ``(filesize - data_start) // strip_bytes``.  The remainder is the trailer
    (padding / stale IFDs), not a partial frame.
    """
    if strip_bytes <= 0 or filesize <= data_start:
        return 0, 0, False
    frames, trailer = divmod(filesize - data_start, strip_bytes)
    return frames, trailer, frames > 0


def normalize_exif_datetime(value: object) -> str | None:
    """EXIF ``2009:02:25 11:55:09`` (or ISO) -> ISO ``datetime`` string."""
    if not value:
        return None
    text = str(value)
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            # EXIF times are camera-local and carry no zone; keep them naive
            # so gaps are computed against the camera clock, not a fake UTC
            return datetime.strptime(text, fmt).isoformat()  # noqa: DTZ007
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text).isoformat()
    except ValueError:
        return None


def avi_headers(data: bytes) -> dict[str, object]:
    """Parse ``avih``/``strh``/``strf`` from the leading bytes of an AVI file.

    ``dwMicroSecPerFrame`` and the stream rate are *playback* timing written by
    the encoder; they are recorded as such and never used as a capture
    interval.
    """
    info: dict[str, object] = {}
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"AVI ":
        raise ValueError("not an AVI RIFF file")

    def walk(start: int, end: int) -> None:
        i = start
        while i + 8 <= end:
            cid = data[i:i + 4]
            size = struct.unpack("<I", data[i + 4:i + 8])[0]
            payload, pend = i + 8, i + 8 + size
            if cid == b"LIST" and payload + 4 <= len(data):
                ltype = data[payload:payload + 4]
                if ltype in (b"hdrl", b"strl"):
                    walk(payload + 4, min(pend, len(data)))
            elif cid == b"avih" and size >= 40:
                (usec, _, _, _, total, _, streams, _, width,
                 height) = struct.unpack("<10I", data[payload:payload + 40])
                info.update(microsec_per_frame=usec, total_frames=total,
                            streams=streams, avih_width=width,
                            avih_height=height)
            elif cid == b"strh" and size >= 56:
                body = data[payload:payload + 56]
                scale, rate, _, length = struct.unpack("<4I", body[20:36])
                info.update(
                    fcc=body[0:4].decode("latin1"),
                    handler=body[4:8].decode("latin1"),
                    scale=scale, rate=rate, strh_frames=length,
                    playback_fps=(rate / scale if scale else None),
                )
            elif cid == b"strf" and size >= 40:
                _, width, height, _, bitcount, compression = struct.unpack(
                    "<IiiHHI", data[payload:payload + 20]
                )
                info.update(bmp_width=width, bmp_height=abs(height),
                            bitcount=bitcount, compression=hex(compression))
            i = pend + (size & 1)

    total = struct.unpack("<I", data[4:8])[0]
    walk(12, min(12 + total, len(data)))
    return info


def iter_movi_chunks(path: Path) -> list[tuple[str, int, int]]:
    """``(ckid, payload_offset, size)`` for every video chunk in ``movi``.

    Walking the RIFF structure with seeks reads only headers; the frame bytes
    themselves are read later, when hashing.
    """
    chunks: list[tuple[str, int, int]] = []
    with path.open("rb") as handle:
        head = handle.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"AVI ":
            raise ValueError(f"not an AVI file: {path}")
        end = 12 + struct.unpack("<I", head[4:8])[0]
        stack = [(12, end)]
        while stack:
            start, stop = stack.pop()
            i = start
            while i + 8 <= stop:
                handle.seek(i)
                hdr = handle.read(8)
                if len(hdr) < 8:
                    break
                cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
                payload = i + 8
                if cid == b"LIST":
                    handle.seek(payload)
                    ltype = handle.read(4)
                    if ltype in (b"movi", b"hdrl", b"strl", b"INFO"):
                        stack.append((payload + 4, min(payload + size, stop)))
                elif cid[:2].isdigit() and cid[2:4] in (b"dc", b"db", b"pc"):
                    chunks.append(
                        (cid.decode("latin1"), payload,
                         min(size, stop - payload))
                    )
                i = payload + size + (size & 1)
    return chunks


def hash_region(path: Path, offset: int, size: int) -> str:
    """MD5 of ``size`` bytes at ``offset`` (streamed, constant memory)."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = size
        while remaining > 0:
            block = handle.read(min(1 << 20, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def duplicate_stats(hashes: list[str]) -> dict[str, object]:
    """How many items, distinct items, shared groups, redundant copies."""
    counts = Counter(hashes)
    groups = {h: n for h, n in counts.items() if n > 1}
    return {
        "n": len(hashes),
        "distinct": len(counts),
        "duplicate_groups": len(groups),
        "items_in_duplicate_groups": sum(groups.values()),
        "redundant_copies": len(hashes) - len(counts),
        "largest_group": max(groups.values(), default=1),
    }


def timestamp_stats(ns_values: list[int]) -> dict[str, object]:
    """Order and interval facts for a per-frame UTC-nanosecond series."""
    if len(ns_values) < 2:
        return {"n": len(ns_values), "strictly_increasing": len(ns_values) == 1}
    gaps = [b - a for a, b in pairwise(ns_values)]
    seconds = sorted(g / 1e9 for g in gaps)
    mid = len(seconds) // 2
    median = (seconds[mid] if len(seconds) % 2
              else (seconds[mid - 1] + seconds[mid]) / 2)

    def iso(ns: int) -> str:
        return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()

    return {
        "n": len(ns_values),
        "strictly_increasing": all(g > 0 for g in gaps),
        "first_utc": iso(ns_values[0]),
        "last_utc": iso(ns_values[-1]),
        "span_days": round((ns_values[-1] - ns_values[0]) / 86400e9, 3),
        "gap_min_s": round(seconds[0], 3),
        "gap_median_s": round(float(median), 3),
        "gap_max_s": round(seconds[-1], 3),
        "gap_distinct_s": sorted({round(g, 3) for g in seconds})[:10],
    }


def series_stats(series: object, entries: list[dict[str, object]]) -> dict:
    """Order and interval facts for one specimen's day-ordered images.

    Order is *confirmed* when the EXIF capture times do not go backwards as
    the filename day increases; a series without EXIF is order-neutral.
    """
    ordered = sorted(entries, key=lambda e: int(e["day"]))
    days = [int(e["day"]) for e in ordered]
    day_counts = Counter(days)
    same_day_reshots = sum(n - 1 for n in day_counts.values() if n > 1)

    pairs = [(a, b) for a, b in pairwise(ordered)
             if a.get("datetime") and b.get("datetime")]
    backwards = [
        (int(a["day"]), int(b["day"])) for a, b in pairs
        if str(b["datetime"]) < str(a["datetime"])
    ]
    gaps_days = [
        (datetime.fromisoformat(str(b["datetime"]))
         - datetime.fromisoformat(str(a["datetime"]))).total_seconds() / 86400
        for a, b in pairs
    ]
    if not gaps_days:  # fall back to the filename day axis
        gaps_days = [float(b - a) for a, b in pairwise(days)]

    exif = [str(e["datetime"]) for e in ordered if e.get("datetime")]
    identical_ts = sum(1 for a, b in pairs
                       if str(a["datetime"]) == str(b["datetime"]))
    return {
        "series": list(series),  # type: ignore[arg-type]
        "n_images": len(ordered),
        "days": days,
        "n_days": len(set(days)),
        "with_exif": len(exif),
        "order_confirmed": bool(pairs) and not backwards,
        "order_backwards": backwards,
        "identical_timestamp_pairs": identical_ts,
        "same_day_reshots": same_day_reshots,
        "gaps_days": [round(g, 4) for g in gaps_days],
        "first_exif": exif[0] if exif else None,
        "last_exif": exif[-1] if exif else None,
    }


def aggregate_series(series_list: list[dict]) -> dict[str, object]:
    """Pool per-series facts: confirmations, interval distribution, day span."""
    gaps = [g for s in series_list for g in s["gaps_days"]]
    gaps_sorted = sorted(gaps)
    if not gaps_sorted:
        median = None
    else:
        mid = len(gaps_sorted) // 2
        median = (gaps_sorted[mid] if len(gaps_sorted) % 2
                  else (gaps_sorted[mid - 1] + gaps_sorted[mid]) / 2)
    return {
        "n_series": len(series_list),
        "n_series_multi_day": sum(1 for s in series_list if s["n_days"] > 1),
        "n_order_confirmed": sum(1 for s in series_list
                                 if s["order_confirmed"]),
        "n_order_conflicts": sum(1 for s in series_list
                                 if s["order_backwards"]),
        "n_series_identical_timestamps": sum(
            1 for s in series_list if s["identical_timestamp_pairs"]
        ),
        "n_identical_timestamp_pairs": sum(
            s["identical_timestamp_pairs"] for s in series_list
        ),
        "n_without_exif": sum(1 for s in series_list if not s["with_exif"]),
        "n_same_day_reshots": sum(s["same_day_reshots"] for s in series_list),
        "day_min": min((min(s["days"]) for s in series_list if s["days"]),
                       default=None),
        "day_max": max((max(s["days"]) for s in series_list if s["days"]),
                       default=None),
        "gap_days_median": round(median, 3) if median is not None else None,
        "gap_days_min": round(min(gaps), 3) if gaps else None,
        "gap_days_max": round(max(gaps), 3) if gaps else None,
        "n_gap_obs": len(gaps),
    }


def label_conflicts(
    md5_by_id: dict[str, str], labels: dict[str, tuple[str, str, str]]
) -> dict[str, int]:
    """md5 groups whose copies carry *different* non-blank labels.

    These are the images where the folder-derived condition of #24 disagrees
    with the pixels (#23): one md5 group can only have one true condition.
    ``conflict_groups`` = 20 reproduces the documented figure.
    """
    by_hash: dict[str, list[str]] = defaultdict(list)
    for image_id, md5 in md5_by_id.items():
        by_hash[md5].append(image_id)
    groups = involved = 0
    for ids in by_hash.values():
        if len(ids) < 2:
            continue
        known = {labels[i] for i in ids if i in labels and labels[i][0]}
        if len(known) > 1:
            groups += 1
            involved += len(ids)
    return {"conflict_groups": groups, "images_involved": involved}


def validate_rows(rows: list[dict[str, str]]) -> None:
    """Refuse a capability row whose claims contradict its own evidence."""
    if not rows:
        raise ValueError("capability matrix is empty")
    for row in rows:
        missing = [f for f in CSV_FIELDS if f not in row]
        if missing:
            raise ValueError(f"{row.get('dataset', '?')}: missing {missing}")
        ident = {t.strip() for t in row["estimands_identifiable"].split(";")
                 if t.strip()}
        blocked = {t.strip() for t in row["estimands_blocked"].split(";")
                   if t.strip()}
        unknown = (ident | blocked) - ESTIMAND_TOKENS
        if unknown:
            raise ValueError(f"{row['dataset']}: unknown estimand {unknown}")
        if ident & blocked:
            raise ValueError(f"{row['dataset']}: {ident & blocked} both "
                             "identifiable and blocked")
        if row["ordered_frames"] not in {"yes", "no", "n/a"}:
            raise ValueError(f"{row['dataset']}: "
                             f"ordered_frames={row['ordered_frames']!r}")
        if (row["ordered_frames"] == "no"
                and "delta_per_ordered_step" in ident):
            raise ValueError(f"{row['dataset']}: claims per-step change "
                             "without ordered frames")
        if (row["capture_interval"].startswith("none")
                and "physical_rate" in ident):
            raise ValueError(f"{row['dataset']}: claims a physical rate "
                             "without a capture interval")
        if row["specimen_identity"] == "no" and (
            "intervention_effect" in ident or "delta_per_ordered_step" in ident
        ):
            raise ValueError(f"{row['dataset']}: claims per-specimen work "
                             "without specimen identity")
        if row["specimen_identity"] not in {"yes", "no", "partial", "n/a"}:
            raise ValueError(f"{row['dataset']}: bad specimen_identity")


# --------------------------------------------------------------------------- #
# dataset auditors: each returns the detail dict that backs one matrix row
# --------------------------------------------------------------------------- #


def open_zip(path: Path) -> zipfile.ZipFile:
    """``ZipFile`` that also understands DEFLATE64 members when available.

    ``Zygomycetous1.zip`` (Fricker networks) uses compression method 9, which
    the stdlib refuses; with ``zipfile-deflate64`` installed every zip opens
    the same way, without it those members raise ``NotImplementedError`` and
    are recorded as unreadable instead of silently dropped.
    """
    if HAVE_DEFLATE64:
        return Deflate64ZipFile(path)  # type: ignore[misc]
    return zipfile.ZipFile(path)


def _read_entry_prefix_and_hash(
    zf: zipfile.ZipFile, info: zipfile.ZipInfo, *, hashes: bool
) -> tuple[bytes, str | None]:
    """One streaming pass: first MiB for the JPEG header, md5 of all bytes."""
    prefix = bytearray()
    digest = hashlib.md5() if hashes else None
    with zf.open(info) as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            if digest is not None:
                digest.update(block)
            if len(prefix) < 1 << 20:
                prefix += block[:(1 << 20) - len(prefix)]
    return bytes(prefix), (digest.hexdigest() if digest else None)


def audit_myceliumseg(triage_dir: Path) -> dict | None:
    """Assemble MyceliumSeg provenance from the stored outputs of #21-#25."""
    scan_path = triage_dir / "exif_scan_summary.csv"
    tracks_path = triage_dir / "dish_tracks.csv"
    summary_path = triage_dir / "dish_tracks_summary.csv"
    labels_path = triage_dir / "known_labels.csv"
    if not scan_path.exists() or not tracks_path.exists():
        return None

    sizes: Counter[str] = Counter()
    images = datetimes = 0
    forward: list[float] = []
    with scan_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            images += int(row["images"])
            datetimes += int(row["images_with_datetime"])
            for part in row["top_sizes"].split(";"):
                part = part.strip()
                if not part:
                    continue
                wh, _, count = part.partition("=")
                sizes[wh] += int(count)
            if int(row["images"]) >= 507:
                forward.append(float(row["id_mtime_forward_fraction"]))

    md5_by_id: dict[str, str] = {}
    with tracks_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            md5_by_id[row["id"]] = row["md5"]
    dup = duplicate_stats(list(md5_by_id.values()))

    colony_forward: list[float] = []
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                colony_forward.append(float(row["colony_radius_fwd_steps"]))

    labels: dict[str, tuple[str, str, str]] = {}
    blank = total = 0
    if labels_path.exists():
        with labels_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                total += 1
                species = row["species_if_known"]
                if not species:
                    blank += 1
                labels[row["id"]] = (
                    species, row["medium_if_known"], row["temp_if_known"],
                )

    return {
        "inputs": [p.name for p in
                   (scan_path, tracks_path, summary_path, labels_path)
                   if p.exists()],
        "images": images,
        "images_with_datetime": datetimes,
        "resolutions": dict(sizes),
        "id_mtime_forward_fraction": [round(min(forward), 4),
                                      round(max(forward), 4)] if forward else [],
        "colony_vs_order_forward_fraction": (
            [round(min(colony_forward), 4), round(max(colony_forward), 4)]
            if colony_forward else []
        ),
        "duplicates": dup,
        "labels_known": total - blank,
        "labels_unknown": blank,
        **label_conflicts(md5_by_id, labels),
    }


def audit_grazers(
    raw_dir: Path, *, hashes: bool = True, limit: int = 0
) -> dict | None:
    """Fricker/Boddy grazer series: filenames, EXIF times, sizes, md5."""
    zips = sorted((raw_dir / "fricker_grazers").glob("*.zip"))
    if not zips:
        return None

    files: list[dict[str, object]] = []
    unparsed: list[str] = []
    unreadable: list[str] = []
    sizes: Counter[str] = Counter()
    rejected_sizes: Counter[str] = Counter()
    for zip_path in zips:
        seen = 0
        with open_zip(zip_path) as zf:
            for info in zf.infolist():
                base = info.filename.split("/")[-1]
                if not base.lower().endswith(".jpg"):
                    continue
                if limit and seen >= limit:
                    break
                seen += 1
                parsed = parse_grazer_name(base)
                if parsed is None:
                    unparsed.append(info.filename)
                    continue
                try:
                    prefix, digest = _read_entry_prefix_and_hash(
                        zf, info, hashes=hashes
                    )
                except NotImplementedError:
                    unreadable.append(info.filename)
                    continue
                header = parse_jpeg_header(prefix, allow_truncation=True)
                rejected = "/rejected/" in info.filename
                files.append(
                    {
                        "zip": zip_path.name,
                        "entry": info.filename,
                        "rejected": rejected,
                        "width": header.width,
                        "height": header.height,
                        "datetime": normalize_exif_datetime(
                            header.exif.get("ExifIFD.DateTimeOriginal")
                        ),
                        "md5": digest,
                        **parsed,
                    }
                )
                key = f"{header.width}x{header.height}"
                (rejected_sizes if rejected else sizes)[key] += 1

    kept = [f for f in files if not f["rejected"]]
    series_map: dict[tuple, list[dict]] = defaultdict(list)
    for f in kept:
        key = (f["species"], f["block"], f["treatment"], f["replicate"])
        series_map[key].append(f)
    series = [series_stats(list(key), value)
              for key, value in sorted(series_map.items())]
    agg = aggregate_series(series)
    exif_times = [str(f["datetime"]) for f in kept if f["datetime"]]

    dup_stats = None
    dup_detail: dict[str, list[str]] | None = None
    if hashes:
        dup_stats = duplicate_stats([str(f["md5"]) for f in kept])
        by_md5: dict[str, list[str]] = defaultdict(list)
        for f in kept:
            by_md5[str(f["md5"])].append(str(f["entry"]))
        dup_detail = {h: entries for h, entries in by_md5.items()
                      if len(entries) > 1}

    return {
        "zips": [p.name for p in zips],
        "images_total": len(files),
        "images_kept": len(kept),
        "images_rejected": len(files) - len(kept),
        "unparsed_names": unparsed,
        "unreadable_entries": unreadable,
        "resolutions": dict(sizes),
        "rejected_resolutions": dict(rejected_sizes),
        "images_with_exif_datetime": len(exif_times),
        "exif_datetime_min": min(exif_times, default=None),
        "exif_datetime_max": max(exif_times, default=None),
        "treatments": sorted({str(f["treatment"]) for f in kept}),
        "species": sorted({str(f["species"]) for f in kept}),
        "duplicates": dup_stats,
        "duplicate_groups_detail": dup_detail,
        "series": series,
        "series_aggregate": agg,
    }


def audit_networks(raw_dir: Path, *, hashes: bool = True) -> dict | None:
    """Fricker network snapshots: one colony, one image, one day label."""
    base = raw_dir / "fricker_networks"
    zips = [p for p in sorted(base.glob("*.zip"))
            if p.name != "parameters.zip"]
    if not zips:
        return None

    files: list[dict[str, object]] = []
    unparsed: list[str] = []
    unreadable: list[str] = []
    sizes: Counter[str] = Counter()
    for zip_path in zips:
        with open_zip(zip_path) as zf:
            for info in zf.infolist():
                name = info.filename.split("/")[-1]
                if not name.lower().endswith((".jpg", ".jpeg")):
                    continue
                parsed = parse_network_name(name.rsplit(".", 1)[0])
                if parsed is None:
                    unparsed.append(info.filename)
                    continue
                try:
                    prefix, digest = _read_entry_prefix_and_hash(
                        zf, info, hashes=hashes
                    )
                except NotImplementedError:
                    unreadable.append(info.filename)
                    continue
                header = parse_jpeg_header(prefix, allow_truncation=True)
                sizes[f"{header.width}x{header.height}"] += 1
                files.append(
                    {
                        "zip": zip_path.name,
                        "entry": info.filename,
                        "width": header.width,
                        "height": header.height,
                        "md5": digest,
                        **parsed,
                    }
                )

    colonies: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        colonies[str(f["colony"])].append(f)
    repeated = {c: len(v) for c, v in colonies.items() if len(v) > 1}

    return {
        "zips": [p.name for p in zips],
        "images": len(files),
        "unparsed_names": unparsed,
        "unreadable_entries": unreadable,
        "resolutions": dict(sizes),
        "colonies": len(colonies),
        "colonies_with_multiple_images": repeated,
        "day_labels": sorted({int(f["day"]) for f in files}),
        "duplicates": (duplicate_stats([str(f["md5"]) for f in files])
                       if hashes else None),
        "files": [{k: f[k] for k in ("zip", "colony", "day")}
                  for f in files],
    }


def order_check(frames: list) -> dict[str, float] | None:
    """Adjacent frames should differ less than random pairs, if ordered."""
    import numpy as np

    if len(frames) < 3:
        return None
    arrs = [np.asarray(f, dtype=np.int16)[::8, ::8] for f in frames]
    adjacent = float(np.mean(
        [np.abs(b - a).mean() for a, b in pairwise(arrs)]
    ))
    pairs = [
        float(np.abs(arrs[i] - arrs[j]).mean())
        for i in range(len(arrs)) for j in range(i + 2, len(arrs))
    ]
    non_adjacent = float(np.mean(pairs)) if pairs else float("nan")
    return {"adjacent_mean_abs_diff": round(adjacent, 4),
            "non_adjacent_mean_abs_diff": round(non_adjacent, 4)}


def audit_avi(path: Path, *, hashes: bool = True) -> dict:
    """One MJPEG movie: headers, frame completeness, duplicate frames."""
    with path.open("rb") as handle:
        headers = avi_headers(handle.read(16 << 20))
    chunks = iter_movi_chunks(path)
    dup = None
    if hashes:
        dup = duplicate_stats([hash_region(path, off, size)
                               for _, off, size in chunks])
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "frames_header": headers.get("total_frames"),
        "frames_stored": len(chunks),
        "order_complete": headers.get("total_frames") == len(chunks),
        "width": headers.get("bmp_width") or headers.get("avih_width"),
        "height": headers.get("bmp_height") or headers.get("avih_height"),
        "handler": headers.get("handler"),
        "playback_fps": headers.get("playback_fps"),
        "microsec_per_frame": headers.get("microsec_per_frame"),
        "duplicates": dup,
    }


def tiff_ifd0(path: Path) -> dict[str, object]:
    """IFD0 facts of a classic TIFF: geometry, single-strip layout, comment."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(8)
        bo = "<" if header[:2] == b"II" else ">"
        magic, ifd0 = struct.unpack(bo + "HI", header[2:8])
        if magic != 42:
            raise ValueError(f"not a classic TIFF: {path}")
        handle.seek(ifd0)
        (count,) = struct.unpack(bo + "H", handle.read(2))
        entries = handle.read(12 * count)
        tags: dict[int, tuple[int, int, int]] = {}
        for k in range(count):
            tag, typ, n, val = struct.unpack(bo + "HHII",
                                             entries[k * 12:(k + 1) * 12])
            tags[tag] = (typ, n, val)
        (next_ifd,) = struct.unpack(bo + "I", handle.read(4))

        def inline(tag: int) -> int | None:
            if tag not in tags:
                return None
            typ, _, val = tags[tag]
            if typ == 3:  # SHORT: value sits in the low half (II) / high (MM)
                return (val & 0xFFFF) if bo == "<" else (val >> 16)
            return val

        def counted(tag: int) -> int | None:
            if tag not in tags:
                return None
            typ, n, val = tags[tag]
            if n == 1:
                return inline(tag)
            handle.seek(val)
            fmt = bo + ("H" if typ == 3 else "I")
            width = 2 if typ == 3 else 4
            return struct.unpack(fmt, handle.read(width))[0]

        description = None
        if 270 in tags:
            typ, n, val = tags[270]
            if n <= 4:
                description = (struct.pack(bo + "I", val)[:n]
                               .split(b"\x00")[0])
            else:
                handle.seek(val)
                description = handle.read(n).split(b"\x00")[0].decode(
                    "utf-8", "replace"
                )
        return {
            "size": size,
            "width": inline(256),
            "height": inline(257),
            "strip_offset": counted(273),
            "strip_bytes": counted(279),
            "description": description,
            "next_ifd": next_ifd,
        }


def audit_tiff_stack(path: Path, *, hashes: bool = True) -> dict:
    """ImageJ TIFF stack whose IFD chain overflows: arithmetic page count."""
    meta = tiff_ifd0(path)
    desc = parse_imagej_description(meta["description"] or "")
    frames, trailer, ok = imagej_frame_layout(
        meta["size"], meta["strip_offset"], meta["strip_bytes"]
    )
    consistent = bool(ok) and desc.get("images") in (None, frames)

    dup = None
    order = None
    blank_pages = None
    frames_with_content = None
    if hashes:
        import numpy as np

        content_digests: list[str] = []
        downsampled: list = []
        blank = 0
        width, height = meta["width"], meta["height"]
        with path.open("rb") as handle:
            for k in range(frames):
                handle.seek(int(meta["strip_offset"])
                            + k * int(meta["strip_bytes"]))
                raw = handle.read(int(meta["strip_bytes"]))
                arr = np.frombuffer(raw, dtype=np.uint8)
                if arr.size and not arr.any():
                    blank += 1  # allocated but zero-filled page
                    continue
                content_digests.append(hashlib.md5(raw).hexdigest())
                if arr.size == width * height:
                    downsampled.append(arr.reshape(height, width))
        blank_pages = blank
        frames_with_content = frames - blank
        dup = duplicate_stats(content_digests)
        order = order_check(downsampled)
    return {
        "file": path.name,
        "bytes": meta["size"],
        "frames": frames,
        "frames_with_content": frames_with_content,
        "blank_pages": blank_pages,
        "trailer_bytes": trailer,
        "imagej_images": desc.get("images"),
        "layout_consistent": consistent,
        "width": meta["width"],
        "height": meta["height"],
        "description": (meta["description"] or "").strip(),
        "next_ifd": meta["next_ifd"],
        "duplicates": dup,
        "order_check": order,
    }


def audit_tiff_pages(path: Path, *, hashes: bool = True) -> dict:
    """A small multi-page TIFF read page by page (net2019b)."""
    import tifffile

    with tifffile.TiffFile(path) as tf:
        pages = list(tf.pages)
        arrays = [page.asarray() for page in pages]
        tag = pages[0].tags.get("ImageDescription") if pages else None
        description = str(tag.value) if tag else ""
    desc = parse_imagej_description(description)
    dup = None
    order = None
    if hashes:
        digests = [hashlib.md5(a.tobytes()).hexdigest() for a in arrays]
        dup = duplicate_stats(digests)
        order = order_check(arrays)
    shape = arrays[0].shape if arrays else ()
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "frames": len(arrays),
        "imagej_images": desc.get("images"),
        "width": shape[1] if shape else None,
        "height": shape[0] if shape else None,
        "channels": shape[2] if len(shape) == 3 else None,
        "dtype": str(arrays[0].dtype) if arrays else None,
        "description": description.strip(),
        "duplicates": dup,
        "order_check": order,
    }


def zarr_array_meta(zf: zipfile.ZipFile, key: str) -> dict:
    """Shape / dtype / chunk facts of a Zarr v3 array without reading data."""
    meta = json.loads(zf.read(f"{key}/zarr.json"))
    return {
        "shape": tuple(meta["shape"]),
        "data_type": meta["data_type"],
        "chunk_shape": tuple(meta["chunk_grid"]["configuration"]["chunk_shape"]),
        "codecs": [c["name"] for c in meta["codecs"]],
        "fill_value": meta.get("fill_value", 0),
    }


def read_zarr_v3_array(zf: zipfile.ZipFile, key: str) -> dict:
    """Read one single-chunk Zarr v3 array (bytes codec + optional zstd)."""
    import numpy as np
    from numcodecs.zstd import Zstd

    meta = zarr_array_meta(zf, key)
    if any(c < s for c, s in zip(meta["chunk_shape"], meta["shape"])):
        raise NotImplementedError(f"{key}: multi-chunk array not supported")
    dtype = {"int64": "<i8", "uint8": "u1", "float32": "<f4"}[
        meta["data_type"]
    ]
    members = sorted(n for n in zf.namelist()
                     if n.startswith(f"{key}/c/") and not n.endswith(".json"))
    if not members:
        # Zarr v3 omits chunks that equal the fill value (all-zero arrays)
        arr = np.full(meta["shape"], meta["fill_value"], dtype=dtype)
        return {"shape": meta["shape"], "data_type": meta["data_type"],
                "array": arr}
    if len(members) != 1:
        raise NotImplementedError(f"{key}: expected one chunk, got {members}")
    raw = zf.read(members[0])
    if "zstd" in meta["codecs"]:
        raw = Zstd().decode(raw)
    arr = np.frombuffer(raw, dtype=dtype).reshape(meta["shape"])
    return {"shape": meta["shape"], "data_type": meta["data_type"],
            "array": arr}


def audit_physarum(raw_dir: Path) -> dict | None:
    """OME-Zarr dish series: timestamps, per-frame sha256, qc, geometry."""
    base = raw_dir / "physarum"
    archives = sorted(base.glob("physarum-*.ome.zarr.zip"))
    if not archives:
        return None
    path = archives[0]
    with open_zip(path) as zf:
        root = json.loads(zf.read("zarr.json"))
        multiscale = root["attributes"]["ome"]["multiscales"][0]
        scale = multiscale["datasets"][0]["coordinateTransformations"][0][
            "scale"
        ]
        provenance = root["attributes"].get("physarum_zenodo", {})
        pixels = zarr_array_meta(zf, "s0")  # sharded: metadata only
        stamps = read_zarr_v3_array(zf, "acquisition/timestamp_utc_ns")
        shas = read_zarr_v3_array(zf, "acquisition/source_sha256")
        qc = read_zarr_v3_array(zf, "acquisition/qc_flags")
        circle = read_zarr_v3_array(zf, "registration/dish_circle_xyr_px")

    hashes = [bytes(row).hex() for row in shas["array"]]
    ns_values = [int(v) for v in stamps["array"]]
    return {
        "archives_on_disk": [p.name for p in archives],
        "release_archives": 24,
        "axes": [a["name"] for a in multiscale["axes"]],
        "frames": pixels["shape"][0],
        "width": pixels["shape"][3],
        "height": pixels["shape"][2],
        "dtype": pixels["data_type"],
        "nominal_interval_s": scale[0],
        "mm_per_px": round(scale[3], 6),
        "timestamps": timestamp_stats(ns_values),
        "source_sha256": duplicate_stats(hashes),
        "qc_flagged_frames": int((qc["array"] != 0).sum()),
        "frames_with_dish_circle": int((circle["array"][:, 2] > 0).sum()),
        "experiment": provenance.get("experiment"),
        "dish": provenance.get("dish"),
        "first_source": provenance.get("source", {}).get("first_source_name"),
        "last_source": provenance.get("source", {}).get("last_source_name"),
        "license": provenance.get("license"),
    }


def audit_aureobasidium(raw_dir: Path) -> dict | None:
    """The Zenodo record holds a conference book, not data."""
    base = raw_dir / "aureobasidium"
    if not base.exists():
        return None
    files = sorted(p.name for p in base.iterdir() if p.is_file())
    return {"files": files,
            "data_files": [f for f in files
                           if not f.lower().endswith(".pdf")]}


# --------------------------------------------------------------------------- #
# capability matrix
# --------------------------------------------------------------------------- #


def build_row(**cells: str) -> dict[str, str]:
    """Fill a row with empty defaults; ``validate_rows`` checks the result."""
    row = {field: "" for field in CSV_FIELDS}
    row.update(cells)
    return row


def sizes_cell(sizes: dict[str, int], top: int = 4) -> str:
    """``4608x3456 x18448; 1382x1037 x1728`` sorted by frequency, capped."""
    ordered = sorted(sizes.items(), key=lambda kv: -kv[1])
    parts = [f"{k} x{v}" for k, v in ordered[:top]]
    if len(ordered) > top:
        parts.append(f"+{len(ordered) - top} more sizes")
    return "; ".join(parts)


def dup_cell(dup: dict | None) -> str:
    """Human form of :func:`duplicate_stats` for the matrix cell."""
    if dup is None:
        return "not checked (content hashing skipped)"
    if dup["duplicate_groups"] == 0:
        return f"0 in {dup['n']} content hashes (all distinct)"
    return (f"{dup['items_in_duplicate_groups']} items in "
            f"{dup['duplicate_groups']} hash groups; "
            f"{dup['redundant_copies']} redundant copies")


def rows_from_details(details: dict[str, dict]) -> list[dict[str, str]]:
    """Turn per-dataset audit details into the capability matrix rows."""
    rows: list[dict[str, str]] = []

    if (d := details.get("myceliumseg")) is not None:
        fwd = d["id_mtime_forward_fraction"]
        colony = d["colony_vs_order_forward_fraction"]
        rows.append(build_row(
            dataset="myceliumseg",
            kind="stills archive (cross-sectional)",
            location="data/raw/myceliumseg (11 zips)",
            n_units=str(d["images"]),
            n_specimens="unknown (no dish identity)",
            ordered_frames="no",
            order_evidence=(f"id->mtime forward {fwd[0]}-{fwd[1]}; "
                            f"colony size vs file order {colony[0]}-"
                            f"{colony[1]} (#22, #23)"
                            if fwd and colony else "see #22, #23"),
            specimen_identity="no",
            identity_basis=("rim dot fills 720/720 angle bins; every embedding "
                            "pair above 0.99 is a byte-duplicate (#23)"),
            capture_interval="none",
            interval_source=(f"0/{d['images']} images carry EXIF; zip mtimes "
                             "are export sessions (#21)"),
            resolution=sizes_cell(d["resolutions"]),
            resolution_source="JPEG SOF headers (exif_scan_summary, #21)",
            duplicates=dup_cell(d["duplicates"]),
            duplicate_rule="one representative per md5 group (estimands.md §7)",
            condition_labels=(f"species+medium+temp known for "
                              f"{d['labels_known']}; {d['labels_unknown']} "
                              f"blank; {d['conflict_groups']} md5 label "
                              "conflicts (#23, #24)"),
            estimands_identifiable="image_morphology",
            estimands_blocked=("delta_per_ordered_step;physical_rate;"
                               "intervention_effect"),
            evidence="#21 #22 #23 #24 #25 #26; notes/gate.md; "
                     "notes/estimands.md",
        ))

    if (d := details.get("fricker_grazers")) is not None:
        agg = d["series_aggregate"]
        if agg["n_gap_obs"]:
            gap = (f"{agg['gap_days_min']}-{agg['gap_days_max']} d "
                   f"(median {agg['gap_days_median']})")
        else:
            gap = "none"
        same_second = ""
        if agg["n_series_identical_timestamps"]:
            same_second = (f"; {agg['n_series_identical_timestamps']} series "
                           f"capture consecutive days in the same second")
        rows.append(build_row(
            dataset="fricker_grazers",
            kind="image series (still camera, day resolution)",
            location="data/raw/fricker_grazers (4 zips)",
            n_units=str(d["images_kept"]),
            n_specimens=str(agg["n_series"]),
            ordered_frames="yes",
            order_evidence=(f"filename day axis; EXIF DateTimeOriginal agrees "
                            f"in {agg['n_order_confirmed']}/"
                            f"{agg['n_series']} series, "
                            f"{agg['n_order_conflicts']} conflicts"
                            f"{same_second}"),
            specimen_identity="yes",
            identity_basis="species+treatment+replicate in every filename",
            capture_interval=f"days: {gap}",
            interval_source="filename day label + EXIF capture times (2009)",
            resolution=sizes_cell(d["resolutions"]),
            resolution_source="JPEG SOF headers",
            duplicates=dup_cell(d["duplicates"]),
            duplicate_rule="md5 over kept images; one image per (series, day)",
            condition_labels=(f"grazer-treatment labels (allocation procedure not "
                              f"documented in reviewed primary methods): "
                              f"{', '.join(d['treatments'])}; day; replicate; "
                              f"{d['images_rejected']} rejected thumbnails "
                              "excluded"),
            estimands_identifiable=("image_morphology;delta_per_ordered_step;"
                                    "physical_rate"),
            estimands_blocked=("intervention_effect: treatment allocation or "
                               "exchangeability not documented in reviewed "
                               "primary methods"),
            evidence="data/INVENTORY.md; Zenodo 14931953; "
                     "notes/provenance.md",
        ))

    if (d := details.get("fricker_networks")) is not None:
        rows.append(build_row(
            dataset="fricker_networks",
            kind="stills archive (cross-sectional snapshots)",
            location="data/raw/fricker_networks (4 image zips)",
            n_units=str(d["images"]),
            n_specimens=str(d["colonies"]),
            ordered_frames="n/a",
            order_evidence=("one image per colony; day labels differ across "
                            "colonies, not within them"),
            specimen_identity="partial",
            identity_basis=("taxon+replicate identifies the tray, but each "
                            "tray appears exactly once"),
            capture_interval="none",
            interval_source="single day label per image (t08-t25)",
            resolution=sizes_cell(d["resolutions"]),
            resolution_source="JPEG SOF headers",
            duplicates=dup_cell(d["duplicates"]),
            duplicate_rule="md5 over all network images",
            condition_labels="taxon, replicate, day, some treatment labels",
            estimands_identifiable="image_morphology",
            estimands_blocked=("delta_per_ordered_step;physical_rate;"
                               "intervention_effect"),
            evidence="data/INVENTORY.md; Zenodo 5725751; "
                     "notes/provenance.md",
        ))

    if (d := details.get("podospora_thallus")) is not None:
        movies = d["movies"]
        frames = sorted(int(m["frames_header"]) for m in movies)
        resolutions = sorted({f"{m['width']}x{m['height']}" for m in movies})
        rows.append(build_row(
            dataset="podospora_thallus",
            kind="movies (MJPEG AVI)",
            location="data/raw/podospora_thallus (6 avi)",
            n_units=(f"{frames[0]}-{frames[-1]} frames per movie "
                     f"x {len(movies)} movies"),
            n_specimens=str(len(movies)),
            ordered_frames="yes",
            order_evidence=(f"container frame order complete: header frames "
                            f"== stored chunks in "
                            f"{sum(1 for m in movies if m['order_complete'])}/"
                            f"{len(movies)} movies"),
            specimen_identity="yes",
            identity_basis=("one colony per movie; medium M0/M2 + replicate "
                            "in the filename"),
            capture_interval="18 min/frame (relative to first frame)",
            interval_source=("Figshare 22666138 description; first-frame "
                             "origin relative to germination unknown; 5 fps is playback"),
            resolution="; ".join(resolutions),
            resolution_source=("AVI strf BITMAPINFOHEADER; 1.63 um/source-pixel "
                               "from Figshare 22666138"),
            duplicates=dup_cell(d["duplicates"]),
            duplicate_rule="md5 of compressed frame bytes, pooled over movies",
            condition_labels=("medium M0 (3 movies) vs M2 (3 movies), "
                              "replicate 1-3"),
            estimands_identifiable=("image_morphology;delta_per_ordered_step;"
                                    "physical_rate;medium_condition_association"),
            estimands_blocked="causal_intervention_effect",
            evidence=("Figshare 22666138; Ledoux et al. 2023; "
                      "notes/podospora_growth_model.md"),
        ))

    if (d := details.get("podospora_m0_raw")) is not None:
        order_evidence = (
            f"contiguous pages after IFD0: ImageJ images="
            f"{d['imagej_images']} == arithmetic count {d['frames']}"
            if d["layout_consistent"] else
            "page count disagrees with the ImageJ header -- recheck"
        )
        if d.get("blank_pages"):
            order_evidence += (f"; only {d['frames_with_content']} of "
                               f"{d['frames']} pages carry image data")
        if d.get("order_check"):
            order_evidence += ("; adjacent content frames differ less than "
                               "distant ones")
        if d.get("blank_pages"):
            n_units = (f"{d['frames']} pages "
                       f"({d['frames_with_content']} with content)")
            dup_text = (f"{dup_cell(d['duplicates'])}; "
                        f"{d['blank_pages']} pages are zero-filled")
        else:
            n_units = str(d["frames"])
            dup_text = dup_cell(d["duplicates"])
        rows.append(build_row(
            dataset="podospora_m0_raw",
            kind="stack (ImageJ TIFF)",
            location="data/raw/podospora_m0_raw/M0_1.tif",
            n_units=n_units,
            n_specimens="1",
            ordered_frames="yes",
            order_evidence=order_evidence,
            specimen_identity="yes",
            identity_basis="one colony (M0_1) in the file",
            capture_interval="none (in files)",
            interval_source=("~18 min per frame from Ledoux et al. 2023, "
                             "external to the file"),
            resolution=f"{d['width']}x{d['height']}",
            resolution_source="TIFF IFD0 (tags 256/257)",
            duplicates=dup_text,
            duplicate_rule="md5 of page bytes",
            condition_labels="medium M0, single movie (M0_1)",
            estimands_identifiable=("image_morphology;"
                                    "delta_per_ordered_step"),
            estimands_blocked="physical_rate;intervention_effect",
            evidence=("Figshare 22140692; data/INVENTORY.md; "
                      "notes/provenance.md"),
        ))

    if (d := details.get("podospora_net2019a")) is not None:
        rows.append(build_row(
            dataset="podospora_net2019a",
            kind="movie (MJPEG AVI, downsized)",
            location="data/raw/podospora_net2019a/experiment_a_downsized.avi",
            n_units=str(d["frames_header"]),
            n_specimens="1",
            ordered_frames="yes",
            order_evidence=(f"container frame order complete: "
                            f"{d['frames_header']} header frames == "
                            f"{d['frames_stored']} stored chunks"
                            if d["order_complete"] else
                            "header/pointer frame counts disagree"),
            specimen_identity="yes",
            identity_basis="one network in the file",
            capture_interval="none (in files)",
            interval_source="frame labels only; 4 fps is playback rate",
            resolution=f"{d['width']}x{d['height']}",
            resolution_source="AVI strf BITMAPINFOHEADER",
            duplicates=dup_cell(d["duplicates"]),
            duplicate_rule="md5 of compressed frame bytes",
            condition_labels="species P. anserina; medium not in the file",
            estimands_identifiable=("image_morphology;"
                                    "delta_per_ordered_step"),
            estimands_blocked="physical_rate;intervention_effect",
            evidence=("Figshare 9897056; data/INVENTORY.md; "
                      "notes/provenance.md"),
        ))

    if (d := details.get("podospora_net2019b")) is not None:
        if d.get("order_check"):
            order_evidence = (f"adjacent pages differ less than distant ones "
                              f"(adj {d['order_check']['adjacent_mean_abs_diff']}"
                              f" vs distant "
                              f"{d['order_check']['non_adjacent_mean_abs_diff']})")
        else:
            order_evidence = "content-hash order check skipped"
        rows.append(build_row(
            dataset="podospora_net2019b",
            kind="stack (multi-page TIFF)",
            location="data/raw/podospora_net2019b/superpose_50-60_raw.tif",
            n_units=str(d["frames"]),
            n_specimens="1",
            ordered_frames="yes",
            order_evidence=order_evidence,
            specimen_identity="yes",
            identity_basis=("one network; filename suggests frames 50-60 of "
                            "a run"),
            capture_interval="none (in files)",
            interval_source="ImageJ header only; no timestamps",
            resolution=(f"{d['width']}x{d['height']}"
                        + (f"x{d['channels']}" if d.get("channels") else "")),
            resolution_source="TIFF page geometry + ImageJ description",
            duplicates=dup_cell(d["duplicates"]),
            duplicate_rule="md5 of page bytes",
            condition_labels="medium M2 (record title); no per-frame labels",
            estimands_identifiable=("image_morphology;"
                                    "delta_per_ordered_step"),
            estimands_blocked="physical_rate;intervention_effect",
            evidence=("Figshare 10626968; data/INVENTORY.md; "
                      "notes/provenance.md"),
        ))

    if (d := details.get("physarum_rig")) is not None:
        ts = d["timestamps"]
        rows.append(build_row(
            dataset="physarum_rig",
            kind="OME-Zarr time-lapse (controlled rig template)",
            location=(f"data/raw/physarum/{d['archives_on_disk'][0]} "
                      f"(1 of {d['release_archives']} release archives)"),
            n_units=str(d["frames"]),
            n_specimens="1 on disk (6 per experiment; 24 in release)",
            ordered_frames="yes",
            order_evidence=(f"timestamps strictly increasing: "
                            f"{'yes' if ts['strictly_increasing'] else 'NO'} "
                            f"over {ts['n']} frames"),
            specimen_identity="yes",
            identity_basis="dish id in filename; per-frame fitted dish circle",
            capture_interval=(f"{ts['gap_min_s']}-{ts['gap_max_s']} s "
                              f"(nominal {d['nominal_interval_s']} s)"),
            interval_source="acquisition/timestamp_utc_ns per frame",
            resolution=(f"{d['width']}x{d['height']}x3 @ "
                        f"{d['mm_per_px']} mm/px"),
            resolution_source="OME scale transform (90-mm dish fit)",
            duplicates=dup_cell(d["source_sha256"]),
            duplicate_rule="per-frame source PNG sha256 (release manifest)",
            condition_labels=(f"experiment+dish id; oat-flake arrangement; "
                              f"qc flags: {d['qc_flagged_frames']} flagged"),
            estimands_identifiable=("image_morphology;delta_per_ordered_step;"
                                    "physical_rate"),
            estimands_blocked="intervention_effect",
            evidence=("Zenodo 22868368 README/QC-NOTES; "
                      "notes/provenance.md"),
        ))

    if (d := details.get("aureobasidium")) is not None:
        rows.append(build_row(
            dataset="aureobasidium",
            kind="unavailable (no data in record)",
            location="data/raw/aureobasidium",
            n_units="0",
            n_specimens="0",
            ordered_frames="n/a",
            order_evidence="record contains only a conference book PDF",
            specimen_identity="n/a",
            capture_interval="none",
            resolution="",
            duplicates="n/a (no data)",
            condition_labels="",
            estimands_identifiable="",
            estimands_blocked=("image_morphology;delta_per_ordered_step;"
                               "physical_rate;intervention_effect"),
            evidence="data/INVENTORY.md; Zenodo 14001310",
        ))

    return rows


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def _merge_movie_dups(movies: list[dict]) -> dict | None:
    """Combine per-movie duplicate stats across the thallus movies."""
    groups = [m["duplicates"] for m in movies if m.get("duplicates")]
    if not groups:
        return None
    return {
        "n": sum(g["n"] for g in groups),
        "distinct": sum(g["distinct"] for g in groups),
        "duplicate_groups": sum(g["duplicate_groups"] for g in groups),
        "items_in_duplicate_groups": sum(
            g["items_in_duplicate_groups"] for g in groups
        ),
        "redundant_copies": sum(g["redundant_copies"] for g in groups),
        "largest_group": max(g["largest_group"] for g in groups),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--triage-dir", type=Path, default=DEFAULT_TRIAGE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="capability matrix CSV")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT,
                        help="per-file evidence JSON")
    parser.add_argument("--no-hashes", dest="hashes", action="store_false",
                        help="skip content hashing (duplicates not checked)")
    parser.add_argument("--limit", type=int, default=0,
                        help="images per archive (0 = all); smoke tests")
    args = parser.parse_args(argv)

    details: dict[str, dict] = {}

    def record(name: str, detail: dict | None) -> None:
        if detail is None:
            print(f"skip {name}: inputs not found under {args.raw_dir}")
        else:
            details[name] = detail
            print(f"audited {name}")

    record("myceliumseg", audit_myceliumseg(args.triage_dir))
    record("fricker_grazers",
           audit_grazers(args.raw_dir, hashes=args.hashes, limit=args.limit))
    record("fricker_networks",
           audit_networks(args.raw_dir, hashes=args.hashes))

    thallus_dir = args.raw_dir / "podospora_thallus"
    movies = []
    for path in sorted(thallus_dir.glob("*.avi")) if thallus_dir.exists() else []:
        movie = audit_avi(path, hashes=args.hashes)
        m = PODOSPORA_RX.match(path.name)
        if m:
            movie["medium"] = m.group("medium")
            movie["replicate"] = int(m.group("rep"))
        movies.append(movie)
    record("podospora_thallus",
           {"movies": movies,
            "duplicates": _merge_movie_dups(movies),
            "metadata_source": "Figshare 22666138 description",
            "capture_interval_minutes_external_to_avi": 18,
            "pixel_size_um_external_to_avi": 1.63,
            "time_origin": "first AVI frame; relation to ascospore germination unspecified",
            "license": "CC BY 4.0"} if movies else None)

    path = args.raw_dir / "podospora_m0_raw" / "M0_1.tif"
    record("podospora_m0_raw",
           audit_tiff_stack(path, hashes=args.hashes) if path.exists()
           else None)

    path = args.raw_dir / "podospora_net2019a" / "experiment_a_downsized.avi"
    record("podospora_net2019a",
           audit_avi(path, hashes=args.hashes) if path.exists() else None)

    path = args.raw_dir / "podospora_net2019b" / "superpose_50-60_raw.tif"
    record("podospora_net2019b",
           audit_tiff_pages(path, hashes=args.hashes) if path.exists()
           else None)

    record("physarum_rig", audit_physarum(args.raw_dir))
    record("aureobasidium", audit_aureobasidium(args.raw_dir))

    rows = rows_from_details(details)
    validate_rows(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue": 61,
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "hashes": args.hashes,
        "deflate64_supported": HAVE_DEFLATE64,
        "limit": args.limit,
        "datasets": details,
    }
    args.json_out.write_text(json.dumps(payload, indent=1, default=str),
                             encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} rows) and {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
