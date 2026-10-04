#!/usr/bin/env python3
"""Scan every MyceliumSeg image for acquisition-time metadata, reading from the zips.

For each ``*.jpg`` in ``data/raw/myceliumseg/*.zip`` this reads only the leading
JPEG marker segments (a few hundred bytes out of a ~3 MB image) and records
EXIF / XMP / IPTC / JPEG COM / JFIF content plus the zip entry mtime.  Nothing is
ever extracted to disk and no archive is opened for writing.

Outputs
    data/triage/exif_scan.csv           one row per image
    data/triage/exif_scan_summary.csv   one row per archive (counts + verdict inputs)

Usage
    .venv/bin/python scripts/triage/exif_scan.py
    .venv/bin/python scripts/triage/exif_scan.py --jobs 4 --limit 20   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import struct
import sys
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

try:  # executed as a script: this directory is already on sys.path
    from exif_parse import (
        ENTRY_APPLEDOUBLE,
        ENTRY_DIR,
        ENTRY_DS_STORE,
        ENTRY_IMAGE,
        ENTRY_MASK,
        ENTRY_OTHER,
        JpegParseError,
        classify_entry,
        clean_text,
        condition_tag_from_entry,
        find_vendor_hits,
        forward_step_fraction,
        group_sessions,
        image_id_from_entry,
        median,
        parse_appledouble,
        parse_png_text,
        parse_zip_extra,
        read_header,
        vendor_tags,
    )
except ImportError:  # imported as scripts.triage.exif_scan
    from .exif_parse import (  # type: ignore[no-redef]
        ENTRY_APPLEDOUBLE,
        ENTRY_DIR,
        ENTRY_DS_STORE,
        ENTRY_IMAGE,
        ENTRY_MASK,
        ENTRY_OTHER,
        JpegParseError,
        classify_entry,
        clean_text,
        condition_tag_from_entry,
        find_vendor_hits,
        forward_step_fraction,
        group_sessions,
        image_id_from_entry,
        median,
        parse_appledouble,
        parse_png_text,
        parse_zip_extra,
        read_header,
        vendor_tags,
    )

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "myceliumseg"
DEFAULT_OUT = ROOT / "data" / "triage" / "exif_scan.csv"
DEFAULT_SUMMARY = ROOT / "data" / "triage" / "exif_scan_summary.csv"

# Anything a camera, a scanner vendor or an image tool would leave behind.
DEFAULT_VENDOR_KEYWORDS = (
    "fpheno", "phenomatics", "phenomatrix", "fuji", "nikon", "canon", "olympus",
    "leica", "epson", "perkinelmer", "hamamatsu", "pco.", "pixera", "basler",
    "zeiss", "sartorius", "evident", "olympus life science", "micromanager",
    "micro-manager", "labview", "matlab", "imagemagick", "libjpeg", "cjpeg",
    "gimp", "photoshop", "adobe", "affinity", "snapseed", "darktable", "pil",
    "pillow", "opencv", "python", "windows photo", "software",
)

CSV_COLUMNS = [
    "archive", "zip_entry", "image_id", "path_tag", "entry_size_bytes",
    "zip_compress_type", "zip_mtime", "has_exif", "exif_datetime",
    "exif_datetimeoriginal", "exif_datetime_digitized",
    "exif_offset_time_original", "exif_subsectime_original", "exif_make",
    "exif_model", "exif_software", "exif_artist", "exif_copyright",
    "exif_description", "exif_usercomment", "exif_makernote", "exif_vendor_tags",
    "has_xmp", "xmp_fields", "has_iptc", "iptc_fields", "n_com_comments",
    "com_comments", "vendor_hits", "jfif_version", "jfif_units", "jfif_x_density",
    "jfif_y_density", "width", "height", "components", "precision", "sof",
    "marker_signature", "icc_profile_bytes", "reached_sos", "header_bytes",
    "truncated", "parse_error", "deep_markers", "zip_mtime_utc", "zip_mtime_source",
    "zip_local_utc_offset_h", "zip_uid", "zip_gid", "appledouble_quarantine",
    "appledouble_downloaded", "appledouble_agent",
]

SUMMARY_COLUMNS = [
    "archive", "zip_bytes", "zip_file_mtime", "entries", "images", "masks",
    "appledouble", "ds_store", "dirs", "other", "image_id_min", "image_id_max",
    "images_with_exif", "images_with_xmp", "images_with_iptc", "images_with_com",
    "images_with_any_metadata", "images_with_datetime", "images_vendor_hit",
    "deep_marker_hits", "marker_signatures", "top_marker_signature", "distinct_sizes",
    "top_sizes", "size_4608x3456", "mtime_min", "mtime_max", "mtime_span_days",
    "mtime_utc_min", "mtime_utc_max", "mtime_utc_span_days",
    "distinct_mtimes", "max_files_one_mtime", "mtime_source", "local_utc_offset_h",
    "n_sessions", "top_session_files", "top_session_start", "top_session_end",
    "top_session_span_s", "median_gap_s", "id_mtime_forward_fraction",
    "mtime_dates", "quarantine_files", "quarantine_sidecars", "quarantine_min",
    "quarantine_max",
    "quarantine_agents", "mask_text_files", "mask_text_chunks", "mask_time_chunks",
    "ds_store_md5s", "zip_comment",
]

DATETIME_EXIF_KEYS = (
    "DateTime", "DateTimeOriginal", "DateTimeDigitized",
    "OffsetTimeOriginal", "SubSecTimeOriginal",
)


@dataclass(frozen=True)
class ScanConfig:
    """Knobs shared by the worker processes."""

    header_budget: int = 1 << 20
    chunk: int = 8192
    png_bytes: int = 1 << 16
    session_gap_s: float = 60.0
    limit: int = 0                     # 0 = no limit
    deep_scan: bool = True             # also grep raw bytes for stray metadata
    vendor_keywords: tuple[str, ...] = DEFAULT_VENDOR_KEYWORDS


def _zip_mtime(info: zipfile.ZipInfo) -> datetime | None:
    """Entry DOS mtime as a naive datetime, or ``None`` if the header is invalid.

    Deliberately naive: the DOS field is local time with no timezone attached.
    :func:`parse_zip_extra` supplies the same instant in UTC, and the difference
    between the two recovers the packing machine's offset.
    """
    try:
        return datetime(*info.date_time)  # noqa: DTZ001 - local time by design
    except (ValueError, TypeError):
        return None


def _flatten(mapping: dict, limit: int = 300) -> str:
    """``{"a": 1, "b": "x"}`` -> ``"a=1; b=x"`` for a spreadsheet cell."""
    return clean_text("; ".join(f"{k}={v}" for k, v in mapping.items()), limit)


def _flag(value: object) -> int:
    """CSV-friendly boolean: 1 or 0, so ``pandas.read_csv`` can sum the column."""
    return int(bool(value))


def _iso(moment: datetime | None) -> str:
    return moment.isoformat() if moment else ""


def _local_offset_hours(info: zipfile.ZipInfo, exact: datetime | None) -> float | str:
    """Offset between the zip's DOS mtime (local, 2 s resolution) and UTC.

    The DOS field carries no timezone, so the packing machine's offset can only
    be recovered by comparing it with the extended UTC timestamp.
    """
    dos = _zip_mtime(info)
    if dos is None or exact is None:
        return ""
    return round((dos - exact.replace(tzinfo=None)).total_seconds() / 3600, 2)


# Byte patterns that would betray metadata which no marker segment declares --
# a vendor tool appending its own block, or a trailer lost in re-encoding.
# Deliberately long and multi-word: a 4-byte needle like ``8BIM`` turns up by
# chance roughly once per megabyte of entropy-coded scan data, so such hits must
# be structurally validated (see :func:`_valid_8bim`) rather than reported raw.
DEEP_MARKERS: tuple[bytes, ...] = (
    b"Exif\x00\x00", b"http://ns.adobe.com/xap", b"Photoshop 3.0",
    b"http://ns.adobe.com/exif", b"Intel(r) DSLR", b"FPheno", b"Phenomatics",
    b"www.getiptc.org", b"8BIM",
)
_DEEP_8BIM = b"8BIM"


def _valid_8bim(prefix: bytes, offset: int) -> bool:
    """True when a plausible Photoshop image-resource block starts at ``offset``.

    Layout is ``8BIM`` + 2-byte resource id + Pascal name + 4-byte size; the
    size must stay inside the buffer, which a coincidental ``8BIM`` in scan
    data almost never does.
    """
    if offset + 12 > len(prefix):
        return False
    name_len = prefix[offset + 6]
    size_start = offset + 7 + name_len + 1
    if size_start + 4 > len(prefix):
        return False
    (size,) = struct.unpack_from(">I", prefix, size_start)
    return size_start + 4 + size <= len(prefix)


def _find_all(haystack: bytes, needle: bytes, *, limit: int) -> list[int]:
    """Offsets of at most ``limit`` occurrences of ``needle`` in ``haystack``."""
    offsets: list[int] = []
    start = haystack.find(needle)
    while start >= 0 and len(offsets) < limit:
        offsets.append(start)
        start = haystack.find(needle, start + 1)
    return offsets


def deep_scan_markers(prefix: bytes) -> list[str]:
    """Names of :data:`DEEP_MARKERS` present anywhere in a raw byte prefix.

    Names are the printable form of each needle.  ``8BIM`` is only reported when
    the surrounding resource block is structurally valid, so entropy-coded data
    that happens to spell it is not mistaken for embedded Photoshop metadata.
    """
    hits: list[str] = []
    for needle in DEEP_MARKERS:
        if needle not in prefix:
            continue
        if needle == _DEEP_8BIM:
            offsets = _find_all(prefix, _DEEP_8BIM, limit=8)
            if not any(_valid_8bim(prefix, offset) for offset in offsets):
                continue
        hits.append(needle.decode("ascii", "replace"))
    return hits


def _row_from_header(
    archive: str,
    info: zipfile.ZipInfo,
    head,
    cfg: ScanConfig,
    error: str = "",
) -> dict[str, object]:
    exif = head.exif
    extra = parse_zip_extra(info.extra)
    return {
        "archive": archive,
        "zip_entry": info.filename,
        "image_id": image_id_from_entry(info.filename),
        "path_tag": condition_tag_from_entry(info.filename),
        "entry_size_bytes": info.file_size,
        "zip_compress_type": "stored" if info.compress_type == 0 else str(info.compress_type),
        "zip_mtime": _iso(_zip_mtime(info)),
        "zip_mtime_utc": _iso(extra.mtime_utc),
        "zip_mtime_source": extra.source,
        "zip_local_utc_offset_h": _local_offset_hours(info, extra.mtime_utc),
        "zip_uid": extra.uid if extra.uid is not None else "",
        "zip_gid": extra.gid if extra.gid is not None else "",
        "has_exif": _flag(exif),
        "exif_datetime": exif.get("DateTime", ""),
        "exif_datetimeoriginal": exif.get("DateTimeOriginal", ""),
        "exif_datetime_digitized": exif.get("DateTimeDigitized", ""),
        "exif_offset_time_original": exif.get("OffsetTimeOriginal", ""),
        "exif_subsectime_original": exif.get("SubSecTimeOriginal", ""),
        "exif_make": exif.get("Make", ""),
        "exif_model": exif.get("Model", ""),
        "exif_software": exif.get("Software", ""),
        "exif_artist": exif.get("Artist", ""),
        "exif_copyright": exif.get("Copyright", ""),
        "exif_description": exif.get("ImageDescription", ""),
        "exif_usercomment": exif.get("UserComment", ""),
        "exif_makernote": clean_text(exif.get("MakerNote", ""), 60),
        "exif_vendor_tags": _flatten(vendor_tags(exif)),
        "has_xmp": _flag(head.xmp),
        "xmp_fields": _flatten(head.xmp),
        "has_iptc": _flag(head.iptc),
        "iptc_fields": _flatten(head.iptc),
        "n_com_comments": len(head.comments),
        "com_comments": _flatten({str(i): c for i, c in enumerate(head.comments)}),
        "vendor_hits": " | ".join(
            find_vendor_hits(head.all_text_values(), cfg.vendor_keywords)
        ),
        "jfif_version": head.jfif.get("version", ""),
        "jfif_units": head.jfif.get("units", ""),
        "jfif_x_density": head.jfif.get("x_density", ""),
        "jfif_y_density": head.jfif.get("y_density", ""),
        "width": head.width,
        "height": head.height,
        "components": head.components,
        "precision": head.precision,
        "sof": head.sof_marker if head.sof_marker is None else f"0x{head.sof_marker:02X}",
        "marker_signature": head.signature,
        "icc_profile_bytes": head.icc_profile_bytes,
        "reached_sos": _flag(head.reached_sos),
        "header_bytes": head.header_bytes,
        "truncated": _flag(head.truncated),
        "parse_error": error,
        "deep_markers": "",
        "appledouble_quarantine": "",
        "appledouble_downloaded": "",
        "appledouble_agent": "",
    }


def _blank_row(archive: str, info: zipfile.ZipInfo, error: str) -> dict[str, object]:
    """Row for an entry whose bytes could not be parsed at all."""
    extra = parse_zip_extra(info.extra)
    row = {column: "" for column in CSV_COLUMNS}
    row.update({
        "archive": archive,
        "zip_entry": info.filename,
        "image_id": image_id_from_entry(info.filename),
        "path_tag": condition_tag_from_entry(info.filename),
        "entry_size_bytes": info.file_size,
        "zip_mtime": _iso(_zip_mtime(info)),
        "zip_mtime_utc": _iso(extra.mtime_utc),
        "zip_mtime_source": extra.source,
        "zip_local_utc_offset_h": _local_offset_hours(info, extra.mtime_utc),
        "zip_uid": extra.uid if extra.uid is not None else "",
        "zip_gid": extra.gid if extra.gid is not None else "",
        "parse_error": error,
    })
    return row


def scan_archive(task: tuple[str, ScanConfig]) -> tuple[str, list[dict], dict]:
    """Scan one archive: returns ``(archive_name, image_rows, summary)``."""
    zip_path, cfg = task
    archive = Path(zip_path).name
    rows: list[dict] = []
    counts: Counter[str] = Counter()
    mtimes: list[datetime] = []
    exact_mtimes: list[datetime] = []
    mtime_sources: Counter[str] = Counter()
    offsets: Counter[float] = Counter()
    id_mtime_pairs: list[tuple[int, datetime]] = []
    signatures: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    quarantine_times: list[datetime] = []
    quarantine_sidecars = 0
    quarantine_agents: Counter[str] = Counter()
    mask_chunks: Counter[str] = Counter()
    mask_with_text = 0
    mask_with_time = 0
    quarantine_by_id: dict[str, dict] = {}
    ds_store_md5: Counter[str] = Counter()
    deep_hits: Counter[str] = Counter()
    images_done = 0
    id_min: int | None = None
    id_max: int | None = None

    with zipfile.ZipFile(zip_path) as zf:
        zip_comment = zf.comment.decode("utf-8", "replace")
        for info in zf.infolist():
            kind = classify_entry(info.filename)
            counts[kind] += 1

            if kind == ENTRY_APPLEDOUBLE:
                # `._name.jpg` is a macOS sidecar, never an image: harvest the
                # download timestamp and attach it to the image it shadows.
                try:
                    entries = parse_appledouble(zf.read(info))
                except JpegParseError:
                    counts["appledouble_unreadable"] += 1
                    continue
                for entry in entries:
                    quarantine = entry.get("quarantine")
                    if not isinstance(quarantine, dict):
                        continue
                    quarantine_sidecars += 1
                    shadowed = info.filename.replace("__MACOSX/", "").replace("/._", "/")
                    if classify_entry(shadowed) != ENTRY_IMAGE:
                        continue      # sidecar for a .DS_Store or a folder
                    quarantine_by_id[image_id_from_entry(shadowed)] = quarantine
                    if quarantine.get("agent"):
                        quarantine_agents[str(quarantine["agent"])] += 1
                    stamp = quarantine.get("downloaded")
                    if stamp:
                        quarantine_times.append(datetime.fromisoformat(stamp))
                continue

            if kind == ENTRY_DS_STORE:
                ds_store_md5[hashlib.md5(zf.read(info)).hexdigest()] += 1
                continue

            if kind == ENTRY_MASK:
                with zf.open(info) as handle:
                    prefix = handle.read(cfg.png_bytes)
                try:
                    parsed = parse_png_text(prefix)
                except JpegParseError:
                    counts["mask_unreadable"] += 1
                    continue
                mask_chunks.update(parsed["chunks"])  # type: ignore[arg-type]
                if parsed["text"]:
                    mask_with_text += 1
                if parsed["time"]:
                    mask_with_time += 1
                continue

            if kind != ENTRY_IMAGE:
                continue
            if cfg.limit and images_done >= cfg.limit:
                counts["images_skipped_by_limit"] += 1
                continue
            images_done += 1

            try:
                with zf.open(info) as handle:
                    head = read_header(
                        handle.read, chunk=cfg.chunk, budget=cfg.header_budget
                    )
            except (JpegParseError, zipfile.BadZipFile, RuntimeError) as exc:
                rows.append(_blank_row(archive, info, f"{type(exc).__name__}: {exc}"))
            else:
                row = _row_from_header(
                    archive, info, head, cfg,
                    error="header truncated" if head.truncated else "",
                )
                if cfg.deep_scan:
                    with zf.open(info) as handle:
                        hits = deep_scan_markers(handle.read(cfg.header_budget))
                    if hits:
                        row["deep_markers"] = " | ".join(hits)
                        deep_hits.update(hits)
                        counts["deep_marker_hits"] += 1
                rows.append(row)
                signatures[head.signature] += 1
                if head.width and head.height:
                    sizes[f"{head.width}x{head.height}"] += 1
                if head.exif:
                    counts["images_with_exif"] += 1
                if head.xmp:
                    counts["images_with_xmp"] += 1
                if head.iptc:
                    counts["images_with_iptc"] += 1
                if head.comments:
                    counts["images_with_com"] += 1
                if head.has_any_metadata:
                    counts["images_with_any_metadata"] += 1
                if any(head.exif.get(key) for key in DATETIME_EXIF_KEYS):
                    counts["images_with_datetime"] += 1
                if find_vendor_hits(head.all_text_values(), cfg.vendor_keywords):
                    counts["images_vendor_hit"] += 1

            image_id = image_id_from_entry(info.filename)
            dos = _zip_mtime(info)
            extra = parse_zip_extra(info.extra)
            stamp = extra.mtime_utc if extra.mtime_utc is not None else dos
            if dos is not None:
                mtimes.append(dos)
            if stamp is not None:
                exact_mtimes.append(stamp)
                mtime_sources[extra.source or "dos"] += 1
                offset = _local_offset_hours(info, extra.mtime_utc)
                if offset != "":
                    offsets[offset] += 1
            if image_id:
                numeric = int(image_id)
                id_min = numeric if id_min is None else min(id_min, numeric)
                id_max = numeric if id_max is None else max(id_max, numeric)
                if stamp is not None:
                    id_mtime_pairs.append((numeric, stamp))

    # attach sidecar quarantine stamps to their images
    for row in rows:
        quarantine = quarantine_by_id.get(str(row["image_id"]))
        if quarantine:
            row["appledouble_quarantine"] = quarantine.get("value", "")
            row["appledouble_downloaded"] = quarantine.get("downloaded", "")
            row["appledouble_agent"] = quarantine.get("agent", "")

    sessions = group_sessions(exact_mtimes, gap_s=cfg.session_gap_s)
    top_session = max(sessions, key=lambda s: s.count) if sessions else None
    ordered = sorted(exact_mtimes)
    gaps = [(b - a).total_seconds() for a, b in pairwise(ordered)]
    stamp_counts = Counter(ordered)
    date_counts = Counter(stamp.date().isoformat() for stamp in ordered)
    id_mtime_pairs.sort()
    zip_stat = Path(zip_path).stat()
    median_gap = median(gaps)
    forward = forward_step_fraction([stamp for _, stamp in id_mtime_pairs])

    summary = {
        "archive": archive,
        "zip_bytes": zip_stat.st_size,
        # Local time on purpose: it is when the archive file itself was written
        # on this machine, which is what a human reading the note wants.
        "zip_file_mtime": datetime.fromtimestamp(
            zip_stat.st_mtime, tz=timezone.utc
        ).astimezone().isoformat(timespec="seconds"),
        "entries": sum(counts[kind] for kind in
                       (ENTRY_IMAGE, ENTRY_MASK, ENTRY_APPLEDOUBLE, ENTRY_DS_STORE,
                        ENTRY_DIR, ENTRY_OTHER)),
        "images": images_done,
        "masks": counts[ENTRY_MASK],
        "appledouble": counts[ENTRY_APPLEDOUBLE],
        "ds_store": counts[ENTRY_DS_STORE],
        "dirs": counts[ENTRY_DIR],
        "other": counts[ENTRY_OTHER],
        "image_id_min": id_min if id_min is not None else "",
        "image_id_max": id_max if id_max is not None else "",
        "images_with_exif": counts["images_with_exif"],
        "images_with_xmp": counts["images_with_xmp"],
        "images_with_iptc": counts["images_with_iptc"],
        "images_with_com": counts["images_with_com"],
        "images_with_any_metadata": counts["images_with_any_metadata"],
        "images_with_datetime": counts["images_with_datetime"],
        "images_vendor_hit": counts["images_vendor_hit"],
        "deep_marker_hits": counts["deep_marker_hits"],
        "marker_signatures": len(signatures),
        "top_marker_signature": (
            f"{signatures.most_common(1)[0][0]} ({signatures.most_common(1)[0][1]})"
            if signatures else ""
        ),
        "distinct_sizes": len(sizes),
        "top_sizes": _flatten(dict(sizes.most_common(3))),
        "size_4608x3456": sizes.get("4608x3456", 0),
        "mtime_min": min(mtimes).isoformat() if mtimes else "",
        "mtime_max": max(mtimes).isoformat() if mtimes else "",
        "mtime_span_days": round((max(mtimes) - min(mtimes)).total_seconds() / 86400, 2)
        if len(mtimes) > 1 else 0,
        "mtime_utc_min": ordered[0].isoformat() if ordered else "",
        "mtime_utc_max": ordered[-1].isoformat() if ordered else "",
        "mtime_utc_span_days": round((ordered[-1] - ordered[0]).total_seconds() / 86400, 2)
        if len(ordered) > 1 else 0,
        "distinct_mtimes": len(stamp_counts),
        "max_files_one_mtime": max(stamp_counts.values()) if stamp_counts else 0,
        "mtime_source": _flatten(dict(mtime_sources)),
        "local_utc_offset_h": _flatten({str(k): v for k, v in offsets.items()}),
        "mtime_dates": _flatten(dict(date_counts.most_common())),
        "n_sessions": len(sessions),
        "top_session_files": top_session.count if top_session else 0,
        "top_session_start": top_session.start.isoformat() if top_session else "",
        "top_session_end": top_session.end.isoformat() if top_session else "",
        "top_session_span_s": round(top_session.span_seconds, 1) if top_session else "",
        "median_gap_s": round(median_gap, 2) if median_gap is not None else "",
        "id_mtime_forward_fraction": round(forward, 4) if forward is not None else "",
        "quarantine_files": len(quarantine_times),
        "quarantine_sidecars": quarantine_sidecars,
        "quarantine_min": min(quarantine_times).isoformat() if quarantine_times else "",
        "quarantine_max": max(quarantine_times).isoformat() if quarantine_times else "",
        "quarantine_agents": _flatten(dict(quarantine_agents)),
        "mask_text_files": mask_with_text,
        "mask_text_chunks": sum(count for chunk, count in mask_chunks.items()
                                if chunk != "IDAT"),
        "mask_time_chunks": mask_with_time,
        "ds_store_md5s": len(ds_store_md5),
        "zip_comment": clean_text(zip_comment, 200),
    }

    return archive, rows, summary


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--header-budget", type=int, default=1 << 20,
                        help="max bytes read per image header")
    parser.add_argument("--session-gap", type=float, default=60.0,
                        help="seconds without a zip mtime that starts a new session")
    parser.add_argument("--jobs", type=int, default=11,
                        help="parallel worker processes (one archive each)")
    parser.add_argument("--limit", type=int, default=0,
                        help="scan at most N images per archive (smoke tests)")
    parser.add_argument("--no-deep-scan", dest="deep_scan", action="store_false",
                        help="skip the raw-byte grep for undeclared metadata blocks")
    parser.add_argument("--vendor-keyword", action="append", default=None,
                        help="extra case-insensitive substring to hunt for (repeatable)")
    args = parser.parse_args(argv)

    zips = sorted(args.raw_dir.glob("*.zip"))
    if not zips:
        parser.error(f"no *.zip under {args.raw_dir}")

    keywords = DEFAULT_VENDOR_KEYWORDS + tuple(args.vendor_keyword or ())
    cfg = ScanConfig(
        header_budget=args.header_budget,
        session_gap_s=args.session_gap,
        limit=args.limit,
        deep_scan=args.deep_scan,
        vendor_keywords=keywords,
    )

    tasks = [(str(path), cfg) for path in zips]
    rows: list[dict] = []
    summaries: list[dict] = []
    jobs = max(1, args.jobs)
    print(f"scanning {len(zips)} archives with {jobs} worker(s) "
          f"(header budget {cfg.header_budget // 1024} KiB/image)", flush=True)

    if jobs == 1:
        results = [scan_archive(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(scan_archive, tasks))

    for archive, archive_rows, summary in results:
        rows.extend(archive_rows)
        summaries.append(summary)
        print(f"  {archive:26s} images={summary['images']:6d} "
          f"exif={summary['images_with_exif']} xmp={summary['images_with_xmp']} "
          f"iptc={summary['images_with_iptc']} com={summary['images_with_com']} "
          f"sessions={summary['n_sessions']} "
          f"utc={summary['mtime_utc_min']}..{summary['mtime_utc_max']}", flush=True)

    rows.sort(key=lambda row: (str(row["archive"]), str(row["image_id"]),
                               str(row["zip_entry"])))
    summaries.sort(key=lambda row: str(row["archive"]))
    write_csv(args.out, CSV_COLUMNS, rows)
    write_csv(args.summary_out, SUMMARY_COLUMNS, summaries)

    images = sum(int(s["images"]) for s in summaries)
    with_metadata = sum(int(s["images_with_any_metadata"]) for s in summaries)
    with_exif = sum(int(s["images_with_exif"]) for s in summaries)
    with_datetime = sum(int(s["images_with_datetime"]) for s in summaries)
    with_deep = sum(int(s["deep_marker_hits"]) for s in summaries)
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    print(f"wrote {len(summaries)} rows -> {args.summary_out}")
    print(f"images scanned: {images}")
    print(f"  EXIF: {with_exif}  any EXIF/XMP/IPTC/COM: {with_metadata}  "
          f"any capture datetime: {with_datetime}")
    if args.deep_scan:
        print(f"  raw-byte metadata markers (deep scan): {with_deep}")
    if len(rows) != images:
        print(f"WARNING: {len(rows)} csv rows for {images} images", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
