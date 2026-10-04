#!/usr/bin/env python3
"""Check whether sequential file numbering follows time within dish (thumbnail strips).

Within each archive, compare file-number order against:
  (a) zip mtimes
  (b) simple image signals that change monotonically with growth within a dish
      (e.g. fraction of bright pixels inside the dish rim, computed on a downscaled image)

Report: fraction of forward steps, run-length structure, any periodicity in IDs.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

# Reuse the parsing helpers from exif_parse
try:
    from exif_parse import (
        ENTRY_IMAGE,
        classify_entry,
        image_id_from_entry,
        parse_zip_extra,
    )
except ImportError:
    from .exif_parse import (  # type: ignore[no-redef]
        ENTRY_IMAGE,
        classify_entry,
        image_id_from_entry,
        parse_zip_extra,
    )

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "myceliumseg"
DEFAULT_OUT = ROOT / "data" / "triage" / "numbering_signals.csv"
DEFAULT_NOTES = ROOT / "notes" / "triage_numbering.md"

# Downscale factor for fast brightness analysis
DOWNSCALE = 16  # 4608/16 = 288, 3456/16 = 216


@dataclass(frozen=True)
class ScanConfig:
    """Knobs shared by worker processes."""
    header_budget: int = 1 << 20
    downscale: int = DOWNSCALE
    brightness_threshold: int = 200  # 0-255, pixels above this are "bright"
    limit: int = 0  # 0 = no limit


def _zip_mtime_utc(info: zipfile.ZipInfo) -> datetime | None:
    """Entry UTC mtime from extra field, fallback to DOS field as naive UTC."""
    extra = parse_zip_extra(info.extra)
    if extra.mtime_utc is not None:
        return extra.mtime_utc
    # DOS field: naive local time, treat as UTC for ordering only
    try:
        return datetime(*info.date_time, tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _bright_fraction(image_data: bytes, downscale: int, threshold: int) -> float | None:
    """Compute fraction of pixels above threshold inside the dish area.
    
    Reads the JPEG from bytes, downscales, finds the orange dish rim by hue,
    and computes bright pixel fraction inside it.
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        img.load()  # Force load
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Downscale for speed
        small = img.resize(
            (img.width // downscale, img.height // downscale),
            Image.Resampling.BOX
        )
        arr = np.array(small, dtype=np.float32)  # H, W, 3
        h, w = arr.shape[:2]
        
        # Find dish rim: orange pixels (high R, medium G, low B)
        # In HSV: orange is around hue 30° = R high, G ~0.5*R, B low
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        # Orange-ish: R > G > B, R dominant
        orange_mask = (r > g * 1.3) & (g > b * 1.3) & (r > 100)
        
        if not orange_mask.any():
            # Fallback: use center region as proxy for dish area
            cx, cy = w // 2, h // 2
            radius = min(w, h) // 3
            y, x = np.ogrid[:h, :w]
            dish_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        else:
            # Find bounding box of orange rim, use interior
            rows = np.where(orange_mask.any(axis=1))[0]
            cols = np.where(orange_mask.any(axis=0))[0]
            if len(rows) < 10 or len(cols) < 10:
                # Fallback
                cx, cy = w // 2, h // 2
                radius = min(w, h) // 3
                y, x = np.ogrid[:h, :w]
                dish_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            else:
                rmin, rmax = rows[0], rows[-1]
                cmin, cmax = cols[0], cols[-1]
                # Shrink by 10% to get inside the rim
                pad_r = max(1, (rmax - rmin) // 10)
                pad_c = max(1, (cmax - cmin) // 10)
                dish_mask = np.zeros((h, w), dtype=bool)
                dish_mask[rmin + pad_r:rmax - pad_r, cmin + pad_c:cmax - pad_c] = True
        
        # Bright pixels inside dish
        if not dish_mask.any():
            return None
        
        # Brightness = max of RGB (or luminance)
        brightness = np.maximum(np.maximum(r, g), b)
        bright_inside = brightness[dish_mask] > threshold
        return float(bright_inside.mean())
    except Exception:
        return None


def forward_step_fraction(values: Sequence[float | datetime | int]) -> float | None:
    """Fraction of consecutive pairs where values does not decrease."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    steps = sum(1 for a, b in pairwise(vals) if b >= a)
    return steps / (len(vals) - 1)


def run_lengths(values: Sequence) -> list[int]:
    """Lengths of consecutive runs of equal values."""
    if not values:
        return []
    lengths = []
    current = values[0]
    count = 1
    for v in values[1:]:
        if v == current:
            count += 1
        else:
            lengths.append(count)
            current = v
            count = 1
    lengths.append(count)
    return lengths


def analyze_archive(task: tuple[Path, ScanConfig]) -> list[dict]:
    """Scan one archive for numbering signals."""
    zip_path, cfg = task
    archive = zip_path.name
    rows: list[dict] = []
    
    with zipfile.ZipFile(zip_path) as zf:
        # Collect image entries in zip order (which is ID order)
        image_infos: list[zipfile.ZipInfo] = []
        for info in zf.infolist():
            if classify_entry(info.filename) == ENTRY_IMAGE:
                image_infos.append(info)
                if cfg.limit and len(image_infos) >= cfg.limit:
                    break
        
        # Sort by image ID (which is the file number)
        image_infos.sort(key=lambda info: image_id_from_entry(info.filename))
        
        # Process each image
        for info in image_infos:
            image_id = image_id_from_entry(info.filename)
            mtime_utc = _zip_mtime_utc(info)
            
            # Read image data and compute bright fraction
            with zf.open(info) as handle:
                img_data = handle.read()
            
            bright_frac = _bright_fraction(img_data, cfg.downscale, cfg.brightness_threshold)
            
            rows.append({
                "id": image_id,
                "archive": archive,
                "zip_mtime": mtime_utc.isoformat() if mtime_utc else "",
                "bright_frac": f"{bright_frac:.6f}" if bright_frac is not None else "",
            })
    
    return rows


def analyze_all(archives: list[Path], cfg: ScanConfig) -> list[dict]:
    """Analyze all archives, optionally in parallel."""
    all_rows: list[dict] = []
    jobs = min(len(archives), 11)  # max 11 archives
    tasks = [(p, cfg) for p in archives]
    
    if jobs == 1:
        for task in tasks:
            all_rows.extend(analyze_archive(task))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = pool.map(analyze_archive, tasks)
            for rows in results:
                all_rows.extend(rows)
    
    return all_rows


def write_signals_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "archive", "zip_mtime", "bright_frac"])
        writer.writeheader()
        writer.writerows(rows)


def compute_summary(rows: list[dict]) -> dict:
    """Compute summary statistics across all rows."""
    by_archive: dict[str, list[dict]] = {}
    for row in rows:
        by_archive.setdefault(row["archive"], []).append(row)
    
    summary = {}
    for archive, arch_rows in by_archive.items():
        arch_rows.sort(key=lambda r: r["id"])
        
        # Numeric IDs
        ids = [int(r["id"]) for r in arch_rows]
        # MTimes
        mtimes = [datetime.fromisoformat(r["zip_mtime"].replace("Z", "+00:00")) 
                  if r["zip_mtime"] else None for r in arch_rows]
        # Bright fractions
        bright = [float(r["bright_frac"]) if r["bright_frac"] else None for r in arch_rows]
        
        # Forward step fractions
        fwd_id_mtime = forward_step_fraction(mtimes)
        fwd_id_bright = forward_step_fraction(bright)
        
        # Run lengths of consecutive IDs (should be 1 for all if contiguous)
        id_diffs = [ids[i+1] - ids[i] for i in range(len(ids)-1)]
        run_lens = run_lengths(id_diffs)
        
        # Periodicity check: look for patterns in ID differences
        # e.g., blocks of N = one dish or one session
        # Check autocorrelation of bright fraction at various lags
        bright_valid = [b for b in bright if b is not None]
        autocorr = {}
        if len(bright_valid) > 1:
            for lag in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
                if len(bright_valid) > lag:
                    a = np.array(bright_valid[:-lag])
                    b = np.array(bright_valid[lag:])
                    # Avoid numpy warning when either slice has zero variance
                    if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                        corr = np.corrcoef(a, b)[0, 1]
                        autocorr[lag] = float(corr) if not np.isnan(corr) else 0.0
                    else:
                        autocorr[lag] = 1.0 if np.allclose(a, b) else 0.0
        
        summary[archive] = {
            "images": len(arch_rows),
            "id_min": min(ids),
            "id_max": max(ids),
            "id_gaps": sum(1 for d in id_diffs if d != 1),
            "fwd_id_mtime": fwd_id_mtime,
            "fwd_id_bright": fwd_id_bright,
            "run_lengths": run_lens[:20],  # first 20
            "max_run": max(run_lens) if run_lens else 0,
            "bright_autocorr": autocorr,
            "bright_mean": float(np.nanmean(bright_valid)) if bright_valid else None,
            "bright_std": float(np.nanstd(bright_valid)) if bright_valid else None,
        }
    
    return summary


def write_notes(path: Path, summary: dict, total_rows: int) -> None:
    """Write findings note in markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", encoding="utf-8") as f:
        f.write("# Triage: Sequential File Numbering vs Time/Dish (#22)\n\n")
        f.write("**Question.** Does the sequential file numbering (00000001–00020176) "
                "encode capture time, dish identity, or both?\n\n")
        
        f.write("## Summary\n\n")
        
        # Overall forward fraction vs mtime
        all_mtimes = []
        all_bright = []
        all_ids = []
        for archive, arch_rows in summary.items():
            pass  # We'll compute from rows directly
        
        f.write("| Archive | Images | ID Range | Fwd steps (ID→mtime) | Fwd steps (ID→bright) | Bright autocorr@1 |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        for archive, stats in summary.items():
            fwd_m = f"{stats['fwd_id_mtime']:.4f}" if stats['fwd_id_mtime'] is not None else "N/A"
            fwd_b = f"{stats['fwd_id_bright']:.4f}" if stats['fwd_id_bright'] is not None else "N/A"
            ac1 = f"{stats['bright_autocorr'].get(1, 0):.4f}"
            f.write(f"| {archive} | {stats['images']} | {stats['id_min']}–{stats['id_max']} | "
                    f"{fwd_m} | {fwd_b} | {ac1} |\n")
        
        f.write("\n## Detailed Analysis\n\n")
        
        for archive, stats in summary.items():
            f.write(f"### {archive}\n\n")
            f.write(f"- **Images**: {stats['images']}\n")
            f.write(f"- **ID range**: {stats['id_min']}–{stats['id_max']} "
                    f"(gaps: {stats['id_gaps']})\n")
            f.write(f"- **Forward steps (ID → mtime)**: "
                    f"{stats['fwd_id_mtime']:.4f}" if stats['fwd_id_mtime'] is not None 
                    else "- **Forward steps (ID → mtime)**: N/A")
            f.write("\n")
            f.write(f"- **Forward steps (ID → bright_frac)**: "
                    f"{stats['fwd_id_bright']:.4f}" if stats['fwd_id_bright'] is not None
                    else "- **Forward steps (ID → bright_frac)**: N/A")
            f.write("\n")
            f.write(f"- **Run lengths of ID diffs**: {stats['run_lengths']}\n")
            f.write(f"- **Max run length**: {stats['max_run']}\n")
            f.write(f"- **Bright fraction mean ± std**: "
                    f"{stats['bright_mean']:.4f} ± {stats['bright_std']:.4f}\n"
                    if stats['bright_mean'] is not None else "- **Bright fraction**: no data\n")
            f.write(f"- **Bright autocorrelation**: {stats['bright_autocorr']}\n\n")
        
        # Overall verdict
        f.write("## Verdict\n\n")
        
        # Check if any archive shows strong correlation
        max_fwd_mtime = max((s['fwd_id_mtime'] or 0) for s in summary.values())
        max_fwd_bright = max((s['fwd_id_bright'] or 0) for s in summary.values())
        
        if max_fwd_mtime > 0.9 and max_fwd_bright > 0.9:
            verdict = "BOTH"
        elif max_fwd_mtime > 0.9:
            verdict = "TIME (via zip mtime)"
        elif max_fwd_bright > 0.9:
            verdict = "DISH/GROWTH (via bright fraction)"
        else:
            verdict = "NEITHER"
        
        f.write(f"**Numbering encodes: {verdict}**\n\n")
        
        if verdict == "NEITHER":
            f.write("The sequential file numbering is shuffled relative to both "
                    "export time (zip mtime) and growth progression (bright fraction). "
                    "No ordering signal recovered.\n")
        elif verdict == "TIME (via zip mtime)":
            f.write("Numbering correlates with zip mtimes, suggesting files were "
                    "packed in capture order. However zip mtimes are export times, "
                    "not necessarily capture times.\n")
        elif verdict == "DISH/GROWTH (via bright fraction)":
            f.write("Numbering correlates with bright fraction, which increases "
                    "monotonically with colony growth. This suggests files within "
                    "a dish are ordered by time.\n")
        else:
            f.write("Numbering correlates with both zip mtime and bright fraction. "
                    "This suggests the original capture order is preserved.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--downscale", type=int, default=DOWNSCALE,
                        help="downscale factor for brightness analysis")
    parser.add_argument("--brightness-threshold", type=int, default=200,
                        help="pixel value threshold for bright pixels (0-255)")
    parser.add_argument("--limit", type=int, default=0,
                        help="scan at most N images per archive (smoke test)")
    parser.add_argument("--jobs", type=int, default=11,
                        help="parallel worker processes (one per archive)")
    args = parser.parse_args(argv)
    
    zips = sorted(args.raw_dir.glob("*.zip"))
    if not zips:
        parser.error(f"no *.zip under {args.raw_dir}")
    
    cfg = ScanConfig(
        downscale=args.downscale,
        brightness_threshold=args.brightness_threshold,
        limit=args.limit,
    )
    
    print(f"analyzing {len(zips)} archives (downscale={cfg.downscale}, "
          f"threshold={cfg.brightness_threshold})", flush=True)
    
    rows = analyze_all(zips, cfg)
    
    print(f"writing {len(rows)} rows to {args.out}", flush=True)
    write_signals_csv(args.out, rows)
    
    summary = compute_summary(rows)
    print("computing summary...", flush=True)
    write_notes(args.notes, summary, len(rows))
    print(f"wrote notes to {args.notes}", flush=True)
    
    # Print key findings
    for archive, stats in summary.items():
        fwd_m = f"{stats['fwd_id_mtime']:.4f}" if stats['fwd_id_mtime'] is not None else "N/A"
        fwd_b = f"{stats['fwd_id_bright']:.4f}" if stats['fwd_id_bright'] is not None else "N/A"
        print(f"  {archive:26s} fwd_id→mtime={fwd_m}  fwd_id→bright={fwd_b}", flush=True)
    
    return 0


if __name__ == "__main__":
    import io
    sys.exit(main())