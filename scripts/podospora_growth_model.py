#!/usr/bin/env python3
"""Extract a reproducible hyphal-occupancy proxy and fit the Podospora movies (#47).

This is an image-derived ridge mask, not a trained or manually validated semantic
segmenter. It deliberately reports foreground occupancy and spread proxies rather
than calling ridge pixels hyphal length or counting biological tips.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data/raw/podospora_thallus"
OUT_DIR = REPO / "data/derived/podospora_growth_model"
FRAME_STEP_MIN = 18.0
PIXEL_UM = 1.63
HALF_PIXEL_UM = 2.0 * PIXEL_UM
RIDGE_SIGMA_HALF_PX = 2.5
RIDGE_BETA = 0.5
BACKGROUND_QUANTILES = (0.99, 0.995, 0.999, 0.9995)
QUANTILE_COLUMNS = {
    0.99: "raw_ridge_px_q099",
    0.995: "raw_ridge_px_q0995",
    0.999: "raw_ridge_px_q0999",
    0.9995: "raw_ridge_px_q09995",
}
PRIMARY_QUANTILE = 0.99
MIN_COMPONENT_PIXELS = 1
COMMON_WINDOW_HOURS = 20.0
INITIAL_BASELINE_FRAMES = 5
EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)


def probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
        "-of", "json", str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
        "playback_rate": str(stream.get("r_frame_rate", "")),
    }


def exact_read(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def ridge_response(gray: np.ndarray) -> np.ndarray:
    """Bright, line-like Hessian response at the resampled hyphal width."""
    image = gray.astype(np.float32, copy=False)
    sigma = RIDGE_SIGMA_HALF_PX
    dxx = gaussian_filter(image, sigma=sigma, order=(2, 0), mode="reflect") * sigma**2
    dyy = gaussian_filter(image, sigma=sigma, order=(0, 2), mode="reflect") * sigma**2
    dxy = gaussian_filter(image, sigma=sigma, order=(1, 1), mode="reflect") * sigma**2
    trace = dxx + dyy
    discriminant = np.sqrt(np.maximum((dxx - dyy) ** 2 + 4.0 * dxy**2, 0.0))
    eigen_small = 0.5 * (trace - discriminant)
    eigen_large = 0.5 * (trace + discriminant)
    ratio = np.abs(eigen_large) / (np.abs(eigen_small) + 1e-6)
    return np.maximum(-eigen_small, 0.0) * np.exp(-(ratio**2) / (2.0 * RIDGE_BETA**2))


def spatial_slices(height: int, width: int) -> tuple[tuple[slice, slice], tuple[slice, slice, slice, slice]]:
    """Use a central 70% field for colony features and four outer corners for noise."""
    roi_y0, roi_y1 = int(0.15 * height), int(0.85 * height)
    roi_x0, roi_x1 = int(0.15 * width), int(0.85 * width)
    corner_h, corner_w = int(0.10 * height), int(0.10 * width)
    corners = (
        (slice(0, corner_h), slice(0, corner_w)),
        (slice(0, corner_h), slice(width - corner_w, width)),
        (slice(height - corner_h, height), slice(0, corner_w)),
        (slice(height - corner_h, height), slice(width - corner_w, width)),
    )
    return (slice(roi_y0, roi_y1), slice(roi_x0, roi_x1)), corners


def background_samples(response: np.ndarray, corners: tuple[tuple[slice, slice], ...]) -> np.ndarray:
    return np.concatenate([response[y, x].ravel() for y, x in corners])


def mask_and_moments(
    response_roi: np.ndarray,
    threshold: float,
    min_component: int,
) -> tuple[int, int, float | None, float | None, float | None, float | None]:
    labels, _ = label(response_roi > threshold, structure=EIGHT_CONNECTED)
    component_sizes = np.bincount(labels.ravel())
    if len(component_sizes):
        component_sizes[0] = 0
    selected_components = component_sizes >= min_component
    selected_components[0] = False
    selected = selected_components[labels]
    count = int(selected.sum())
    component_count = int(selected_components.sum())
    if count < 3:
        return count, component_count, None, None, None, None
    y, x = np.nonzero(selected)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    dx = x.astype(np.float64) - x_mean
    dy = y.astype(np.float64) - y_mean
    cov_xx = float(np.mean(dx * dx))
    cov_yy = float(np.mean(dy * dy))
    cov_xy = float(np.mean(dx * dy))
    eigenvalues = np.linalg.eigvalsh(np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]]))
    radius_minor = math.sqrt(max(float(eigenvalues[0]), 0.0)) * HALF_PIXEL_UM
    radius_major = math.sqrt(max(float(eigenvalues[1]), 0.0)) * HALF_PIXEL_UM
    # These radii describe the distribution of detected pixels, not apex locations.
    return count, component_count, radius_major, radius_minor, x_mean, y_mean


def overlay(gray_roi: np.ndarray, mask: np.ndarray, output: Path) -> None:
    image = Image.fromarray(np.asarray(gray_roi, dtype=np.uint8))
    scale = min(1.0, 1500.0 / max(image.size))
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    base = np.asarray(image.resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)
    # BOX downsampling marks any detected source pixel, keeping thin hyphae visible.
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    small_mask = np.asarray(mask_image.resize(size, Image.Resampling.BOX), dtype=np.uint8) > 8
    rgb = np.repeat(base[:, :, None], 3, axis=2)
    rgb[small_mask] = np.array([255, 28, 75], dtype=np.uint8)
    result = Image.fromarray(rgb)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output, optimize=True)


def write_figure(rows: list[dict[str, Any]], fits: list[dict[str, Any]], output: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["movie"], []).append(row)
    colors = {"M0": "#2878B5", "M2": "#D95F02"}
    fit_by_movie = {r["movie"]: r for r in fits}
    width, height = 1400, 560
    left = (80, 75, 810, 425)
    right = (925, 75, 390, 425)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222} .axis{stroke:#555;stroke-width:1} .grid{stroke:#ddd;stroke-width:1}</style>',
        '<text x="700" y="32" text-anchor="middle" font-size="19" font-weight="bold">Podospora thallus: image-derived hyphal foreground occupancy</text>',
    ]
    x0, y0, plot_w, plot_h = left
    y_values = [math.log10(float(r["signal_mask_area_um2_proxy"])) for r in rows
                if float(r["signal_mask_area_um2_proxy"]) > 0 and float(r["time_from_first_h"]) <= COMMON_WINDOW_HOURS]
    y_min = math.floor(min(y_values)) if y_values else 0
    y_max = math.ceil(max(y_values)) if y_values else 1
    y_min = min(y_min, y_max - 1)
    for y_tick in range(y_min, y_max + 1):
        py = y0 + plot_h - (y_tick - y_min) / (y_max - y_min) * plot_h
        parts.append(f'<line class="grid" x1="{x0}" y1="{py:.1f}" x2="{x0+plot_w}" y2="{py:.1f}"/>')
        parts.append(f'<text x="{x0-12}" y="{py+4:.1f}" text-anchor="end" font-size="12">10^{y_tick}</text>')
    for hour in range(0, 21, 4):
        px = x0 + hour / COMMON_WINDOW_HOURS * plot_w
        parts.append(f'<line class="grid" x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+plot_h}"/>')
        parts.append(f'<text x="{px:.1f}" y="{y0+plot_h+22}" text-anchor="middle" font-size="12">{hour}</text>')
    parts.extend([
        f'<line class="axis" x1="{x0}" y1="{y0+plot_h}" x2="{x0+plot_w}" y2="{y0+plot_h}"/>',
        f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+plot_h}"/>',
        f'<text x="{x0+plot_w/2}" y="{y0+plot_h+48}" text-anchor="middle" font-size="13">Hours since first frame (18 min/frame)</text>',
        f'<text x="20" y="{y0+plot_h/2}" transform="rotate(-90 20 {y0+plot_h/2})" text-anchor="middle" font-size="13">Baseline-corrected ridge area proxy (µm²)</text>',
        f'<line x1="{x0+plot_w}" y1="{y0}" x2="{x0+plot_w}" y2="{y0+plot_h}" stroke="#777" stroke-dasharray="4 4"/>',
    ])
    for movie, movie_rows in sorted(grouped.items()):
        movie_rows.sort(key=lambda r: int(r["frame_index"]))
        points = []
        for row in movie_rows:
            time = float(row["time_from_first_h"])
            area = float(row["signal_mask_area_um2_proxy"])
            if time > COMMON_WINDOW_HOURS or area <= 0:
                continue
            px = x0 + time / COMMON_WINDOW_HOURS * plot_w
            py = y0 + plot_h - (math.log10(area) - y_min) / (y_max - y_min) * plot_h
            points.append(f'{px:.1f},{py:.1f}')
        if points:
            parts.append(f'<polyline fill="none" stroke="{colors[movie[:2]]}" stroke-width="1.7" opacity="0.8" points="{" ".join(points)}"/>')
    for i, (movie, color) in enumerate((("M0_1", colors["M0"]), ("M0_2", colors["M0"]), ("M0_3", colors["M0"]),
                                        ("M2_1", colors["M2"]), ("M2_2", colors["M2"]), ("M2_3", colors["M2"]))):
        col = i % 3
        row = i // 3
        lx = x0 + 12 + col * 115
        ly = y0 - 18 + row * 17
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+18}" y2="{ly}" stroke="{color}" stroke-width="3"/><text x="{lx+23}" y="{ly+4}" font-size="11">{movie}</text>')

    rx, ry, rw, rh = right
    successful = [r for r in fits if r.get("status") == "ok"]
    slopes = [float(r["occupancy_exp_slope_per_h"]) for r in successful]
    s_min = min(0.0, min(slopes) if slopes else 0.0)
    s_max = max(0.5, max(slopes) if slopes else 0.5)
    for tick in np.linspace(s_min, s_max, 6):
        py = ry + rh - (tick - s_min) / (s_max - s_min) * rh
        parts.append(f'<line class="grid" x1="{rx}" y1="{py:.1f}" x2="{rx+rw}" y2="{py:.1f}"/>')
        parts.append(f'<text x="{rx-10}" y="{py+4:.1f}" text-anchor="end" font-size="11">{tick:.2f}</text>')
    parts.extend([
        f'<line class="axis" x1="{rx}" y1="{ry+rh}" x2="{rx+rw}" y2="{ry+rh}"/>',
        f'<line class="axis" x1="{rx}" y1="{ry}" x2="{rx}" y2="{ry+rh}"/>',
        f'<text x="{rx+rw/2}" y="{ry-28}" text-anchor="middle" font-size="14" font-weight="bold">Movie-specific exponential slopes</text>',
        f'<text x="{rx+rw/2}" y="{ry+rh+48}" text-anchor="middle" font-size="13">Per-movie log area slope (h⁻¹)</text>',
        f'<text x="{rx-52}" y="{ry+rh/2}" transform="rotate(-90 {rx-52} {ry+rh/2})" text-anchor="middle" font-size="12">Slope, log area per hour</text>',
    ])
    for condition, xloc in (("M0", rx + rw * 0.34), ("M2", rx + rw * 0.70)):
        values = [float(r["occupancy_exp_slope_per_h"]) for r in successful if r["condition"] == condition]
        if not values:
            continue
        xs = np.linspace(xloc - 18, xloc + 18, len(values)) if len(values) > 1 else [xloc]
        for movie_value, point_x in zip(values, xs):
            py = ry + rh - (movie_value - s_min) / (s_max - s_min) * rh
            parts.append(f'<circle cx="{point_x:.1f}" cy="{py:.1f}" r="5" fill="{colors[condition]}"/>')
        mean = float(np.mean(values))
        py = ry + rh - (mean - s_min) / (s_max - s_min) * rh
        parts.append(f'<path d="M {xloc-18:.1f} {py:.1f} L {xloc+18:.1f} {py:.1f}" stroke="{colors[condition]}" stroke-width="3"/>')
        parts.append(f'<text x="{xloc}" y="{ry+rh+22}" text-anchor="middle" font-size="13">{condition}</text>')
    parts.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")


def movie_frames(path: Path, info: dict[str, Any]):
    half_width = info["width"] // 2
    half_height = info["height"] // 2
    frame_bytes = half_width * half_height
    vf = f"scale={half_width}:{half_height}:flags=area,format=gray"
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
        "-i", str(path), "-vf", vf, "-frames:v", str(info.get("processed_frames", info["frames"])),
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    target_frames = int(info.get("processed_frames", info["frames"]))
    for index in range(target_frames):
        chunk = exact_read(process.stdout, frame_bytes)
        if len(chunk) != frame_bytes:
            process.kill()
            raise RuntimeError(f"short frame read from {path.name} at index {index}")
        yield index, np.frombuffer(chunk, dtype=np.uint8).reshape(half_height, half_width)
    process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed on {path.name}: {stderr[-2000:]}")


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ coef
    residual = y - prediction
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else 1.0
    return float(coef[0]), float(coef[1]), r2


def robust_baseline(values: np.ndarray) -> tuple[float, float, float]:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    robust_sd = 1.4826 * mad
    cutoff = center + max(3.0 * robust_sd, 50.0)
    return center, robust_sd, cutoff


def slope_models(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["movie"], []).append(row)
    summaries: list[dict[str, Any]] = []
    for movie, movie_rows in sorted(grouped.items()):
        movie_rows.sort(key=lambda r: r["frame_index"])
        common = [r for r in movie_rows if r["time_from_first_h"] <= COMMON_WINDOW_HOURS]
        baseline_rows = common[:INITIAL_BASELINE_FRAMES]
        areas = np.asarray([r["mask_area_px_resampled"] for r in common], dtype=float)
        base, baseline_robust_sd, onset_cut = robust_baseline(areas[:len(baseline_rows)])
        above = areas > onset_cut
        onset = next((j for j in range(max(0, len(above) - 2)) if bool(np.all(above[j:j + 3]))), None)
        if onset is None:
            summaries.append({
                "movie": movie,
                "condition": movie[:2],
                "frame_count": len(movie_rows),
                "common_window_frames": len(common),
                "baseline_mask_area_px": base,
                "baseline_robust_sd_px": baseline_robust_sd,
                "onset_frame": None,
                "status": "no 3-frame signal above baseline",
            })
            continue
        active = common[onset:]
        elapsed = np.asarray([r["time_from_first_h"] - onset * FRAME_STEP_MIN / 60.0 for r in active])
        signal = np.asarray([max(r["mask_area_px_resampled"] - base, 1.0) for r in active])
        log_signal = np.log(signal)
        _, exp_slope, exp_r2 = fit_line(elapsed, log_signal)
        log_time = np.log(elapsed + FRAME_STEP_MIN / 60.0)
        _, power_exponent, power_r2 = fit_line(log_time, log_signal)

        split = max(4, int(len(active) * 0.70))
        holdout = len(active) - split
        exp_tail_rmse = power_tail_rmse = float("nan")
        if holdout >= 2:
            exp_a, exp_b, _ = fit_line(elapsed[:split], log_signal[:split])
            power_a, power_b, _ = fit_line(log_time[:split], log_signal[:split])
            exp_tail_rmse = float(np.sqrt(np.mean((log_signal[split:] - (exp_a + exp_b * elapsed[split:])) ** 2)))
            power_tail_rmse = float(np.sqrt(np.mean((log_signal[split:] - (power_a + power_b * log_time[split:])) ** 2)))
        summaries.append({
            "movie": movie,
            "condition": movie[:2],
            "frame_count": len(movie_rows),
            "common_window_frames": len(common),
            "baseline_mask_area_px": base,
            "baseline_robust_sd_px": baseline_robust_sd,
            "onset_frame": onset,
            "onset_relative_h": round(onset * FRAME_STEP_MIN / 60.0, 6),
            "active_points_in_common_window": len(active),
            "occupancy_exp_slope_per_h": exp_slope,
            "occupancy_exp_doubling_h": math.log(2) / exp_slope if exp_slope > 0 else None,
            "occupancy_exp_log_r2": exp_r2,
            "occupancy_power_exponent": power_exponent,
            "occupancy_power_log_r2": power_r2,
            "exp_tail_log_rmse": exp_tail_rmse,
            "power_tail_log_rmse": power_tail_rmse,
            "status": "ok",
        })

    successful = [r for r in summaries if r.get("status") == "ok"]
    grouped_slopes: dict[str, list[float]] = {
        condition: [float(r["occupancy_exp_slope_per_h"]) for r in successful if r["condition"] == condition]
        for condition in ("M0", "M2")
    }
    effect = estimate_group_contrast(grouped_slopes["M2"], grouped_slopes["M0"])
    return summaries, effect


def estimate_group_contrast(m2: list[float], m0: list[float]) -> dict[str, Any]:
    if len(m2) < 2 or len(m0) < 2:
        return {"status": "insufficient movie slopes", "m2_n": len(m2), "m0_n": len(m0)}
    m2a, m0a = np.asarray(m2, dtype=float), np.asarray(m0, dtype=float)
    delta = float(m2a.mean() - m0a.mean())
    pooled_var = (
        float(np.sum((m2a - m2a.mean()) ** 2)) + float(np.sum((m0a - m0a.mean()) ** 2))
    ) / (len(m2a) + len(m0a) - 2)
    se = math.sqrt(pooled_var * (1.0 / len(m2a) + 1.0 / len(m0a)))
    df = len(m2a) + len(m0a) - 2
    critical = float(student_t.ppf(0.975, df))
    all_values = np.concatenate([m2a, m0a])
    perm_diffs = []
    for selected in itertools.combinations(range(len(all_values)), len(m2a)):
        chosen = set(selected)
        a = np.asarray([value for index, value in enumerate(all_values) if index in chosen])
        b = np.asarray([value for index, value in enumerate(all_values) if index not in chosen])
        perm_diffs.append(float(a.mean() - b.mean()))
    p_exact = float(np.mean(np.abs(perm_diffs) >= abs(delta) - 1e-12))
    return {
        "status": "ok",
        "m2_n": len(m2a),
        "m0_n": len(m0a),
        "m2_mean_log_slope_per_h": float(m2a.mean()),
        "m0_mean_log_slope_per_h": float(m0a.mean()),
        "m2_minus_m0_log_slope_per_h": delta,
        "log_slope_difference_95ci": [delta - critical * se, delta + critical * se],
        "per_hour_area_multiplier_ratio_m2_to_m0": math.exp(delta),
        "per_hour_multiplier_ratio_95ci": [math.exp(delta - critical * se), math.exp(delta + critical * se)],
        "exact_two_sided_permutation_p": p_exact,
        "m2_movie_slopes": [float(v) for v in m2a],
        "m0_movie_slopes": [float(v) for v in m0a],
        "interpretation": "exploratory contrast of image-derived occupancy slopes; three movies per condition",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit-frames", type=int, default=None, help="debug-only cap per movie")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="rebuild the SVG chart from existing CSV outputs")
    args = parser.parse_args()
    if args.plot_only:
        with (args.output_dir / "frame_measurements.csv").open(newline="") as handle:
            saved_rows = list(csv.DictReader(handle))
        with (args.output_dir / "movie_growth_fits.csv").open(newline="") as handle:
            saved_fits = list(csv.DictReader(handle))
        write_figure(saved_rows, saved_fits, args.output_dir / "podospora_growth_trajectories.svg")
        print(f"Wrote {args.output_dir / 'podospora_growth_trajectories.svg'}")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "qc_overlays"

    all_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    movies = sorted(args.data_dir.glob("M[02]_[123].avi"))
    if len(movies) != 6:
        raise RuntimeError(f"expected six M0/M2 triplicate movies, found {len(movies)} in {args.data_dir}")

    for path in movies:
        info = probe(path)
        n_frames = min(info["frames"], args.limit_frames) if args.limit_frames else info["frames"]
        info["processed_frames"] = n_frames
        inventory.append({
            "movie": path.stem,
            "condition": path.stem[:2],
            "replicate": int(path.stem[-1]),
            "width": info["width"],
            "height": info["height"],
            "frame_count": info["frames"],
            "playback_rate": info["playback_rate"],
            "capture_interval_min": FRAME_STEP_MIN,
            "capture_interval_source": "Figshare article 22666138 description",
            "pixel_size_um": PIXEL_UM,
            "license": "CC BY 4.0",
        })
        print(f"[{path.stem}] processing {n_frames} frames at half resolution", flush=True)
        half_h, half_w = info["height"] // 2, info["width"] // 2
        (roi_y, roi_x), corners = spatial_slices(half_h, half_w)
        overlay_indices = set() if args.no_overlays else {
            0, max(0, n_frames // 3), max(0, 2 * n_frames // 3), n_frames - 1,
        }
        for frame_index, gray in movie_frames(path, info):
            response = ridge_response(gray)
            samples = background_samples(response, corners)
            thresholds = {q: float(np.quantile(samples, q)) for q in BACKGROUND_QUANTILES}
            response_roi = response[roi_y, roi_x]
            threshold = thresholds[PRIMARY_QUANTILE]
            mask_px, components, radius_major, radius_minor, centroid_x, centroid_y = mask_and_moments(
                response_roi, threshold, MIN_COMPONENT_PIXELS
            )
            quantile_counts = {
                q: int(np.count_nonzero(response_roi > thresholds[q]))
                for q in BACKGROUND_QUANTILES
            }
            all_rows.append({
                "movie": path.stem,
                "condition": path.stem[:2],
                "replicate": int(path.stem[-1]),
                "frame_index": frame_index,
                "time_from_first_h": frame_index * FRAME_STEP_MIN / 60.0,
                "capture_interval_min": FRAME_STEP_MIN,
                "pixel_size_um_source": PIXEL_UM,
                "playback_rate_fps": info["playback_rate"],
                "threshold_background_q099": thresholds[0.99],
                "threshold_background_q0995": thresholds[0.995],
                "threshold_background_q0999": thresholds[0.999],
                "threshold_background_q09995": thresholds[0.9995],
                "mask_area_px_resampled": mask_px,
                "mask_area_px_original_equiv": mask_px * 4,
                "mask_area_um2_proxy": mask_px * HALF_PIXEL_UM**2,
                "mask_area_fraction_in_central_roi": mask_px / response_roi.size,
                "mask_component_count": components,
                "spread_r_major_um_proxy": radius_major,
                "spread_r_minor_um_proxy": radius_minor,
                "spread_centroid_x_half_px": centroid_x,
                "spread_centroid_y_half_px": centroid_y,
                **{QUANTILE_COLUMNS[q]: quantile_counts[q] for q in BACKGROUND_QUANTILES},
                "common_20h_window": frame_index * FRAME_STEP_MIN / 60.0 <= COMMON_WINDOW_HOURS,
            })
            if frame_index in overlay_indices:
                mask = response_roi > threshold
                labels, _ = label(mask, structure=EIGHT_CONNECTED)
                component_sizes = np.bincount(labels.ravel())
                if len(component_sizes):
                    component_sizes[0] = 0
                selected_components = component_sizes >= MIN_COMPONENT_PIXELS
                selected_components[0] = False
                mask = selected_components[labels]
                overlay(gray[roi_y, roi_x], mask, overlay_dir / f"{path.stem}_frame{frame_index:03d}.png")
            if frame_index and frame_index % 20 == 0:
                print(f"[{path.stem}] frame {frame_index}/{n_frames}", flush=True)

    # Movie-specific baseline correction is computed only after every frame is read.
    by_movie: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_movie.setdefault(row["movie"], []).append(row)
    for movie_rows in by_movie.values():
        movie_rows.sort(key=lambda row: row["frame_index"])
        baseline_rows = movie_rows[:min(INITIAL_BASELINE_FRAMES, len(movie_rows))]
        baseline = float(np.median([r["mask_area_px_resampled"] for r in baseline_rows]))
        for row in movie_rows:
            row["baseline_mask_area_px_resampled"] = baseline
            row["signal_mask_area_px_resampled"] = max(row["mask_area_px_resampled"] - baseline, 0.0)
            row["signal_mask_area_um2_proxy"] = row["signal_mask_area_px_resampled"] * HALF_PIXEL_UM**2

    movie_summaries, effect = slope_models(all_rows)
    sensitivity: dict[str, Any] = {}
    for q, column in QUANTILE_COLUMNS.items():
        slopes_by_condition: dict[str, list[float]] = {"M0": [], "M2": []}
        for movie, movie_rows in by_movie.items():
            common = [r for r in movie_rows if r["time_from_first_h"] <= COMMON_WINDOW_HOURS]
            raw = np.asarray([float(r[column]) for r in common])
            baseline, _, cutoff = robust_baseline(raw[:INITIAL_BASELINE_FRAMES])
            areas = np.maximum(raw - baseline, 1.0)
            above = raw > cutoff
            onset = next((j for j in range(max(0, len(above) - 2)) if bool(np.all(above[j:j + 3]))), None)
            if onset is None or len(common) - onset < 6:
                continue
            time = np.asarray([r["time_from_first_h"] for r in common[onset:]])
            _, slope, _ = fit_line(time, np.log(areas[onset:]))
            slopes_by_condition[movie[:2]].append(slope)
        sensitivity[str(q)] = estimate_group_contrast(slopes_by_condition["M2"], slopes_by_condition["M0"])

    out = args.output_dir
    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = out / name
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv("movie_inventory.csv", inventory)
    write_csv("frame_measurements.csv", all_rows)
    write_csv("movie_growth_fits.csv", movie_summaries)
    summary = {
        "dataset": "Podospora anserina M0/M2 six-movie triplicate dataset",
        "n_movies": len(movies),
        "n_frames_processed": len(all_rows),
        "capture_interval_min": FRAME_STEP_MIN,
        "time_origin": "relative to first AVI frame; not asserted to be ascospore germination",
        "pixel_size_um": PIXEL_UM,
        "analysis_resolution": "half-size grayscale; 3.26 um per resampled pixel",
        "measurement": "bright, line-like Hessian ridge candidates above each frame's outer-corner background quantile; primary q=0.99, no component-size filter",
        "primary_background_quantile": PRIMARY_QUANTILE,
        "common_condition_comparison_window_h": COMMON_WINDOW_HOURS,
        "model": "per-movie log foreground occupancy slope; M2-M0 slope contrast is a log per-hour multiplier ratio",
        "growth_form_selection": "compare exponential occupancy (log area linear in time) with power-law occupancy using a 30% within-movie temporal tail holdout",
        "nutrient_effect": effect,
        "threshold_sensitivity": sensitivity,
        "limitations": [
            "No human-labeled hyphal masks are released; ridge foreground is a candidate occupancy proxy.",
            "Foreground pixels are not a validated count of hyphal length, branches, or apexes.",
            "Movie replicates, not frames, are the independent units; n=3 per condition.",
            "Relative capture time is known from frame order and Figshare metadata; first frame is not assumed to equal germination time.",
            "The paper's published apex-count and apex-spread outcomes cannot be exactly reproduced from this foreground mask.",
        ],
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_figure(all_rows, movie_summaries, out / "podospora_growth_trajectories.svg")
    print(json.dumps({
        "output_dir": str(out),
        "frames": len(all_rows),
        "movie_fits": movie_summaries,
        "nutrient_effect": effect,
        "threshold_sensitivity": sensitivity,
    }, indent=2))


if __name__ == "__main__":
    main()
