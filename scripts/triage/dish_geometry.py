#!/usr/bin/env python3
"""Pure geometry helpers for locating a petri dish in a top-down dish photo.

Everything in this module is array-in / dict-out and free of file I/O so it can
be unit-tested against synthetic dishes (``tests/test_dish_geometry.py``).

Why a dedicated module
----------------------
Chainlink #23 asks whether the MyceliumSeg images can be grouped into per-dish
tracks.  The only physical handle on "which dish is this" is the dish itself: its
rim, the marker dot painted on that rim, and where the inoculum was placed.  All
three need the same first step -- find the dish circle -- so it lives here.

How the circle is found
-----------------------
The dishes are transparent plastic on a dark stage, lit from above.  The rim is a
bright annulus; outside it is near-black stage.  Two properties follow:

* the rim's *outer* edge is the steepest outward drop in the angular-median
  radial profile, and
* the drop is found from the profile's derivative rather than from an absolute
  brightness cut-off, because the stage illumination is visibly uneven (there are
  broad vertical falloff bands in some images).  A fixed threshold lands in a
  different place on every frame; the derivative does not.

Sub-pixel refinement is a parabola through the derivative minimum.  Note the
sub-pixel value is only ever a ~1 px correction, so downstream callers should not
treat the third decimal place as meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

# Working resolution.  The rim is tens of pixels across at full size, so
# decoding large is wasted work: 900 px on the long side keeps it unambiguous
# while making a full pass over the release affordable.
TARGET_LONG_SIDE = 900

# ``find_rim_dots`` samples the rim on this many angular steps, so the measured
# dot angle is quantised to 360 / RIM_DOT_N_ANG = 0.5 deg unless the peak spans
# two steps and is reported as their mean.  Anything finer than this in the
# angle statistics is an artefact of the sampling grid, not of the dish.
RIM_DOT_N_ANG = 720

# The dish fills roughly 30% of the frame width (verified across all 11
# archives), so the search window below is generous rather than tight.
RADIUS_FRACTION_LO = 0.16
RADIUS_FRACTION_HI = 0.44

# How far the dish centre may sit from the middle of the frame.  The camera is
# fixed, so this is a safety margin around an already-centred object; it is not
# a claim that the dish moves that far.
CENTRE_SEARCH_SPAN = 60.0

# Fraction of frame width used as the flank when scoring a radial ring.  The
# rim annulus is ~0.02 W thick, so this is comfortably wider than the feature it
# must bracket and much narrower than the colony.
RING_FLANK_FRACTION = 0.03


@dataclass(frozen=True)
class Disc:
    """A dish located in one frame.

    ``cx``/``cy`` are in pixels of the *working* image (see
    :func:`load_rgb`), ``r`` is the outer rim radius in the same units.
    """

    cx: float
    cy: float
    r: float
    edge: float      # profile derivative at the edge; negative, sharper = better
    background: float
    width: int
    height: int
    confidence: float = 1.0   # 0 = no usable rim found, 1 = unambiguous

    @property
    def r_over_w(self) -> float:
        """Rim radius as a fraction of image width.

        Nearly constant across the release, so it doubles as a cheap check that
        a frame was decoded at the expected scale.
        """
        return self.r / self.width

    @property
    def pixels_per_mm(self) -> float:
        """Scale assuming the dish is a 90 mm standard plate (#25).

        Kept here because every later step wants a physical length, but the
        90 mm assumption is only as good as the dish being a standard plate --
        see ``notes/triage_dish_tracks.md``.
        """
        return (2.0 * self.r) / 90.0


def background_level(lum: np.ndarray, border: int = 6) -> float:
    """Median of the image border: the dark stage the dish sits on."""
    b = min(border, max(1, min(lum.shape) // 8))
    edge = np.concatenate([
        lum[:b, :].ravel(), lum[-b:, :].ravel(),
        lum[:, :b].ravel(), lum[:, -b:].ravel(),
    ])
    return float(np.median(edge))


def polar_median(
    lum: np.ndarray,
    cx: float,
    cy: float,
    radii: np.ndarray,
    n_ang: int = 240,
) -> np.ndarray:
    """Median luminance sampled on circles of each radius around ``(cx, cy)``.

    The median over angle is what makes this robust: a single pen mark, a
    reflection, or a speck of dust on the stage perturbs one or two samples,
    while a real rim brightens the whole ring at once.
    """
    angles = np.arange(n_ang) / n_ang * 2.0 * np.pi
    xs = np.clip(
        (cx + radii[None, :] * np.cos(angles)[:, None]).astype(int),
        0, lum.shape[1] - 1,
    )
    ys = np.clip(
        (cy + radii[None, :] * np.sin(angles)[:, None]).astype(int),
        0, lum.shape[0] - 1,
    )
    return np.median(lum[ys, xs], axis=0)


def polar_median_many(
    lum: np.ndarray,
    centres: np.ndarray,
    radii: np.ndarray,
    n_ang: int = 240,
) -> np.ndarray:
    """Angular-median radial profile for many centres at once.

    Returns ``(len(centres), len(radii))``.  Evaluating the centre sweep one
    candidate at a time spends nearly all its time inside ``np.median``, which
    cannot be amortised across candidates; doing the whole sweep in one gather
    turns the search from minutes per image into milliseconds.
    """
    angles = np.arange(n_ang) / n_ang * 2.0 * np.pi
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    n_c = len(centres)
    # (n_c, n_ang, n_r) is too large to materialise, so chunk over radii
    out = np.empty((n_c, len(radii)), dtype=np.float32)
    cxs = centres[:, 0]
    cys = centres[:, 1]
    for i, r in enumerate(radii):
        xs = np.clip(
            (cxs[:, None] + r * cos_a[None, :]).astype(np.intp), 0, lum.shape[1] - 1,
        )
        ys = np.clip(
            (cys[:, None] + r * sin_a[None, :]).astype(np.intp), 0, lum.shape[0] - 1,
        )
        out[:, i] = np.median(lum[ys, xs], axis=1)
    return out


def radial_profile(
    lum: np.ndarray,
    cx: float,
    cy: float,
    n_ang: int = 240,
    smooth: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed angular-median radial profile and its derivative."""
    radii = np.arange(0.0, max(lum.shape) * 0.48, 1.0)
    med = polar_median(lum, cx, cy, radii, n_ang)
    sm = np.convolve(med, np.ones(smooth) / smooth, mode="same")
    return sm, np.gradient(sm)


def stage_level(profile: np.ndarray, width: int) -> float:
    """Brightness of the bare stage, read from the far side of the dish.

    Measured locally rather than from the image border because the illumination
    across the stage is uneven; a global background level lands in the wrong
    place on a fair number of frames (see notes/triage_dish_tracks.md).
    """
    lo = int(0.42 * width)
    hi = min(len(profile), int(0.48 * width))
    if hi <= lo:
        return float(np.min(profile))
    return float(np.median(profile[lo:hi]))


def _outermost_above_stage(profile: np.ndarray, derivative: np.ndarray,
                           width: int) -> int:
    """Outermost *sharp* falloff whose outside is the bare stage.

    Two simpler rules both fail on this dataset, which is why this is a bit more
    careful:

    * "outermost radius above the stage" picks the *colony* edge, because the agar
      inside a dish is only slightly brighter than the stage and the colony is
      far brighter, so both are "above stage" and the colony is the inner one.
    * "sharpest falloff" also picks the colony edge on a well-covered plate.

    What identifies the rim is that it is a sharp edge onto the stage.  The
    sharpness test is essential rather than decorative: beyond the rim the
    profile sits on the stage for hundreds of pixels with derivative
    fluctuations around zero, and every one of those wobbles otherwise looks
    like a perfectly good "edge onto stage" and the outermost one wins.
    """
    stage = stage_level(profile, width)
    lo = int(RADIUS_FRACTION_LO * width)
    hi = min(int(RADIUS_FRACTION_HI * width), len(profile) - 1)

    contrast = float(np.percentile(profile[lo:hi], 99.0)) - stage
    post_margin = max(2.0, 0.08 * contrast)
    # A real edge, not stage-level noise.  Deliberately modest: on a well-covered
    # plate the colony's own edge is *sharper* than the rim, and the stage test
    # below is what rejects it.  A threshold scaled to the sharpest edge in the
    # frame would throw the rim away along with the noise.
    sharp_floor = max(1.0, 0.03 * contrast)
    f1 = max(2, int(0.02 * width))
    f2 = max(f1 + 2, int(0.08 * width))

    interior = derivative[lo + 1:hi]
    is_min = np.ones(len(interior), dtype=bool)
    is_min[1:] &= interior[1:] <= interior[:-1]
    is_min[:-1] &= interior[:-1] <= interior[1:]
    candidates = np.nonzero(is_min)[0] + lo + 1

    best = -1
    fallback = -1
    for i in candidates:
        outer_end = min(len(profile), i + f2)
        if outer_end - i < f1 + 1:
            continue
        if derivative[i] > -sharp_floor:
            continue
        post = float(np.median(profile[i + f1:outer_end]))
        if post > stage + post_margin:
            continue                            # lands on agar, not on the stage
        if fallback < 0:
            fallback = int(i)
        if best < 0 or i > best:
            best = int(i)
    return best if best >= 0 else fallback


def _smooth_profiles(profiles: np.ndarray, smooth: int) -> tuple[np.ndarray, np.ndarray]:
    """Smooth each row and differentiate along the radius axis."""
    kernel = np.ones(smooth) / smooth
    sm = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), 1, profiles,
    )
    return sm, np.gradient(sm, axis=1)


def find_disc(
    lum: np.ndarray,
    n_ang_fine: int = 360,
    smooth: int = 5,
    search_span: float = CENTRE_SEARCH_SPAN,
) -> Disc:
    """Locate the dish by its outer rim edge.

    A coarse-to-fine search over candidate centres.  The camera is fixed so the
    dish sits near the middle of the frame and the first pass only has to sweep a
    generous window to find it; the later passes localise it, which matters
    because the rim-marker's angle is measured against this centre and a few
    pixels of error rotates every reported angle.

    Each pass sweeps all its candidates in one batched gather, and the coarse
    pass deliberately samples few angles and a coarse radius grid -- it only has
    to find the dish, not measure it.  Evaluating every pass at full resolution
    costs ~0.6 s per image, which is the whole scan budget for nothing.
    """
    height, width = lum.shape
    lo = int(RADIUS_FRACTION_LO * width)
    hi = int(RADIUS_FRACTION_HI * width)

    cx, cy = float(width / 2), float(height / 2)
    # (span, centre step, n_ang, radius step)
    passes = (
        (search_span, 4.0, 96, 2.0),
        (5.0, 1.5, 180, 1.0),
        (1.5, 0.5, n_ang_fine, 1.0),
    )
    for span, step, angles, radius_step in passes:
        radii = np.arange(0.0, max(lum.shape) * 0.48, radius_step)
        xs = np.arange(cx - span, cx + span + 0.1, step)
        ys = np.arange(cy - span, cy + span + 0.1, step)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        centres = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        profiles = polar_median_many(lum, centres, radii, n_ang=angles)
        sm, _ = _smooth_profiles(profiles, smooth)
        lo_i = int(lo / radius_step)
        hi_i = min(int(hi / radius_step), profiles.shape[1] - 1)
        # Score each candidate centre by the best *ring* in its profile: a bright
        # annulus with darker material on both sides.  Minimising the sharpest
        # falloff instead is what lets an off-centre candidate win by putting the
        # colony's own edge in frame; a colony is a plateau, so its ring score is
        # ~0 wherever it is, while the rim scores strongly.
        #
        # The flank subtraction has to run on the *whole* profile and only then be
        # sliced: rolling a slice wraps its own ends around and manufactures rings
        # that are not in the image.
        flank = max(3, int(RING_FLANK_FRACTION * width / radius_step))
        full_ring = sm - 0.5 * (np.roll(sm, flank, axis=1) + np.roll(sm, -flank, axis=1))
        ring = full_ring[:, lo_i:hi_i]
        radii_here = np.arange(lo_i, hi_i) * radius_step
        usable = radii_here > (0.18 * width)
        masked = np.where(usable[None, :], ring, -np.inf)
        score = np.where(np.isfinite(masked).any(axis=1),
                         masked.max(axis=1), ring.max(axis=1))
        best = int(np.argmax(score))
        cx, cy = float(centres[best, 0]), float(centres[best, 1])

    radii = np.arange(0.0, max(lum.shape) * 0.48, 1.0)
    sm, d = _smooth_profiles(
        polar_median_many(lum, np.array([[cx, cy]]), radii, n_ang=n_ang_fine), smooth,
    )
    sm, d = sm[0], d[0]
    stage = stage_level(sm, width)
    contrast = float(np.percentile(sm[lo:min(hi, len(sm) - 1)], 99.0)) - stage
    margin = max(3.0, 0.15 * contrast)
    flank = max(3, int(RING_FLANK_FRACTION * width))

    # A colony is a bright *disc*, so its edge also looks ring-like and on a
    # well-covered plate it can score as strongly as the rim.  What separates
    # them is that the rim is the outermost ring: everything past it is stage
    # for good, while past the colony edge there is still agar and then the rim.
    # So take the outermost radius whose ring is significant, not the strongest.
    ring = sm - 0.5 * (np.roll(sm, flank) + np.roll(sm, -flank))
    band = np.zeros(len(sm), dtype=bool)
    band[lo:min(hi, len(sm) - 1)] = True
    if not band.any():
        band[:] = True
    significant = band & (ring >= 0.5 * float(ring[band].max())) & (sm > stage + margin)
    indices = np.nonzero(significant)[0]
    if len(indices):
        i = int(indices.max())
    else:
        i = _outermost_above_stage(sm, d, width)
    if i < 0:
        i = lo + int(np.argmin(d[lo:min(hi, len(d) - 1)]))

    r = float(radii[i])
    if 0 < i < len(d) - 1:                      # parabolic sub-pixel refinement
        y0, y1, y2 = d[i - 1], d[i], d[i + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-9:
            r += float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))

    # Confidence: how ring-like the winning radius looks, relative to the best
    # candidate seen during the search.  A plate whose colony covers almost the
    # whole dish can leave the rim barely brighter than the agar, and the radius
    # then degrades towards the agar surface -- callers need to be able to tell.
    contrast = max(float(np.percentile(sm[lo:min(hi, len(sm) - 1)], 99.0)) - stage, 1e-6)
    confidence = float(np.clip(ring[i] / (0.5 * contrast), 0.0, 1.0))

    return Disc(
        cx=cx, cy=cy, r=r, edge=float(d[i]),
        background=background_level(lum), width=int(width), height=int(height),
        confidence=confidence,
    )


def angular_samples(rgb: np.ndarray, disc: Disc, rho_lo: float, rho_hi: float,
                    n_rho: int, n_ang: int) -> np.ndarray:
    """Unwrap an annulus around the dish into ``(n_ang, n_rho, 3)``.

    Image coordinates: row 0 is the top, so angles run clockwise on screen and
    0 deg points along +x (image right).  Callers that want a compass angle
    should negate y themselves.
    """
    heights, width = rgb.shape[:2]
    angles = np.arange(n_ang) / n_ang * 2.0 * np.pi
    rhos = np.linspace(rho_lo, rho_hi, n_rho) * disc.r
    xs = np.clip((disc.cx + rhos[None, :] * np.cos(angles)[:, None]).astype(int), 0, width - 1)
    ys = np.clip((disc.cy + rhos[None, :] * np.sin(angles)[:, None]).astype(int), 0, heights - 1)
    return rgb[ys, xs]


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Mean direction of angles in degrees, in [0, 360).

    Averaging angles naively breaks at the 0/360 wrap -- 350 and 10 must average
    to 0, not 180.  The modulo alone is not enough: ``-1e-15 % 360`` evaluates to
    exactly ``360.0`` in IEEE-754, which would put a valid direction outside the
    documented range and break the angular-difference arithmetic downstream.
    """
    if len(angles_deg) == 0:
        return float("nan")
    radians = np.radians(np.asarray(angles_deg, dtype=float))
    degrees = math.degrees(math.atan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360.0
    return 0.0 if degrees >= 360.0 else degrees


def circular_resultant(angles_deg: np.ndarray) -> float:
    """Length of the mean resultant vector: 1 = perfectly tight, 0 = uniform."""
    if len(angles_deg) == 0:
        return 0.0
    radians = np.radians(np.asarray(angles_deg, dtype=float))
    return float(math.hypot(np.sin(radians).mean(), np.cos(radians).mean()))


def find_rim_dots(
    rgb: np.ndarray,
    disc: Disc,
    n_ang: int = RIM_DOT_N_ANG,
    rho_lo: float = 0.78,
    rho_hi: float = 0.98,
    sat_floor: float = 45.0,
    min_sep_deg: float = 15.0,
    max_dots: int = 3,
) -> list[dict[str, float]]:
    """Locate the saturated marker dot(s) painted on the dish rim.

    Each dish carries a small yellow dot on the rim.  It is the single most
    dish-specific feature in the release, so it is worth isolating precisely.

    The dot is only a few pixels wide.  Averaging along the radial band would
    dilute it below any useful threshold, so for each angle we keep the *most
    yellow* sample found anywhere in the band rather than the mean, then look for
    a circular peak in that profile.  ``rho`` reports where in the band the
    strongest response sat, which is a useful sanity check: the dot sits near
    0.9 of the rim radius, not out at the edge.
    """
    heights, width = rgb.shape[:2]
    angles = np.arange(n_ang) / n_ang * 2.0 * np.pi
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    best_sat = np.zeros(n_ang)
    best_rho = np.zeros(n_ang)
    for rho in np.arange(rho_lo, rho_hi + 1e-9, 0.01):
        xs = np.clip((disc.cx + disc.r * rho * cos_a).astype(int), 0, width - 1)
        ys = np.clip((disc.cy + disc.r * rho * sin_a).astype(int), 0, heights - 1)
        red, green, blue = rgb[ys, xs, 0], rgb[ys, xs, 1], rgb[ys, xs, 2]
        lit = (red > 80) & (green > 55) & (blue < 90)
        # yellow-ness: green high relative to blue, which is what separates the
        # marker from the warm-tinted agar and from neutral stage grey
        sat = np.where(lit, green - blue, 0.0)
        take = sat > best_sat
        best_sat = np.where(take, sat, best_sat)
        best_rho = np.where(take, rho, best_rho)

    if best_sat.max() < sat_floor:
        return []

    peaks: list[dict[str, float]] = []
    floor = max(sat_floor, 0.55 * float(best_sat.max()))
    for idx in np.argsort(-best_sat):
        value = float(best_sat[idx])
        if value < floor:
            break
        window = int(n_ang * min_sep_deg / 360.0)
        span = (np.arange(idx - window, idx + window + 1)) % n_ang
        if value < float(best_sat[span].max()) - 1e-6:
            continue                            # not a local maximum
        members = span[best_sat[span] >= max(0.45 * value, sat_floor)]
        degrees = np.degrees(angles[members])
        mean_deg = circular_mean_deg(degrees)
        if any(abs((mean_deg - p["angle_deg"] + 180.0) % 360.0 - 180.0) < min_sep_deg
               for p in peaks):
            continue
        peaks.append({
            "angle_deg": mean_deg,
            "saturation": value,
            "concentration": circular_resultant(degrees),
            "rho": float(np.median(best_rho[members])),
        })
        if len(peaks) >= max_dots:
            break
    return peaks


def colony_metrics(rgb: np.ndarray, disc: Disc) -> dict[str, float]:
    """Measure the colony against the dark agar around it.

    The threshold is derived from the agar annulus rather than from a percentile
    of the whole disc.  A percentile is the obvious shortcut and it is wrong: it
    returns the same area no matter what the colony does, because the top N% of
    pixels are bright whether the colony is a speck or covers the plate.
    """
    lum = rgb.mean(axis=2)
    heights, width = lum.shape
    ys, xs = np.mgrid[0:heights, 0:width]
    radius = np.hypot(xs - disc.cx, ys - disc.cy)

    band = (radius > disc.r * 0.70) & (radius < disc.r * 0.80)
    inside = radius < disc.r * 0.80
    if band.sum() < 200 or inside.sum() < 200:
        return {
            "area_frac": 0.0,
            "radius_frac": 0.0,
            "inoculum_offset": 0.0,
            "inoculum_angle_deg": float("nan"),
            "agar_background": float("nan"),
            "agar_noise": float("nan"),
        }

    agar_bg = float(np.median(lum[band]))
    agar_sd = float(np.std(lum[band]))
    mask = inside & (lum > agar_bg + max(12.0, 5.0 * agar_sd))

    area_frac = float(mask.sum() / max(int(inside.sum()), 1))
    if mask.sum() < 100:
        return {
            "area_frac": 0.0,
            "radius_frac": 0.0,
            "inoculum_offset": 0.0,
            "inoculum_angle_deg": float("nan"),
            "agar_background": agar_bg,
            "agar_noise": agar_sd,
        }

    mx = float(xs[mask].mean())
    my = float(ys[mask].mean())
    return {
        "area_frac": area_frac,
        "radius_frac": float(np.sqrt(mask.sum() / np.pi) / disc.r),
        "inoculum_offset": float(np.hypot(mx - disc.cx, my - disc.cy) / disc.r),
        "inoculum_angle_deg": float(
            math.degrees(math.atan2(-(my - disc.cy), mx - disc.cx)) % 360.0
        ),
        "agar_background": agar_bg,
        "agar_noise": agar_sd,
    }


def detrend_circular(signal: np.ndarray, window: int = 31) -> np.ndarray:
    """Subtract a circular moving average from a closed angular signal.

    The stage illumination has broad angular structure (a smooth ramp, plus
    vertical falloff bands).  Left in, that low-frequency component dominates any
    correlation between frames and drowns the sharp features -- the marker dot,
    pen marks -- that would actually identify a dish.
    """
    if len(signal) < window:
        return signal - float(signal.mean())
    half = window // 2
    padded = np.concatenate([signal[-half:], signal, signal[:half]])
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="same")[half:half + len(signal)]
    return signal - smoothed


def rim_signature(rgb: np.ndarray, disc: Disc, n_ang: int = 180,
                  rho_lo: float = 0.86, rho_hi: float = 1.00,
                  n_rho: int = 6, detrend_window: int = 31) -> np.ndarray:
    """An ``2 * n_ang`` fingerprint of the rim, detrended against illumination.

    Yellow-ness and brightness, each detrended, so what remains is the sharp
    per-dish structure.  Rotation-*sensitive*: two frames of the same dish only
    match if the dish did not turn between shots.
    """
    stack = angular_samples(rgb, disc, rho_lo, rho_hi, n_rho, n_ang)
    yellow = stack[:, :, 1].mean(axis=1) - stack[:, :, 2].mean(axis=1)
    bright = stack.mean(axis=2).mean(axis=1)
    return np.concatenate([detrend_circular(yellow, detrend_window),
                           detrend_circular(bright, detrend_window)])


def rotation_invariant_signature(
    rgb: np.ndarray,
    disc: Disc,
    n_ang: int = 256,
    harmonics: int = 8,
    rho_lo: float = 0.84,
    rho_hi: float = 1.00,
    n_rho: int = 6,
) -> np.ndarray:
    """Fingerprint of the rim that does not care how far the dish was rotated.

    Magnitudes of the angular Fourier coefficients: rotating the dish only
    changes the phases.  This is the right feature to ask "are these the same
    dish?" when the dish may have been picked up and turned.
    """
    stack = angular_samples(rgb, disc, rho_lo, rho_hi, n_rho, n_ang)
    yellow = stack[:, :, 1].mean(axis=1) - stack[:, :, 2].mean(axis=1)
    bright = stack.mean(axis=2).mean(axis=1)
    parts = []
    for channel in (yellow, bright):
        centred = channel - channel.mean()
        parts.append(np.log1p(np.abs(np.fft.rfft(centred))[1:harmonics + 1]))
    return np.concatenate(parts)


def zscore(matrix: np.ndarray) -> np.ndarray:
    """Column-wise standardisation, guarding against constant columns."""
    return (matrix - matrix.mean(axis=0)) / np.maximum(matrix.std(axis=0), 1e-6)


def correlation_matrix(features: np.ndarray) -> np.ndarray:
    """Pearson correlation between rows, with the diagonal set to ``-inf``.

    The normalisation divides by the *outer product* of the row norms.  Dividing
    by a single broadcast norm is a silent, spectacular bug: it leaves values far
    outside [-1, 1] and every downstream threshold becomes meaningless.
    """
    norms = np.linalg.norm(features, axis=1)
    norms = np.maximum(norms, 1e-12)
    corr = (features @ features.T) / np.outer(norms, norms)
    np.fill_diagonal(corr, -np.inf)
    return corr


def gaussian_null(features: np.ndarray, seed: int = 0) -> np.ndarray:
    """Nearest-neighbour correlations for rows drawn with the same covariance.

    The null that matters here is not "random noise": the signatures have strong
    shared low-frequency structure, so unstructured noise would give a
    spuriously low baseline and make any real grouping look significant.  Drawing
    from a Gaussian with the observed covariance reproduces exactly the part of
    the similarity structure that is *not* dish identity.
    """
    rng = np.random.default_rng(seed)
    centred = features - features.mean(axis=0)
    cov = np.cov(centred, rowvar=False) + 1e-9 * np.eye(features.shape[1])
    factor = np.linalg.cholesky(cov)
    draws = rng.standard_normal((len(features), features.shape[1])) @ factor.T
    return gaussian_null_nn(draws)


def gaussian_null_nn(draws: np.ndarray) -> np.ndarray:
    """Nearest-neighbour correlation of each row of ``draws`` against the rest."""
    norms = np.maximum(np.linalg.norm(draws, axis=1), 1e-12)
    corr = (draws @ draws.T) / np.outer(norms, norms)
    np.fill_diagonal(corr, -np.inf)
    return corr.max(axis=1)


def forward_step_fraction(values: list[float]) -> float | None:
    """Fraction of consecutive pairs that do not decrease.

    Reused from :mod:`exif_parse` for the same reason it was written there: a
    genuinely ordered series sits near 1.0, a shuffled one near 0.5.
    """
    clean = [v for v in values if not math.isnan(v)]  # a missing dot angle is NaN
    if len(clean) < 2:
        return None
    steps = sum(1 for a, b in pairwise(clean) if b >= a)
    return steps / (len(clean) - 1)