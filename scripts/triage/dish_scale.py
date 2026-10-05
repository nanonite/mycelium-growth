#!/usr/bin/env python3
"""px -> mm scale calibration from the 90-mm dish rim (Chainlink #25).

Phase 1 of PLAN.md turns mask areas into mm^2, which needs one number per
image: pixels per millimetre.  A standard dish is 90 mm across, so the rim
radius that ``dish_geometry.find_disc`` measures *is* the calibration::

    pixels_per_mm = 2 * rim_radius_px / DISH_DIAMETER_MM

Three things stop that from being one global constant:

* **The release was shot at several scales.**  #23 measured
  ``dish_r / image_width`` = 0.302 (GL), 0.331 (PO/TS/MYG) and 0.222 (GS), so
  one constant is wrong by up to ~36%.  The scale therefore belongs to the
  *condition block* (:func:`condition_block`), which is the unit a camera setup
  was constant over.
* **One block is not always one scale.**  The flat ``unlabeled-GS_PO_TS`` shard
  mixes GS-scale frames with PO/TS-scale ones -- every one of the ten
  byte-identical copies of the labeled-GS images sits in a tight r/W = 0.222
  cluster inside it -- so a block is split further into :func:`scale_modes`.
* **The rim fit fails on ~3% of images.**  It locks onto the colony edge or an
  inner reflection and reports a radius tens of percent too small (checked by
  eye, see ``notes/triage_calibration.md``).  A mode's *median* is unaffected by
  those, and an image whose own fit disagrees with the mode it was given is
  flagged rather than trusted.

Everything here is numbers-in / numbers-out with no file I/O, so
``tests/test_dish_scale.py`` can exercise it on synthetic distributions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

# The rim of a standard plate is 90 mm across; every conversion below is
# anchored on that one physical fact.
DISH_DIAMETER_MM = 90.0

# The calibration is estimated on the working images of dish_geometry.py, which
# are decoded to 900 px on the long side.  px/mm scales linearly with the frame
# width, so anything measured at another resolution goes through
# :func:`rescale_px_per_mm` rather than being reused as-is.
REFERENCE_WIDTH = 900

# An image whose own rim fit agrees with its mode to within this relative
# distance is counted as evidence for the mode; a bigger disagreement means the
# fit failed and the mode's constant is used instead of the image's own number.
FIT_TOLERANCE = 0.05

STATUS_OK = "ok"
STATUS_OFF_SCALE = "off_scale"

_GL_ARCHIVES = ("labeled-GL.zip",) + tuple(f"unlabeled-GL{i}.zip" for i in range(1, 8))
_SPECIES_FOLDERS = ("GS", "PO", "TS")
_MEDIUM_FOLDERS = ("MYG", "PDA", "TEMP15")


def condition_block(archive: str, entry: str) -> str:
    """The condition block whose camera scale an image belongs to.

    Blocks follow the r/W table of ``notes/triage_dish_tracks.md`` (§1): the
    seven unlabeled GL shards and ``labeled-GL`` share one scale, each labeled
    condition folder is its own block, and the flat unlabeled shard is a block
    of its own because nothing in its paths says what it holds.

    Raises ``ValueError`` for an archive this scheme does not know: silently
    filing an image under the wrong block would silently corrupt its scale.
    """
    if archive in _GL_ARCHIVES:
        return "GL"
    if archive == "labeled-GS_PO_TS.zip":
        return _folder_block(entry, _SPECIES_FOLDERS, index=1)
    if archive == "labeled-MYG_PDA_TEMP.zip":
        return _folder_block(entry, _MEDIUM_FOLDERS, index=0)
    if archive == "unlabeled-GS_PO_TS.zip":
        return "GS_PO_TS_flat"
    raise ValueError(f"no condition block for archive {archive!r} (entry {entry!r})")


def _folder_block(entry: str, allowed: tuple[str, ...], index: int) -> str:
    parts = entry.split("/")
    if len(parts) <= index or parts[index] not in allowed:
        raise ValueError(
            f"entry {entry!r} has no {'/'.join(allowed)} folder at {index}"
        )
    return parts[index]


@dataclass(frozen=True)
class ScaleMode:
    """One camera setup inside a block: a tight cluster of fitted rim scales."""

    r_over_w: float  # median rim radius / image width of the cluster
    n_cluster: int  # images that formed it


@dataclass(frozen=True)
class ScaleSummary:
    """Scale statistics for one group of images, in CSV-ready form."""

    n: int
    r_over_w_median: float
    r_over_w_iqr: float
    px_per_mm: float
    px_per_mm_p25: float
    px_per_mm_p75: float
    spread_pct: float

    def as_row(self) -> dict[str, object]:
        return asdict(self)


# --- scale modes ------------------------------------------------------------

# A camera setup leaves the dish at one size in frame, so a setup shows up as a
# *tight* cluster in r/W, while a spread of r/W is the fit failing.  The gaps
# these numbers sit in, measured on the release (IQR as a fraction of the
# median): real setups 0.00% (TEMP15) ... 1.91% (TS); failure smears 5.5%
# (flat's low tail) and 6.1% (GL's).  Nothing real lives between 3% and 5%.
MODE_WINDOW = 0.02  # neighbour window, log distance: +/- ~2%
MODE_MIN_IMAGES = 6  # a mode needs at least this many images ...
MODE_MIN_SHARE = 0.005  # ... and at least this share of its block
MODE_MAX_IQR = 0.03  # wider than this is a smear, not a setup
# Below this ratio two tight clusters cannot be two camera heights -- they are
# one setup with fit noise on either side -- so they are merged.  Observed on
# the release: a genuine second setup is 1.49 away (0.222 vs 0.331), the tight
# failure clusters sit at 1.08-1.20 from their own block's mode.  A merge is
# only committed while the union stays tight (``MODE_MAX_IQR``), so merging can
# never manufacture a constant that contradicts the images under it.
MODE_MIN_SEPARATION = 1.25


def scale_modes(
    r_over_w: list[float] | np.ndarray,
    *,
    window: float = MODE_WINDOW,
    min_images: int = MODE_MIN_IMAGES,
    min_share: float = MODE_MIN_SHARE,
    max_iqr: float = MODE_MAX_IQR,
    min_separation: float = MODE_MIN_SEPARATION,
) -> list[ScaleMode]:
    """Find the camera setups inside one block of fitted rim scales.

    Density-based clustering in one dimension: a point is *dense* when at least
    ``max(min_images, min_share * n)`` images sit within ``+/- window`` of it,
    dense points are chained where consecutive ones are within ``window``, the
    resulting clusters are checked for tightness (``max_iqr``), and clusters
    closer together than ``min_separation`` are merged -- but only while the
    union is still tight, so a merge can never invent a constant that disagrees
    with the images filed under it.

    Modes come back dominant first, which is what :func:`assign_scales` needs to
    give a failed fit somewhere sensible to land.  An empty list means "nothing
    here is a cluster" -- see :func:`block_modes` for the fallback.
    """
    values = np.asarray(
        [v for v in np.atleast_1d(r_over_w) if math.isfinite(v) and v > 0], dtype=float
    )
    if values.size == 0:
        return []
    values = np.sort(values)
    log_values = np.log(values)

    min_points = max(min_images, math.ceil(min_share * values.size))
    dense = _dense_mask(log_values, min_points, window)
    clusters = _chain(dense, log_values, window)
    tight = [c for c in clusters if _relative_iqr(values[c]) <= max_iqr]
    survivors = _merge_close(values, tight, min_separation, max_iqr)

    modes = [
        ScaleMode(r_over_w=float(np.median(values[c])), n_cluster=int(c.size))
        for c in survivors
    ]
    modes.sort(key=lambda mode: (-mode.n_cluster, mode.r_over_w))
    return modes


def block_modes(r_over_w: list[float] | np.ndarray) -> list[ScaleMode]:
    """Scale modes for one block, always able to answer.

    A block too small (or too scattered) to yield a density cluster still needs
    a constant, so it falls back to a single mode at the block median -- which
    is the whole calibration of #25 before the mode splitting is applied.
    """
    modes = scale_modes(r_over_w)
    if modes:
        return modes
    values = [v for v in np.atleast_1d(r_over_w) if math.isfinite(v) and v > 0]
    if not values:
        return []
    return [ScaleMode(r_over_w=float(np.median(values)), n_cluster=len(values))]


def _dense_mask(log_values: np.ndarray, min_points: int, window: float) -> np.ndarray:
    lower = np.searchsorted(log_values, log_values - window, side="left")
    upper = np.searchsorted(log_values, log_values + window, side="right")
    return (upper - lower) >= min_points


def _chain(
    dense: np.ndarray, log_values: np.ndarray, window: float
) -> list[np.ndarray]:
    """Single-linkage clusters of dense points, gap between neighbours <= window."""
    index = np.flatnonzero(dense)
    if index.size == 0:
        return []
    gaps = np.diff(log_values[index])
    breaks = np.flatnonzero(gaps > window) + 1
    return [part for part in np.split(index, breaks) if part.size]


def _relative_iqr(sorted_values: np.ndarray) -> float:
    """IQR as a fraction of the median: the spread of one candidate setup."""
    q25, q50, q75 = np.percentile(sorted_values, [25, 50, 75])
    if q50 <= 0:
        return math.inf
    return float((q75 - q25) / q50)


def _merge_close(
    sorted_values: np.ndarray,
    clusters: list[np.ndarray],
    min_separation: float,
    max_iqr: float,
) -> list[np.ndarray]:
    """Merge cluster pairs that are too close to be two camera heights.

    Two tight clusters a few percent apart are one setup with fit noise on
    either side; past ``min_separation`` the gap can only be a second camera
    height.  A merge is committed only while the union stays within
    ``max_iqr`` -- merging two genuinely different populations would produce a
    constant sitting in the gap between them, which describes no image at all.
    """
    remaining = list(clusters)
    while len(remaining) > 1:
        medians = [float(np.median(sorted_values[c])) for c in remaining]
        candidates = sorted(
            (
                (max(medians[i], medians[j]) / min(medians[i], medians[j]), i, j)
                for i in range(len(remaining))
                for j in range(i + 1, len(remaining))
            ),
            key=lambda pair: pair[0],
        )
        merged_any = False
        for ratio, i, j in candidates:
            if ratio >= min_separation:
                break
            union = np.sort(np.concatenate([remaining[i], remaining[j]]))
            if _relative_iqr(sorted_values[union]) > max_iqr:
                continue
            remaining = [c for k, c in enumerate(remaining) if k not in (i, j)]
            remaining.append(union)
            merged_any = True
            break
        if not merged_any:
            break
    return remaining


# --- assignment and summaries ----------------------------------------------


def assign_scales(
    r_over_w: list[float] | np.ndarray,
    modes: list[ScaleMode],
    *,
    tolerance: float = FIT_TOLERANCE,
) -> list[tuple[int, str]]:
    """Give every image a mode: the nearest one it agrees with, else the dominant.

    An image further than ``tolerance`` from *every* mode has no readable scale
    of its own -- that is a failed fit, not a fourth camera setup -- so it takes
    the dominant mode's constant.  The flag says whether the image corroborated
    the mode it was filed under; the driver republishes it against the mode's
    constant so the two columns of ``image_scale.csv`` cannot disagree.
    """
    if not modes:
        raise ValueError("assign_scales needs at least one mode")
    assignments: list[tuple[int, str]] = []
    for value in np.atleast_1d(r_over_w):
        best, best_distance = 0, math.inf
        for index, mode in enumerate(modes):
            distance = abs(float(value) / mode.r_over_w - 1.0)
            if distance < best_distance:
                best, best_distance = index, distance
        if best_distance <= tolerance:
            assignments.append((best, STATUS_OK))
        else:
            assignments.append((0, STATUS_OFF_SCALE))
    return assignments


def scale_summary(r_over_w: list[float], pixels_per_mm: list[float]) -> ScaleSummary:
    """Robust scale statistics for the images of one group.

    Medians and percentiles throughout: the group's own failed fits must not be
    able to move the constant they are being repaired with.
    """
    if len(r_over_w) != len(pixels_per_mm) or not r_over_w:
        raise ValueError("scale_summary needs matching, non-empty inputs")
    radii = np.asarray(r_over_w, dtype=float)
    scale = np.asarray(pixels_per_mm, dtype=float)
    radius_q25, radius_q50, radius_q75 = np.percentile(radii, [25, 50, 75])
    scale_q25, scale_q50, scale_q75 = np.percentile(scale, [25, 50, 75])
    # rounded here, not at the CSV boundary: image_scale.csv publishes the same
    # constant it flags images against, so the two can never disagree by more
    # than the printed precision
    return ScaleSummary(
        n=len(r_over_w),
        r_over_w_median=round(float(radius_q50), 6),
        r_over_w_iqr=round(float(radius_q75 - radius_q25), 6),
        px_per_mm=round(float(scale_q50), 4),
        px_per_mm_p25=round(float(scale_q25), 4),
        px_per_mm_p75=round(float(scale_q75), 4),
        spread_pct=round(float((scale_q75 - scale_q25) / scale_q50 * 100.0), 3),
    )


# --- unit conversion --------------------------------------------------------


def rim_pixels_per_mm(rim_radius_px: float) -> float:
    """px/mm from a rim radius measured on the working image (the 90-mm rim)."""
    if rim_radius_px <= 0:
        raise ValueError(f"rim radius must be positive, got {rim_radius_px}")
    return 2.0 * rim_radius_px / DISH_DIAMETER_MM


def px_to_mm(length_px: float, px_per_mm: float) -> float:
    """Convert a length in pixels to millimetres at a known scale."""
    if px_per_mm <= 0:
        raise ValueError(f"px_per_mm must be positive, got {px_per_mm}")
    return length_px / px_per_mm


def area_px_to_mm2(area_px2: float, px_per_mm: float) -> float:
    """Convert a mask area in px^2 to mm^2 (a length scale squared)."""
    return px_to_mm(1.0, px_per_mm) ** 2 * area_px2


def rescale_px_per_mm(
    px_per_mm: float, width: int, reference_width: int = REFERENCE_WIDTH
) -> float:
    """Move a calibration measured at ``reference_width`` to another width."""
    if reference_width <= 0 or width <= 0:
        raise ValueError(f"widths must be positive, got {width}/{reference_width}")
    return px_per_mm * width / reference_width
