"""Unit tests for the dish-geometry primitives used to recover dish tracks.

The measurements only mean something if they survive the failure modes that are
actually present in this dataset, so the synthetic fixtures below reproduce them
deliberately:

* a **non-uniform illumination ramp** across the stage, which defeats any
  absolute brightness threshold;
* a **colony of varying size**, which defeats a percentile threshold;
* an **off-centre dish**, since the rim-dot angle is measured against the
  fitted centre and a wrong centre silently rotates every angle;
* **marker dots at known angles**, to check the detector recovers the angle
  rather than merely returning *something*.

A test that only ever sees clean, centred, evenly lit dishes would pass against
an implementation that is wrong on real frames.
"""

from __future__ import annotations

import math

import dish_geometry as dg
import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# synthetic dish fixture
# --------------------------------------------------------------------------- #

STAGE = 18.0          # dark stage grey
RIM_GAIN = 90.0       # how much brighter the plastic rim is than the stage
DISH_R = 130          # rim radius in pixels
FRAME = (520, 390)    # (width, height) -- landscape, like the real 4608x3456


def make_dish(
    width: int = FRAME[0],
    height: int = FRAME[1],
    cx: float | None = None,
    cy: float | None = None,
    radius: float = DISH_R,
    colony_radius: float = 0.0,
    colony_offset: tuple[float, float] = (0.0, 0.0),
    dot_angle_deg: float | None = None,
    dot_rho: float = 0.90,
    ramp: bool = True,
    pen_marks: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Render a synthetic top-down dish photo.

    Returns the RGB float array and the ground-truth geometry, so tests can
    assert against the values used to draw the image rather than magic numbers.
    """
    rng = np.random.default_rng(seed)
    cx = width / 2.0 if cx is None else cx
    cy = height / 2.0 if cy is None else cy

    ys, xs = np.mgrid[0:height, 0:width]
    radius_map = np.hypot(xs - cx, ys - cy)
    image = np.full((height, width, 3), STAGE, dtype=np.float32)

    # even illumination, unless a ramp is requested
    if ramp:
        ramp_x = 14.0 * (xs / max(width - 1, 1))
        ramp_y = 7.0 * (ys / max(height - 1, 1))
        image += (ramp_x + ramp_y)[..., None]

    image += rng.normal(0.0, 1.2, image.shape).astype(np.float32)

    # agar disc: slightly brighter than the stage (scattered light)
    agar = radius_map < radius * 0.86
    image[agar] += 16.0

    # the rim itself: a bright annulus a few pixels thick
    rim = (radius_map >= radius * 0.86) & (radius_map <= radius)
    image[rim] += RIM_GAIN

    # the colony
    if colony_radius > 0:
        ox = cx + colony_offset[0] * radius
        oy = cy + colony_offset[1] * radius
        colony = np.hypot(xs - ox, ys - oy) < colony_radius * radius
        image[colony] = np.array([235.0, 232.0, 228.0], dtype=np.float32)

    # pen marks: short dark radial strokes on the lid
    for _ in range(pen_marks):
        theta = rng.uniform(0, 2 * math.pi)
        r0 = rng.uniform(0.2, 0.7) * radius
        length = rng.uniform(0.08, 0.18) * radius
        along = np.arange(r0, r0 + length)
        mx = np.clip((cx + along * math.cos(theta)).astype(int), 0, width - 1)
        my = np.clip((cy + along * math.sin(theta)).astype(int), 0, height - 1)
        image[my, mx] = STAGE + 4.0

    # the marker dot on the rim
    if dot_angle_deg is not None:
        theta = math.radians(dot_angle_deg)
        for rr in np.linspace(radius * 0.82, radius, 9):
            for tt in np.linspace(theta - 0.06, theta + 0.06, 7):
                px = round(cx + rr * math.cos(tt))
                py = round(cy + rr * math.sin(tt))
                if 0 <= px < width and 0 <= py < height:
                    image[py, px] = np.array([236.0, 190.0, 30.0], dtype=np.float32)

    truth = {"cx": cx, "cy": cy, "r": radius}
    return np.clip(image, 0, 255), truth


# --------------------------------------------------------------------------- #
# background / circular helpers
# --------------------------------------------------------------------------- #


def test_background_level_reads_the_dark_stage() -> None:
    lum = np.full((80, 100), 200.0)
    lum[10:70, 10:90] = 12.0                  # a bright object away from the border
    assert dg.background_level(lum) == pytest.approx(200.0, abs=1e-6)


def test_background_level_ignores_a_bright_interior() -> None:
    lum = np.full((80, 100), 18.0)
    lum[30:50, 40:60] = 255.0
    assert dg.background_level(lum) == pytest.approx(18.0, abs=1e-6)


def test_circular_mean_wraps_around_zero() -> None:
    # 350 and 10 degrees are 20 degrees apart, not 180
    assert dg.circular_mean_deg(np.array([350.0, 10.0])) == pytest.approx(0.0, abs=1e-6)
    assert dg.circular_mean_deg(np.array([90.0, 90.0])) == pytest.approx(90.0, abs=1e-6)


def test_circular_resultant_bounds() -> None:
    assert dg.circular_resultant(np.array([10.0, 11.0, 9.0])) == pytest.approx(1.0, abs=0.02)
    # a uniform spread has no preferred direction
    uniform = np.arange(0.0, 360.0, 5.0)
    assert dg.circular_resultant(uniform) < 0.1


def test_circular_mean_of_empty_is_nan() -> None:
    assert math.isnan(dg.circular_mean_deg(np.array([])))
    assert dg.circular_resultant(np.array([])) == 0.0


# --------------------------------------------------------------------------- #
# disc detection
# --------------------------------------------------------------------------- #


def test_find_disc_recovers_geometry_with_illumination_ramp() -> None:
    """The ramp is what makes an absolute threshold fail; check it still works."""
    image, truth = make_dish(ramp=True, seed=1)
    disc = dg.find_disc(image.mean(axis=2))
    assert disc.cx == pytest.approx(truth["cx"], abs=6)
    assert disc.cy == pytest.approx(truth["cy"], abs=6)
    assert disc.r == pytest.approx(truth["r"], abs=6)
    assert disc.edge < 0, "the rim edge should be a falloff, i.e. negative"


def test_find_disc_handles_an_offcentre_dish() -> None:
    image, truth = make_dish(cx=228.0, cy=214.0, seed=2)
    disc = dg.find_disc(image.mean(axis=2))
    assert disc.cx == pytest.approx(truth["cx"], abs=6)
    assert disc.cy == pytest.approx(truth["cy"], abs=6)
    assert disc.r == pytest.approx(truth["r"], abs=6)


def test_find_disc_is_unmoved_by_a_large_colony() -> None:
    """A colony covering much of the plate must not be mistaken for the rim."""
    image, truth = make_dish(colony_radius=0.72, seed=3)
    disc = dg.find_disc(image.mean(axis=2))
    assert disc.r == pytest.approx(truth["r"], abs=6)


def test_find_disc_survives_pen_marks() -> None:
    image, truth = make_dish(pen_marks=6, seed=4)
    disc = dg.find_disc(image.mean(axis=2))
    assert disc.r == pytest.approx(truth["r"], abs=6)


def test_pixels_per_mm_assumes_a_90_mm_dish() -> None:
    image, _ = make_dish(radius=145.0)
    disc = dg.find_disc(image.mean(axis=2))
    # 2r px across 90 mm
    assert disc.pixels_per_mm == pytest.approx(2 * disc.r / 90.0, rel=1e-9)


def test_r_over_w_is_a_stable_scale_check() -> None:
    image, _ = make_dish(radius=DISH_R)
    disc = dg.find_disc(image.mean(axis=2))
    assert disc.r_over_w == pytest.approx(disc.r / disc.width, rel=1e-12)


# --------------------------------------------------------------------------- #
# rim marker dots
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 180.0, 270.0, 330.0])
def test_find_rim_dots_recovers_the_angle(angle: float) -> None:
    image, _ = make_dish(dot_angle_deg=angle, seed=5)
    disc = dg.find_disc(image.mean(axis=2))
    dots = dg.find_rim_dots(image, disc)
    assert dots, f"no dot found for a dot drawn at {angle} deg"
    recovered = dots[0]["angle_deg"]
    # circular distance: 359 and 1 are 2 degrees apart, not 358
    delta = abs((recovered - angle + 180.0) % 360.0 - 180.0)
    assert delta < 4.0, f"expected ~{angle}, got {recovered}"


def test_find_rim_dots_places_the_dot_near_the_rim() -> None:
    image, _ = make_dish(dot_angle_deg=123.0, dot_rho=0.90, seed=6)
    disc = dg.find_disc(image.mean(axis=2))
    dot = dg.find_rim_dots(image, disc)[0]
    assert dot["rho"] == pytest.approx(0.90, abs=0.12)
    assert dot["concentration"] > 0.5, "a single dot should give a tight angle"
    assert dot["saturation"] > 45.0


def test_find_rim_dots_returns_nothing_when_there_is_none() -> None:
    image, _ = make_dish(seed=7)
    disc = dg.find_disc(image.mean(axis=2))
    assert dg.find_rim_dots(image, disc) == []


def test_find_rim_dots_is_not_fooled_by_a_bright_colony() -> None:
    """The colony is bright but neutral; only the dot is strongly yellow."""
    image, _ = make_dish(colony_radius=0.8, seed=8)
    disc = dg.find_disc(image.mean(axis=2))
    assert dg.find_rim_dots(image, disc) == []


def test_find_rim_dots_prefers_the_strongest_of_two() -> None:
    """Two dots at different strengths: the yellower one wins."""
    image, _ = make_dish(dot_angle_deg=200.0, seed=9)
    disc = dg.find_disc(image.mean(axis=2))
    # a second, much duller dot
    theta = math.radians(40.0)
    ys, xs = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    rr = np.hypot(xs - disc.cx, ys - disc.cy)
    near = np.exp(-((rr - disc.r * 0.90) ** 2) / (2 * (disc.r * 0.02) ** 2))
    wedge = np.cos(theta) * (xs - disc.cx) + np.sin(theta) * (ys - disc.cy)
    dull = (near * (wedge > disc.r * 0.85) * (wedge < disc.r * 0.95)) > 0.2
    image[dull] = image[dull] * 0.55 + np.array([120.0, 95.0, 55.0]) * 0.45
    assert dull.any(), "the dull dot was never drawn, so the test would be vacuous"
    dots = dg.find_rim_dots(image, disc)
    assert dots
    recovered = dots[0]["angle_deg"]
    delta = abs((recovered - 200.0 + 180.0) % 360.0 - 180.0)
    assert delta < 4.0


# --------------------------------------------------------------------------- #
# colony / inoculum measurements
# --------------------------------------------------------------------------- #


def test_colony_metrics_area_scales_with_the_colony() -> None:
    small, _ = make_dish(colony_radius=0.2, seed=10)
    large, _ = make_dish(colony_radius=0.6, seed=10)
    disc_s = dg.find_disc(small.mean(axis=2))
    disc_l = dg.find_disc(large.mean(axis=2))
    area_s = dg.colony_metrics(small, disc_s)["area_frac"]
    area_l = dg.colony_metrics(large, disc_l)["area_frac"]
    # a percentile threshold would report the same area for both
    assert area_l > area_s * 2.0
    assert 0.0 < area_s < area_l < 1.0


def test_colony_metrics_reports_the_inoculum_offset() -> None:
    image, _ = make_dish(colony_radius=0.25, colony_offset=(0.30, 0.0), seed=11)
    disc = dg.find_disc(image.mean(axis=2))
    metrics = dg.colony_metrics(image, disc)
    assert metrics["inoculum_offset"] == pytest.approx(0.30, abs=0.06)
    # offset is along +x, so the compass angle is 0.  Compare circularly: a
    # naive difference would call 358.6 deg "off by 358", not "off by 1.4".
    delta = abs((metrics["inoculum_angle_deg"] - 0.0 + 180.0) % 360.0 - 180.0)
    assert delta < 6.0


def test_colony_metrics_on_an_empty_dish() -> None:
    image, _ = make_dish(seed=12)
    disc = dg.find_disc(image.mean(axis=2))
    metrics = dg.colony_metrics(image, disc)
    assert metrics["area_frac"] == 0.0
    assert math.isnan(metrics["inoculum_angle_deg"])


def test_colony_metrics_radius_frac_matches_the_drawn_colony() -> None:
    image, _ = make_dish(colony_radius=0.5, seed=13)
    disc = dg.find_disc(image.mean(axis=2))
    metrics = dg.colony_metrics(image, disc)
    assert metrics["radius_frac"] == pytest.approx(0.5, abs=0.08)


# --------------------------------------------------------------------------- #
# signatures
# --------------------------------------------------------------------------- #


def test_detrend_removes_a_ramp_but_keeps_a_sharp_feature() -> None:
    angles = np.linspace(0, 2 * math.pi, 256, endpoint=False)
    # A boxcar of width w only attenuates harmonics well below the sample rate,
    # so test the first harmonic -- the broad illumination ramp this is meant to
    # kill.  (The 2nd harmonic would pass a 31-wide boxcar almost intact, which
    # is correct behaviour for a moving average, not a defect to hide.)
    ramp = 20.0 * np.sin(angles)
    sharp = np.zeros_like(angles)
    sharp[np.abs(angles - 1.0) < 0.08] = 50.0
    assert sharp.sum() == 50.0 * 7, "the marker should span about 7 samples"
    detrended = dg.detrend_circular(ramp + sharp, window=31)
    # Away from the marker, the smooth ramp is gone: subtracting a boxcar that
    # is wider than the feature removes the baseline but not the feature, so
    # there is no residual slope left behind.
    away = np.abs(angles - 1.0) > 0.5
    assert float(np.ptp(detrended[away])) < 2.0
    # The marker itself survives at most of its amplitude.  It is attenuated a
    # little because the moving average absorbs part of it; for a feature of
    # width w in a window of W the retained fraction is about 1 - w/W.
    assert 30.0 < float(detrended.max()) <= 50.0


def test_detrend_leaves_higher_harmonics_alone() -> None:
    """Documented limit: detrending is a low-pass subtraction, not a whitening.

    Worth pinning down because it sets what the rim signature can be used for.
    """
    angles = np.linspace(0, 2 * math.pi, 256, endpoint=False)
    high = 10.0 * np.cos(8 * angles)
    detrended = dg.detrend_circular(high, window=31)
    assert float(np.ptp(detrended)) > 8.0


def test_detrend_handles_a_short_signal() -> None:
    short = np.linspace(0, 1, 10)
    assert len(dg.detrend_circular(short, window=31)) == 10


def test_rim_signature_has_expected_width() -> None:
    image, _ = make_dish(dot_angle_deg=45.0, pen_marks=4, seed=14)
    disc = dg.find_disc(image.mean(axis=2))
    assert dg.rim_signature(image, disc, n_ang=180).shape == (360,)


def test_rim_signature_distinguishes_different_dot_angles() -> None:
    """Two dishes that differ only in where the dot sits must not match."""
    a, _ = make_dish(dot_angle_deg=10.0, seed=15)
    b, _ = make_dish(dot_angle_deg=190.0, seed=15)
    disc_a = dg.find_disc(a.mean(axis=2))
    disc_b = dg.find_disc(b.mean(axis=2))
    sig_a = dg.zscore(np.stack([dg.rim_signature(a, disc_a),
                                dg.rim_signature(a, disc_a)]))
    sig_b = dg.zscore(np.stack([dg.rim_signature(b, disc_b),
                                dg.rim_signature(b, disc_b)]))
    assert abs(float(sig_a[0] @ sig_b[0])) < 0.5


def test_rotation_invariant_signature_ignores_rotation() -> None:
    """Rotating the same dish must leave the rotation-invariant feature alone."""
    image, _ = make_dish(dot_angle_deg=30.0, seed=16)
    rotated = np.rot90(image, k=1).copy()
    disc_a = dg.find_disc(image.mean(axis=2))
    disc_b = dg.find_disc(rotated.mean(axis=2))
    sig_a = dg.rotation_invariant_signature(image, disc_a)
    sig_b = dg.rotation_invariant_signature(rotated, disc_b)
    correlation = float(sig_a @ sig_b / (np.linalg.norm(sig_a) * np.linalg.norm(sig_b)))
    assert correlation > 0.9


def test_rotation_sensitive_signature_does_change_under_rotation() -> None:
    image, _ = make_dish(dot_angle_deg=30.0, seed=17)
    rotated = np.rot90(image, k=1).copy()
    disc_a = dg.find_disc(image.mean(axis=2))
    disc_b = dg.find_disc(rotated.mean(axis=2))
    sig_a = dg.zscore(dg.rim_signature(image, disc_a).reshape(1, -1))
    sig_b = dg.zscore(dg.rim_signature(rotated, disc_b).reshape(1, -1))
    assert abs(float((sig_a @ sig_b.T)[0, 0])) < 0.5


# --------------------------------------------------------------------------- #
# similarity helpers
# --------------------------------------------------------------------------- #


def test_correlation_matrix_is_bounded_and_self_excluded() -> None:
    features = dg.zscore(np.random.default_rng(0).standard_normal((40, 12)))
    corr = dg.correlation_matrix(features)
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    assert off_diagonal.max() <= 1.0 + 1e-6
    assert off_diagonal.min() >= -1.0 - 1e-6
    assert np.all(np.isneginf(np.diag(corr))), "a row must not match itself"


def test_correlation_matrix_normalises_by_both_row_norms() -> None:
    """Regression guard: dividing by one broadcast norm yields values > 1."""
    features = np.array([[3.0, 4.0], [1.0, 0.0], [4.0, -3.0]])
    corr = dg.correlation_matrix(features)
    assert corr.max() <= 1.0 + 1e-6
    # rows 0 and 2 are perpendicular, so their correlation is exactly zero
    assert corr[0, 2] == pytest.approx(0.0, abs=1e-6)


def test_zscore_guards_constant_columns() -> None:
    constant = np.column_stack([np.arange(10.0), np.ones(10)])
    scaled = dg.zscore(constant)
    assert np.all(np.isfinite(scaled))
    assert np.allclose(scaled[:, 1], 0.0)


def test_gaussian_null_tracks_the_observed_correlation() -> None:
    """A shared covariance means the null must NOT collapse to zero.

    If the null came out near zero while the data were strongly similar, any
    apparent clustering would be an artefact of the comparison, not a signal.
    """
    rng = np.random.default_rng(3)
    common = rng.standard_normal((200, 1))
    features = common @ rng.standard_normal((1, 10)) + rng.standard_normal((200, 10)) * 0.2
    standard = dg.zscore(features)
    observed = dg.correlation_matrix(standard).max(axis=1)
    null = dg.gaussian_null(standard, seed=1)
    assert float(np.median(null)) > 0.2
    assert float(np.median(observed)) == pytest.approx(float(np.median(null)), abs=0.15)


def test_forward_step_fraction_separates_ordered_from_shuffled() -> None:
    assert dg.forward_step_fraction([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)
    assert dg.forward_step_fraction([4.0, 3.0, 2.0, 1.0]) == pytest.approx(0.0)
    assert dg.forward_step_fraction([1.0]) is None
    # ties count as forward: a constant signal is "not decreasing"
    assert dg.forward_step_fraction([2.0, 2.0, 2.0]) == pytest.approx(1.0)