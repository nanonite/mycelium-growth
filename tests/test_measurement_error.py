"""Unit tests for the measurement-error model (Chainlink #31).

The mask half needs only numpy + PIL, so what is tested here is the part that
can silently produce a *plausible but wrong* number: the feature arithmetic on
known shapes, the convex hull, the candidate-protrusion detector, the
bias/SD error model and the #60 target check.  An integration test runs the
full driver on the real #29 masks when they are present (skipped otherwise).

The real-data run is recorded in notes/measurement_error.md.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import measurement_error as me  # needs the sys.path line above

# --- synthetic masks ---------------------------------------------------------


def _filled_rectangle(w: int, h: int) -> np.ndarray:
    return np.ones((h, w), dtype=np.uint8)


def _filled_circle(r: int, *, size: int | None = None) -> np.ndarray:
    n = size or (2 * r + 1)
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    return ((xx - c) ** 2 + (yy - c) ** 2 <= r**2).astype(np.uint8)


def _circle_with_bump(r: int = 50, *, bump: int = 20, size: int = 220) -> np.ndarray:
    """A circle with a rectangular protrusion on the +x (angle-0) side."""
    mask = _filled_circle(r, size=size)
    c = (size - 1) / 2.0
    x0 = int(c + r) - 2
    x1 = x0 + bump
    y0 = int(c - 5)
    y1 = int(c + 5)
    mask[y0:y1, x0:x1] = 1
    return mask


# --- feature computation ----------------------------------------------------


def test_rectangle_features_match_the_geometric_values():
    w, h = 100, 50
    feats = me.mask_features(_filled_rectangle(w, h), px_per_mm=None, dish_r_900=None, fit_ok=False)
    assert feats["area_px"] == float(w * h)
    # 4-connected boundary of a filled rectangle: the outer ring.
    assert feats["perimeter_px"] == float(2 * w + 2 * h - 4)
    assert feats["equiv_radius_px"] == pytest.approx(math.sqrt(w * h / math.pi))
    assert feats["solidity"] == pytest.approx(1.0)  # a rectangle is convex
    expected_circ = 4.0 * math.pi * w * h / (2 * w + 2 * h - 4) ** 2
    assert feats["circularity"] == pytest.approx(expected_circ)
    expected_rough = (2 * w + 2 * h - 4) / (2.0 * math.pi * math.sqrt(w * h / math.pi))
    assert feats["edge_roughness"] == pytest.approx(expected_rough)


def test_circle_features_are_close_to_the_continuous_values():
    r = 50
    feats = me.mask_features(_filled_circle(r), px_per_mm=None, dish_r_900=None, fit_ok=False)
    # A digital circle of radius r has area ~ pi r^2.  Its 4-connected
    # boundary is ~0.89 x 2 pi r (the discrete staircase undercounts the
    # circumference by ~11%), and the region hull area is ~ pi r^2, so the
    # solidity of a convex disk is 1.
    assert feats["area_px"] == pytest.approx(math.pi * r**2, rel=0.02)
    assert feats["perimeter_px"] == pytest.approx(2.0 * math.pi * r, rel=0.15)
    assert feats["equiv_radius_px"] == pytest.approx(r, rel=0.02)
    assert feats["solidity"] == pytest.approx(1.0, abs=0.02)
    # edge_roughness = P / (2 pi r_eq); for a disk P ~ 0.89 x 2 pi r.
    assert feats["edge_roughness"] == pytest.approx(0.89, rel=0.1)


def test_calibrated_features_need_fit_ok_and_constants():
    mask = _filled_rectangle(100, 50)
    # No calibration without fit_ok, even if the constants are present.
    feats = me.mask_features(mask, px_per_mm=6.0, dish_r_900=270.0, fit_ok=False)
    assert "area_mm2" not in feats
    assert "area_fraction" not in feats
    # With fit_ok the calibrated features appear.
    feats = me.mask_features(mask, px_per_mm=6.0, dish_r_900=270.0, fit_ok=True)
    assert feats["area_mm2"] == pytest.approx(5000.0 / 6.0**2)
    assert feats["equiv_radius_mm"] == pytest.approx(math.sqrt(5000.0 / math.pi) / 6.0)
    assert feats["perimeter_mm"] == pytest.approx(296.0 / 6.0)
    assert feats["area_fraction"] == pytest.approx(5000.0 / (math.pi * 270.0**2))
    assert feats["log_area_mm2"] == pytest.approx(math.log(5000.0 / 6.0**2))


def test_empty_mask_is_flagged():
    feats = me.mask_features(np.zeros((10, 10), dtype=np.uint8), px_per_mm=None, dish_r_900=None, fit_ok=False)
    assert feats["empty"] == 1.0
    assert feats["area_px"] == 0.0


# --- convex hull -------------------------------------------------------------


def test_convex_hull_of_a_square_with_an_interior_point():
    pts = np.array(
        [[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [3, 7]], dtype=float
    )
    hull = me.convex_hull(pts)
    assert set(hull) == {(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)}
    assert me.polygon_area(hull) == pytest.approx(100.0)


def test_convex_hull_of_a_degenerate_set():
    assert me.convex_hull(np.array([[1.0, 2.0]])) == [(1.0, 2.0)]
    assert me.convex_hull(np.array([[0.0, 0.0], [1.0, 1.0]])) == [(0.0, 0.0), (1.0, 1.0)]


# --- candidate protrusions ---------------------------------------------------


def test_plain_circle_has_no_candidate_protrusions():
    assert me.radial_protrusions(_filled_circle(50)) == []


def test_circle_with_a_bump_has_one_candidate_protrusion():
    props = me.radial_protrusions(_circle_with_bump())
    assert len(props) == 1
    # The bump is on the +x side, i.e. at angle 0.
    assert abs(props[0]["peak_angle"]) < 10.0
    assert props[0]["prominence"] > me.PROTRUSION_ABS_THRESH


def test_protrusion_matching_counts_true_and_false_events():
    ref = [{"peak_angle": 0.0}, {"peak_angle": 90.0}]
    pred = [{"peak_angle": 2.0}, {"peak_angle": 45.0}, {"peak_angle": 91.0}]
    match = me.match_protrusions(pred, ref, match_deg=15.0)
    assert match["tp"] == 2  # 0->0 and 90->91
    assert match["fp"] == 1  # the 45-degree excursion is invented
    assert match["fn"] == 0
    assert match["false_event_rate"] == pytest.approx(1.0 / 3.0)


def test_protrusion_matching_is_one_to_one():
    ref = [{"peak_angle": 0.0}]
    pred = [{"peak_angle": 1.0}, {"peak_angle": 2.0}]
    match = me.match_protrusions(pred, ref, match_deg=15.0)
    assert match["tp"] == 1
    assert match["fp"] == 1


# --- error model -------------------------------------------------------------


def test_error_values_relative_and_log():
    pairs = [(110.0, 100.0), (90.0, 100.0)]
    rel = me.error_values(pairs, mode="relative")
    assert rel == pytest.approx([0.1, -0.1])
    log = me.error_values(pairs, mode="log")
    assert log == pytest.approx([math.log(1.1), math.log(0.9)])
    # A zero reference is skipped for the relative error.
    assert me.error_values([(5.0, 0.0)], mode="relative") == []


def test_summarise_errors_reports_bias_sd_and_target():
    # A segmenter that is exactly 10% high on every image: bias 10%, SD 0%.
    pairs = [(1.1, 1.0)] * 20
    stats = me.summarise_errors(pairs, mode="relative", kind="size")
    assert stats["bias_pct"] == pytest.approx(10.0)
    assert stats["sd_pct"] == pytest.approx(0.0, abs=1e-9)
    # Size target is bias <= 3%, SD <= 10%: the bias fails.
    assert stats["target"]["meets"] is False


def test_summarise_errors_clears_the_target_when_both_clear():
    # Errors centred on 0 with a small spread.
    rng = np.random.default_rng(7)
    errs = rng.normal(0.0, 0.02, size=500)
    pairs = [(1.0 + e, 1.0) for e in errs]
    stats = me.summarise_errors(pairs, mode="relative", kind="size")
    assert stats["target"]["meets"] is True


def test_summarise_errors_with_too_few_pairs():
    stats = me.summarise_errors([(1.0, 1.0)], mode="relative", kind="size")
    assert stats["n"] == 1
    assert stats["bias"] is None
    assert stats["target"] is None


# --- mask loading ------------------------------------------------------------


def test_load_mask_working_resizes_to_900_width(tmp_path: Path):
    from PIL import Image

    big = np.zeros((675, 4608), dtype=np.uint8)
    big[100:200, 100:200] = 1
    path = tmp_path / "mask.png"
    Image.fromarray(big * 255, mode="L").save(path)
    mask = me.load_mask_working(path)
    # The height scales with the width: 675 * 900 / 4608 ~ 132.
    assert mask.shape == (round(675 * 900 / 4608), 900)
    assert mask.sum() > 0


# --- integration on the real #29 masks (skipped when absent) -----------------

WORK = REPO / "work_dirs" / "segformer-b0-repro" / "inference"
REAL_DATA = (WORK / "records.json").is_file() and (WORK / "model_meta.json").is_file()


@pytest.mark.skipif(not REAL_DATA, reason="real #29 inference outputs not present")
def test_analyze_on_real_holdouts_produces_finite_error_model():
    records = json.loads((WORK / "records.json").read_text())
    result = me.analyze(records, WORK)
    evidence = result["evidence"]
    pop = evidence["population"]
    # 110 hold-out, 90 representatives, 88 calibrated (the #29 structure).
    assert pop["n_holdout"] == 110
    assert pop["n_representatives"] == 90
    assert pop["n_calibrated"] == 88
    # Every feature has a finite bias and SD on the de-duplicated population.
    for feature, stats in evidence["features"].items():
        if stats["n"] >= 2:
            assert math.isfinite(stats["bias_pct"]), feature
            assert math.isfinite(stats["sd_pct"]), feature
    # The size features that clear the target are the radius/log-area family.
    assert evidence["features"]["equiv_radius_mm"]["target"]["meets"] is True
    assert evidence["features"]["log_area_mm2"]["target"]["meets"] is True
    # Repeatability is honestly reported as not estimable.
    assert evidence["repeatability"]["estimable_on_this_release"] is False
    # The protrusion false-event rate is high and robust across the grid.
    assert evidence["protrusions"]["false_event_rate"] > 0.5
    rates = [c["false_event_rate"] for c in evidence["protrusion_sensitivity"]]
    assert min(rates) > 0.5


@pytest.mark.skipif(not REAL_DATA, reason="real #29 inference outputs not present")
def test_real_error_model_writes_auditable_outputs(tmp_path: Path):
    records = json.loads((WORK / "records.json").read_text())
    result = me.analyze(records, WORK)
    csv_path = tmp_path / "measurement_error.csv"
    json_path = tmp_path / "evidence.json"
    me.write_error_csv(csv_path, result["rows"])
    me.write_evidence_json(json_path, result["evidence"])
    assert csv_path.is_file()
    assert json_path.is_file()
    # The CSV carries one row per hold-out image with the provenance columns.
    import csv

    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 110
    assert {"id", "source_set", "duplicate_role", "pred_area_px", "ref_area_px"} <= set(rows[0])
