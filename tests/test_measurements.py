"""Unit tests for the observable measurement table (Chainlink #30).

The mask half needs only numpy + PIL, so what is tested here is the part that
can silently produce a *plausible but wrong* number: the intensity summaries
on a known image, the off_scale exclusion of calibrated features, the
forbidden-inference guard, and the CSV's column contract.  An integration
test runs the full driver on the real #29 masks when they are present
(skipped otherwise).

The real-data run is recorded in notes/measurements.md.
"""

from __future__ import annotations

import csv
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

import measurements as mm  # needs the sys.path line above

# --- synthetic images -------------------------------------------------------


def _checkerboard(size: int = 200) -> np.ndarray:
    """A deterministic 8-bit image with a known intensity distribution."""
    yy, xx = np.mgrid[0:size, 0:size]
    return (((xx // 10) + (yy // 10)) % 2 * 255).astype(np.uint8)


def _disc_mask(r: int = 60, *, size: int = 200) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    return ((xx - c) ** 2 + (yy - c) ** 2 <= r**2).astype(np.uint8)


# --- intensity summaries ----------------------------------------------------


def test_intensity_summaries_match_the_pixel_values():
    image = _checkerboard()
    mask = _disc_mask()
    summaries = mm.intensity_summaries(image, mask)
    pixels = image[mask > 0]
    assert summaries["mean_intensity"] == pytest.approx(float(pixels.mean()))
    assert summaries["intensity_median"] == pytest.approx(float(np.median(pixels)))
    assert summaries["intensity_std"] == pytest.approx(float(pixels.std()))
    assert summaries["intensity_min"] == float(pixels.min())
    assert summaries["intensity_max"] == float(pixels.max())


def test_intensity_summaries_empty_mask_is_none_not_zero():
    image = _checkerboard()
    mask = np.zeros_like(image)
    summaries = mm.intensity_summaries(image, mask)
    # An empty mask has no pixels: the summaries must be None (an empty cell),
    # never a 0.0 that would read as a real measurement of a black colony.
    assert summaries["mean_intensity"] is None
    assert summaries["intensity_median"] is None
    assert summaries["intensity_std"] is None
    assert summaries["intensity_min"] is None
    assert summaries["intensity_max"] is None


def test_intensity_summaries_shape_mismatch_raises():
    with pytest.raises(mm.MeasurementError):
        mm.intensity_summaries(np.zeros((10, 10)), np.zeros((20, 20)))


def test_load_image_working_resizes_to_the_working_width(tmp_path):
    from PIL import Image

    # A 4608x3456 image (the released MyceliumSeg size) resized to 900 px.
    path = tmp_path / "big.jpg"
    Image.fromarray(np.zeros((3456, 4608), dtype=np.uint8)).save(path)
    out = mm.load_image_working(path)
    assert out.shape == (675, 900)


# --- row assembly -----------------------------------------------------------


def _record(**overrides):
    """A minimal #29-shaped record for the row-assembly tests."""
    base = {
        "id": "00000001",
        "archive": "labeled-GL.zip",
        "source_set": "test",
        "block": "GL",
        "species": "GL",
        "medium": "MYG",
        "temp": "25C",
        "condition": "GL/MYG/25C",
        "condition_known": True,
        "condition_source": "folder:labeled-GL + paper Table 1",
        "label_conflict": False,
        "md5": "abc123",
        "duplicate_role": "representative",
        "duplicate_group_size": 1,
        "duplicate_representative": "00000001",
        "fit_status": "ok",
        "px_per_mm": 6.0444,
        "dish_disc_radius_900": 272.0,
        "frame_order_status": "not_demonstrated",
        "time_unit_status": "no_capture_interval",
        "selection": "labeled_holdout",
        "selection_rule": "paper Table 6 hold-out",
        "estimands_identifiable": "image_morphology",
        "estimands_blocked": "delta_per_ordered_step;physical_rate;intervention_effect",
        "image_path": "",
        "mask_path": "",
        "mask_sha256": "",
        "width": 900,
        "height": 675,
        "mask_width": 4608,
        "mask_height": 3456,
    }
    base.update(overrides)
    return base


def test_build_measurement_rows_carries_provenance_and_measurements(tmp_path):
    from PIL import Image

    # A 900x675 image and a matching mask, so no resize is needed.
    image = np.zeros((675, 900), dtype=np.uint8)
    image[100:400, 100:400] = 200
    mask = np.zeros((675, 900), dtype=np.uint8)
    mask[100:400, 100:400] = 1
    image_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask * 255).save(mask_path)

    rec = _record(image_path=str(image_path), mask_path=str(mask_path))
    rows = mm.build_measurement_rows([rec])
    assert len(rows) == 1
    row = rows[0]
    # provenance carried verbatim
    assert row["id"] == "00000001"
    assert row["archive"] == "labeled-GL.zip"
    assert row["frame_order_status"] == "not_demonstrated"
    assert row["time_unit_status"] == "no_capture_interval"
    assert row["working_width"] == mm.WORKING_WIDTH
    # primary size features present and correct
    assert row["area_px"] == pytest.approx(300 * 300)
    assert row["area_mm2"] == pytest.approx(300 * 300 / 6.0444**2)
    assert row["equiv_radius_px"] == pytest.approx(math.sqrt(300 * 300 / math.pi))
    assert row["log_area_px2"] == pytest.approx(math.log(300 * 300))
    # boundary features present
    assert row["perimeter_px"] == pytest.approx(4 * 300 - 4)
    assert row["edge_roughness"] == pytest.approx((4 * 300 - 4) / (2 * math.pi * math.sqrt(300 * 300 / math.pi)))
    # intensity summaries present
    assert row["mean_intensity"] == pytest.approx(200.0)
    assert row["intensity_std"] == pytest.approx(0.0)


def test_build_measurement_rows_off_scale_lacks_calibrated_features(tmp_path):
    from PIL import Image

    image = np.zeros((675, 900), dtype=np.uint8)
    image[100:400, 100:400] = 200
    mask = np.zeros((675, 900), dtype=np.uint8)
    mask[100:400, 100:400] = 1
    image_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask * 255).save(mask_path)

    rec = _record(
        image_path=str(image_path),
        mask_path=str(mask_path),
        fit_status="off_scale",
    )
    rows = mm.build_measurement_rows([rec])
    row = rows[0]
    # calibrated features excluded per the #60 inclusion rule
    assert row["area_mm2"] is None
    assert row["area_fraction"] is None
    assert row["equiv_radius_mm"] is None
    assert row["perimeter_mm"] is None
    assert row["log_area_mm2"] is None
    # px^2 and dimensionless shape features kept
    assert row["area_px"] == pytest.approx(300 * 300)
    assert row["equiv_radius_px"] == pytest.approx(math.sqrt(300 * 300 / math.pi))
    assert row["perimeter_px"] == pytest.approx(4 * 300 - 4)
    assert row["edge_roughness"] is not None
    assert row["solidity"] is not None


def test_build_measurement_rows_missing_mask_raises(tmp_path):
    rec = _record(image_path=str(tmp_path / "img.jpg"), mask_path=str(tmp_path / "nope.png"))
    with pytest.raises(mm.MeasurementError):
        mm.build_measurement_rows([rec])


# --- guards -----------------------------------------------------------------


def test_assert_no_inferred_time_rejects_forbidden_fields():
    bad = [{"id": "x", "growth_rate": 1.0, "mm_per_day": 2.0}]
    with pytest.raises(mm.MeasurementError):
        mm._assert_no_inferred_time(bad)


def test_assert_no_inferred_time_allows_provenance_fields():
    # frame_order_status and time_unit_status are provenance facts, not
    # inferences, so they must pass the guard.
    good = [{"id": "x", "frame_order_status": "not_demonstrated", "time_unit_status": "no_capture_interval"}]
    mm._assert_no_inferred_time(good)  # must not raise


# --- CSV contract -----------------------------------------------------------


def test_csv_header_is_provenance_then_measurements():
    header = list(mm.CSV_HEADER)
    provenance = list(mm.PROVENANCE_COLUMNS)
    measurement_cols = [col for col, _, _ in mm.MEASUREMENT_COLUMNS]
    assert header == provenance + measurement_cols


def test_measurement_columns_have_a_tier_and_a_note():
    for col, tier, note in mm.MEASUREMENT_COLUMNS:
        assert tier in ("primary", "secondary")
        assert note


def test_primary_features_are_the_size_family():
    primary = {col for col, tier, _ in mm.MEASUREMENT_COLUMNS if tier == "primary"}
    assert primary == {
        "area_px",
        "area_mm2",
        "area_fraction",
        "equiv_radius_px",
        "equiv_radius_mm",
        "log_area_px2",
        "log_area_mm2",
    }


def test_secondary_features_are_boundary_and_intensity():
    secondary = {col for col, tier, _ in mm.MEASUREMENT_COLUMNS if tier == "secondary"}
    assert secondary == {
        "perimeter_px",
        "perimeter_mm",
        "edge_roughness",
        "circularity",
        "solidity",
        "mean_intensity",
        "intensity_median",
        "intensity_std",
        "intensity_min",
        "intensity_max",
    }


# --- integration on the real #29 masks --------------------------------------


def _real_work() -> Path:
    return REPO / "work_dirs" / "segformer-b0-repro" / "inference"


@pytest.mark.skipif(
    not (_real_work() / "records.json").is_file(),
    reason="#29 inference outputs not present",
)
def test_real_run_writes_one_row_per_image(tmp_path):
    work = _real_work()
    records = json.loads((work / "records.json").read_text())
    rows = mm.build_measurement_rows(records)
    assert len(rows) == len(records) == 125
    # every row keeps the Gate C provenance
    for row in rows:
        assert row["frame_order_status"] == "not_demonstrated"
        assert row["time_unit_status"] == "no_capture_interval"
    # the three off_scale images lack calibrated features, the rest have them
    off_scale = {r["id"] for r in rows if r["fit_status"] != "ok"}
    assert off_scale == {"00018438", "00018456", "00018458"}
    for row in rows:
        if row["id"] in off_scale:
            assert row["area_mm2"] is None
        else:
            assert row["area_mm2"] is not None
    # the primary size features are present on every row
    for row in rows:
        assert row["area_px"] is not None
        assert row["equiv_radius_px"] is not None
        assert row["log_area_px2"] is not None
    # the mask-level content hash is carried from the #29 record's
    # `predicted_mask_sha256` and joins on `id`
    rec_by_id = {str(r["id"]): r for r in records}
    for row in rows:
        assert row["mask_sha256"], f"row {row['id']} has an empty mask_sha256"
        assert row["mask_sha256"] == rec_by_id[row["id"]]["predicted_mask_sha256"]


@pytest.mark.skipif(
    not (_real_work() / "records.json").is_file(),
    reason="#29 inference outputs not present",
)
def test_real_run_values_match_the_error_model(tmp_path):
    """The primary features must agree with #31's per-image table."""
    work = _real_work()
    records = json.loads((work / "records.json").read_text())
    rows = mm.build_measurement_rows(records)
    by_id = {r["id"]: r for r in rows}

    with (REPO / "data" / "triage" / "measurement_error.csv").open() as handle:
        err_rows = {r["id"]: r for r in csv.DictReader(handle)}

    # a few spot-checked images: the measurement table and the error model
    # must report the same predicted-mask features
    for image_id in ("00000458", "00000459", "00018428"):
        assert by_id[image_id]["area_px"] == pytest.approx(float(err_rows[image_id]["pred_area_px"]))
        assert by_id[image_id]["area_mm2"] == pytest.approx(float(err_rows[image_id]["pred_area_mm2"]))
        assert by_id[image_id]["equiv_radius_mm"] == pytest.approx(float(err_rows[image_id]["pred_equiv_radius_mm"]))
        assert by_id[image_id]["perimeter_mm"] == pytest.approx(float(err_rows[image_id]["pred_perimeter_mm"]))
