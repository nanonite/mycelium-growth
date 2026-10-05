"""Tests for the calibration driver (Chainlink #25).

What is tested here is the plumbing between the mode detection and the published
tables, because that is where a run can produce a plausible-looking artefact
that is silently wrong: a failed rim fit getting to keep its own number, a
mixed block collapsing back into one constant, or a committed CSV that no
longer matches the rim fits it claims to summarise.
"""

from __future__ import annotations

import csv
from pathlib import Path

import calibrate as cl
import pytest

REPO = Path(__file__).resolve().parent.parent
TRACKS = REPO / "data" / "triage" / "dish_tracks.csv"
CALIBRATION = REPO / "data" / "triage" / "calibration.csv"
IMAGE_SCALE = REPO / "data" / "triage" / "image_scale.csv"

WORKING_WIDTH = 900


def make_row(index: int, archive: str, folder: str, r_over_w: float) -> dict[str, str]:
    """One dish_tracks-shaped row: the fit and the px/mm it implies."""
    radius_px = r_over_w * WORKING_WIDTH
    entry = f"{folder}/{index:08d}.jpg" if folder else f"{index:08d}.jpg"
    return {
        "id": f"{index:08d}",
        "archive": archive,
        "entry": entry,
        "dish_r_over_w": f"{r_over_w:.5f}",
        "pixels_per_mm": f"{2 * radius_px / 90.0:.4f}",
    }


def fixture_rows() -> list[dict[str, str]]:
    """Two blocks, one of them mixed, plus a failed fit in each."""
    gl = [
        make_row(i, "unlabeled-GL1.zip", "unlabeled_data1", 0.30222) for i in range(40)
    ]
    gl += [
        make_row(500 + i, "unlabeled-GL1.zip", "unlabeled_data1", 0.20)
        for i in range(3)
    ]
    flat = [
        make_row(1000 + i, "unlabeled-GS_PO_TS.zip", "", 0.33100) for i in range(30)
    ]
    flat += [
        make_row(1100 + i, "unlabeled-GS_PO_TS.zip", "", 0.22222) for i in range(8)
    ]
    flat += [
        make_row(1200 + i, "unlabeled-GS_PO_TS.zip", "", 0.19000) for i in range(2)
    ]
    return gl + flat


def rows_by_block(rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {(row["block"], row["scale_group"]): row for row in rows}


# --------------------------------------------------------------------------- #
# calibration table
# --------------------------------------------------------------------------- #


def test_a_mixed_block_is_calibrated_once_per_setup() -> None:
    calibration, _, _ = cl.calibrate(fixture_rows())

    by_key = rows_by_block(calibration)
    assert sorted({row["block"] for row in calibration}) == ["GL", "GS_PO_TS_flat"]
    assert len(calibration) == 3, "the flat block must split into two setups"

    flat_main = by_key[("GS_PO_TS_flat", 1)]
    flat_gs = by_key[("GS_PO_TS_flat", 2)]
    assert flat_main["dominant"] == 1
    assert flat_gs["dominant"] == 0
    assert flat_main["n_images"] == 32  # 30 setup images + 2 failed fits
    assert flat_gs["n_images"] == 8
    assert flat_main["px_per_mm"] == pytest.approx(6.62, abs=0.01)
    assert flat_gs["px_per_mm"] == pytest.approx(4.4444, abs=0.01)


def test_the_global_constant_error_is_reported_per_group() -> None:
    calibration, _, global_px_per_mm = cl.calibrate(fixture_rows())

    assert global_px_per_mm == pytest.approx(6.0444, abs=0.01)
    gs = rows_by_block(calibration)[("GS_PO_TS_flat", 2)]
    # one global constant would be wrong by ~36% for the GS-scale images
    assert gs["global_err_pct"] == pytest.approx(36.0, abs=1.0)
    gl = rows_by_block(calibration)[("GL", 1)]
    assert gl["global_err_pct"] == pytest.approx(0.0, abs=0.1)


def test_failed_fits_are_counted_against_the_group_they_fell_into() -> None:
    calibration, _, _ = cl.calibrate(fixture_rows())

    by_key = rows_by_block(calibration)
    assert by_key[("GL", 1)]["n_off_scale"] == 3
    assert by_key[("GS_PO_TS_flat", 1)]["n_off_scale"] == 2
    assert by_key[("GS_PO_TS_flat", 2)]["n_off_scale"] == 0
    # the three failed GL fits are members of the group, so the counts add up
    assert sum(row["n_images"] for row in calibration) == len(fixture_rows())


def test_calibration_never_reports_a_constant_below_the_split() -> None:
    """Every group's constant must come from images that agree with it."""
    calibration, image_rows, _ = cl.calibrate(fixture_rows())

    for row in calibration:
        members = [
            image
            for image in image_rows
            if (image["block"], image["scale_group"])
            == (row["block"], row["scale_group"])
        ]
        agreeing = [m for m in members if m["fit_status"] == "ok"]
        assert agreeing, f"group {row['block']}/{row['scale_group']} has no evidence"
        assert row["n_off_scale"] == len(members) - len(agreeing)


# --------------------------------------------------------------------------- #
# per-image table
# --------------------------------------------------------------------------- #


def test_a_failed_fit_gets_the_constant_not_its_own_number() -> None:
    _, image_rows, _ = cl.calibrate(fixture_rows())

    broken = next(row for row in image_rows if row["id"] == "00001200")
    assert broken["fit_status"] == "off_scale"
    assert broken["block"] == "GS_PO_TS_flat"
    assert broken["px_per_mm"] == pytest.approx(6.62, abs=0.01)  # the group's
    assert broken["px_per_mm_fit"] == pytest.approx(3.80, abs=0.01)  # its own
    assert broken["fit_rel_err_pct"] == pytest.approx(-42.6, abs=0.5)


def test_an_image_keeps_the_setup_it_was_photographed_at() -> None:
    _, image_rows, _ = cl.calibrate(fixture_rows())

    gs_scale = next(row for row in image_rows if row["id"] == "00001103")
    assert gs_scale["block"] == "GS_PO_TS_flat"
    assert gs_scale["scale_group"] == 2
    assert gs_scale["px_per_mm"] == pytest.approx(4.4444, abs=0.01)
    assert gs_scale["fit_status"] == "ok"


def test_read_tracks_counts_rows_without_a_fit(tmp_path: Path) -> None:
    path = tmp_path / "dish_tracks.csv"
    rows = fixture_rows() + [
        {
            "id": "00000999",
            "archive": "unlabeled-GL1.zip",
            "entry": "unlabeled_data1/00000999.jpg",
            "dish_r_over_w": "",
            "pixels_per_mm": "",
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    usable, skipped = cl.read_tracks(path)

    assert len(usable) == len(fixture_rows())
    assert skipped == 1


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #


def test_main_writes_both_tables(tmp_path: Path) -> None:
    tracks = tmp_path / "dish_tracks.csv"
    rows = fixture_rows()
    with tracks.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "calibration.csv"
    per_image = tmp_path / "image_scale.csv"

    code = cl.main(
        ["--tracks", str(tracks), "--out", str(out), "--per-image", str(per_image)]
    )

    assert code == 0
    assert out.exists() and per_image.exists()
    with out.open(encoding="utf-8") as handle:
        calibration = list(csv.DictReader(handle))
    assert [field for field in calibration[0]] == cl.CALIBRATION_FIELDS
    assert len(calibration) == 3
    with per_image.open(encoding="utf-8") as handle:
        images = list(csv.DictReader(handle))
    assert len(images) == len(rows)
    assert {row["fit_status"] for row in images} == {"ok", "off_scale"}


def test_main_refuses_a_missing_tracks_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cl.main(
            [
                "--tracks",
                str(tmp_path / "nope.csv"),
                "--out",
                str(tmp_path / "out.csv"),
                "--per-image",
                str(tmp_path / "per.csv"),
            ]
        )
    assert excinfo.value.code == 2


# --------------------------------------------------------------------------- #
# the published tables
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not TRACKS.exists(), reason="dish_tracks.csv has not been run")
def test_committed_tables_reproduce_from_the_rim_fits(tmp_path: Path) -> None:
    """The published calibration must be exactly what the fits say today.

    Byte-for-byte: the tables are derived data, so a re-run of this script on
    the same rim fits has to be indistinguishable from what is committed, or
    nobody can tell which of the two a reader is looking at.
    """
    rows, _ = cl.read_tracks(TRACKS)
    calibration, images, _ = cl.calibrate(rows)

    written = tmp_path / "calibration.csv"
    per_image = tmp_path / "image_scale.csv"
    cl.write_csv(written, calibration, cl.CALIBRATION_FIELDS)
    cl.write_csv(per_image, images, cl.IMAGE_FIELDS)

    assert written.read_text(encoding="utf-8") == CALIBRATION.read_text(
        encoding="utf-8"
    )
    assert per_image.read_text(encoding="utf-8") == IMAGE_SCALE.read_text(
        encoding="utf-8"
    )

    # the premise of the issue: one global constant is badly wrong somewhere
    worst = max(abs(row["global_err_pct"]) for row in calibration)
    assert worst >= 30.0
