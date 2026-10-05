"""Unit tests for the px -> mm calibration primitives (Chainlink #25).

The calibration is only worth publishing if the two failure modes it exists to
handle are actually handled, so the fixtures below reproduce them from the
release rather than from clean data:

* **several camera scales** -- the same 90-mm dish is fitted at r/W = 0.302,
  0.331 or 0.222 depending on the condition block, and one block
  (``unlabeled-GS_PO_TS``) mixes two of them;
* **failed rim fits** -- ``find_disc`` locks onto a colony edge or an inner
  reflection and reports a radius tens of percent too small, which must not
  move the constant and must not be mistaken for a fourth camera setup.

A suite built only from tight, single-scale fixtures would pass against an
implementation that reports one global constant and calls it calibrated.
"""

from __future__ import annotations

import math

import dish_scale as ds
import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# condition blocks
# --------------------------------------------------------------------------- #

BLOCK_CASES = [
    ("labeled-GL.zip", "labeled-GL/trainset/image/00000001.jpg", "GL"),
    ("unlabeled-GL4.zip", "unlabeled_data4/00009715.jpg", "GL"),
    ("labeled-GS_PO_TS.zip", "labeled-GS_PO_TS/GS/image/00018428.jpg", "GS"),
    ("labeled-GS_PO_TS.zip", "labeled-GS_PO_TS/PO/image/00018441.jpg", "PO"),
    ("labeled-GS_PO_TS.zip", "labeled-GS_PO_TS/TS/image/00018450.jpg", "TS"),
    ("labeled-MYG_PDA_TEMP.zip", "MYG/image/00018458.jpg", "MYG"),
    ("labeled-MYG_PDA_TEMP.zip", "PDA/image/00018468.jpg", "PDA"),
    ("labeled-MYG_PDA_TEMP.zip", "TEMP15/image/00018478.jpg", "TEMP15"),
    ("unlabeled-GS_PO_TS.zip", "00018488.jpg", "GS_PO_TS_flat"),
]


@pytest.mark.parametrize("archive,entry,expected", BLOCK_CASES)
def test_condition_block_maps_every_archive_shape(
    archive: str, entry: str, expected: str
) -> None:
    assert ds.condition_block(archive, entry) == expected


def test_condition_block_refuses_an_unknown_archive() -> None:
    """Silently filing an image under the wrong block corrupts its scale."""
    with pytest.raises(ValueError, match="no condition block"):
        ds.condition_block("mystery.zip", "a/b.jpg")


def test_condition_block_refuses_a_path_without_a_condition_folder() -> None:
    with pytest.raises(ValueError, match="GS/PO/TS"):
        ds.condition_block("labeled-GS_PO_TS.zip", "labeled-GS_PO_TS/image/1.jpg")


# --------------------------------------------------------------------------- #
# scale modes
# --------------------------------------------------------------------------- #


def test_one_setup_plus_a_failure_smear_stays_one_mode() -> None:
    """GL: 18k images at r/W 0.302 and a smear of failed fits below it."""
    rng = np.random.default_rng(0)
    setup = rng.normal(0.3022, 0.0004, 4000)
    smear = rng.uniform(0.17, 0.29, 400)
    values = np.concatenate([setup, smear])

    modes = ds.scale_modes(values)

    assert len(modes) == 1, "a diffuse smear is not a second camera setup"
    assert modes[0].r_over_w == pytest.approx(0.3022, abs=0.001)
    # the smear is assigned to the setup rather than promoted to one of its own,
    # and every member further than the tolerance away is flagged, not trusted
    groups = ds.assign_scales(smear, modes)
    assert {group for group, _ in groups} == {0}
    expected_off = sum(
        1 for value in smear if abs(value / modes[0].r_over_w - 1) > ds.FIT_TOLERANCE
    )
    assert (
        sum(1 for _, status in groups if status == ds.STATUS_OFF_SCALE) == expected_off
    )
    assert expected_off > 300, "the fixture would be vacuous without a real tail"


def test_a_smear_cannot_become_a_mode_but_a_tight_cluster_can() -> None:
    """The flat shard: one setup, a second setup, and failed fits in between."""
    rng = np.random.default_rng(1)
    values = np.concatenate(
        [
            rng.normal(0.3310, 0.0035, 1400),  # PO/TS-scale frames
            rng.normal(0.2222, 0.0025, 90),  # GS-scale frames (verified by eye)
            rng.uniform(0.188, 0.211, 40),  # failed fits, smeared
            rng.uniform(0.273, 0.315, 60),  # inner-ring failures, smeared
        ]
    )

    modes = ds.scale_modes(values)

    assert [mode.n_cluster for mode in modes] == sorted(
        (mode.n_cluster for mode in modes), reverse=True
    ), "modes must come back dominant first"
    assert len(modes) == 2
    assert modes[0].r_over_w == pytest.approx(0.3310, abs=0.005)
    assert modes[1].r_over_w == pytest.approx(0.2222, abs=0.005)


def test_a_tight_failure_cluster_is_absorbed_into_the_block_mode() -> None:
    """Inner-ring failures cluster tightly, ~20% under the real rim.

    Left alone they would be published as a second "setup" with a constant that
    is wrong for the images behind it, so the separation floor merges them into
    the block's mode whenever the union is still tight.
    """
    rng = np.random.default_rng(2)
    values = np.concatenate(
        [rng.normal(0.331, 0.003, 800), rng.normal(0.275, 0.003, 40)]
    )

    modes = ds.scale_modes(values)

    assert len(modes) == 1
    assert modes[0].r_over_w == pytest.approx(0.331, abs=0.005)
    # both populations end up under the one constant; only points too isolated
    # to be dense (the tail of the failure cluster) stay outside it
    assert modes[0].n_cluster >= 830


def test_two_tight_populations_keep_their_own_constants() -> None:
    """The separation floor allows a merge; it never forces one.

    Merging populations whose union is wide would park the constant in the gap
    between them, where no image lives.  The release has no such block -- its
    setups sit 49% apart -- so this is the guard rail, not a case in the data.
    """
    rng = np.random.default_rng(3)
    values = np.concatenate(
        [rng.normal(0.302, 0.001, 60), rng.normal(0.325, 0.001, 60)]
    )

    modes = ds.scale_modes(values)

    assert len(modes) == 2
    assert sorted(mode.r_over_w for mode in modes) == pytest.approx(
        [0.302, 0.325], abs=0.002
    )


def test_a_ten_image_block_yields_the_one_setup_it_has() -> None:
    """Every labeled condition block is ten images; one of them is a failure."""
    values = [
        0.32778,
        0.33471,
        0.32444,
        0.32783,
        0.33254,
        0.32667,
        0.33444,
        0.33333,
        0.27889,
        0.33302,
    ]

    modes = ds.block_modes(values)

    assert len(modes) == 1
    assert modes[0].r_over_w == pytest.approx(0.330, abs=0.004)


def test_a_block_too_small_to_cluster_falls_back_to_its_median() -> None:
    values = [0.20, 0.24, 0.28, 0.32]

    assert ds.scale_modes(values) == []
    modes = ds.block_modes(values)
    assert len(modes) == 1
    assert modes[0].r_over_w == pytest.approx(0.26)
    assert modes[0].n_cluster == 4


def test_empty_input_is_not_a_mode() -> None:
    assert ds.scale_modes([]) == []
    assert ds.block_modes([]) == []


# --------------------------------------------------------------------------- #
# assignment
# --------------------------------------------------------------------------- #


def test_assignment_takes_the_setup_it_agrees_with() -> None:
    modes = [ds.ScaleMode(0.331, 1500), ds.ScaleMode(0.222, 90)]

    groups = ds.assign_scales([0.332, 0.2215, 0.19], modes)

    assert groups == [(0, ds.STATUS_OK), (1, ds.STATUS_OK), (0, ds.STATUS_OFF_SCALE)]


def test_a_failed_fit_takes_the_dominant_setup_not_the_nearest_one() -> None:
    """A fit 40% off the mark carries no evidence about which setup it came from.

    The prior is the block's main setup: 1,598 of 1,689 flat-shard images are
    PO/TS-scale, so 0.19 belongs to that group unless something says otherwise.
    """
    modes = [ds.ScaleMode(0.331, 1598), ds.ScaleMode(0.222, 91)]

    group, status = ds.assign_scales([0.19], modes)[0]

    assert (group, status) == (0, ds.STATUS_OFF_SCALE)


def test_assignment_without_a_mode_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one mode"):
        ds.assign_scales([0.3], [])


# --------------------------------------------------------------------------- #
# summaries and conversion
# --------------------------------------------------------------------------- #


def test_scale_summary_is_median_based() -> None:
    """One pathological fit must not be able to move the constant."""
    summary = ds.scale_summary([0.302] * 9 + [0.05], [6.044] * 9 + [1.11])

    assert summary.n == 10
    assert summary.r_over_w_median == pytest.approx(0.302)
    assert summary.px_per_mm == pytest.approx(6.044)
    assert summary.r_over_w_iqr == pytest.approx(0.0)
    assert summary.spread_pct == pytest.approx(0.0)


def test_scale_summary_rejects_mismatched_or_empty_input() -> None:
    with pytest.raises(ValueError):
        ds.scale_summary([0.3, 0.4], [6.0])
    with pytest.raises(ValueError):
        ds.scale_summary([], [])


def test_pixels_per_mm_comes_from_the_90_mm_rim() -> None:
    # the fitted rim of the GL blocks is ~272 px across on the working image
    assert ds.rim_pixels_per_mm(272.0) == pytest.approx(6.0444, abs=1e-4)
    with pytest.raises(ValueError):
        ds.rim_pixels_per_mm(0.0)


def test_px_to_mm_is_the_calibration_applied_in_both_directions() -> None:
    assert ds.px_to_mm(6.0444, 6.0444) == pytest.approx(1.0)
    # 272 px of rim is 45 mm by construction, so the inverse is 45 mm too
    scale = ds.rim_pixels_per_mm(272.0)
    assert ds.px_to_mm(272.0, scale) == pytest.approx(45.0)
    with pytest.raises(ValueError):
        ds.px_to_mm(10.0, 0.0)


def test_the_dish_itself_converts_back_to_its_physical_area() -> None:
    """End-to-end check: a 90-mm disc measured in px comes out as pi*45^2 mm^2."""
    radius_px = 272.0
    scale = ds.rim_pixels_per_mm(radius_px)

    area_mm2 = ds.area_px_to_mm2(math.pi * radius_px**2, scale)

    assert area_mm2 == pytest.approx(math.pi * 45.0**2, rel=1e-9)


def test_area_conversion_needs_a_length_scale() -> None:
    assert ds.area_px_to_mm2(900.0, 3.0) == pytest.approx(100.0)
    with pytest.raises(ValueError):
        ds.area_px_to_mm2(900.0, 0.0)


def test_px_per_mm_rescales_with_the_frame_width() -> None:
    """The constant is measured at 900 px wide; full resolution is 4608 px."""
    assert ds.rescale_px_per_mm(6.0444, 900) == pytest.approx(6.0444)
    assert ds.rescale_px_per_mm(6.0444, 4608) == pytest.approx(6.0444 * 5.12)
    with pytest.raises(ValueError):
        ds.rescale_px_per_mm(6.0444, 0)
