"""Tests for the measurement contract (Chainlink #60).

The contract exists to stop a class of mistakes that produce plausible numbers:
differencing two frames whose order was never demonstrated, dividing a
per-frame change by nothing and calling it mm/day, logging an area whose unit
was dropped, counting a byte-identical copy as a second observation, or writing
a rule in ``notes/estimands.md`` that ``scripts/measurement_contract.py`` does
not enforce. So what is tested here is (a) the guards fire on the inputs they
are supposed to refuse, (b) the contract is internally consistent and agrees
with the committed triage tables it quotes, and (c) the note and the module
still name the same estimands, units and prohibitions.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import measurement_contract as mc  # needs the sys.path line above

NOTE = REPO / "notes" / "estimands.md"
ASSUMPTIONS_NOTE = REPO / "notes" / "assumptions.md"
DISH_TRACKS = REPO / "data" / "triage" / "dish_tracks.csv"
CALIBRATION = REPO / "data" / "triage" / "calibration.csv"

# calibration.csv (block, scale_group) -> the key the contract publishes it under.
CALIBRATION_KEYS = {
    ("GL", "1"): "GL",
    ("GS", "1"): "GS",
    ("GS_PO_TS_flat", "1"): "GS_PO_TS_flat_setup1",
    ("GS_PO_TS_flat", "2"): "GS_PO_TS_flat_setup2",
    ("MYG", "1"): "MYG_PO",
    ("PO", "1"): "MYG_PO",
    ("PDA", "1"): "PDA",
    ("TEMP15", "1"): "TEMP15",
    ("TS", "1"): "TS",
}

# Headings the note has to carry for the contract to be readable as one document.
NOTE_SECTIONS = (
    "## 1. Population",
    "## 2. Inclusion rule",
    "## 3. Primary image-level estimands",
    "## 4. Conditional frame-level estimand",
    "## 5. Units",
    "## 6. Calibration",
    "## 7. Duplicate handling",
    "## 8. Uncertainty target",
    "## 9. Reporting rules",
    "## Reproducing",
)


# --- the contract is internally consistent -----------------------------------


def test_validate_contract_reports_no_problems() -> None:
    assert mc.validate_contract() == []


def test_every_required_section_is_populated() -> None:
    for key in mc.REQUIRED_CONTRACT_KEYS:
        assert key in mc.CONTRACT, f"missing contract section {key!r}"
        assert mc.CONTRACT[key], f"contract section {key!r} is empty"


def test_population_arithmetic_is_self_consistent() -> None:
    population = mc.CONTRACT["population"]
    assert (
        population["n_condition_known"] + population["n_condition_unknown"]
        == population["n_images"]
    )
    assert population["n_labeled_masks"] + population["n_unlabeled"] == 20176
    duplicates = mc.CONTRACT["duplicate_handling"]
    # 62 groups, 133 images in them: 133 - 62 = 71 copies are dropped.
    assert duplicates["n_after_deduplication"] == population["n_images"] - (
        duplicates["n_duplicate_images"] - duplicates["n_duplicate_groups"]
    )


def test_every_estimand_is_fully_specified() -> None:
    for estimand in mc.ALL_ESTIMANDS:
        for field in mc.REQUIRED_ESTIMAND_FIELDS:
            value = getattr(estimand, field)
            assert value, f"{estimand.name}: {field} is empty"
        assert estimand.level in mc.LEVELS
        assert estimand.status in mc.STATUSES
        assert estimand.precondition in mc.PRECONDITIONS
        assert estimand.strata, f"{estimand.name}: no strata"


def test_primary_image_estimands_cover_the_required_quantities() -> None:
    """Area fraction, log area, equivalent radius and calibrated shape features."""
    primary = {e.name for e in mc.IMAGE_ESTIMANDS if e.status == mc.STATUS_PRIMARY}
    assert {"area_fraction", "log_area", "equivalent_radius"} <= primary
    shape = primary & {"perimeter", "circularity", "edge_roughness", "solidity"}
    assert len(shape) >= 3, f"expected calibrated shape features, got {sorted(shape)}"
    assert mc.ESTIMANDS_BY_NAME["mean_intensity"].status == mc.STATUS_SECONDARY


def test_estimands_are_stratified_by_species_and_known_condition() -> None:
    for estimand in mc.IMAGE_ESTIMANDS:
        if estimand.status == mc.STATUS_SECONDARY:
            continue
        assert "species" in estimand.strata, estimand.name
        assert "condition" in estimand.strata, estimand.name


def test_frame_estimand_is_conditional_and_stays_per_frame() -> None:
    estimand = mc.ESTIMANDS_BY_NAME["delta_log_area_per_frame"]
    assert estimand.level == mc.LEVEL_FRAME
    assert estimand.status == mc.STATUS_CONDITIONAL
    assert estimand.precondition == mc.PRECONDITION_FRAME_ORDER
    assert "frame" in estimand.unit
    assert estimand.name in mc.CONTRACT["units"]


def test_no_estimand_unit_is_a_rate() -> None:
    for name, unit in mc.CONTRACT["units"].items():
        lowered = unit.lower()
        for word in mc.TIME_WORDS:
            assert word not in lowered, f"{name} is reported in a time unit ({unit})"


def test_units_in_the_contract_match_the_registry() -> None:
    assert mc.CONTRACT["units"] == mc.UNITS
    assert set(mc.CONTRACT["units"]) == {e.name for e in mc.ALL_ESTIMANDS}


# --- the guards fire ---------------------------------------------------------


def test_log_area_requires_an_explicit_reference_unit() -> None:
    with pytest.raises(mc.ContractViolation):
        mc.log_area(1000.0, reference_unit="")
    with pytest.raises(mc.ContractViolation):
        mc.log_area(1000.0, reference_unit="cm^2")
    assert mc.log_area(1000.0) == pytest.approx(math.log(1000.0))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_log_area_refuses_non_positive_or_non_finite_area(bad: float) -> None:
    with pytest.raises(mc.ContractViolation):
        mc.log_area(bad)


def test_frame_delta_refuses_undemonstrated_order() -> None:
    with pytest.raises(mc.ContractViolation) as excinfo:
        mc.frame_delta_log_area(100.0, 110.0, frame_order_demonstrated=False)
    assert "frame order" in str(excinfo.value)


def test_frame_delta_computes_once_order_is_demonstrated() -> None:
    delta = mc.frame_delta_log_area(
        100.0, 110.0, frame_order_demonstrated=True, reference_unit="mm^2"
    )
    assert delta == pytest.approx(math.log(1.10))
    # Order matters: reversing the frames flips the sign, so an unordered pair
    # cannot be differenced "either way round" and still mean something.
    reverse = mc.frame_delta_log_area(
        110.0, 100.0, frame_order_demonstrated=True, reference_unit="mm^2"
    )
    assert reverse == pytest.approx(-delta)


def test_physical_rate_refuses_without_a_capture_interval() -> None:
    for interval in (None, 0.0, -3.0, float("nan")):
        with pytest.raises(mc.ContractViolation) as excinfo:
            mc.to_physical_rate(0.1, interval)
        assert "capture interval" in str(excinfo.value)


def test_physical_rate_converts_only_with_a_capture_interval() -> None:
    assert mc.to_physical_rate(0.2, 2.0, time_unit="hour") == pytest.approx(0.1)
    assert mc.to_physical_rate(0.2, 0.5, time_unit="day") == pytest.approx(0.4)


def test_physical_rate_rejects_an_unknown_time_unit() -> None:
    with pytest.raises(mc.ContractViolation):
        mc.to_physical_rate(0.1, 1.0, time_unit="fortnight")


# --- uncertainty target ------------------------------------------------------


def test_uncertainty_target_has_both_limits_per_kind() -> None:
    target = mc.CONTRACT["uncertainty_target"]
    assert target["confidence_level"] == 0.95
    for kind in mc.KINDS:
        limits = target["targets_pct"][kind]
        assert limits["relative_bias"] > 0
        assert limits["relative_sd"] >= limits["relative_bias"]


def test_size_and_boundary_targets_are_compared_correctly() -> None:
    assert mc.meets_uncertainty_target(
        "size", relative_bias_pct=2.0, relative_sd_pct=8.0
    )
    assert not mc.meets_uncertainty_target(
        "size", relative_bias_pct=4.0, relative_sd_pct=8.0
    )
    assert not mc.meets_uncertainty_target(
        "boundary", relative_bias_pct=1.0, relative_sd_pct=25.0
    )
    assert mc.meets_uncertainty_target(
        "boundary", relative_bias_pct=1.0, relative_sd_pct=15.0
    )
    with pytest.raises(ValueError):
        mc.meets_uncertainty_target("tip", relative_bias_pct=1.0, relative_sd_pct=1.0)


def test_min_reportable_delta_is_two_repeatability_sds() -> None:
    sigma = mc.CONTRACT["uncertainty_target"]["frame_detection_sigma"]
    assert sigma == 2.0
    assert mc.min_reportable_delta(0.04) == pytest.approx(0.08)
    assert mc.min_reportable_delta(0.0) == 0.0
    with pytest.raises(mc.ContractViolation):
        mc.min_reportable_delta(-0.1)


# --- duplicate handling ------------------------------------------------------


def test_deduplicate_keeps_the_lowest_representative() -> None:
    rows = [
        {"archive": "b.zip", "id": "00000002", "md5": "same"},
        {"archive": "b.zip", "id": "00000001", "md5": "same"},
        {"archive": "a.zip", "id": "00000009", "md5": "other"},
        {"archive": "b.zip", "id": "00000003", "md5": "same"},
    ]
    kept, dropped = mc.deduplicate(rows)
    assert [row["id"] for row in kept] == ["00000001", "00000009"]
    assert [row["id"] for row in dropped] == ["00000002", "00000003"]


def test_deduplicate_is_deterministic_and_order_preserving() -> None:
    rows = [
        {"archive": "z.zip", "id": "00000005", "md5": "h"},
        {"archive": "a.zip", "id": "00000007", "md5": "h"},
    ]
    kept, dropped = mc.deduplicate(rows)
    assert kept[0]["archive"] == "a.zip"
    assert dropped[0]["archive"] == "z.zip"
    # a generator input must give the same answer as a list input
    kept_again, dropped_again = mc.deduplicate(row for row in rows)
    assert kept_again == kept and dropped_again == dropped


def test_deduplicate_refuses_rows_without_a_hash() -> None:
    with pytest.raises(ValueError):
        mc.deduplicate([{"archive": "a.zip", "id": "00000001"}])


def test_committed_dish_tracks_deduplicate_to_the_contract_counts() -> None:
    """The duplicate rule is checked against the CSV it is written against (#23)."""
    with DISH_TRACKS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    duplicates = mc.CONTRACT["duplicate_handling"]
    assert len(rows) == mc.CONTRACT["population"]["n_images"]
    kept, dropped = mc.deduplicate(rows)
    # 133 images share 62 hashes, so 71 copies are dropped and 20,105 remain --
    # which is also the number of distinct hashes in the file.
    assert (
        len(dropped)
        == duplicates["n_duplicate_images"] - duplicates["n_duplicate_groups"]
    )
    assert len(kept) == duplicates["n_after_deduplication"]
    assert len({row["md5"] for row in rows}) == duplicates["n_after_deduplication"]


def test_contract_calibration_constants_match_the_committed_table() -> None:
    published = mc.CONTRACT["calibration"]["constants_px_per_mm_at_900"]
    seen: set[str] = set()
    with CALIBRATION.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = CALIBRATION_KEYS[(row["block"], row["scale_group"])]
            seen.add(key)
            assert published[key] == round(float(row["px_per_mm"]), 4), key
    assert seen == set(published), "contract and calibration.csv disagree on blocks"


# --- the note and the module say the same thing ------------------------------


@pytest.fixture(scope="module")
def note_text() -> str:
    assert NOTE.is_file(), f"missing measurement contract note at {NOTE}"
    return NOTE.read_text()


def test_note_carries_every_section(note_text: str) -> None:
    for heading in NOTE_SECTIONS:
        assert heading in note_text, f"note is missing heading {heading!r}"


def test_note_names_every_estimand(note_text: str) -> None:
    for estimand in mc.ALL_ESTIMANDS:
        assert estimand.name in note_text, f"note never mentions {estimand.name}"


def test_note_records_the_prohibitions(note_text: str) -> None:
    assert "capture interval" in note_text
    assert "mm/day" in note_text
    assert "growth curve" in note_text
    assert "hyphal tips" in note_text
    assert len(mc.CONTRACT["prohibited"]) >= 5


def test_note_quotes_the_contract_numbers(note_text: str) -> None:
    population = mc.CONTRACT["population"]
    duplicates = mc.CONTRACT["duplicate_handling"]
    assert f"{population['n_images']:,}" in note_text  # 20,176
    assert f"{population['n_labeled_masks']:,}" in note_text  # 567
    assert f"{population['n_condition_unknown']:,}" in note_text  # 1,689
    assert f"{duplicates['n_after_deduplication']:,}" in note_text  # 20,105
    level = mc.CONTRACT["uncertainty_target"]["confidence_level"]
    assert f"{round(level * 100)}%" in note_text  # 95%


def test_note_and_module_agree_on_the_frame_condition(note_text: str) -> None:
    """Both halves must say the frame estimand needs demonstrated order."""
    assert "demonstrated frame order" in note_text or "frame order" in note_text
    assert mc.ESTIMANDS_BY_NAME["delta_log_area_per_frame"].precondition == (
        mc.PRECONDITION_FRAME_ORDER
    )
    assert "not estimated" in note_text


# --- selection-bias record (#35) -----------------------------------------------


def test_validate_selection_record_reports_no_problems() -> None:
    assert mc.validate_selection_record() == []


def test_selection_record_carries_exactly_the_two_claims() -> None:
    assert set(mc.SELECTION_BIAS["claims"]) == {
        "independent_of_observed_morphology",
        "independent_of_latent_time_frame",
    }


def test_selection_record_names_the_documented_mechanism() -> None:
    assert "distinctive edge morphology" in mc.SELECTION_BIAS["mechanism"]


def test_selection_independence_from_observed_morphology_is_falsified() -> None:
    # Selection is explicitly based on distinctive edge morphology, so the
    # release over-represents the estimands' own features by construction.
    assert mc.selection_claim("independent_of_observed_morphology") == "falsified"


def test_selection_independence_from_latent_time_frame_is_untestable() -> None:
    # Gate C: no dish identity, no timestamps, no ordered sequences, so no
    # in-release evidence can speak to temporal dependence.
    assert mc.selection_claim("independent_of_latent_time_frame") == "untestable"


def test_no_selection_claim_is_supported() -> None:
    # Neither half may be reported as independent: one is falsified, the other
    # untestable, and no evidence for temporal independence may be invented.
    for claim, status in mc.SELECTION_BIAS["claims"].items():
        assert status != "supported", claim
        assert status in mc.SELECTION_STATUSES, claim


def test_selection_claim_refuses_an_unknown_half() -> None:
    with pytest.raises(mc.ContractViolation):
        mc.selection_claim("independent_of_species")


def test_selection_record_bounds_interpretation_to_the_selected_release() -> None:
    supported = mc.SELECTION_BIAS["supported"]
    assert "selected release" in supported
    assert "never a plate population" in supported


def test_selection_record_preserves_the_gate_c_prohibition() -> None:
    prohibition = mc.SELECTION_BIAS["gate_c_prohibition"]
    assert "causal temperature or medium" in prohibition
    assert "growth-rate" in prohibition


# --- the assumptions note and the module say the same thing --------------------


@pytest.fixture(scope="module")
def assumptions_text() -> str:
    assert ASSUMPTIONS_NOTE.is_file(), f"missing assumption register at {ASSUMPTIONS_NOTE}"
    return ASSUMPTIONS_NOTE.read_text()


def test_assumptions_note_records_the_35_split(assumptions_text: str) -> None:
    """The register must separate the falsified half from the untestable half."""
    assert "#35" in assumptions_text
    assert "falsified" in assumptions_text
    assert "untestable" in assumptions_text
    assert "selected release" in assumptions_text


def test_note_and_module_agree_on_the_selection_mechanism(assumptions_text: str) -> None:
    assert mc.SELECTION_BIAS["mechanism"] in assumptions_text


def test_note_does_not_assert_temporal_independence(assumptions_text: str) -> None:
    """No evidence for temporal independence may be invented (#35)."""
    lowered = assumptions_text.lower()
    assert "no evidence for temporal independence" in lowered
    # the untestable half is stated as untestable, never as independent
    assert mc.selection_claim("independent_of_latent_time_frame") == "untestable"
