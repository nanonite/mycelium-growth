"""Unit tests for the 15 °C image-count audit (Chainlink #34).

These tests verify the audit's core invariants against the committed data
artifacts: the 10 TEMP15 images are unique, correctly labelled, calibrated,
present in the measurement table, and free of cross-archive collisions and
label conflicts.  The 25 °C GL context count is checked for internal
consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_15c_images as a15


@pytest.fixture(scope="module")
def audit_result() -> dict:
    return a15.audit()


class TestTemp15Count:
    """The 15 °C GL image count is 10."""

    def test_n_images(self, audit_result):
        assert audit_result["temp15"]["n_images"] == 10

    def test_image_ids(self, audit_result):
        expected = {f"000184{n}" for n in range(78, 88)}
        assert set(audit_result["temp15"]["image_ids"]) == expected

    def test_source_archive(self, audit_result):
        assert audit_result["temp15"]["source_archive"] == "labeled-MYG_PDA_TEMP.zip"

    def test_source_folder(self, audit_result):
        assert audit_result["temp15"]["source_folder"] == "TEMP15"


class TestTemp15Uniqueness:
    """All 10 images are byte-distinct and appear in only one archive."""

    def test_distinct_md5(self, audit_result):
        assert audit_result["temp15"]["n_distinct_md5"] == 10

    def test_no_duplicate_ids(self, audit_result):
        assert audit_result["temp15"]["duplicate_ids"] == []

    def test_no_cross_archive_collisions(self, audit_result):
        assert audit_result["temp15"]["cross_archive_collisions"] == []

    def test_no_cross_temperature_overlap(self, audit_result):
        assert audit_result["cross_temperature_md5_overlap"] == 0


class TestTemp15Labels:
    """All 10 images carry the correct GL/MYG/15C labels."""

    def test_no_label_issues(self, audit_result):
        assert audit_result["temp15"]["label_issues"] == []

    def test_no_label_conflicts(self, audit_result):
        assert audit_result["temp15"]["label_conflicts"] == []


class TestTemp15Quality:
    """All 10 images pass the px→mm calibration (fit_status=ok)."""

    def test_no_off_scale(self, audit_result):
        assert audit_result["temp15"]["off_scale_ids"] == []


class TestTemp15Measurements:
    """All 10 images are present in the measurement table."""

    def test_none_missing(self, audit_result):
        assert audit_result["temp15"]["missing_from_measurements"] == []

    def test_all_eligible(self, audit_result):
        assert audit_result["temp15"]["n_eligible"] == 10


class TestGl25Context:
    """The 25 °C GL context count is internally consistent."""

    def test_n_images(self, audit_result):
        assert audit_result["gl25_context"]["n_images"] == 18437

    def test_distinct_md5(self, audit_result):
        assert audit_result["gl25_context"]["n_distinct_md5"] == 18437

    def test_off_scale_plus_ok_equals_total(self, audit_result):
        g = audit_result["gl25_context"]
        assert g["n_off_scale"] + g["n_ok"] == g["n_images"]

    def test_off_scale_count(self, audit_result):
        # 435 GL block + 1 MYG block = 436
        assert audit_result["gl25_context"]["n_off_scale"] == 436


class TestFeasibility:
    """The audit correctly states descriptive-only feasibility."""

    def test_descriptive_feasible(self, audit_result):
        assert audit_result["feasibility"]["descriptive_comparison"] == "feasible"

    def test_no_causal_claim(self, audit_result):
        assert "not identified" in audit_result["feasibility"]["causal_temperature_effect"]

    def test_no_rate_claim(self, audit_result):
        assert "not identifiable" in audit_result["feasibility"]["physical_growth_rate"]

    def test_no_power_claim(self, audit_result):
        assert "not computable" in audit_result["feasibility"]["dish_level_power"]


class TestAuditAgainstLiveData:
    """Cross-check the audit against the raw CSV artifacts directly."""

    def test_temp15_ids_match_known_labels(self):
        labels = a15.load_known_labels()
        temp15_from_labels = {
            i for i, lbl in labels.items()
            if lbl["species"] == "GL" and lbl["temp"] == "15C"
        }
        assert len(temp15_from_labels) == 10

    def test_temp15_md5s_all_distinct(self):
        md5 = a15.load_md5()
        temp15_ids = {f"000184{n}" for n in range(78, 88)}
        md5s = {md5[i] for i in temp15_ids if i in md5}
        assert len(md5s) == 10

    def test_temp15_all_fit_ok(self):
        scale = a15.load_image_scale()
        temp15_ids = {f"000184{n}" for n in range(78, 88)}
        for i in temp15_ids:
            assert scale[i]["fit_status"] == "ok"

    def test_temp15_all_in_measurements(self):
        meas = a15.load_measurements()
        temp15_ids = {f"000184{n}" for n in range(78, 88)}
        for i in temp15_ids:
            assert i in meas
            assert meas[i]["label_conflict"] == "False"
