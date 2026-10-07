"""Unit tests for the inference + exploratory smoke-test driver (Chainlink #29).

The model half needs torch/mmcv and a GPU, so what is tested here is the part
that can silently produce a *plausible but wrong* result without one: how the
committed triage tables become per-image provenance, how the unlabeled sample
is made deterministic and auditable, how area and area fraction are computed
under the #60 contract, and — most importantly — that the exploratory sort
never smuggles in elapsed time, a growth rate, a dish id or a biological tip.
The real GPU run is recorded in notes/segformer_inference.md.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import segformer_inference as si  # needs the sys.path line above

# --- fixtures ----------------------------------------------------------------


def _track(image_id, archive, *, md5, dish_r=270.0, width=900, height=675):
    return {
        "id": image_id,
        "archive": archive,
        "entry": f"{archive}/{image_id}.jpg",
        "width": str(width),
        "height": str(height),
        "dish_r": str(dish_r),
        "md5": md5,
    }


def _scale(image_id, archive, *, block="GL", px_per_mm=6.0, fit_status="ok", group="1"):
    return {
        "id": image_id,
        "archive": archive,
        "block": block,
        "scale_group": group,
        "px_per_mm": str(px_per_mm),
        "fit_status": fit_status,
    }


def _label(image_id, species, medium, temp="25C"):
    return {
        "id": image_id,
        "species_if_known": species,
        "medium_if_known": medium,
        "temp_if_known": temp,
        "source": "fixture",
    }


CAPABILITY = {
    "dataset": "myceliumseg",
    "ordered_frames": "no",
    "capture_interval": "none",
    "estimands_identifiable": "image_morphology",
    "estimands_blocked": "delta_per_ordered_step;physical_rate;intervention_effect",
}


@pytest.fixture()
def provenance() -> dict[str, si.Provenance]:
    tracks = [
        _track("00000002", "b.zip", md5="same"),
        _track("00000001", "a.zip", md5="same"),
        _track("00000003", "a.zip", md5="other"),
        _track("00000004", "a.zip", md5="unique", dish_r=0.0),
    ]
    scales = [
        _scale("00000002", "b.zip"),
        _scale("00000001", "a.zip"),
        _scale("00000003", "a.zip", fit_status="off_scale"),
        _scale("00000004", "a.zip"),
    ]
    labels = [
        _label("00000001", "GL", "MYG"),
        _label("00000002", "PO", "PDA"),  # disagrees with 00000001: a conflict
        _label("00000003", "GS", "MYG"),
        _label("00000004", "", "", ""),
    ]
    return si.build_provenance(tracks, scales, labels, CAPABILITY)


# --- provenance --------------------------------------------------------------


def test_duplicate_role_follows_the_lowest_archive_and_id(provenance):
    assert provenance["00000001"].duplicate_role == "representative"
    assert provenance["00000002"].duplicate_role == "copy"
    assert provenance["00000001"].duplicate_group_size == 2
    assert provenance["00000002"].duplicate_representative == "00000001"
    # A unique image is its own representative.
    assert provenance["00000003"].duplicate_role == "representative"
    assert provenance["00000003"].duplicate_group_size == 1


def test_label_conflict_is_flagged_from_the_md5_group(provenance):
    # 00000001 (GL/MYG) and 00000002 (PO/PDA) are one byte-identical group with
    # two different non-blank labels: both rows carry the conflict flag.
    assert provenance["00000001"].label_conflict is True
    assert provenance["00000002"].label_conflict is True
    # A unique image can never conflict with itself.
    assert provenance["00000003"].label_conflict is False


def test_frame_order_and_time_unit_status_come_from_the_capability_row(provenance):
    for row in provenance.values():
        assert row.frame_order_status == "not_demonstrated"
        assert row.time_unit_status == "no_capture_interval"
        assert row.ordered_frames == "no"
        assert row.capture_interval == "none"


def test_provenance_as_dict_has_no_inferred_time_field(provenance):
    si.assert_no_inferred_time([row.as_dict() for row in provenance.values()])
    payload = provenance["00000001"].as_dict()
    assert payload["condition"] == "GL/MYG/25C"
    assert payload["condition_known"] is True
    assert provenance["00000004"].condition_known is False


def test_build_provenance_reads_the_committed_tables():
    loaded = si.load_provenance()
    assert len(loaded) == 20176
    assert loaded["00000001"].archive == "labeled-GL.zip"
    assert loaded["00000001"].frame_order_status == "not_demonstrated"


# --- deterministic unlabeled sample -----------------------------------------


def _sample_provenance() -> dict[str, si.Provenance]:
    tracks = []
    scales = []
    labels = []
    for index in range(10):
        image_id = f"unlabeled_data{index:05d}"  # placeholder, replaced below
        del image_id
    # Two archives, ten ids each, ids spaced by 1 so the stride is predictable.
    for base, archive, block in ((5000, "unlabeled-GL2.zip", "GL"), (9000, "flat.zip", "GS_PO_TS_flat")):
        for offset in range(10):
            image_id = f"{base + offset:08d}"
            tracks.append(_track(image_id, archive, md5=f"h-{image_id}"))
            scales.append(_scale(image_id, archive, block=block))
            labels.append(_label(image_id, "", "", "25C"))
    return si.build_provenance(tracks, scales, labels, CAPABILITY)


def test_select_unlabeled_sample_is_deterministic_and_auditable():
    provenance = _sample_provenance()
    spec = {
        "unlabeled-GL2.zip": {"stride": 3, "limit": 2},
        "flat.zip": {"stride": 4, "limit": 1},
    }
    sample = si.select_unlabeled_sample(provenance, stride_spec=spec, probes=())
    assert sample == si.select_unlabeled_sample(provenance, stride_spec=spec, probes=())
    picked = [(item["id"], item["archive"], item["selection"]) for item in sample]
    # every 3rd of 00005000..00005009, capped at 2
    assert ("00005000", "unlabeled-GL2.zip", "stride") in picked
    assert ("00005003", "unlabeled-GL2.zip", "stride") in picked
    assert ("00009000", "flat.zip", "stride") in picked
    assert len([item for item in sample if item["archive"] == "unlabeled-GL2.zip"]) == 2
    # each row carries the exact zip member it came from
    for item in sample:
        assert item["entry"] == f"{item['archive']}/{item['id']}.jpg"
        assert item["selection_rule"]


def test_select_unlabeled_sample_adds_duplicate_probes_once():
    provenance = _sample_provenance()
    spec = {"unlabeled-GL2.zip": {"stride": 100, "limit": 1}}
    sample = si.select_unlabeled_sample(
        provenance, stride_spec=spec, probes=("00009005", "00009005", "99999999")
    )
    probes = [item for item in sample if item["selection"] == "duplicate_probe"]
    assert [item["id"] for item in probes] == ["00009005"]  # deduplicated, missing skipped


# --- area / area fraction under the contract ---------------------------------


def test_area_fraction_uses_the_full_frame_width(provenance):
    prov = provenance["00000001"]  # 900-px working width, px/mm 6, dish_r 270
    frame_width = 1800  # twice the working width
    measured = si.derive_measurements(100, prov, frame_width=frame_width)
    # px/mm scales linearly with the frame, so at 1800 px it is 12.
    assert measured["predicted_area_mm2"] == pytest.approx(100 / 12.0**2)
    dish_r_full = 270.0 * frame_width / 900.0
    assert measured["predicted_area_fraction"] == pytest.approx(
        100 / (math.pi * dish_r_full**2)
    )
    assert measured["predicted_equiv_radius_px"] == pytest.approx(math.sqrt(100 / math.pi))
    assert measured["area_exclusion"] == ""


def test_off_scale_image_keeps_px_but_loses_calibrated_area(provenance):
    prov = provenance["00000003"]  # fit_status == off_scale
    measured = si.derive_measurements(100, prov, frame_width=900)
    assert measured["predicted_area_px"] == 100
    assert measured["predicted_area_fraction"] is None
    assert "fit_status=off_scale" in measured["area_exclusion"]


def test_missing_dish_disc_excludes_area_fraction(provenance):
    prov = provenance["00000004"]  # dish_r == 0
    measured = si.derive_measurements(100, prov, frame_width=900)
    assert measured["predicted_area_fraction"] is None
    assert "no_dish_disc" in measured["area_exclusion"]


def test_empty_mask_is_not_logged_and_is_flagged():
    tracks = [_track("00000009", "a.zip", md5="x", dish_r=270.0)]
    scales = [_scale("00000009", "a.zip")]
    labels = [_label("00000009", "GL", "MYG")]
    prov = si.build_provenance(tracks, scales, labels, CAPABILITY)["00000009"]
    measured = si.derive_measurements(0, prov, frame_width=900)
    assert measured["predicted_empty_mask"] is True
    assert measured["predicted_log_area_px2"] is None
    assert measured["predicted_log_area_mm2"] is None
    assert measured["predicted_area_fraction"] == 0.0


# --- the exploratory sort and its guardrails ---------------------------------


def _record(image_id, archive, block, area_fraction, *, md5=None, role="representative"):
    return {
        "id": image_id,
        "archive": archive,
        "block": block,
        "md5": md5 or f"md5-{image_id}",
        "duplicate_role": role,
        "duplicate_group_size": 1,
        "fit_status": "ok",
        "predicted_area_fraction": area_fraction,
        "predicted_area_mm2": area_fraction,
        "predicted_empty_mask": False,
        "predicted_mask_sha256": f"sha-{image_id}",
        "predicted_over_reference_area": None,
        "label_conflict": False,
    }


def test_explore_sorts_inside_each_archive_block_stratum():
    records = [
        _record("00000003", "a.zip", "GL", 0.5),
        _record("00000001", "a.zip", "GL", 0.1),
        _record("00000002", "a.zip", "GL", 0.9),
        _record("00000004", "b.zip", "GS", 0.2),
    ]
    exploration = si.explore_records(records)
    by_block = {(b["archive"], b["block"]): b for b in exploration["blocks"]}
    assert list(by_block) == [("a.zip", "GL"), ("b.zip", "GS")]
    gl = by_block[("a.zip", "GL")]
    assert [row["id"] for row in gl["sorted"]] == ["00000001", "00000003", "00000002"]
    assert [row["rank"] for row in gl["sorted"]] == [1, 2, 3]
    assert by_block[("b.zip", "GS")]["sorted"][0]["id"] == "00000004"


def test_explore_detects_gap_bands():
    # Three tiny values then a jump: the gap factor should split two bands.
    records = [_record(f"{i:08d}", "a.zip", "GL", area) for i, area in enumerate(
        [0.01, 0.02, 0.03, 0.30, 0.31, 0.32], start=1
    )]
    exploration = si.explore_records(records, gap_factor=4.0)
    block = exploration["blocks"][0]
    assert block["n_bands"] == 2
    assert [band["n"] for band in block["bands"]] == [3, 3]


def test_explore_flags_duplicate_groups_with_identical_masks():
    records = [
        _record("00000001", "a.zip", "GL", 0.4, md5="same", role="representative"),
        _record("00000002", "b.zip", "GL", 0.4, md5="same", role="copy"),
    ]
    records[0]["predicted_mask_sha256"] = "identical"
    records[1]["predicted_mask_sha256"] = "identical"
    exploration = si.explore_records(records)
    assert len(exploration["duplicate_groups"]) == 1
    group = exploration["duplicate_groups"][0]
    assert group["ids"] == ["00000001", "00000002"]
    assert group["predicted_mask_sha256_identical"] is True


def test_explore_records_consecutive_id_runs_as_exploratory_only():
    records = [
        _record("00000010", "a.zip", "GL", 0.1),
        _record("00000011", "a.zip", "GL", 0.2),
        _record("00000020", "a.zip", "GL", 0.3),
    ]
    exploration = si.explore_records(records)
    assert exploration["blocks"][0]["id_runs"] == [["00000010", "00000011"]]


def test_explore_keeps_excluded_rows_out_of_the_bands():
    record = _record("00000001", "a.zip", "GL", 0.4)
    record["predicted_area_fraction"] = None
    record["area_exclusion"] = "fit_status=off_scale"
    exploration = si.explore_records([record])
    block = exploration["blocks"][0]
    assert block["n_scored"] == 0
    assert block["n_excluded"] == 1
    assert block["excluded"][0]["area_exclusion"] == "fit_status=off_scale"


def test_explore_output_carries_the_caveats():
    exploration = si.explore_records([_record("00000001", "a.zip", "GL", 0.4)])
    text = " ".join(exploration["caveats"])
    assert "no ordered frames" in text
    assert "not a duration" in text or "elapsed time" in text
    assert "hyphal tip" in text


def test_forbidden_inference_fields_are_a_hard_error():
    for key in ("elapsed_time", "growth_rate", "dish_id", "hyphal_tip_count"):
        with pytest.raises(si.InferenceError):
            si.assert_no_inferred_time([{"id": "00000001", key: 1}])
    # the status fields this task *does* record must pass
    si.assert_no_inferred_time(
        [{"id": "00000001", "frame_order_status": "not_demonstrated",
          "time_unit_status": "no_capture_interval", "capture_interval": "none"}]
    )


def test_exploration_csv_header_names_no_forbidden_quantity():
    header = si.explore_csv_header()
    for column in header:
        for forbidden in si.FORBIDDEN_COLUMN_SUBSTRINGS:
            assert forbidden not in column, column
    # the provenance/status columns the task must preserve are present
    for required in (
        "archive",
        "id",
        "block",
        "duplicate_role",
        "condition",
        "frame_order_status",
        "time_unit_status",
        "predicted_area_px",
        "predicted_area_fraction",
    ):
        assert required in header


def test_write_explore_csv_round_trips(tmp_path: Path):
    records = [
        _record("00000001", "a.zip", "GL", 0.1),
        _record("00000002", "a.zip", "GL", 0.5, md5="same", role="copy"),
    ]
    exploration = si.explore_records(records)
    out = tmp_path / "explore.csv"
    si.write_explore_csv(out, records, exploration)
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == ["00000001", "00000002"]
    assert rows[0]["explore_rank"] == "1"
    assert rows[1]["duplicate_role"] == "copy"
    assert "frame_order_status" in rows[0]
    assert set(rows[0]) == set(si.explore_csv_header())


def test_write_explore_csv_keeps_excluded_rows_with_provenance(tmp_path: Path):
    scored = _record("00000001", "a.zip", "GL", 0.1)
    excluded = _record("00000002", "a.zip", "GL", 0.0)
    excluded["predicted_area_fraction"] = None
    excluded["fit_status"] = "off_scale"
    excluded["area_exclusion"] = "fit_status=off_scale"
    records = [scored, excluded]
    exploration = si.explore_records(records)
    out = tmp_path / "explore.csv"
    si.write_explore_csv(out, records, exploration)
    with out.open(newline="") as handle:
        rows = {row["id"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {"00000001", "00000002"}
    assert rows["00000002"]["explore_rank"] == ""
    assert rows["00000002"]["fit_status"] == "off_scale"
    assert "off_scale" in rows["00000002"]["explore_flags"]


# --- evidence ----------------------------------------------------------------


def test_collect_evidence_counts_records_and_records_caveats(tmp_path: Path):
    records = [
        _record("00000001", "a.zip", "GL", 0.1),
        _record("00000002", "a.zip", "GL", 0.2),
    ]
    for record in records:
        record["source_set"] = "test"
    exploration = si.explore_records(records)
    evidence = si.collect_evidence(
        tmp_path,
        {"checkpoint_sha256": si.CHECKPOINT_SHA256, "device": "cuda:0"},
        records,
        exploration,
    )
    assert evidence["counts"] == {"test": 2}
    assert evidence["checkpoint_guard"]["expected_sha256"] == si.CHECKPOINT_SHA256
    assert evidence["exploration"]["n_duplicate_groups"] == 0
    assert evidence["caveats"]


def test_checkpoint_constant_matches_the_committed_evidence():
    evidence = json.loads((REPO / "notes" / "segformer_evidence.json").read_text())
    assert evidence["checkpoint"]["sha256"] == si.CHECKPOINT_SHA256
    assert evidence["protocol"]["config_sha256"] == si.CONFIG_SHA256


def test_committed_exploration_table_matches_the_header_and_guard():
    """The committed smoke-test table must be exactly what the driver writes."""
    if not si.EXPLORE_CSV.is_file():
        pytest.skip("inference_explore.csv not generated in this checkout")
    with si.EXPLORE_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    assert header == si.explore_csv_header()
    for column in header:
        for forbidden in si.FORBIDDEN_COLUMN_SUBSTRINGS:
            assert forbidden not in column
    assert rows, "the committed table should not be empty"
    # Every row must carry the provenance the task promises.
    for row in rows:
        assert row["archive"]
        assert row["frame_order_status"] == "not_demonstrated"
        assert row["time_unit_status"] == "no_capture_interval"
        assert row["capture_interval"] == "none"
        assert row["duplicate_role"] in ("representative", "copy")
    # The two deliberate duplicate probes share a group with their hold-out
    # representatives and must be marked as copies.
    by_id = {row["id"]: row for row in rows}
    for probe, representative in (("00019773", "00018428"), ("00019301", "00018429")):
        if probe in by_id:
            assert by_id[probe]["duplicate_role"] == "copy"
            assert by_id[probe]["duplicate_representative"] == representative
            assert by_id[probe]["md5"] == by_id[representative]["md5"]
