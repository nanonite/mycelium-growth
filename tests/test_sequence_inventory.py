"""Tests for the sequence/frame provenance inventory (Chainlink #61).

What is tested is the machinery that turns containers into provenance facts,
because that is where a run can silently produce a *plausible-looking* matrix:

* filename parsers for the three coexisting naming schemes;
* AVI / TIFF / OME-Zarr readers, built as tiny synthetic containers;
* the series statistics (order confirmation, capture interval, duplicates);
* :func:`validate_rows`, which refuses a row whose estimand claims contradict
  its own evidence -- the guard that keeps "no interval" from ever carrying a
  physical rate;
* the committed capability matrix itself, so ``notes/provenance.md`` and
  ``data/triage/sequence_inventory.csv`` cannot drift into claiming a growth
  curve for MyceliumSeg.
"""

from __future__ import annotations

import csv
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
import sequence_inventory as si

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# fixtures: tiny JPEG / AVI / TIFF / zarr builders
# --------------------------------------------------------------------------- #


def make_jpeg(width: int = 4, height: int = 3) -> bytes:
    """A JPEG with SOI + JFIF + SOF0 and no SOS (parsers allow truncation)."""
    app0 = (b"\xff\xe0" + struct.pack(">H", 16)
            + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    sof_payload = struct.pack(">BHHB", 8, height, width, 1) + bytes(
        [1, 0x11, 0]
    )
    sof = b"\xff\xc0" + struct.pack(">H", len(sof_payload) + 2) + sof_payload
    return b"\xff\xd8" + app0 + sof


def riff_chunk(cid: bytes, body: bytes) -> bytes:
    padded = body + (b"\x00" if len(body) % 2 else b"")
    return cid + struct.pack("<I", len(body)) + padded


def make_avi(path: Path, frames: list[bytes], *, total_frames: int | None = None,
             width: int = 8, height: int = 8) -> None:
    """Smallest structurally valid AVI: hdrl (avih+strl) + movi + idx1."""
    avih = struct.pack(
        "<10I", 200000, 0, 0, 0x10, total_frames or len(frames), 0, 1, 0,
        width, height,
    )
    strh = (b"vids" + b"MJPG" + struct.pack("<I", 0)
            + struct.pack("<IIIIIIIIII", 0, 0, 1, 5, 0, len(frames),
                          1 << 20, 100, 0, 0)
            + struct.pack("<i", 0))  # second half of rcFrame: 56 bytes total
    strf = struct.pack("<IiiHHIIIIII", 40, width, height, 1, 24,
                       0x47504A4D, width * height * 3, 0, 0, 0, 0)
    strl = riff_chunk(
        b"LIST", b"strl" + riff_chunk(b"strh", strh)
        + riff_chunk(b"strf", strf)
    )
    hdrl = riff_chunk(
        b"LIST", b"hdrl" + riff_chunk(b"avih", avih) + strl
    )
    movi = riff_chunk(b"LIST", b"movi" + b"".join(
        riff_chunk(b"00dc", frame) for frame in frames
    ))
    idx1 = riff_chunk(b"idx1", b"\x00" * 16)
    body = b"AVI " + hdrl + movi + idx1
    path.write_bytes(riff_chunk(b"RIFF", body))


def make_imagej_tiff(
    path: Path,
    *,
    width: int = 4,
    height: int = 3,
    pages: list[bytes],
    endian: str = "<",
    description: str | None = None,
) -> None:
    """ImageJ-style stack: one IFD, then contiguous pages, no IFD chain."""
    if description is None:
        description = (f"ImageJ=1.52t\nimages={len(pages)}\n"
                       f"slices={len(pages)}\nloop=false\n")
    desc_bytes = description.encode() + b"\x00"
    strip_bytes = width * height
    tags = [
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, 1, 8),
        (262, 3, 1, 1),
        (270, 2, len(desc_bytes), None),  # offset filled below
        (273, 4, 1, None),
        (277, 3, 1, 1),
        (278, 4, 1, height),
        (279, 4, 1, strip_bytes),
    ]
    ifd_start = 8
    desc_start = ifd_start + 2 + 12 * len(tags) + 4
    strip_start = desc_start + len(desc_bytes)
    entries = bytearray()
    for tag, typ, count, value in tags:
        if value is None:
            value = desc_start if tag == 270 else strip_start
        entries += struct.pack(endian + "HHII", tag, typ, count, value)
    order_mark = b"II" if endian == "<" else b"MM"
    header = (order_mark + struct.pack(endian + "H", 42)
              + struct.pack(endian + "I", ifd_start))
    ifd = (struct.pack(endian + "H", len(tags)) + bytes(entries)
           + struct.pack(endian + "I", 0))
    path.write_bytes(header + ifd + desc_bytes + b"".join(pages))


def make_zarr_zip(path: Path, *, timestamps: list[int]) -> None:
    """Minimal OME-Zarr v3 store with the provenance arrays #61 reads."""
    from numcodecs.zstd import Zstd

    frames = len(timestamps)
    root = {
        "attributes": {
            "ome": {
                "multiscales": [{
                    "name": "synthetic",
                    "axes": [{"name": "t", "type": "time", "unit": "second"},
                             {"name": "c", "type": "channel"},
                             {"name": "y", "type": "space"},
                             {"name": "x", "type": "space"}],
                    "datasets": [{
                        "path": "s0",
                        "coordinateTransformations": [{
                            "type": "scale",
                            "scale": [300.0, 1.0, 0.09, 0.09],
                        }],
                    }],
                }],
            },
            "physarum_zenodo": {
                "experiment": "001",
                "dish": {"index": 0},
                "license": "CC-BY-4.0",
                "source": {"first_source_name": "frame_000.png"},
            },
        },
    }

    def array_meta(shape, data_type, fill_value=0, codecs=None):
        return {
            "shape": list(shape),
            "data_type": data_type,
            "chunk_grid": {"name": "regular",
                           "configuration": {"chunk_shape": list(shape)}},
            "fill_value": fill_value,
            "codecs": codecs or [{"name": "bytes",
                                  "configuration": {"endian": "little"}}],
            "node_type": "array",
            "zarr_format": 3,
        }

    sha_rows = [(i if i != 1 else 0) for i in range(frames)]
    sha = np.zeros((frames, 32), dtype=np.uint8)
    for row, tag in enumerate(sha_rows):
        sha[row, 0] = tag  # frame 1 reuses frame 0's hash -> duplicate pair
    circles = np.zeros((frames, 3), dtype=np.float32)
    circles[:, 2] = 50.0

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("zarr.json", json.dumps(root))
        zf.writestr("s0/zarr.json",
                    json.dumps(array_meta([frames, 3, 8, 8], "uint8")))
        zstd = [{"name": "bytes",
                 "configuration": {"endian": "little"}},
                {"name": "zstd", "configuration": {"level": 0}}]
        zf.writestr("acquisition/zarr.json", json.dumps({"node_type": "group"}))
        zf.writestr("acquisition/timestamp_utc_ns/zarr.json",
                    json.dumps(array_meta([frames], "int64", codecs=zstd)))
        zf.writestr(
            "acquisition/timestamp_utc_ns/c/0",
            Zstd().encode(np.asarray(timestamps, dtype="<i8").tobytes()),
        )
        zf.writestr("acquisition/source_sha256/zarr.json",
                    json.dumps(array_meta([frames, 32], "uint8",
                                          codecs=zstd)))
        zf.writestr("acquisition/source_sha256/c/0/0",
                    Zstd().encode(sha.tobytes()))
        # qc_flags: no chunk member at all -> every value is the fill value
        zf.writestr("acquisition/qc_flags/zarr.json",
                    json.dumps(array_meta([frames], "uint8", fill_value=0)))
        zf.writestr("registration/zarr.json",
                    json.dumps({"node_type": "group"}))
        zf.writestr("registration/dish_circle_xyr_px/zarr.json",
                    json.dumps(array_meta([frames, 3], "float32",
                                          codecs=zstd)))
        zf.writestr("registration/dish_circle_xyr_px/c/0/0",
                    Zstd().encode(circles.tobytes()))


# --------------------------------------------------------------------------- #
# filename parsers
# --------------------------------------------------------------------------- #


def test_grazer_names_cover_every_real_pattern() -> None:
    """Hf carries the DD2 block and both day markers; Pi/Pv/Rb neither."""
    parsed = si.parse_grazer_name("Hf_DD2_coll_D1_1.jpg")
    assert parsed == {"species": "Hf", "block": "DD2", "treatment": "coll",
                      "day": 1, "replicate": 1}
    assert si.parse_grazer_name("Hf_DD2_nema_d16_5.jpg") == {
        "species": "Hf", "block": "DD2", "treatment": "nema", "day": 16,
        "replicate": 5,
    }
    assert si.parse_grazer_name("Pi_wl2_d38_3.jpg") == {
        "species": "Pi", "block": "", "treatment": "wl2", "day": 38,
        "replicate": 3,
    }
    assert si.parse_grazer_name("Rb_ctrl1_d1_4.jpg")["treatment"] == "ctrl1"


def test_grazer_name_rejects_what_it_cannot_parse() -> None:
    assert si.parse_grazer_name("Pi_coll_d16.jpg") is None  # no replicate
    assert si.parse_grazer_name("readme.txt") is None
    assert si.parse_grazer_name("Pv_coll_d1_1.png") is None  # wrong extension


def test_network_names_cover_all_three_schemes() -> None:
    first = si.parse_network_name("C41(1)_t25_c")
    assert first == {"colony": "C41(1)", "day": 25, "condition": ""}
    second = si.parse_network_name("Pv_ctrl1_d8_1")
    assert second == {"colony": "Pv_ctrl1_1", "day": 8, "condition": "ctrl1"}
    third = si.parse_network_name("Pi 18C day 8 001")
    assert third == {"colony": "Pi_18C_1", "day": 8, "condition": "18C"}
    assert si.parse_network_name("not a filename") is None


# --------------------------------------------------------------------------- #
# container readers
# --------------------------------------------------------------------------- #


def test_avi_headers_and_movi_walk_on_synthetic_file(tmp_path: Path) -> None:
    path = tmp_path / "movie.avi"
    make_avi(path, [b"frame-one!", b"frame-two!", b"frame-one!"])
    headers = si.avi_headers(path.read_bytes())
    assert headers["total_frames"] == 3
    assert headers["bmp_width"] == 8 and headers["bmp_height"] == 8
    assert headers["playback_fps"] == 5.0
    assert headers["microsec_per_frame"] == 200000  # playback, not capture

    chunks = si.iter_movi_chunks(path)
    assert len(chunks) == 3
    for _, offset, size in chunks:
        assert path.read_bytes()[offset:offset + size].startswith(b"frame")
    with pytest.raises(ValueError, match="not an AVI"):
        si.avi_headers(b"not a riff file at all")


def test_audit_avi_counts_stored_frames_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "movie.avi"
    make_avi(path, [b"same", b"same", b"other"])
    detail = si.audit_avi(path)
    assert detail["frames_header"] == detail["frames_stored"] == 3
    assert detail["order_complete"] is True
    assert detail["duplicates"]["duplicate_groups"] == 1
    assert detail["duplicates"]["redundant_copies"] == 1
    # a header that lies about the frame count must be visible as a mismatch
    lying = tmp_path / "lying.avi"
    make_avi(lying, [b"x"], total_frames=99)
    assert si.audit_avi(lying)["order_complete"] is False


def test_imagej_frame_layout_recovers_85_pages(tmp_path: Path) -> None:
    """The arithmetic behind the real M0_1.tif: 85 pages + a stale trailer."""
    frames, trailer, ok = si.imagej_frame_layout(
        4_261_178_698, 3_789, 50_131_345
    )
    assert (frames, trailer, ok) == (85, 10_584, True)
    assert si.imagej_frame_layout(100, 100, 10) == (0, 0, False)
    assert si.imagej_frame_layout(100, 0, -1) == (0, 0, False)


def test_parse_imagej_description() -> None:
    parsed = si.parse_imagej_description(
        "ImageJ=1.52t\nimages=85\nslices=85\nloop=false\n"
    )
    # non-integer values (loop=false, version strings) are skipped, not fatal
    assert parsed == {"images": 85, "slices": 85}
    assert si.parse_imagej_description("no equals sign") == {}


def test_tiff_ifd0_reads_both_byte_orders(tmp_path: Path) -> None:
    pages = [bytes([i + 1]) * 12 for i in range(3)]
    for endian in ("<", ">"):
        path = tmp_path / f"tiff_{endian}.tif"
        make_imagej_tiff(path, pages=pages, endian=endian)
        meta = si.tiff_ifd0(path)
        assert (meta["width"], meta["height"]) == (4, 3)
        assert meta["strip_bytes"] == 12
        assert "images=3" in (meta["description"] or "")


def test_audit_tiff_stack_detects_zero_filled_pages(tmp_path: Path) -> None:
    """M0_1.tif's real defect: 85 allocated pages, only 15 with content."""
    content = [bytes([11]) * 12, bytes([11]) * 12, bytes([22]) * 12]
    path = tmp_path / "stack.tif"
    make_imagej_tiff(path, pages=content + [b"\x00" * 12] * 5)
    detail = si.audit_tiff_stack(path)
    assert detail["frames"] == 8
    assert detail["frames_with_content"] == 3
    assert detail["blank_pages"] == 5
    assert detail["layout_consistent"] is True
    assert detail["imagej_images"] == 8
    # two of the three content pages are byte-identical
    assert detail["duplicates"]["duplicate_groups"] == 1
    # order_check runs on content pages only (three of them -> it computes)
    order = detail["order_check"]
    assert order is not None
    assert set(order) == {"adjacent_mean_abs_diff",
                          "non_adjacent_mean_abs_diff"}


def test_audit_tiff_stack_flags_a_mismatched_header(tmp_path: Path) -> None:
    path = tmp_path / "lying.tif"
    make_imagej_tiff(path, pages=[b"\x01" * 12, b"\x02" * 12],
                     description="ImageJ=1.52t\nimages=99\n")
    detail = si.audit_tiff_stack(path, hashes=False)
    assert detail["frames"] == 2
    assert detail["layout_consistent"] is False  # header says 99


def test_order_check_prefers_adjacent_frames() -> None:
    base = np.arange(32 * 32, dtype=np.int16).reshape(32, 32)
    frames = [base + i for i in range(6)]  # slow monotone drift
    stats = si.order_check(frames)
    assert stats["adjacent_mean_abs_diff"] < stats["non_adjacent_mean_abs_diff"]
    assert si.order_check([base, base + 1]) is None  # too few frames


def test_read_zarr_v3_array_and_fill_only_arrays(tmp_path: Path) -> None:
    path = tmp_path / "series.ome.zarr.zip"
    make_zarr_zip(path, timestamps=[1_000_000_000, 1_300_000_000,
                                    1_601_000_000, 1_902_000_000])
    with zipfile.ZipFile(path) as zf:
        stamps = si.read_zarr_v3_array(zf, "acquisition/timestamp_utc_ns")
        assert list(stamps["array"]) == [1_000_000_000, 1_300_000_000,
                                         1_601_000_000, 1_902_000_000]
        qc = si.read_zarr_v3_array(zf, "acquisition/qc_flags")
        assert list(qc["array"]) == [0, 0, 0, 0]  # fill-only member
        meta = si.zarr_array_meta(zf, "s0")
        assert meta["shape"] == (4, 3, 8, 8)


def test_audit_physarum_reads_provenance_arrays(tmp_path: Path) -> None:
    base = tmp_path / "physarum"
    base.mkdir()
    path = base / "physarum-001-dish0-r0c0.ome.zarr.zip"
    t0 = 1_693_074_049_000_000_000  # ns, as in the real release
    make_zarr_zip(path, timestamps=[t0, t0 + 300_000_000_000,
                                    t0 + 601_000_000_000,
                                    t0 + 902_000_000_000])
    detail = si.audit_physarum(tmp_path)
    assert detail["frames"] == 4
    assert detail["nominal_interval_s"] == 300.0
    ts = detail["timestamps"]
    assert ts["strictly_increasing"] is True
    assert (ts["gap_min_s"], ts["gap_max_s"]) == (300.0, 301.0)
    # frames 0 and 1 share a source sha256 in the fixture
    assert detail["source_sha256"]["duplicate_groups"] == 1
    assert detail["qc_flagged_frames"] == 0
    assert detail["frames_with_dish_circle"] == 4
    assert si.audit_physarum(tmp_path / "empty") is None


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def test_duplicate_stats_counts_groups_and_redundancy() -> None:
    stats = si.duplicate_stats(["a", "b", "a", "a", "c"])
    assert stats == {"n": 5, "distinct": 3, "duplicate_groups": 1,
                     "items_in_duplicate_groups": 3, "redundant_copies": 2,
                     "largest_group": 3}
    assert si.duplicate_stats(["x", "y"])["duplicate_groups"] == 0


def test_timestamp_stats_reports_order_and_interval() -> None:
    ns = [0, 300_000_000_000, 601_000_000_000]
    stats = si.timestamp_stats(ns)
    assert stats["strictly_increasing"] is True
    assert (stats["gap_min_s"], stats["gap_max_s"]) == (300.0, 301.0)
    assert stats["span_days"] >= 0
    broken = si.timestamp_stats([5, 4])
    assert broken["strictly_increasing"] is False
    assert si.timestamp_stats([7])["strictly_increasing"] is True


def test_series_stats_confirms_order_from_exif() -> None:
    entries = [
        {"day": 1, "datetime": "2009-02-25T11:00:00"},
        {"day": 2, "datetime": "2009-02-26T11:00:00"},
        {"day": 4, "datetime": "2009-02-28T11:00:00"},
    ]
    stats = si.series_stats(["Hf", "", "coll", 1], entries)
    assert stats["order_confirmed"] is True
    assert stats["gaps_days"] == [1.0, 2.0]
    assert stats["n_days"] == 3
    assert stats["identical_timestamp_pairs"] == 0

    # timestamps that contradict the filename day axis are a conflict
    conflict = si.series_stats(["x"], [
        {"day": 1, "datetime": "2009-02-28T11:00:00"},
        {"day": 2, "datetime": "2009-02-26T11:00:00"},
    ])
    assert conflict["order_confirmed"] is False
    assert conflict["order_backwards"] == [(1, 2)]

    same_ts = [dict(entries[0], datetime="2009-02-25T11:00:00"),
               dict(entries[1], datetime="2009-02-25T11:00:00")]
    assert si.series_stats(["x"], same_ts)["identical_timestamp_pairs"] == 1

    # no EXIF -> the filename day axis is the only interval source
    no_exif = si.series_stats(["x"], [{"day": 1}, {"day": 4}])
    assert no_exif["order_confirmed"] is False
    assert no_exif["gaps_days"] == [3.0]


def test_aggregate_series_pools_the_interval_distribution() -> None:
    one = si.series_stats(["a"], [{"day": 1}, {"day": 2}, {"day": 5}])
    two = si.series_stats(["b"], [{"day": 1}, {"day": 3}])
    agg = si.aggregate_series([one, two])
    assert agg["n_series"] == 2
    assert agg["n_order_confirmed"] == 0  # no EXIF in the fixtures
    assert agg["n_without_exif"] == 2
    assert agg["day_min"] == 1 and agg["day_max"] == 5
    assert agg["gap_days_min"] == 1.0 and agg["gap_days_max"] == 3.0
    assert agg["n_gap_obs"] == 3
    assert si.aggregate_series([])["n_gap_obs"] == 0


def test_label_conflicts_need_two_different_non_blank_labels() -> None:
    md5_by_id = {"1": "h1", "2": "h1", "3": "h2", "4": "h3"}
    labels = {
        "1": ("PO", "PDA", "25C"),
        "2": ("GL", "MYG", "25C"),   # same pixels, different folder label
        "3": ("GL", "MYG", "25C"),
        "4": ("", "", "25C"),        # blank species is not a conflict
    }
    conflicts = si.label_conflicts(md5_by_id, labels)
    assert conflicts == {"conflict_groups": 1, "images_involved": 2}
    # identical labels across copies are not a conflict either
    agree = si.label_conflicts({"1": "h9", "2": "h9"},
                               {"1": ("GL", "MYG", "25C"),
                                "2": ("GL", "MYG", "25C")})
    assert agree["conflict_groups"] == 0


def test_normalize_exif_datetime() -> None:
    assert si.normalize_exif_datetime("2009:02:25 11:55:09") == \
        "2009-02-25T11:55:09"
    assert si.normalize_exif_datetime("2023-08-26T14:20:49") == \
        "2023-08-26T14:20:49"
    assert si.normalize_exif_datetime(None) is None
    assert si.normalize_exif_datetime("garbage") is None


def test_size_and_duplicate_cells_format_for_humans() -> None:
    assert si.sizes_cell({"4608x3456": 18448, "1382x1037": 1728}) == \
        "4608x3456 x18448; 1382x1037 x1728"
    crowded = si.sizes_cell({f"{i}x{i}": 1 for i in range(7)})
    assert crowded.endswith("+3 more sizes")
    assert si.dup_cell(None) == "not checked (content hashing skipped)"
    assert si.dup_cell(si.duplicate_stats(["a", "a"])) == \
        "2 items in 1 hash groups; 1 redundant copies"


# --------------------------------------------------------------------------- #
# auditors on synthetic inputs
# --------------------------------------------------------------------------- #


def _write_grazer_zip(path: Path) -> None:
    rejected = make_jpeg(600, 450)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Hf/Hf_DD2_coll_D1_1.jpg", make_jpeg())
        zf.writestr("Hf/Hf_DD2_coll_D2_1.jpg", make_jpeg(601, 451))
        zf.writestr("Hf/Hf_DD2_coll_D3_1.jpg", make_jpeg())  # == D1 bytes
        zf.writestr("Hf/Hf_DD2_ctrl1_D1_1.jpg", make_jpeg(602, 452))
        zf.writestr("Hf/Hf_DD2_ctrl2_D1_1.jpg", make_jpeg(603, 453))
        zf.writestr("Hf/not_a_grazer_name.jpg", make_jpeg())
        zf.writestr("Pv/rejected/Pv_coll_d1_1.jpg", rejected)
        zf.writestr("Hf/results/notes.txt", "not an image")


def test_audit_grazers_series_duplicates_and_rejection(tmp_path: Path) -> None:
    raw = tmp_path / "fricker_grazers"
    raw.mkdir()
    _write_grazer_zip(raw / "Hf.zip")

    detail = si.audit_grazers(tmp_path)
    assert detail["images_total"] == 6  # parsed jpgs incl. rejected + odd
    assert detail["images_kept"] == 5
    assert detail["images_rejected"] == 1
    assert detail["unparsed_names"] == ["Hf/not_a_grazer_name.jpg"]
    assert detail["unreadable_entries"] == []
    assert detail["rejected_resolutions"] == {"600x450": 1}
    # coll rep 1 has three days, one of them a byte-identical copy
    assert detail["duplicates"]["duplicate_groups"] == 1
    assert detail["duplicates"]["redundant_copies"] == 1
    only_group = next(iter(detail["duplicate_groups_detail"].values()))
    assert only_group == [
        "Hf/Hf_DD2_coll_D1_1.jpg", "Hf/Hf_DD2_coll_D3_1.jpg",
    ]
    agg = detail["series_aggregate"]
    assert agg["n_series"] == 3  # coll rep1, ctrl1 rep1, ctrl2 rep1
    assert agg["day_min"] == 1 and agg["day_max"] == 3
    assert agg["n_without_exif"] == 3  # one per series (fixtures carry no EXIF)
    assert detail["images_with_exif_datetime"] == 0
    assert set(detail["treatments"]) == {"coll", "ctrl1", "ctrl2"}

    limited = si.audit_grazers(tmp_path, limit=1, hashes=False)
    assert limited["images_total"] == 1  # one entry per archive
    assert limited["duplicates"] is None  # hashing skipped


def test_audit_networks_counts_colonies_once(tmp_path: Path) -> None:
    raw = tmp_path / "fricker_networks"
    raw.mkdir()
    with zipfile.ZipFile(raw / "mixed.zip", "w") as zf:
        zf.writestr("C41(1)_t25_c.jpg", make_jpeg(10, 10))
        zf.writestr("C41(2)_t25_c.jpg", make_jpeg(11, 11))
        zf.writestr("Pi 18C day 8 001.jpg", make_jpeg(12, 12))
    with zipfile.ZipFile(raw / "parameters.zip", "w") as zf:
        zf.writestr("C41(1)_t25_c_param.mat", b"not an image")

    detail = si.audit_networks(tmp_path)
    assert detail["images"] == 3
    assert detail["colonies"] == 3
    assert detail["colonies_with_multiple_images"] == {}  # snapshots only
    assert detail["day_labels"] == [8, 25]
    assert detail["duplicates"]["duplicate_groups"] == 0
    assert si.audit_networks(tmp_path / "empty") is None


def test_audit_aureobasidium_is_data_free(tmp_path: Path) -> None:
    base = tmp_path / "aureobasidium"
    base.mkdir()
    (base / "book_of_abstracts.pdf").write_bytes(b"%PDF-1.4")
    detail = si.audit_aureobasidium(tmp_path)
    assert detail["data_files"] == []
    assert si.audit_aureobasidium(tmp_path / "missing") is None


# --------------------------------------------------------------------------- #
# the capability matrix and its guard
# --------------------------------------------------------------------------- #


def base_row(**overrides: str) -> dict[str, str]:
    row = {field: "" for field in si.CSV_FIELDS}
    row.update(
        dataset="synthetic",
        ordered_frames="no",
        specimen_identity="no",
        capture_interval="none",
        estimands_identifiable="image_morphology",
        estimands_blocked=("delta_per_ordered_step;physical_rate;"
                           "intervention_effect"),
    )
    row.update(overrides)
    return row


def test_validate_rows_accepts_the_honest_default() -> None:
    si.validate_rows([base_row()])


def test_validate_rows_refuses_claims_without_evidence() -> None:
    with pytest.raises(ValueError, match="empty"):
        si.validate_rows([])
    with pytest.raises(ValueError, match="without ordered frames"):
        si.validate_rows([base_row(
            ordered_frames="no",
            estimands_identifiable="image_morphology;delta_per_ordered_step",
            estimands_blocked="physical_rate;intervention_effect",
        )])
    with pytest.raises(ValueError, match="without a capture interval"):
        si.validate_rows([base_row(
            capture_interval="none",
            estimands_identifiable="image_morphology;physical_rate",
            estimands_blocked="delta_per_ordered_step;intervention_effect",
        )])
    with pytest.raises(ValueError, match="without specimen identity"):
        si.validate_rows([base_row(
            specimen_identity="no",
            estimands_identifiable="image_morphology;intervention_effect",
            estimands_blocked="delta_per_ordered_step;physical_rate",
        )])
    with pytest.raises(ValueError, match="unknown estimand"):
        si.validate_rows([base_row(estimands_identifiable="growth_curve")])
    with pytest.raises(ValueError, match="both identifiable and blocked"):
        si.validate_rows([base_row(
            estimands_blocked="image_morphology",
        )])
    with pytest.raises(ValueError, match="ordered_frames"):
        si.validate_rows([base_row(ordered_frames="maybe")])


def test_rows_from_details_builds_the_myceliumseg_row() -> None:
    details = {
        "myceliumseg": {
            "images": 20176,
            "images_with_datetime": 0,
            "resolutions": {"4608x3456": 18448, "1382x1037": 1728},
            "id_mtime_forward_fraction": [0.4891, 0.5474],
            "colony_vs_order_forward_fraction": [0.4828, 0.5517],
            "duplicates": si.duplicate_stats(["a", "a", "b"]),
            "labels_known": 18487,
            "labels_unknown": 1689,
            "conflict_groups": 20,
            "images_involved": 40,
        },
    }
    rows = si.rows_from_details(details)
    si.validate_rows(rows)
    row = rows[0]
    assert row["dataset"] == "myceliumseg"
    assert row["ordered_frames"] == "no"
    assert row["capture_interval"] == "none"
    assert row["estimands_identifiable"] == "image_morphology"
    assert "physical_rate" in row["estimands_blocked"]
    assert row["resolution"] == "4608x3456 x18448; 1382x1037 x1728"
    assert "20 md5 label conflicts" in row["condition_labels"]


def test_audit_myceliumseg_aggregates_the_stored_triage(tmp_path: Path) -> None:
    (tmp_path / "exif_scan_summary.csv").write_text(
        "archive,images,images_with_datetime,top_sizes,"
        "id_mtime_forward_fraction\n"
        "labeled-GL.zip,507,0,4608x3456=507,0.5474\n"
        "unlabeled-GL2.zip,2240,0,4608x3456=2052; 1382x1037=188,0.5083\n",
        encoding="utf-8",
    )
    (tmp_path / "dish_tracks.csv").write_text(
        "id,archive,entry,width,height,md5\n"
        "00000001,a.zip,e1.jpg,900,675,aaa\n"
        "00000002,a.zip,e2.jpg,900,675,aaa\n"
        "00000003,b.zip,e3.jpg,900,675,bbb\n"
        "00000004,b.zip,e4.jpg,900,675,ccc\n",
        encoding="utf-8",
    )
    (tmp_path / "dish_tracks_summary.csv").write_text(
        "archive,images,colony_radius_fwd_steps,duplicated_images\n"
        "labeled-GL.zip,3,0.5079,2\n"
        "unlabeled-GL2.zip,5,0.5163,0\n",
        encoding="utf-8",
    )
    (tmp_path / "known_labels.csv").write_text(
        "id,species_if_known,medium_if_known,temp_if_known,source\n"
        "00000001,GL,MYG,25C,folder:a\n"
        "00000002,PO,PDA,25C,folder:b\n"   # same md5 as 1, different label
        "00000003,GL,MYG,25C,folder:c\n"
        "00000004,,,25C,folder:d\n",        # blank species: not a conflict
        encoding="utf-8",
    )

    detail = si.audit_myceliumseg(tmp_path)
    assert detail["images"] == 2747
    assert detail["images_with_datetime"] == 0
    assert detail["resolutions"] == {"4608x3456": 2559, "1382x1037": 188}
    assert detail["id_mtime_forward_fraction"] == [0.5083, 0.5474]
    assert detail["colony_vs_order_forward_fraction"] == [0.5079, 0.5163]
    assert detail["duplicates"]["duplicate_groups"] == 1
    assert detail["labels_known"] == 3 and detail["labels_unknown"] == 1
    assert detail["conflict_groups"] == 1
    assert si.audit_myceliumseg(tmp_path / "missing") is None


def test_main_writes_matrix_and_json(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    networks = raw / "fricker_networks"
    networks.mkdir(parents=True)
    with zipfile.ZipFile(networks / "one.zip", "w") as zf:
        zf.writestr("C41(1)_t25_c.jpg", make_jpeg())
        zf.writestr("C41(2)_t25_c.jpg", make_jpeg(5, 5))
    aureo = raw / "aureobasidium"
    aureo.mkdir()
    (aureo / "book.pdf").write_bytes(b"%PDF-1.4")

    out = tmp_path / "matrix.csv"
    json_out = tmp_path / "detail.json"
    code = si.main([
        "--raw-dir", str(raw),
        "--triage-dir", str(tmp_path / "no-triage"),
        "--out", str(out),
        "--json-out", str(json_out),
    ])
    assert code == 0

    with out.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    si.validate_rows(rows)
    # only the two datasets on disk produce rows; the rest are announced
    assert sorted(r["dataset"] for r in rows) == [
        "aureobasidium", "fricker_networks",
    ]
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["issue"] == 61
    assert sorted(payload["datasets"]) == ["aureobasidium", "fricker_networks"]
    assert payload["datasets"]["fricker_networks"]["colonies"] == 2


# --------------------------------------------------------------------------- #
# the committed artifacts
# --------------------------------------------------------------------------- #


def test_committed_matrix_satisfies_its_own_guards() -> None:
    """The published CSV must survive the guard that generated it."""
    csv_path = REPO / "data" / "triage" / "sequence_inventory.csv"
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    si.validate_rows(rows)

    by_name = {row["dataset"]: row for row in rows}
    # MyceliumSeg: the cross-sectional fallback, never a growth curve
    seg = by_name["myceliumseg"]
    assert seg["ordered_frames"] == "no"
    assert seg["specimen_identity"] == "no"
    assert seg["capture_interval"] == "none"
    assert seg["estimands_identifiable"] == "image_morphology"
    assert seg["estimands_blocked"] == (
        "delta_per_ordered_step;physical_rate;intervention_effect"
    )
    # the nominated dynamic / intervention datasets do claim them
    grazers = by_name["fricker_grazers"]
    assert grazers["ordered_frames"] == "yes"
    assert "physical_rate" in grazers["estimands_identifiable"]
    assert "intervention_effect" in grazers["estimands_identifiable"]
    physarum = by_name["physarum_rig"]
    assert "physical_rate" in physarum["estimands_identifiable"]
    assert "intervention_effect" in physarum["estimands_blocked"]


def test_provenance_note_mentions_every_matrix_row() -> None:
    """notes/provenance.md mirrors the CSV row for row (no silent drops)."""
    note = (REPO / "notes" / "provenance.md").read_text(encoding="utf-8")
    with (REPO / "data" / "triage" / "sequence_inventory.csv").open(
        encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        assert f"`{row['dataset']}`" in note, (
            f"dataset {row['dataset']} missing from notes/provenance.md"
        )
    # the headline verdicts are stated in prose, not only in the CSV
    assert "no usable longitudinal sequence" in note
    assert "Fricker" in note and "Podospora" in note


def test_sequence_inventory_json_is_complete_and_honest() -> None:
    payload = json.loads(
        (REPO / "notes" / "sequence_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["hashes"] is True  # committed run hashed content
    assert payload["deflate64_supported"] is True
    datasets = payload["datasets"]
    assert len(datasets) == 9
    seg = datasets["myceliumseg"]
    assert seg["images"] == 20176
    assert seg["images_with_datetime"] == 0
    assert seg["conflict_groups"] == 20
    grazers = datasets["fricker_grazers"]
    assert grazers["images_kept"] == 777
    assert grazers["series_aggregate"]["n_order_conflicts"] == 0
    # the four byte-identical Pi ctrl2 day pairs are named, not just counted
    groups = grazers["duplicate_groups_detail"]
    assert len(groups) == 4
    for entries in groups.values():
        assert len(entries) == 2
        assert all(e.startswith("Pi/Pi_ctrl2_") for e in entries)
        assert {e.split("_d")[1][0] for e in entries} == {"1", "2"}
    m0 = datasets["podospora_m0_raw"]
    assert m0["frames"] == 85 and m0["frames_with_content"] == 15
    assert m0["blank_pages"] == 70
