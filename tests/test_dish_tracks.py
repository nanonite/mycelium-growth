"""Tests for the driver that recovers dish tracks (Chainlink #23).

The geometry has its own suite; what is tested here is the plumbing that turns
per-image measurements into the published report, because that is where a run
can silently produce a *plausible-looking* artefact:

* ``--report-only`` is documented as rebuilding the statistics without
  rescanning.  The clustering block lives in the embeddings file, not in the
  CSV, so if the embeddings are not reloaded the notes are rewritten with an
  empty ``clusters`` section -- same shape, no content, easy to miss.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import dish_tracks as dt
import numpy as np
import pytest


def write_fixture(
    root: Path,
    *,
    embeddings: bool = True,
    n: int = 12,
    width: int = 4,
) -> dict[str, Path]:
    """Create the smallest set of files ``main(--report-only)`` can run on."""
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    # the archive list is checked before --report-only short-circuits the scan
    with zipfile.ZipFile(raw / "dummy.zip", "w") as zf:
        zf.writestr("dummy/image/00000001.jpg", b"not scanned in report-only mode")

    out = root / "dish_tracks.csv"
    rng = np.random.default_rng(0)
    rows = [{field: "" for field in dt.CSV_FIELDS} for _ in range(n)]
    for i, row in enumerate(rows):
        # rows 0 and 1 are byte-identical, so the duplicate report has work to do
        row.update(
            id=f"{i:08d}",
            archive="dummy.zip",
            entry=f"dummy/image/{i:08d}.jpg",
            md5=f"{0:032x}" if i < 2 else f"{i:032x}",
            width=900,
            height=675,
            dish_r_over_w=f"{0.30 + 0.001 * i:.5f}",
            dot_angle_deg=f"{360.0 * i / n:.2f}",
            colony_radius_frac=f"{0.1 + 0.01 * i:.5f}",
            inoculum_offset="0.0200",
        )
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=dt.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    paths = {
        "raw": raw,
        "out": out,
        "summary": root / "summary.csv",
        "notes": root / "notes.md",
        "embeddings": root / "dish_embeddings.npz",
    }
    if embeddings:
        np.savez_compressed(
            paths["embeddings"],
            keys=np.array([f"dummy.zip|{r['entry']}" for r in rows]),
            rim=rng.standard_normal((n, width)).astype(np.float32),
            rim_rot=rng.standard_normal((n, 3)).astype(np.float32),
        )
    return paths


def run_report_only(paths: dict[str, Path]) -> dict:
    code = dt.main(
        [
            "--raw-dir",
            str(paths["raw"]),
            "--out",
            str(paths["out"]),
            "--summary",
            str(paths["summary"]),
            "--notes",
            str(paths["notes"]),
            "--embeddings",
            str(paths["embeddings"]),
            "--report-only",
        ]
    )
    assert code == 0
    return json.loads(paths["notes"].with_suffix(".json").read_text(encoding="utf-8"))


def test_report_only_rebuilds_the_clustering_block(tmp_path: Path) -> None:
    """Regression: the report must reload the saved embeddings, not drop them."""
    paths = write_fixture(tmp_path)
    report = run_report_only(paths)
    assert report["images"] == 12
    rim = report["clusters"]["rim"]
    # 12 rows -> 12 * 11 / 2 off-diagonal pairs
    assert rim["n"] == 12
    assert rim["pairs_total"] == 66
    assert len(rim["nn_observed"]) == 4


def test_report_only_without_embeddings_degrades_visibly(tmp_path: Path) -> None:
    """A missing embeddings file must not fail the run, but must be announced."""
    paths = write_fixture(tmp_path, embeddings=False)
    report = run_report_only(paths)
    assert report["clusters"] == {}
    # duplicates and dot statistics come from the CSV and must still be there
    assert report["dot_angles"]["n"] == 12
    # one shared hash (rows 0 and 1), the rest unique
    assert len(report["duplicates"]) == 1


def test_report_only_summary_matches_the_rows(tmp_path: Path) -> None:
    """The summary CSV is derived from the same rows the JSON was built from."""
    paths = write_fixture(tmp_path)
    run_report_only(paths)
    with paths["summary"].open(encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 1
    assert summary[0]["archive"] == "dummy.zip"
    assert summary[0]["images"] == "12"
    assert summary[0]["dot_angles_distinct"] == "12"


# --------------------------------------------------------------------------- #
# statistics that the write-up quotes
# --------------------------------------------------------------------------- #


def test_duplicate_pairs_are_split_out_of_the_tail(tmp_path: Path) -> None:
    """The table's "of which byte-identical" column must be exact.

    Rows 0 and 1 share an md5 *and* an embedding, so they are a byte-identical
    pair by construction and their similarity is 1.0; everything else is noise.
    """
    paths = write_fixture(tmp_path, width=32)
    rng = np.random.default_rng(1)
    embeddings = rng.standard_normal((12, 32)).astype(np.float32)
    embeddings[1] = embeddings[0]
    np.savez_compressed(
        paths["embeddings"],
        keys=np.array([f"dummy.zip|dummy/image/{i:08d}.jpg" for i in range(12)]),
        rim=embeddings,
        rim_rot=embeddings[:, :8],
    )
    report = run_report_only(paths)
    rim = report["clusters"]["rim"]
    assert rim["duplicate_pairs"] == 1
    assert rim["pairs_gt_0p99"] >= 1
    assert rim["duplicate_pairs_gt_0p99"] == 1
    assert rim["non_duplicate_pairs_gt_0p99"] == rim["pairs_gt_0p99"] - 1
    # and the split is consistent for every threshold
    for label in ("0p90", "0p95", "0p99"):
        assert rim[f"duplicate_pairs_gt_{label}"] <= rim[f"pairs_gt_{label}"]
        assert rim[f"non_duplicate_pairs_gt_{label}"] == (
            rim[f"pairs_gt_{label}"] - rim[f"duplicate_pairs_gt_{label}"]
        )


def _angle_rows(angles: list[float]) -> list[dict[str, object]]:
    return [{"dot_angle_deg": f"{a:.2f}"} for a in angles]


def test_dot_angle_structure_counts_multiplicity() -> None:
    """Reuse shows up as one angle carrying many images -- count it directly."""
    rows = _angle_rows([10.0] * 3 + [20.0])
    stats = dt.dot_angle_structure(rows)
    assert stats["n"] == 4
    assert stats["distinct"] == 2
    assert stats["max_multiplicity"] == 3
    assert stats["max_multiplicity_share"] == pytest.approx(0.75)
    assert stats["occupied_bins"] == 2
    assert stats["total_bins"] == 720
    # three copies of one value are three pairs within 0.05 deg
    assert stats["collisions"]["0.05"]["observed"] == 3


def test_same_angle_pairs_wraps_around_zero() -> None:
    """359.9 and 0.1 are 0.2 degrees apart, not 359.8."""
    observed, expected = dt.same_angle_pairs(np.array([359.9, 0.1]), 0.5)
    assert observed == 1
    assert expected == pytest.approx(1 * 2 * 0.5 / 360.0, abs=1e-4)
    # ... and they are outside a 0.1 degree window
    observed, _ = dt.same_angle_pairs(np.array([359.9, 0.1]), 0.1)
    assert observed == 0


def test_duplicate_pair_similarities_only_counts_shared_hashes() -> None:
    """A hash shared by three images is three pairs, and nothing else."""
    features = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 4.0]], dtype=np.float32
    )
    keys = np.array(["a", "b", "c", "d"])
    md5_by_key = {"a": "h1", "b": "h1", "c": "h1", "d": "h2"}
    sims = dt.duplicate_pair_similarities(features, keys, md5_by_key)
    assert len(sims) == 3, "C(3, 2) = 3 pairs inside the one shared hash"
    # a and b are identical, a/c and b/c are orthogonal; the singleton hash
    # contributes nothing
    assert sorted(sims) == pytest.approx([0.0, 0.0, 1.0], abs=1e-5)
