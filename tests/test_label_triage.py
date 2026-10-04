#!/usr/bin/env python3
"""Unit tests for label_triage.py parsing helpers."""

import pytest
from scripts.triage.label_triage import (
    get_condition_key,
    image_id_from_entry,
    iter_archive_entries,
    SPECIES_FROM_FOLDER,
    MEDIUM_FROM_FOLDER,
    TEMP_FROM_FOLDER,
)


class TestImageIdFromEntry:
    """Tests for image_id_from_entry helper."""

    def test_standard_jpg(self):
        assert image_id_from_entry("labeled-GL/trainset/image/00000001.jpg") == "00000001"
        assert image_id_from_entry("unlabeled_data1/00000508.jpg") == "00000508"
        assert image_id_from_entry("00020176.jpg") == "00020176"

    def test_uppercase_extension(self):
        assert image_id_from_entry("image/00000001.JPG") == "00000001"
        assert image_id_from_entry("image/00000001.Jpeg") == "00000001"

    def test_no_id(self):
        assert image_id_from_entry("README.txt") == ""
        assert image_id_from_entry("folder/") == ""


class TestGetConditionKey:
    """Tests for get_condition_key mapping."""

    def test_labeled_gl(self):
        assert get_condition_key("labeled-GL.zip", "") == "labeled-GL"
        assert get_condition_key("labeled-GL.zip", "labeled-GL/trainset/image/00000001.jpg") == "labeled-GL"

    def test_labeled_gs_po_ts_species_folders(self):
        assert get_condition_key("labeled-GS_PO_TS.zip", "labeled-GS_PO_TS/GS/image/00018428.jpg") == "labeled-GS_PO_TS/GS"
        assert get_condition_key("labeled-GS_PO_TS.zip", "labeled-GS_PO_TS/PO/image/00018438.jpg") == "labeled-GS_PO_TS/PO"
        assert get_condition_key("labeled-GS_PO_TS.zip", "labeled-GS_PO_TS/TS/image/00018448.jpg") == "labeled-GS_PO_TS/TS"

    def test_labeled_myg_pda_temp_condition_folders(self):
        assert get_condition_key("labeled-MYG_PDA_TEMP.zip", "MYG/image/00018458.jpg") == "labeled-MYG_PDA_TEMP/MYG"
        assert get_condition_key("labeled-MYG_PDA_TEMP.zip", "PDA/image/00018468.jpg") == "labeled-MYG_PDA_TEMP/PDA"
        assert get_condition_key("labeled-MYG_PDA_TEMP.zip", "TEMP15/image/00018478.jpg") == "labeled-MYG_PDA_TEMP/TEMP15"

    def test_unlabeled_gl_shards(self):
        assert get_condition_key("unlabeled-GL1.zip", "unlabeled_data1/00000508.jpg") == "unlabeled-GL1"
        assert get_condition_key("unlabeled-GL7.zip", "unlabeled_data7/00018427.jpg") == "unlabeled-GL7"

    def test_unlabeled_gs_po_ts(self):
        assert get_condition_key("unlabeled-GS_PO_TS.zip", "00019002.jpg") == "unlabeled-GS_PO_TS"


class TestLookupTables:
    """Tests that lookup tables have expected keys."""

    def test_species_keys_exist(self):
        # Labeled archives
        assert "labeled-GL" in SPECIES_FROM_FOLDER
        assert "labeled-GS_PO_TS/GS" in SPECIES_FROM_FOLDER
        assert "labeled-GS_PO_TS/PO" in SPECIES_FROM_FOLDER
        assert "labeled-GS_PO_TS/TS" in SPECIES_FROM_FOLDER
        assert "labeled-MYG_PDA_TEMP/MYG" in SPECIES_FROM_FOLDER
        assert "labeled-MYG_PDA_TEMP/PDA" in SPECIES_FROM_FOLDER
        assert "labeled-MYG_PDA_TEMP/TEMP15" in SPECIES_FROM_FOLDER
        # Unlabeled
        for i in range(1, 8):
            assert f"unlabeled-GL{i}" in SPECIES_FROM_FOLDER
        assert "unlabeled-GS_PO_TS" in SPECIES_FROM_FOLDER

    def test_medium_keys_exist(self):
        assert "labeled-GL" in MEDIUM_FROM_FOLDER
        assert "labeled-GS_PO_TS/GS" in MEDIUM_FROM_FOLDER
        assert "labeled-GS_PO_TS/PO" in MEDIUM_FROM_FOLDER
        assert "labeled-GS_PO_TS/TS" in MEDIUM_FROM_FOLDER
        assert "labeled-MYG_PDA_TEMP/MYG" in MEDIUM_FROM_FOLDER
        assert "labeled-MYG_PDA_TEMP/PDA" in MEDIUM_FROM_FOLDER
        assert "labeled-MYG_PDA_TEMP/TEMP15" in MEDIUM_FROM_FOLDER
        for i in range(1, 8):
            assert f"unlabeled-GL{i}" in MEDIUM_FROM_FOLDER
        assert "unlabeled-GS_PO_TS" in MEDIUM_FROM_FOLDER

    def test_temp_keys_exist(self):
        assert "labeled-GL" in TEMP_FROM_FOLDER
        assert "labeled-GS_PO_TS/GS" in TEMP_FROM_FOLDER
        assert "labeled-GS_PO_TS/PO" in TEMP_FROM_FOLDER
        assert "labeled-GS_PO_TS/TS" in TEMP_FROM_FOLDER
        assert "labeled-MYG_PDA_TEMP/MYG" in TEMP_FROM_FOLDER
        assert "labeled-MYG_PDA_TEMP/PDA" in TEMP_FROM_FOLDER
        assert "labeled-MYG_PDA_TEMP/TEMP15" in TEMP_FROM_FOLDER
        for i in range(1, 8):
            assert f"unlabeled-GL{i}" in TEMP_FROM_FOLDER
        assert "unlabeled-GS_PO_TS" in TEMP_FROM_FOLDER


class TestIterArchiveEntries:
    """Tests for iter_archive_entries (requires actual archive files)."""

    def test_labeled_gl_count(self):
        """Verify labeled-GL.zip yields 507 image entries (no MACOSX)."""
        from pathlib import Path
        archive = Path("/mnt/sharedOs/mycelium-growth-workspace/data/raw/myceliumseg/labeled-GL.zip")
        entries = list(iter_archive_entries(archive))
        assert len(entries) == 507, f"Expected 507, got {len(entries)}"
        # All should be .jpg
        for arch, name in entries:
            assert name.lower().endswith(".jpg")
            assert "__MACOSX/" not in name
            assert not name.split("/")[-1].startswith("._")

    def test_labeled_gs_po_ts_count(self):
        from pathlib import Path
        archive = Path("/mnt/sharedOs/mycelium-growth-workspace/data/raw/myceliumseg/labeled-GS_PO_TS.zip")
        entries = list(iter_archive_entries(archive))
        assert len(entries) == 30, f"Expected 30, got {len(entries)}"

    def test_labeled_myg_pda_temp_count(self):
        from pathlib import Path
        archive = Path("/mnt/sharedOs/mycelium-growth-workspace/data/raw/myceliumseg/labeled-MYG_PDA_TEMP.zip")
        entries = list(iter_archive_entries(archive))
        assert len(entries) == 30, f"Expected 30, got {len(entries)}"

    def test_unlabeled_gl1_count(self):
        from pathlib import Path
        archive = Path("/mnt/sharedOs/mycelium-growth-workspace/data/raw/myceliumseg/unlabeled-GL1.zip")
        entries = list(iter_archive_entries(archive))
        assert len(entries) == 4480, f"Expected 4480, got {len(entries)}"

    def test_unlabeled_gs_po_ts_count(self):
        from pathlib import Path
        archive = Path("/mnt/sharedOs/mycelium-growth-workspace/data/raw/myceliumseg/unlabeled-GS_PO_TS.zip")
        entries = list(iter_archive_entries(archive))
        assert len(entries) == 1689, f"Expected 1689, got {len(entries)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])