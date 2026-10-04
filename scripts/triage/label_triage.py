#!/usr/bin/env python3
"""
Triage per-image metadata labels for MyceliumSeg.

This script derives what metadata labels are implied by the folder structure
of the 11 MyceliumSeg archives and writes data/triage/known_labels.csv

Labels checked:
- species (4 classes): G. lucidum (GL), G. sinense (GS), P. ostreatus (PO), Trametes (TS)
- medium (2 classes): MYG, PDA
- temperature (2 classes): 15°C, 25°C
- growth_stage (4 classes): NOT available per-image
- sclerotium_colour (5 classes): NOT available per-image

Sources:
- Folder structure in labeled archives (explicit)
- Paper Table 1 (culture conditions by species)
- Paper Tables 2-3 (aggregate counts only, not per-image)
- Zenodo record (file listings only)
- Benchmark repo (no per-image labels)
"""

from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class ImageLabel:
    """Per-image label assignment from folder structure + paper inference."""
    image_id: str
    species: str | None = None
    medium: str | None = None
    temperature: str | None = None
    source: str = ""


# ─── Archive → species/medium/temp mapping from folder structure + paper ───

# Paper Table 1: culture conditions by species
# GL: MYG, 15/25°C
# GS: MYG, 25°C
# PO: PDA, 25°C
# TS: PDA, 25°C

SPECIES_FROM_FOLDER = {
    # labeled archives
    "labeled-GL": "GL",
    "labeled-GS_PO_TS/GS": "GS",
    "labeled-GS_PO_TS/PO": "PO",
    "labeled-GS_PO_TS/TS": "TS",
    "labeled-MYG_PDA_TEMP/MYG": "GL",  # paper says GL on MYG
    "labeled-MYG_PDA_TEMP/PDA": "TS",  # paper says TS on PDA
    "labeled-MYG_PDA_TEMP/TEMP15": "GL",  # paper says GL at 15°C
    # unlabeled archives
    "unlabeled-GL1": "GL",
    "unlabeled-GL2": "GL",
    "unlabeled-GL3": "GL",
    "unlabeled-GL4": "GL",
    "unlabeled-GL5": "GL",
    "unlabeled-GL6": "GL",
    "unlabeled-GL7": "GL",
    "unlabeled-GS_PO_TS": "mixed_GS_PO_TS",  # mixed, no per-image breakdown
}

MEDIUM_FROM_FOLDER = {
    "labeled-MYG_PDA_TEMP/MYG": "MYG",
    "labeled-MYG_PDA_TEMP/PDA": "PDA",
    "labeled-MYG_PDA_TEMP/TEMP15": "MYG",  # paper: GL at 15°C on MYG
    # labeled-GL and labeled-GS_PO_TS: infer from paper Table 1
    "labeled-GL": "MYG",
    "labeled-GS_PO_TS/GS": "MYG",
    "labeled-GS_PO_TS/PO": "PDA",
    "labeled-GS_PO_TS/TS": "PDA",
    # unlabeled: no folder info, infer from species
    "unlabeled-GL1": "MYG",
    "unlabeled-GL2": "MYG",
    "unlabeled-GL3": "MYG",
    "unlabeled-GL4": "MYG",
    "unlabeled-GL5": "MYG",
    "unlabeled-GL6": "MYG",
    "unlabeled-GL7": "MYG",
    "unlabeled-GS_PO_TS": "mixed_MYG_PDA",  # mixed
}

TEMP_FROM_FOLDER = {
    "labeled-MYG_PDA_TEMP/TEMP15": "15C",
    # all others 25°C per paper Table 1
    "labeled-GL": "25C",
    "labeled-GS_PO_TS/GS": "25C",
    "labeled-GS_PO_TS/PO": "25C",
    "labeled-GS_PO_TS/TS": "25C",
    "labeled-MYG_PDA_TEMP/MYG": "25C",
    "labeled-MYG_PDA_TEMP/PDA": "25C",
    "unlabeled-GL1": "25C",
    "unlabeled-GL2": "25C",
    "unlabeled-GL3": "25C",
    "unlabeled-GL4": "25C",
    "unlabeled-GL5": "25C",
    "unlabeled-GL6": "25C",
    "unlabeled-GL7": "25C",
    "unlabeled-GS_PO_TS": "25C",
}

SOURCE_DESC = {
    "labeled-GL": "folder:labeled-GL + paper Table 1",
    "labeled-GS_PO_TS/GS": "folder:labeled-GS_PO_TS/GS + paper Table 1",
    "labeled-GS_PO_TS/PO": "folder:labeled-GS_PO_TS/PO + paper Table 1",
    "labeled-GS_PO_TS/TS": "folder:labeled-GS_PO_TS/TS + paper Table 1",
    "labeled-MYG_PDA_TEMP/MYG": "folder:labeled-MYG_PDA_TEMP/MYG + paper Table 1",
    "labeled-MYG_PDA_TEMP/PDA": "folder:labeled-MYG_PDA_TEMP/PDA + paper Table 1",
    "labeled-MYG_PDA_TEMP/TEMP15": "folder:labeled-MYG_PDA_TEMP/TEMP15 + paper Table 1",
    "unlabeled-GL1": "folder:unlabeled-GL1 + paper Table 1",
    "unlabeled-GL2": "folder:unlabeled-GL2 + paper Table 1",
    "unlabeled-GL3": "folder:unlabeled-GL3 + paper Table 1",
    "unlabeled-GL4": "folder:unlabeled-GL4 + paper Table 1",
    "unlabeled-GL5": "folder:unlabeled-GL5 + paper Table 1",
    "unlabeled-GL6": "folder:unlabeled-GL6 + paper Table 1",
    "unlabeled-GL7": "folder:unlabeled-GL7 + paper Table 1",
    "unlabeled-GS_PO_TS": "folder:unlabeled-GS_PO_TS + paper Table 1 (mixed species)",
}

_ID_RE = re.compile(r"(\d{4,8})(?=\.[^.]+$)")


def iter_archive_entries(archive_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (archive_name, entry_name) for image entries only."""
    with zipfile.ZipFile(archive_path, "r") as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            # Skip AppleDouble sidecars
            if "__MACOSX/" in name or name.split("/")[-1].startswith("._"):
                continue
            lowered = name.lower()
            if lowered.endswith((".jpg", ".jpeg")):
                yield archive_path.name, name


def get_condition_key(archive_name: str, entry_name: str) -> str:
    """Map archive + entry path to a condition key for lookup tables."""
    # labeled-GL.zip -> labeled-GL/trainset/... or labeled-GL/testset/...
    # labeled-GS_PO_TS.zip -> labeled-GS_PO_TS/GS/...
    # labeled-MYG_PDA_TEMP.zip -> MYG/... or PDA/... or TEMP15/...
    # unlabeled-GL*.zip -> unlabeled_dataN/...
    # unlabeled-GS_PO_TS.zip -> flat
    
    if archive_name == "labeled-GL.zip":
        return "labeled-GL"
    elif archive_name == "labeled-GS_PO_TS.zip":
        # Extract species folder: GS/, PO/, TS/
        parts = entry_name.split("/")
        if len(parts) >= 2 and parts[1] in ("GS", "PO", "TS"):
            return f"labeled-GS_PO_TS/{parts[1]}"
        return "labeled-GS_PO_TS"
    elif archive_name == "labeled-MYG_PDA_TEMP.zip":
        # Extract condition folder: MYG/, PDA/, TEMP15/
        parts = entry_name.split("/")
        if len(parts) >= 1 and parts[0] in ("MYG", "PDA", "TEMP15"):
            return f"labeled-MYG_PDA_TEMP/{parts[0]}"
        return "labeled-MYG_PDA_TEMP"
    elif archive_name.startswith("unlabeled-GL"):
        # unlabeled-GL1.zip through unlabeled-GL7.zip
        num = archive_name.replace("unlabeled-GL", "").replace(".zip", "")
        return f"unlabeled-GL{num}"
    elif archive_name == "unlabeled-GS_PO_TS.zip":
        return "unlabeled-GS_PO_TS"
    return "unknown"


def image_id_from_entry(entry_name: str) -> str:
    """Extract zero-padded numeric ID from entry name."""
    match = _ID_RE.search(entry_name.rsplit("/", 1)[-1])
    return match.group(1) if match else ""


def process_archives(raw_dir: Path) -> list[ImageLabel]:
    """Scan all archives and build per-image label assignments."""
    archives = sorted(raw_dir.glob("*.zip"))
    results: list[ImageLabel] = []
    
    for archive_path in archives:
        for arch_name, entry_name in iter_archive_entries(archive_path):
            img_id = image_id_from_entry(entry_name)
            if not img_id:
                continue
            
            condition_key = get_condition_key(archive_path.name, entry_name)
            
            species = SPECIES_FROM_FOLDER.get(condition_key)
            medium = MEDIUM_FROM_FOLDER.get(condition_key)
            temperature = TEMP_FROM_FOLDER.get(condition_key)
            source = SOURCE_DESC.get(condition_key, f"folder:{condition_key}")
            
            # For mixed archives, we can't assign per-image species/medium
            is_mixed = species and species.startswith("mixed")
            
            if is_mixed:
                results.append(ImageLabel(
                    image_id=img_id,
                    species=None,
                    medium=None,
                    temperature=temperature,
                    source=source + " (per-image unknown)"
                ))
            else:
                results.append(ImageLabel(
                    image_id=img_id,
                    species=species,
                    medium=medium,
                    temperature=temperature,
                    source=source
                ))
    
    return results


def write_known_labels(labels: list[ImageLabel], output_path: Path) -> None:
    """Write known_labels.csv with columns: id, species_if_known, medium_if_known, temp_if_known, source"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "species_if_known", "medium_if_known", "temp_if_known", "source"])
        for lbl in labels:
            writer.writerow([
                lbl.image_id,
                lbl.species or "",
                lbl.medium or "",
                lbl.temperature or "",
                lbl.source
            ])


def main() -> None:
    raw_dir = Path("/mnt/sharedOs/mycelium-growth-workspace/data/raw/myceliumseg")
    output_path = Path("/mnt/sharedOs/mycelium-growth-workspace/data/triage/known_labels.csv")
    
    print(f"Scanning archives in {raw_dir}...")
    labels = process_archives(raw_dir)
    print(f"Found {len(labels)} images")
    
    write_known_labels(labels, output_path)
    print(f"Wrote {output_path}")
    
    # Summary stats
    species_known = sum(1 for l in labels if l.species)
    medium_known = sum(1 for l in labels if l.medium)
    temp_known = sum(1 for l in labels if l.temperature)
    print(f"Species known: {species_known}/{len(labels)}")
    print(f"Medium known: {medium_known}/{len(labels)}")
    print(f"Temperature known: {temp_known}/{len(labels)}")
    
    # Per-species breakdown
    from collections import Counter
    sp_counts = Counter(l.species for l in labels if l.species)
    print(f"Species breakdown: {dict(sp_counts)}")


if __name__ == "__main__":
    main()