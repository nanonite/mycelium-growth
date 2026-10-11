#!/usr/bin/env python3
"""Train a final SegFormer on every unique labeled MyceliumSeg pair.

The benchmark reproduction (#28) preserves its original train/test split.
This script builds a separate all-labeled training set from the three labeled
release archives, collapses exact duplicate image/mask pairs, and trains on
all remaining pairs. Because every labeled pair is used for training, it does
not report a held-out score for the resulting checkpoint; use #28 as the
independently evaluated baseline.

Run from the GPU development shell:

    nix develop .#gpu --command python scripts/segformer_all_labeled.py \
        --tag all-unique --max-iters 50000 --batch-size 16 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import segformer_repro as repro


ALL_LABELED_ROOT = repro.SEG_ROOT / "all-labeled"


def prepare_all_labeled(root: Path = ALL_LABELED_ROOT, *, force: bool = False) -> dict:
    """Create a single training split from all unique released annotation pairs."""
    release_root = repro.SEG_ROOT / "release"
    repro.prepare_dataset(release_root)
    plan = repro.plan_splits()

    records: list[dict] = []
    unique: dict[str, dict] = {}
    for source_split, pairs in plan.items():
        for pair in pairs:
            image_path = release_root / source_split / "image" / pair.name
            mask_path = release_root / source_split / "mask" / f"{Path(pair.name).stem}.png"
            if not image_path.is_file() or not mask_path.is_file():
                raise repro.ReproError(f"missing extracted pair: {image_path}, {mask_path}")
            image_sha = repro.sha256_file(image_path)
            mask_sha = repro.sha256_file(mask_path)
            name = f"{image_sha}.jpg"
            prior = unique.get(image_sha)
            if prior is not None and prior["mask_sha256"] != mask_sha:
                raise repro.ReproError(
                    "identical image bytes have different masks: "
                    f"{prior['source']} and {source_split}/{pair.name}"
                )
            if prior is None:
                unique[image_sha] = {
                    "name": name,
                    "image": image_path,
                    "mask": mask_path,
                    "mask_sha256": mask_sha,
                    "source": f"{source_split}/{pair.name}",
                }
            records.append(
                {
                    "source_split": source_split,
                    "archive": pair.archive.name,
                    "image_member": pair.image_member,
                    "mask_member": pair.mask_member,
                    "image_sha256": image_sha,
                    "mask_sha256": mask_sha,
                    "training_name": name,
                    "duplicate": prior is not None,
                }
            )

    names = sorted(value["name"] for value in unique.values())
    manifest = {
        "kind": "all_unique_labeled_myceliumseg_pairs",
        "sources": sorted({record["archive"] for record in records}),
        "source_record_count": len(records),
        "unique_pair_count": len(unique),
        "duplicate_record_count": len(records) - len(unique),
        "splits": {"train": names},
        "source_records": records,
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not force:
        try:
            current = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            current = None
        images_ok = all((root / "train" / "image" / name).is_file() for name in names)
        masks_ok = all(
            (root / "train" / "mask" / f"{Path(name).stem}.png").is_file()
            for name in names
        )
        if current == manifest and images_ok and masks_ok:
            print(
                f"all-labeled dataset up to date: {len(records)} records, "
                f"{len(unique)} unique pairs"
            )
            return manifest
        raise repro.ReproError(f"{root} exists with different data; pass --force to rebuild")

    if root.exists():
        if not force:
            raise repro.ReproError(
                f"{root} exists without a matching manifest; pass --force to rebuild"
            )
        shutil.rmtree(root)
    image_dir = root / "train" / "image"
    mask_dir = root / "train" / "mask"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for value in unique.values():
        image_link = image_dir / value["name"]
        mask_link = mask_dir / f"{Path(value['name']).stem}.png"
        image_link.symlink_to(os.path.relpath(value["image"], image_dir))
        mask_link.symlink_to(os.path.relpath(value["mask"], mask_dir))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"prepared all labeled data: {len(records)} annotation records, "
        f"{len(unique)} unique image/mask pairs, "
        f"{len(records) - len(unique)} exact duplicates collapsed"
    )
    return manifest


def run(args: argparse.Namespace) -> int:
    repro.require_mmcv()
    manifest = prepare_all_labeled(force=args.force)
    max_iters = 2 if args.smoke else args.max_iters
    warmup_iters = min(
        args.warmup_iters if args.warmup_iters is not None else repro.warmup_iters_for(max_iters),
        max(1, max_iters // 5),
    )
    if args.pretrained == "none":
        pretrained = None
    elif args.pretrained == "auto":
        pretrained = repro.convert_pretrained()
    else:
        pretrained = Path(args.pretrained)
        if not pretrained.is_file():
            raise repro.ReproError(f"--pretrained file not found: {pretrained}")
    root_name = "smoke-all-labeled" if args.smoke else "full-all-labeled"
    if args.tag:
        root_name += f"-{args.tag}"
    work = repro.WORK_ROOT / root_name
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / "config.py"
    config_path.write_text(
        repro.render_config(
            data_root=ALL_LABELED_ROOT,
            max_iters=max_iters,
            warmup_iters=warmup_iters,
            pretrained=pretrained,
            log_interval=1 if args.smoke else args.log_interval,
            samples_per_gpu=args.batch_size,
            workers_per_gpu=args.workers,
        )
    )

    train_dir = work / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        str(repro.BENCHMARK / "train.py"),
        str(config_path),
        "--work-dir",
        str(train_dir),
        "--seed",
        "42",
        "--deterministic",
        "--no-validate",
        "--options",
        f"checkpoint_config.interval={max(1, max_iters // 8)}",
    ]
    resume = repro.latest_checkpoint(train_dir)
    if resume is not None:
        argv += ["--resume-from", str(resume)]
        print(f"resuming from {resume}")
    print(
        f"training on {manifest['unique_pair_count']} unique labeled pairs "
        f"from {manifest['source_record_count']} annotation records"
    )
    code = repro.run_logged(argv, train_dir / "train.log")
    if code != 0:
        return code
    checkpoint = repro.latest_checkpoint(train_dir)
    if checkpoint is None:
        raise repro.ReproError(f"training produced no checkpoint in {train_dir}")

    evidence = repro.collect_evidence(work, ALL_LABELED_ROOT)
    evidence["training_scope"] = {
        "annotation_records": manifest["source_record_count"],
        "unique_image_mask_pairs": manifest["unique_pair_count"],
        "duplicate_records_collapsed": manifest["duplicate_record_count"],
        "unlabeled_release_images_used": False,
        "held_out_evaluation": "not run; every released labeled pair is in training",
        "reference_evaluation": "the separate #28 benchmark baseline",
    }
    evidence["release_archives"] = repro.archive_md5s(repro.REPO / "data" / "manifest.csv")
    evidence_path = work / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"checkpoint: {checkpoint}")
    print(f"evidence: {evidence_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="train for 2 iterations as a pipeline check")
    parser.add_argument("--tag", default=None, help="unique run suffix; use a fresh tag for each run")
    parser.add_argument("--force", action="store_true", help="rebuild the generated all-labeled data link tree")
    parser.add_argument("--max-iters", type=int, default=50000)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pretrained", default="auto", help="auto or none")
    parser.set_defaults(func=run)
    return parser


if __name__ == "__main__":
    try:
        parsed = build_parser().parse_args()
        raise SystemExit(parsed.func(parsed))
    except repro.ReproError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
