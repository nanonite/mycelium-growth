"""Reproduce the MyceliumSeg SegFormer-B0 baseline (Chainlink #28).

The paper (Yuan et al., *Scientific Data* 2025, doi:10.1038/s41597-025-06265-1)
trains SegFormer-B0 on the 457-image *G. lucidum* trainset for 50,000 iterations
on four RTX 4090s and reports Mycelium F1 / IoU, Boundary IoU, HD95 and ASSD on
the 50-image testset (their Table 6). This driver reproduces that protocol with
the repository's pinned stack, which is CPU-only (see "Known limitations" in
the README), so the run is a *short* training run with the authors' hyper
parameters scaled only where a CPU budget forces it; every deviation is printed
by ``config`` and recorded in notes/segformer_baseline.md.

Subcommands (each is idempotent and logs to ``work_dirs/segformer-b0-repro/``):

    prepare    extract the labeled release zips into the layout the benchmark
               expects (data/segmentation/{train,test,conditions}/{image,mask})
    config     render the training config and the conditions-eval config
    train      run the benchmark's train.py on the prepared data
    evaluate   run the benchmark's test.py, parse the printed metric tables
    evidence   assemble environment, protocol, training, checkpoint, metrics
               and data hashes into <work>/evidence.json for hand-off
    run        prepare -> config -> train -> evaluate -> evidence

Run it from inside ``nix develop``; ``train``/``evaluate`` need mmcv/mmseg, the
other subcommands only need the standard library:

    nix develop --command python scripts/segformer_repro.py run --smoke
    nix develop --command python scripts/segformer_repro.py run

Exit codes: 0 ok, 1 a step failed, 2 the Python environment cannot run it.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCHMARK = REPO / "external" / "MyceliumSeg-benchmark"
BASE_CONFIG = BENCHMARK / "mycelium_model" / "model" / "segformer.b0.1024x1024.mycelium.507.py"
ARCHIVES = REPO / "data" / "raw" / "myceliumseg"
SEG_ROOT = REPO / "data" / "segmentation"
PRETRAINED_DIR = REPO / "data" / "pretrained"
PRETRAINED_CKPT = PRETRAINED_DIR / "mit_b0.pth"
PRETRAINED_URL = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/"
    "segformer/mit_b0_20220624-7e0fe6dd.pth"
)
SHIM_DIR = REPO / "scripts" / "segformer_shim"
WORK_ROOT = REPO / "work_dirs" / "segformer-b0-repro"

# The benchmark vendors its own mmseg fork (0.11.0 + the mit_b* backbones);
# like the benchmark's own scripts, put its tree first so that it shadows the
# packaged mmsegmentation whenever this driver imports mmseg itself (the
# pretrained-backbone verification below). train/eval subprocesses get the
# same shadowing for free, because sys.path[0] is the script's directory.
sys.path.insert(0, str(BENCHMARK))

# The five metrics the benchmark's script_inference.sh and the paper report.
EVAL_METRICS = ("mIoU", "mDice", "assd", "hd95", "biou")

# Where each labeled zip feeds in. The release ships a GL train/test split and
# six 10-image condition sets; the paper trains on trainset, reports Table 6 on
# testset and evaluates the conditions separately as robustness (Table 7), so
# the same three-way split is kept here.
SPLIT_SOURCES: dict[str, tuple[tuple[str, str], ...]] = {
    "train": (("labeled-GL.zip", "labeled-GL/trainset"),),
    "test": (("labeled-GL.zip", "labeled-GL/testset"),),
    "conditions": (
        ("labeled-GS_PO_TS.zip", "labeled-GS_PO_TS/GS"),
        ("labeled-GS_PO_TS.zip", "labeled-GS_PO_TS/PO"),
        ("labeled-GS_PO_TS.zip", "labeled-GS_PO_TS/TS"),
        ("labeled-MYG_PDA_TEMP.zip", "MYG"),
        ("labeled-MYG_PDA_TEMP.zip", "PDA"),
        ("labeled-MYG_PDA_TEMP.zip", "TEMP15"),
    ),
}
EXPECTED_COUNTS = {"train": 457, "test": 50, "conditions": 60}
SMOKE_LIMITS = {"train": 4, "test": 2, "conditions": 2}

JUNK_PREFIXES = ("__MACOSX/",)
JUNK_NAMES = {".DS_Store"}


class ReproError(RuntimeError):
    """A step failed with a message that is already user-facing."""


@dataclass(frozen=True)
class Pair:
    """One image/mask pair as it sits inside a labeled release zip."""

    archive: Path
    image_member: str
    mask_member: str
    name: str


def _member_is_junk(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    if base in JUNK_NAMES or not base:
        return True
    return any(name.startswith(prefix) for prefix in JUNK_PREFIXES)


def plan_split(archive: Path, prefix: str) -> list[Pair]:
    """List the image/mask pairs under ``prefix`` inside one zip.

    The two folders must hold the same basenames; anything the release added
    for macOS (``__MACOSX``, ``.DS_Store``) is ignored.
    """
    if not archive.is_file():
        raise ReproError(f"missing archive: {archive}")
    pairs: list[Pair] = []
    with zipfile.ZipFile(archive) as zf:
        images: dict[str, str] = {}
        masks: dict[str, str] = {}
        for member in zf.namelist():
            if _member_is_junk(member):
                continue
            stem, sep, base = member.rpartition("/")
            if not sep:
                continue
            if stem == f"{prefix}/image" and base.lower().endswith((".jpg", ".jpeg")):
                images[base] = member
            elif stem == f"{prefix}/mask" and base.lower().endswith(".png"):
                masks[base] = member
        image_stems = {Path(name).stem for name in images}
        mask_stems = {Path(name).stem for name in masks}
        missing_masks = sorted(image_stems - mask_stems)
        missing_images = sorted(mask_stems - image_stems)
        if missing_masks or missing_images:
            raise ReproError(
                f"{archive.name}:{prefix} is not paired: "
                f"{len(missing_masks)} images without mask, "
                f"{len(missing_images)} masks without image"
                + (f" (e.g. {[m + '.png' for m in missing_masks[:3]]})" if missing_masks else "")
                + (f" (e.g. {[m + '.jpg' for m in missing_images[:3]]})" if missing_images else "")
            )
        for image_name in sorted(images):
            pairs.append(
                Pair(
                    archive=archive,
                    image_member=images[image_name],
                    mask_member=f"{prefix}/mask/{Path(image_name).stem}.png",
                    name=image_name,
                )
            )
    if not pairs:
        raise ReproError(f"{archive.name}:{prefix} contains no images")
    return pairs


def plan_splits(
    archives: Path = ARCHIVES,
    expected: dict[str, int] | None = None,
) -> dict[str, list[Pair]]:
    """Plan every split, verifying the release still has the expected counts."""
    if expected is None:
        expected = EXPECTED_COUNTS
    plan: dict[str, list[Pair]] = {}
    for split, sources in SPLIT_SOURCES.items():
        pairs: list[Pair] = []
        for zip_name, prefix in sources:
            pairs.extend(plan_split(archives / zip_name, prefix))
        pairs.sort(key=lambda pair: pair.name)
        want = expected.get(split)
        if want is not None and len(pairs) != want:
            raise ReproError(
                f"split {split!r} has {len(pairs)} image/mask pairs, expected {want}"
            )
        plan[split] = pairs
    return plan


def _extract_pairs(pairs: list[Pair], split_dir: Path) -> None:
    """Materialise one split as <split_dir>/{image,mask}/<name>."""
    image_dir = split_dir / "image"
    mask_dir = split_dir / "mask"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    by_archive: dict[Path, list[Pair]] = {}
    for pair in pairs:
        by_archive.setdefault(pair.archive, []).append(pair)
    for archive, group in by_archive.items():
        with zipfile.ZipFile(archive) as zf:
            for pair in group:
                if Path(pair.image_member).stem != Path(pair.mask_member).stem:
                    raise ReproError(f"unpaired members in {archive.name}: {pair}")
                for member, target in (
                    (pair.image_member, image_dir / pair.name),
                    (pair.mask_member, mask_dir / f"{Path(pair.name).stem}.png"),
                ):
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)


def prepare_dataset(
    root: Path,
    archives: Path = ARCHIVES,
    expected: dict[str, int] | None = None,
    smoke: bool = False,
    force: bool = False,
) -> dict:
    """Extract (or verify) the labeled data below ``root``.

    Returns the manifest that was written. A second call with the same inputs
    is a no-op unless ``force`` is set.
    """
    plan = plan_splits(archives=archives, expected=expected)
    if smoke:
        plan = {
            split: pairs[: SMOKE_LIMITS.get(split, len(pairs))]
            for split, pairs in plan.items()
        }
    manifest = {
        "smoke": smoke,
        "sources": sorted({pair.archive.name for pairs in plan.values() for pair in pairs}),
        "splits": {split: [pair.name for pair in pairs] for split, pairs in plan.items()},
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not force:
        try:
            current = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            current = None
        if current == manifest:
            counts = {split: len(pairs) for split, pairs in plan.items()}
            print(f"dataset up to date under {root} ({counts})")
            return manifest
    for split, pairs in plan.items():
        _extract_pairs(pairs, root / split)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    counts = {split: len(pairs) for split, pairs in plan.items()}
    print(f"prepared {root} {counts}")
    return manifest


def warmup_iters_for(max_iters: int) -> int:
    """The authors' 3,000-iteration warmup, scaled down for short runs.

    Full-length runs (50,000 iterations) keep the published value; a run of N
    iterations warms up for min(3000, N // 5) so that the polynomial decay
    still gets a trained model, not a model that spent the run warming up.
    """
    return min(3000, max(1, max_iters // 5))


def render_config(
    data_root: Path,
    max_iters: int,
    warmup_iters: int,
    pretrained: Path | None,
    log_interval: int,
    samples_per_gpu: int | None = None,
    workers_per_gpu: int | None = None,
) -> str:
    """Config that layers this repository's settings on the authors' config.

    ``samples_per_gpu``/``workers_per_gpu`` default to the authors' 4/2 when
    ``None``; passing them lets a single-GPU run keep the published *effective*
    batch (4 GPUs x 4) or trade workers for data-loading throughput, with the
    deviation recorded by ``evidence``.
    """
    pretrained_repr = repr(str(pretrained)) if pretrained else "None"
    data_lines = []
    if samples_per_gpu is not None:
        data_lines.append(f"    samples_per_gpu={samples_per_gpu},\n")
    if workers_per_gpu is not None:
        data_lines.append(f"    workers_per_gpu={workers_per_gpu},\n")
    data_lines.append(f"    train=dict(dataset=dict(data_root={str(data_root)!r})),\n")
    data_lines.append(f"    val=dict(data_root={str(data_root)!r}),\n")
    data_lines.append(f"    test=dict(data_root={str(data_root)!r}),\n")
    return (
        "# Generated by scripts/segformer_repro.py config - do not edit.\n"
        "# Base: the authors' SegFormer-B0 config from the benchmark repo.\n"
        f"_base_ = [{str(BASE_CONFIG)!r}]\n"
        "\n"
        "# The released config points at ./pretrained/mit_b0.pth, which is not\n"
        "# shipped; use the downloaded ImageNet-pretrained MiT-B0 or random init.\n"
        f"model = dict(pretrained={pretrained_repr})\n"
        "\n"
        "data = dict(\n"
        + "".join(data_lines)
        + ")\n"
        "\n"
        f"runner = dict(type='IterBasedRunner', max_iters={max_iters})\n"
        f"lr_config = dict(warmup_iters={warmup_iters})\n"
        f"checkpoint_config = dict(by_epoch=False, interval={max_iters})\n"
        f"evaluation = dict(interval={max_iters}, metric='mIoU')\n"
        f"log_config = dict(interval={log_interval})\n"
        "workflow = [('train', 1)]\n"
    )


def render_conditions_config(config_path: Path) -> str:
    """Eval-only config that swaps the test split for the condition hold-outs."""
    return (
        "# Generated by scripts/segformer_repro.py config - do not edit.\n"
        f"_base_ = [{str(config_path)!r}]\n"
        "\n"
        "data = dict(test=dict(img_dir='conditions/image', ann_dir='conditions/mask'))\n"
    )


def download_pretrained(dest: Path = PRETRAINED_CKPT) -> Path | None:
    """Fetch the ImageNet-pretrained MiT-B0 backbone; None if unavailable."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    print(f"downloading {PRETRAINED_URL}")
    try:
        with urllib.request.urlopen(PRETRAINED_URL, timeout=60) as resp, open(part, "wb") as out:
            shutil.copyfileobj(resp, out)
        if part.stat().st_size < 1_000_000:
            raise OSError(f"only {part.stat().st_size} bytes")
    except OSError as exc:
        part.unlink(missing_ok=True)
        print(f"warning: pretrained backbone unavailable ({exc}); training from scratch")
        return None
    part.replace(dest)
    print(f"saved {dest} ({dest.stat().st_size} bytes)")
    return dest


# Key renames from the downloadable MiT-B0 (timm-style, ``layers....``) to the
# benchmark's fork of the SegFormer backbone (``patch_embed*``/``block*``).
_TIMM_TO_FORK = (
    (r"layers\.(\d)\.0\.projection\.(weight|bias)", r"patch_embed{stage}.proj.{rest}"),
    (r"layers\.(\d)\.0\.norm\.(weight|bias)", r"patch_embed{stage}.norm.{rest}"),
    (r"layers\.(\d)\.2\.(weight|bias)", r"norm{stage}.{rest}"),
    (r"layers\.(\d)\.1\.(\d+)\.norm([12])\.(weight|bias)", r"block{stage}.{block}.norm{norm}.{rest}"),
    (r"layers\.(\d)\.1\.(\d+)\.attn\.attn\.out_proj\.(weight|bias)", r"block{stage}.{block}.attn.proj.{rest}"),
    (r"layers\.(\d)\.1\.(\d+)\.attn\.sr\.(weight|bias)", r"block{stage}.{block}.attn.sr.{rest}"),
    (r"layers\.(\d)\.1\.(\d+)\.attn\.norm\.(weight|bias)", r"block{stage}.{block}.attn.norm.{rest}"),
    (r"layers\.(\d)\.1\.(\d+)\.ffn\.layers\.0\.(weight|bias)", r"block{stage}.{block}.mlp.fc1.{rest}"),
    (r"layers\.(\d)\.1\.(\d+)\.ffn\.layers\.1\.(weight|bias)", r"block{stage}.{block}.mlp.dwconv.dwconv.{rest}"),
    (r"layers\.(\d)\.1\.(\d+)\.ffn\.layers\.4\.(weight|bias)", r"block{stage}.{block}.mlp.fc2.{rest}"),
)
_QKV = r"layers\.(\d)\.1\.(\d+)\.attn\.attn\.in_proj_(weight|bias)"


def map_timm_to_fork(state: dict) -> dict:
    """Rewrite a timm-format MiT-B0 state dict into the benchmark's names.

    The only structural difference is the attention projection: upstream
    stores torch-MultiheadAttention's fused ``[Q;K;V]`` ``in_proj``, while the
    fork splits it into ``q`` (dim rows) and ``kv`` (the remaining 2*dim rows,
    K before V - torch's documented ordering). Everything else is a rename
    plus flattening the 1x1 convs of the Mix-FFN into the Linear weights the
    fork uses. Unrecognised keys are an error, never a silent drop.
    """
    out: dict = {}
    for key, tensor in state.items():
        match = re.fullmatch(_QKV, key)
        if match:
            stage, block, kind = int(match.group(1)) + 1, match.group(2), match.group(3)
            # torch's MHA stores [Q;K;V] stacked on dim 0 of the weight and
            # of the bias alike: 96 = 3 * 32 rows / entries for mit_b0's
            # smallest stage.
            dim = tensor.shape[1] if kind == "weight" else tensor.shape[0] // 3
            out[f"block{stage}.{block}.attn.q.{kind}"] = tensor[:dim]
            out[f"block{stage}.{block}.attn.kv.{kind}"] = tensor[dim:]
            continue
        for pattern, template in _TIMM_TO_FORK:
            match = re.fullmatch(pattern, key)
            if not match:
                continue
            groups = match.groups()
            stage = int(groups[0]) + 1
            rest = groups[-1]
            names = {"stage": stage, "rest": rest}
            if len(groups) == 4:
                names.update(block=groups[1], norm=groups[2])
            elif len(groups) == 3:
                names.update(block=groups[1])
            target = template.format(**names)
            if target.endswith(("fc1.weight", "fc2.weight")):
                tensor = tensor.reshape(tensor.shape[0], tensor.shape[1])
            out[target] = tensor
            break
        else:
            raise ReproError(f"unrecognised pretrained key: {key}")
    return out


def convert_pretrained(verify: bool = True) -> Path | None:
    """Download (if needed), remap and optionally strict-load-verify MiT-B0.

    Returns the path of the converted checkpoint, or None when torch is not
    importable (prepare/config can run outside the Nix shell) or the download
    failed. ``verify`` builds the benchmark's backbone and loads the result
    with ``strict=True``, so a mapping bug can never pass silently.
    """
    src = download_pretrained()
    if src is None:
        return None
    dest = PRETRAINED_DIR / "mit_b0.converted.pth"
    try:
        import torch
    except ImportError:
        print("warning: torch not importable; cannot convert the pretrained backbone")
        return None
    try:
        raw = torch.load(src, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 has no weights_only
        raw = torch.load(src, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    converted = map_timm_to_fork(raw)
    if verify:
        try:
            import mmseg.models.backbones  # noqa: F401 - registers mit_b*
            from mmcv import Config
            from mmseg.models.builder import build_backbone

            backbone_cfg = Config.fromfile(str(BASE_CONFIG)).model["backbone"]
            build_backbone(backbone_cfg).load_state_dict(converted, strict=True)
        except ImportError:
            print("warning: mmseg not importable; converted checkpoint not verified")
        except Exception as exc:  # report any mapping failure verbatim
            raise ReproError(f"converted pretrained backbone failed to load: {exc}") from exc
    torch.save(converted, dest)
    print(f"converted {src.name} -> {dest.name} ({len(converted)} tensors)")
    return dest


def work_dir(smoke: bool, tag: str | None = None) -> Path:
    """Work directory for a run; ``tag`` separates variants (e.g. ``gpu``)."""
    name = "smoke" if smoke else "full"
    if tag:
        name = f"{name}-{tag}"
    return WORK_ROOT / name


def child_env() -> dict[str, str]:
    """Environment for the benchmark subprocesses.

    OMP_NUM_THREADS=1 works around the nixpkgs-torch OpenMP deadlock described
    in README "Known limitations"; PYTHONPATH activates the compatibility shim
    in scripts/segformer_shim (numpy alias + CPU module.cuda).
    """
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SHIM_DIR) + (os.pathsep + existing if existing else "")
    return env


def require_mmcv() -> None:
    if importlib.util.find_spec("mmcv") is None:
        raise ReproError("mmcv not importable; run this inside `nix develop`")


def run_logged(argv: list[str], log_path: Path) -> int:
    """Run one child process, teeing all of its output into ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(part) for part in argv)
    print(f"+ {printable}\n  -> {log_path}", flush=True)
    started = time.monotonic()
    with open(log_path, "w") as log:
        log.write(f"+ {printable}\n\n")
        log.flush()
        proc = subprocess.run(
            argv,
            cwd=REPO,
            env=child_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f"  exit {proc.returncode} after {time.monotonic() - started:.0f}s", flush=True)
    return proc.returncode


def latest_checkpoint(train_dir: Path) -> Path | None:
    """The highest-iteration checkpoint in a train.py work directory."""
    if not train_dir.is_dir():
        return None
    numbered: list[tuple[int, Path]] = []
    for path in train_dir.glob("*.pth"):
        match = re.fullmatch(r"iter_(\d+)\.pth", path.name)
        numbered.append((int(match.group(1)) if match else -1, path))
    if not numbered:
        return None
    return max(numbered)[1]


def _max_iters(config_path: Path) -> int:
    """Read ``runner.max_iters`` back out of a rendered config."""
    match = re.search(r"max_iters=(\d+)", config_path.read_text())
    if match is None:
        raise ReproError(f"no max_iters in {config_path}; re-run `config`")
    return int(match.group(1))


def _parse_ascii_table(section: str) -> list[list[str]]:
    """Rows of a terminaltables ASCII table as stripped cell lists."""
    rows: list[list[str]] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        rows.append([cell.strip() for cell in stripped.strip("|").split("|")])
    return rows


def parse_eval_log(text: str) -> dict:
    """Extract the metrics the benchmark printed.

    Returns ``{"aggregate": {...native units...}, "per_class": [[{...}], ...]}``
    where per-class numbers are divided back out of the x100 the benchmark's
    table applies. Raises ValueError when neither the summary dict nor the
    tables are present, so a silent parse failure cannot look like a result.
    """
    aggregate: dict[str, float] = {}
    dict_match = re.search(r"Built-in metrics:\s*(\{[^{}]*\})", text)
    if dict_match:
        try:
            raw = ast.literal_eval(dict_match.group(1))
        except (SyntaxError, ValueError):
            raw = None
        if isinstance(raw, dict):
            for key, value in raw.items():
                try:
                    aggregate[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

    if not aggregate and "Summary:" in text:
        summary_section = text.split("Summary:", 1)[1]
        rows = _parse_ascii_table(summary_section)
        if len(rows) >= 2:
            for column, cell in zip(rows[0][1:], rows[1][1:]):
                try:
                    aggregate[column] = float(cell) / 100.0
                except ValueError:
                    continue

    per_class: list[dict[str, float]] = []
    marker = "per class results:"
    if marker in text:
        class_section = text.split(marker, 1)[1].split("Summary:", 1)[0]
        rows = _parse_ascii_table(class_section)
        if len(rows) >= 2:
            header = rows[0]
            for row in rows[1:]:
                if len(row) != len(header) or not row[0]:
                    continue
                entry: dict[str, float] = {}
                for column, cell in zip(header[1:], row[1:]):
                    try:
                        entry[column] = float(cell) / 100.0
                    except ValueError:
                        continue
                entry["class"] = row[0]  # type: ignore[assignment]
                per_class.append(entry)
    if not per_class and not aggregate:
        raise ValueError("no evaluation tables found in the log")
    return {"aggregate": aggregate, "per_class": per_class}


def headline_metrics(per_class: list[dict[str, float]]) -> dict[str, float]:
    """The paper's five reported numbers (Table 6) from the per-class table.

    The class table holds the foreground row second ('mycelium'); values are
    ratios for IoU/Dice/Boundary-IoU and pixels for HD95/ASSD.
    """
    if len(per_class) < 2:
        return {}
    foreground = per_class[1]
    wanted = {
        "mycelium_f1": "Dice",
        "mycelium_iou": "IoU",
        "boundary_iou": "iou",
        "hd95_px": "d95",
        "assd_px": "ssd",
    }
    return {
        name: float(foreground[column])
        for name, column in wanted.items()
        if column in foreground
    }


def _write_metrics(path: Path, payload: dict) -> None:
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing[payload["split"]] = payload
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")


def _split_image_count(data_root: Path, split: str) -> int:
    image_dir = data_root / split / "image"
    return len(list(image_dir.glob("*.jpg"))) if image_dir.is_dir() else 0


# --------------------------------------------------------------------------
# evidence collection (Chainlink #57 / #59)
# --------------------------------------------------------------------------

_MEMORY_RE = re.compile(r"memory: (\d+)")
_TIME_RE = re.compile(r"time: ([0-9.]+), data_time: ([0-9.]+)")
_LOSS_RE = re.compile(r"\bloss: ([0-9.eE+-]+)\b")
_ITER_RE = re.compile(r"Iter \[(\d+)/(\d+)\]")


def parse_train_log(text: str) -> dict:
    """Summarise one train.py log: iterations, speed, peak memory, final loss.

    Only the ``TextLoggerHook`` lines are read, so a log without them yields an
    empty summary rather than a fabricated one. ``memory`` is only printed when
    torch reports CUDA, which is exactly the GPU evidence we want to capture.
    """
    stats: dict = {"iterations": 0, "max_iters": None}
    times: list[float] = []
    data_times: list[float] = []
    memories: list[int] = []
    losses: list[float] = []
    for line in text.splitlines():
        match = _ITER_RE.search(line)
        if match:
            stats["iterations"] = int(match.group(1))
            stats["max_iters"] = int(match.group(2))
        match = _TIME_RE.search(line)
        if match:
            times.append(float(match.group(1)))
            data_times.append(float(match.group(2)))
        match = _MEMORY_RE.search(line)
        if match:
            memories.append(int(match.group(1)))
        match = _LOSS_RE.search(line)
        if match:
            losses.append(float(match.group(1)))
    if times:
        stats["mean_iter_seconds"] = round(sum(times) / len(times), 3)
        stats["mean_data_seconds"] = round(sum(data_times) / len(data_times), 3)
    if memories:
        stats["peak_memory_mb"] = max(memories)
    if losses:
        stats["final_loss"] = losses[-1]
    return stats


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hex digest of a file; streamed so a 45 MB checkpoint stays cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(cwd: Path) -> str | None:
    """Commit at ``cwd``, or None when the checkout has no usable git state."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    head = proc.stdout.strip()
    return head if proc.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", head) else None


def nvidia_smi_info() -> list[dict] | None:
    """``nvidia-smi`` rows for every GPU; None when the tool is unavailable."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    gpus = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        gpus.append(
            {
                "name": parts[0],
                "driver": parts[1],
                "memory_total_mib": int(parts[2]),
                "memory_free_mib": int(parts[3]),
            }
        )
    return gpus or None


def collect_environment() -> dict:
    """Hardware/software record: torch+CUDA when present, else the bare facts.

    Must not raise: ``evidence`` also runs in the CPU shell and in ``.venv``,
    where torch/mmcv are absent - those fields simply come back null, which the
    reader has to treat as "not measured here" rather than "no CUDA".
    """
    env: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "torch": None,
        "torch_cuda_build": None,
        "cuda_available": False,
        "cuda_device": None,
        "mmcv": None,
        "mmseg": None,
        "gpus": nvidia_smi_info(),
    }
    try:
        import torch
    except ImportError:
        return env
    env["torch"] = torch.__version__
    env["torch_cuda_build"] = torch.version.cuda
    env["cuda_available"] = bool(torch.cuda.is_available())
    if env["cuda_available"]:
        env["cuda_device"] = torch.cuda.get_device_name(0)
    for name in ("mmcv", "mmseg"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        env[name] = getattr(module, "__version__", "unknown")
    return env


def read_protocol(config_path: Path) -> dict:
    """Protocol facts read back out of the rendered config, never re-typed.

    Values the render leaves at the authors' defaults (batch size, workers)
    live in the base config chain, so those are looked up there as a fallback
    - the record must say what actually ran, not what the CLI was asked for.
    """
    text = config_path.read_text()
    chain = [text]
    for extra in (BASE_CONFIG, BASE_CONFIG.parent.parent / "dataset" / "EPA_mycelium.py"):
        if extra.is_file():
            chain.append(extra.read_text())
    protocol: dict = {"config": str(config_path), "config_sha256": sha256_file(config_path)}
    patterns = {
        "max_iters": r"max_iters=(\d+)",
        "warmup_iters": r"warmup_iters=(\d+)",
        "log_interval": r"log_config = dict\(interval=(\d+)\)",
        "samples_per_gpu": r"samples_per_gpu=(\d+)",
        "workers_per_gpu": r"workers_per_gpu=(\d+)",
    }
    for key, pattern in patterns.items():
        for chunk in chain:
            match = re.search(pattern, chunk)
            if match:
                protocol[key] = int(match.group(1))
                break
    base = re.search(r"_base_ = \[('|\")([^'\"]+)('|\")\]", text)
    if base:
        protocol["base_config"] = base.group(2)
    return protocol


def collect_evidence(work: Path, data_root: Path) -> dict:
    """Everything a reviewer needs to re-run and judge one work directory."""
    evidence: dict = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "work_dir": str(work),
        "environment": collect_environment(),
        "repository_commit": _git_head(REPO),
        "benchmark_commit": _git_head(BENCHMARK),
        "commands": [
            "nix develop .#gpu --command python scripts/segformer_repro.py run",
            "nix develop .#gpu --command python scripts/segformer_repro.py evidence",
            "nix develop --command python scripts/segformer_repro.py run  # CPU profile",
        ],
    }
    config_path = work / "config.py"
    if config_path.is_file():
        evidence["protocol"] = read_protocol(config_path)
    manifest_path = data_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        evidence["dataset"] = {
            "root": str(data_root),
            "manifest_sha256": sha256_file(manifest_path),
            "counts": {split: len(names) for split, names in manifest["splits"].items()},
        }
    train_log = work / "train" / "train.log"
    if train_log.is_file():
        evidence["training"] = parse_train_log(train_log.read_text())
        evidence["training"]["log"] = str(train_log)
    checkpoint = latest_checkpoint(work / "train")
    if checkpoint is not None:
        evidence["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        }
    metrics_path = work / "metrics.json"
    if metrics_path.is_file():
        raw = json.loads(metrics_path.read_text())
        evidence["metrics"] = {
            split: {
                "images": payload.get("images"),
                "wall_seconds": payload.get("wall_seconds"),
                "headline": payload.get("headline"),
                "aggregate": payload.get("aggregate"),
            }
            for split, payload in raw.items()
        }
        evidence["metrics_file"] = str(metrics_path)
    return evidence


def archive_md5s(manifest_csv: Path) -> dict[str, str]:
    """md5 of each labeled release zip from data/manifest.csv, if recorded."""
    if not manifest_csv.is_file():
        return {}
    wanted = {"labeled-GL.zip", "labeled-GS_PO_TS.zip", "labeled-MYG_PDA_TEMP.zip"}
    found: dict[str, str] = {}
    with open(manifest_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("file") in wanted and row.get("md5"):
                found[row["file"]] = row["md5"]
    return found


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> int:
    root = SEG_ROOT / ("_smoke" if args.smoke else "release")
    prepare_dataset(root, smoke=args.smoke, force=args.force)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    root = SEG_ROOT / ("_smoke" if args.smoke else "release")
    if not (root / "manifest.json").exists():
        raise ReproError(f"no dataset at {root}; run `prepare` first")
    max_iters = args.max_iters
    warmup = args.warmup_iters if args.warmup_iters is not None else warmup_iters_for(max_iters)
    if args.pretrained == "none":
        pretrained: Path | None = None
    elif args.pretrained == "auto":
        pretrained = convert_pretrained()
    else:
        pretrained = Path(args.pretrained)
        if not pretrained.is_file():
            raise ReproError(f"--pretrained file not found: {pretrained}")
    if args.smoke:
        max_iters = min(max_iters, 2)
        warmup = min(warmup, 1)
        # 2 iterations would otherwise fall under the default log interval and
        # leave the evidence without a single Iter line (speed/memory/loss).
        log_interval = 1
    else:
        log_interval = args.log_interval
    work = work_dir(args.smoke, args.tag)
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / "config.py"
    config_path.write_text(
        render_config(
            data_root=root,
            max_iters=max_iters,
            warmup_iters=warmup,
            pretrained=pretrained,
            log_interval=log_interval,
            samples_per_gpu=args.batch_size,
            workers_per_gpu=args.workers,
        )
    )
    conditions_path = work / "config.conditions.py"
    conditions_path.write_text(render_conditions_config(config_path))
    print(f"wrote {config_path} (max_iters={max_iters}, warmup_iters={warmup})")
    print(f"wrote {conditions_path}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    require_mmcv()
    work = work_dir(args.smoke, args.tag)
    config_path = work / "config.py"
    if not config_path.exists():
        raise ReproError(f"{config_path} missing; run `config` first")
    train_dir = work / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    data_root = SEG_ROOT / ("_smoke" if args.smoke else "release")
    argv = [
        sys.executable,
        str(BENCHMARK / "train.py"),
        str(config_path),
        "--work-dir",
        str(train_dir),
        "--seed",
        "42",
        "--deterministic",
        "--no-validate",
    ]
    # Checkpoint every 1/8 of the run (the config keeps only the final one) so
    # an interrupted run resumes below instead of restarting from iteration 0.
    argv += ["--options", f"checkpoint_config.interval={max(1, _max_iters(config_path) // 8)}"]
    resume = latest_checkpoint(train_dir)
    if resume is not None:
        argv += ["--resume-from", str(resume)]
        print(f"resuming from {resume}")
    print(f"training on {_split_image_count(data_root, 'train')} images")
    code = run_logged(argv, train_dir / "train.log")
    if code != 0:
        return code
    checkpoint = latest_checkpoint(train_dir)
    if checkpoint is None:
        raise ReproError(f"training produced no checkpoint in {train_dir}")
    print(f"checkpoint: {checkpoint}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    require_mmcv()
    work = work_dir(args.smoke, args.tag)
    split = args.split
    config_path = work / ("config.conditions.py" if split == "conditions" else "config.py")
    if not config_path.exists():
        raise ReproError(f"{config_path} missing; run `config` first")
    checkpoint = (
        Path(args.checkpoint) if getattr(args, "checkpoint", None) else latest_checkpoint(work / "train")
    )
    if checkpoint is None or not Path(checkpoint).exists():
        raise ReproError("no checkpoint; run `train` first or pass --checkpoint")
    data_root = SEG_ROOT / ("_smoke" if args.smoke else "release")
    images = _split_image_count(data_root, split)
    log_path = work / f"eval_{split}.log"
    results_pkl = work / f"results_{split}.pkl"
    argv = [
        sys.executable,
        str(BENCHMARK / "test.py"),
        str(config_path),
        str(checkpoint),
        "--eval",
        *EVAL_METRICS,
        "--eval-options",
        "efficient_test=False",
        # Decode in-process: with workers, torch's DataLoader hands tensor
        # file descriptors over a resource_sharer socket that dies mid-run on
        # this stack (both splits crashed at image ~3 with FileNotFoundError
        # in rebuild_storage_fd). Decoding costs ~1 s against ~105 s of slide
        # inference per image, so workers buy nothing here anyway.
        "--options",
        "data.workers_per_gpu=0",
        "--out",
        str(results_pkl),
    ]
    print(f"evaluating {images} images of split {split!r}")
    started = time.monotonic()
    code = run_logged(argv, log_path)
    if code != 0:
        return code
    parsed = parse_eval_log(log_path.read_text())
    payload = {
        "split": split,
        "images": images,
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "wall_seconds": round(time.monotonic() - started, 1),
        "aggregate": parsed["aggregate"],
        "headline": headline_metrics(parsed["per_class"]),
        "per_class": parsed["per_class"],
    }
    metrics_path = work / "metrics.json"
    _write_metrics(metrics_path, payload)
    results_pkl.unlink(missing_ok=True)
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    if payload["headline"]:
        print("headline (foreground row, native units):")
        print(" ", json.dumps(payload["headline"], indent=2, sort_keys=True))
    print(f"wrote {metrics_path}")
    return 0


def cmd_evidence(args: argparse.Namespace) -> int:
    work = work_dir(args.smoke, args.tag)
    if not work.is_dir():
        raise ReproError(f"{work} missing; run `config` first")
    data_root = SEG_ROOT / ("_smoke" if args.smoke else "release")
    evidence = collect_evidence(work, data_root)
    evidence["release_archives"] = archive_md5s(REPO / "data" / "manifest.csv")
    output = getattr(args, "output", None)
    out = Path(output) if output else work / "evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: evidence[k] for k in ("environment", "protocol") if k in evidence}, indent=2))
    print(f"wrote {out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    for step in (cmd_prepare, cmd_config, cmd_train, cmd_evaluate):
        step_ns = argparse.Namespace(**vars(args))
        if step is cmd_evaluate:
            step_ns.split = "test"
        code = step(step_ns)
        if code != 0:
            return code
    if not args.smoke:
        eval_ns = argparse.Namespace(**vars(args))
        eval_ns.split = "conditions"
        code = cmd_evaluate(eval_ns)
        if code != 0:
            return code
    return cmd_evidence(argparse.Namespace(**vars(args)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str = "") -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        return p

    def add_common(p: argparse.ArgumentParser, *, with_force: bool = False) -> None:
        """Options shared by every subcommand that reads or writes a work dir."""
        p.add_argument("--smoke", action="store_true", help="tiny subset for a pipeline check")
        p.add_argument(
            "--tag",
            default=None,
            help="work-dir suffix separating run variants (e.g. gpu)",
        )
        if with_force:
            p.add_argument("--force", action="store_true", help="re-extract even if up to date")

    prepare = add("prepare", cmd_prepare, "extract the labeled zips")
    add_common(prepare, with_force=True)

    config = add("config", cmd_config, "render train/eval configs")
    add_common(config)
    config.add_argument("--max-iters", type=int, default=2000, help="iterations for the short run")
    config.add_argument("--warmup-iters", type=int, default=None, help="default: max_iters//5")
    config.add_argument("--log-interval", type=int, default=10)
    config.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="samples_per_gpu (default: the authors' 4)",
    )
    config.add_argument(
        "--workers",
        type=int,
        default=None,
        help="workers_per_gpu (default: the authors' 2)",
    )
    config.add_argument(
        "--pretrained",
        default="auto",
        help="'auto' downloads MiT-B0, 'none' trains from scratch, or a path",
    )

    train = add("train", cmd_train, "run the benchmark's train.py")
    add_common(train)

    evaluate = add("evaluate", cmd_evaluate, "run the benchmark's test.py and parse metrics")
    add_common(evaluate)
    evaluate.add_argument("--split", choices=("test", "conditions"), default="test")
    evaluate.add_argument("--checkpoint", default=None)

    evidence = add("evidence", cmd_evidence, "write environment/protocol/metrics evidence JSON")
    add_common(evidence)
    evidence.add_argument("--output", default=None, help="defaults to <work>/evidence.json")

    run = add("run", cmd_run, "prepare + config + train + evaluate + evidence")
    add_common(run, with_force=True)
    run.add_argument("--max-iters", type=int, default=2000)
    run.add_argument("--warmup-iters", type=int, default=None)
    run.add_argument("--log-interval", type=int, default=10)
    run.add_argument("--batch-size", type=int, default=None)
    run.add_argument("--workers", type=int, default=None)
    run.add_argument("--pretrained", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ReproError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
