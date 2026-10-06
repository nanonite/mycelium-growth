"""Unit tests for the SegFormer baseline reproduction driver (Chainlink #28).

The driver itself needs mmcv and torch (it only shells out to them), so what
is tested here is everything that can be wrong *without* a GPU: how the
labeled release zips become the benchmark's directory layout, how the config
text is rendered, how the metric tables are parsed back out of test.py's
output, and that the compatibility shim really fires. Real training/eval runs
are covered by the smoke invocation documented in notes/segformer_baseline.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import segformer_repro as repro  # needs the sys.path line above

# What the fixture below builds, per split, once every prefix is counted.
FAKE_COUNTS = {"train": 4, "test": 3, "conditions": 7}


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)


def build_fake_archives(root: Path) -> Path:
    """Mimic the three labeled release zips, junk files included."""
    root.mkdir(parents=True, exist_ok=True)
    gl: dict[str, bytes] = {}
    for index in range(1, 5):  # trainset 457 -> 4
        stem = f"{index:08d}"
        gl[f"labeled-GL/trainset/image/{stem}.jpg"] = b"img" + bytes([index])
        gl[f"labeled-GL/trainset/mask/{stem}.png"] = b"mask" + bytes([index])
    for index in range(5, 8):  # testset 50 -> 3
        stem = f"{index:08d}"
        gl[f"labeled-GL/testset/image/{stem}.jpg"] = b"img" + bytes([index])
        gl[f"labeled-GL/testset/mask/{stem}.png"] = b"mask" + bytes([index])
    gl["__MACOSX/._junk"] = b"junk"
    gl["labeled-GL/trainset/.DS_Store"] = b"junk"
    _write_zip(root / "labeled-GL.zip", gl)

    species: dict[str, bytes] = {}
    index = 100
    for folder, count in (("GS", 2), ("PO", 1), ("TS", 1)):
        for _ in range(count):
            index += 1
            stem = f"{index:08d}"
            species[f"labeled-GS_PO_TS/{folder}/image/{stem}.jpg"] = b"img"
            species[f"labeled-GS_PO_TS/{folder}/mask/{stem}.png"] = b"mask"
    _write_zip(root / "labeled-GS_PO_TS.zip", species)

    conditions: dict[str, bytes] = {}
    index = 200
    for folder in ("MYG", "PDA", "TEMP15"):
        index += 1
        stem = f"{index:08d}"
        conditions[f"{folder}/image/{stem}.jpg"] = b"img"
        conditions[f"{folder}/mask/{stem}.png"] = b"mask"
    _write_zip(root / "labeled-MYG_PDA_TEMP.zip", conditions)
    return root


@pytest.fixture()
def archives(tmp_path: Path) -> Path:
    return build_fake_archives(tmp_path / "zips")


def test_plan_splits_counts_pairing_and_order(archives: Path) -> None:
    plan = repro.plan_splits(archives=archives, expected=FAKE_COUNTS)
    assert {split: len(pairs) for split, pairs in plan.items()} == FAKE_COUNTS
    for pairs in plan.values():
        names = [pair.name for pair in pairs]
        assert names == sorted(names)
        for pair in pairs:
            assert Path(pair.image_member).stem == Path(pair.mask_member).stem
            assert pair.mask_member.endswith(".png")


def test_plan_splits_rejects_wrong_count(archives: Path) -> None:
    with pytest.raises(repro.ReproError, match="expected"):
        repro.plan_splits(archives=archives, expected={"train": 5})


def test_plan_split_rejects_unpaired_mask(archives: Path) -> None:
    members = {
        "labeled-GL/testset/image/00000009.jpg": b"img",
        "labeled-GL/testset/mask/00000010.png": b"mask",
    }
    _write_zip(archives / "unpaired.zip", members)
    with pytest.raises(repro.ReproError, match="not paired"):
        repro.plan_split(archives / "unpaired.zip", "labeled-GL/testset")


def test_prepare_extracts_layout_and_manifest(archives: Path, tmp_path: Path) -> None:
    root = tmp_path / "seg"
    manifest = repro.prepare_dataset(root, archives=archives, expected=FAKE_COUNTS)
    for split, count in FAKE_COUNTS.items():
        assert len(list((root / split / "image").glob("*.jpg"))) == count
        assert len(list((root / split / "mask").glob("*.png"))) == count
    assert json.loads((root / "manifest.json").read_text()) == manifest
    # The image bytes must have travelled through the zip untouched.
    sample = next((root / "train" / "image").glob("*.jpg"))
    assert sample.read_bytes() == b"img" + bytes([int(sample.stem)])

    # Idempotent: same inputs -> same manifest, no re-extraction needed.
    sample.unlink()
    again = repro.prepare_dataset(root, archives=archives, expected=FAKE_COUNTS)
    assert again == manifest
    assert not sample.exists(), "up-to-date prepare must not re-extract"
    # ... unless forced.
    repro.prepare_dataset(root, archives=archives, expected=FAKE_COUNTS, force=True)
    assert sample.exists()


def test_prepare_smoke_limits_each_split(archives: Path, tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    manifest = repro.prepare_dataset(
        root, archives=archives, expected=FAKE_COUNTS, smoke=True
    )
    assert manifest["smoke"] is True
    assert manifest["splits"]["train"] == [f"{i:08d}.jpg" for i in range(1, 5)]
    assert manifest["splits"]["test"] == [f"{i:08d}.jpg" for i in range(5, 7)]
    assert len(manifest["splits"]["conditions"]) == repro.SMOKE_LIMITS["conditions"]


def test_warmup_iters_for() -> None:
    # A full run keeps the published 3,000-iteration warmup.
    assert repro.warmup_iters_for(50_000) == 3000
    assert repro.warmup_iters_for(15_000) == 3000
    # Short runs scale it down so the model trains before the decay ends.
    assert repro.warmup_iters_for(2000) == 400
    assert repro.warmup_iters_for(10) == 2
    assert repro.warmup_iters_for(4) == 1


def test_render_config_has_every_override(tmp_path: Path) -> None:
    text = repro.render_config(
        data_root=tmp_path / "seg",
        max_iters=2000,
        warmup_iters=400,
        pretrained=None,
        log_interval=10,
    )
    assert str(repro.BASE_CONFIG) in text
    assert f"data_root={str(tmp_path / 'seg')!r}" in text
    assert "max_iters=2000" in text
    assert "warmup_iters=400" in text
    assert "pretrained=None" in text
    assert "interval=2000" in text  # only the final checkpoint is kept
    # Authors' defaults stay in the base config unless a run overrides them.
    assert "samples_per_gpu" not in text
    assert "workers_per_gpu" not in text

    with_pretrained = repro.render_config(
        data_root=tmp_path / "seg",
        max_iters=50_000,
        warmup_iters=3000,
        pretrained=tmp_path / "mit_b0.pth",
        log_interval=50,
    )
    assert repr(str(tmp_path / "mit_b0.pth")) in with_pretrained
    assert "max_iters=50000" in with_pretrained


def test_render_config_overrides_batch_and_workers(tmp_path: Path) -> None:
    text = repro.render_config(
        data_root=tmp_path / "seg",
        max_iters=60,
        warmup_iters=3000,
        pretrained=None,
        log_interval=10,
        samples_per_gpu=16,
        workers_per_gpu=8,
    )
    assert "samples_per_gpu=16" in text
    assert "workers_per_gpu=8" in text
    assert text.count("samples_per_gpu") == 1


def test_work_dir_tag_separates_run_variants() -> None:
    assert repro.work_dir(False) == repro.WORK_ROOT / "full"
    assert repro.work_dir(True) == repro.WORK_ROOT / "smoke"
    assert repro.work_dir(False, "gpu") == repro.WORK_ROOT / "full-gpu"
    assert repro.work_dir(True, "gpu") == repro.WORK_ROOT / "smoke-gpu"
    # Untagged work dirs must not move, or earlier runs lose their evidence.
    assert repro.work_dir(False, None) == repro.work_dir(False)


def test_render_conditions_config_points_at_holdouts(tmp_path: Path) -> None:
    text = repro.render_conditions_config(tmp_path / "config.py")
    assert str(tmp_path / "config.py") in text
    assert "img_dir='conditions/image'" in text
    assert "ann_dir='conditions/mask'" in text


def test_max_iters_read_back_from_rendered_config(tmp_path: Path) -> None:
    config = tmp_path / "config.py"
    config.write_text(
        repro.render_config(
            data_root=tmp_path / "seg",
            max_iters=1200,
            warmup_iters=240,
            pretrained=None,
            log_interval=10,
        )
    )
    assert repro._max_iters(config) == 1200
    config.write_text("runner = dict(type='IterBasedRunner')\n")
    with pytest.raises(repro.ReproError, match="no max_iters"):
        repro._max_iters(config)


SAMPLE_EVAL_LOG = """
Evaluating: ['mIoU', 'mDice', 'assd', 'hd95', 'biou']
ret_metrics: [array(0.975), array([0.99, 0.95])]
per class results:

+------------+-------+-------+---------+---------+-------+-------+
| Class      | IoU   | Dice  | ssd     | d95     | iou   | Acc   |
+------------+-------+-------+---------+---------+-------+-------+
| background | 99.51 | 99.75 | 1020.35 | 4812.00 | 55.12 | 99.83 |
| mycelium   | 93.10 | 96.43 | 1528.40 | 7595.00 | 28.60 | 95.22 |
+------------+-------+-------+---------+---------+-------+-------+
Summary:

+--------+-------+-------+---------+---------+-------+-------+-------+
| Scope  | mIoU  | mDice | mssd    | md95    | miou  | mAcc  | aAcc  |
+--------+-------+-------+---------+---------+-------+-------+-------+
| global | 96.31 | 98.09 | 1324.38 | 6203.50 | 41.86 | 97.53 | 97.53 |
+--------+-------+-------+---------+---------+-------+-------+-------+
>> Built-in metrics: {'mIoU': 0.9631, 'mDice': 0.9809, 'mssd': 13.2438, 'md95': 62.035, 'miou': 0.4186, 'mAcc': 0.9753, 'aAcc': 0.9753}
"""


def test_parse_eval_log_extracts_tables_and_dict() -> None:
    parsed = repro.parse_eval_log(SAMPLE_EVAL_LOG)
    assert parsed["aggregate"]["mIoU"] == pytest.approx(0.9631)
    assert parsed["aggregate"]["mssd"] == pytest.approx(13.2438)
    assert len(parsed["per_class"]) == 2
    headline = repro.headline_metrics(parsed["per_class"])
    assert headline["mycelium_iou"] == pytest.approx(0.9310)
    assert headline["mycelium_f1"] == pytest.approx(0.9643)
    assert headline["boundary_iou"] == pytest.approx(0.2860)
    assert headline["hd95_px"] == pytest.approx(75.95)
    assert headline["assd_px"] == pytest.approx(15.284)


def test_parse_eval_log_falls_back_to_summary_table() -> None:
    without_dict = SAMPLE_EVAL_LOG.split(">> Built-in metrics:")[0]
    parsed = repro.parse_eval_log(without_dict)
    assert parsed["aggregate"]["mIoU"] == pytest.approx(0.9631)
    assert parsed["aggregate"]["md95"] == pytest.approx(62.035)


def test_parse_eval_log_rejects_a_log_without_results() -> None:
    with pytest.raises(ValueError, match="no evaluation tables"):
        repro.parse_eval_log("Traceback: everything exploded\n")


def _timm_sample() -> dict:
    """One block worth of timm-format keys, shapes as in mit_b0's stage 1."""
    import numpy as np

    return {
        "layers.0.0.projection.weight": np.zeros((32, 3, 7, 7)),
        "layers.0.0.projection.bias": np.zeros(32),
        "layers.0.0.norm.weight": np.ones(32),
        "layers.0.0.norm.bias": np.zeros(32),
        "layers.0.2.weight": np.ones(32),
        "layers.0.2.bias": np.zeros(32),
        "layers.0.1.0.norm1.weight": np.ones(32),
        "layers.0.1.0.norm1.bias": np.zeros(32),
        "layers.0.1.0.norm2.weight": np.ones(32),
        "layers.0.1.0.norm2.bias": np.zeros(32),
        "layers.0.1.0.attn.attn.in_proj_weight": np.arange(96 * 32).reshape(96, 32),
        "layers.0.1.0.attn.attn.in_proj_bias": np.arange(96),
        "layers.0.1.0.attn.attn.out_proj.weight": np.zeros((32, 32)),
        "layers.0.1.0.attn.attn.out_proj.bias": np.zeros(32),
        "layers.0.1.0.attn.sr.weight": np.zeros((32, 32, 8, 8)),
        "layers.0.1.0.attn.sr.bias": np.zeros(32),
        "layers.0.1.0.attn.norm.weight": np.ones(32),
        "layers.0.1.0.attn.norm.bias": np.zeros(32),
        "layers.0.1.0.ffn.layers.0.weight": np.zeros((128, 32, 1, 1)),
        "layers.0.1.0.ffn.layers.0.bias": np.zeros(128),
        "layers.0.1.0.ffn.layers.1.weight": np.zeros((128, 1, 3, 3)),
        "layers.0.1.0.ffn.layers.1.bias": np.zeros(128),
        "layers.0.1.0.ffn.layers.4.weight": np.zeros((32, 128, 1, 1)),
        "layers.0.1.0.ffn.layers.4.bias": np.zeros(32),
    }


def test_map_timm_to_fork_covers_every_pattern() -> None:
    import numpy as np

    state = _timm_sample()
    out = repro.map_timm_to_fork(state)

    assert set(out) == {
        "patch_embed1.proj.weight", "patch_embed1.proj.bias",
        "patch_embed1.norm.weight", "patch_embed1.norm.bias",
        "norm1.weight", "norm1.bias",
        "block1.0.norm1.weight", "block1.0.norm1.bias",
        "block1.0.norm2.weight", "block1.0.norm2.bias",
        "block1.0.attn.q.weight", "block1.0.attn.q.bias",
        "block1.0.attn.kv.weight", "block1.0.attn.kv.bias",
        "block1.0.attn.proj.weight", "block1.0.attn.proj.bias",
        "block1.0.attn.sr.weight", "block1.0.attn.sr.bias",
        "block1.0.attn.norm.weight", "block1.0.attn.norm.bias",
        "block1.0.mlp.fc1.weight", "block1.0.mlp.fc1.bias",
        "block1.0.mlp.dwconv.dwconv.weight", "block1.0.mlp.dwconv.dwconv.bias",
        "block1.0.mlp.fc2.weight", "block1.0.mlp.fc2.bias",
    }
    # The fused [Q;K;V] projection splits exactly along torch's ordering.
    in_proj = state["layers.0.1.0.attn.attn.in_proj_weight"]
    in_bias = state["layers.0.1.0.attn.attn.in_proj_bias"]
    assert np.array_equal(out["block1.0.attn.q.weight"], in_proj[:32])
    assert np.array_equal(out["block1.0.attn.kv.weight"], in_proj[32:])
    assert np.array_equal(out["block1.0.attn.q.bias"], in_bias[:32])
    assert np.array_equal(out["block1.0.attn.kv.bias"], in_bias[32:])
    # The Mix-FFN's 1x1 convs flatten to the fork's Linear weights; the
    # depthwise conv keeps its shape.
    assert out["block1.0.mlp.fc1.weight"].shape == (128, 32)
    assert out["block1.0.mlp.fc2.weight"].shape == (32, 128)
    assert out["block1.0.mlp.dwconv.dwconv.weight"].shape == (128, 1, 3, 3)
    assert out["block1.0.attn.proj.weight"].shape == (32, 32)


def test_map_timm_to_fork_rejects_unknown_keys() -> None:
    state = _timm_sample()
    state["layers.0.1.0.attn.attn.weird.weight"] = None
    with pytest.raises(repro.ReproError, match="unrecognised pretrained key"):
        repro.map_timm_to_fork(state)


def test_shim_restores_numpy_alias_and_skips_torchless_envs() -> None:
    probe = "import numpy as np; print(hasattr(np, 'float'))"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repro.SHIM_DIR)
    with_shim = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert with_shim.stdout.strip() == "True"

    import numpy as np  # the point is the *current* interpreter's numpy

    if not hasattr(np, "float"):  # numpy >= 1.24, where the alias was removed
        without = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        assert without.stdout.strip() == "False"


SAMPLE_TRAIN_LOG = """
2026-10-06 02:49:19 - mmseg - INFO - Iter [10/60]\tlr: 1.531e-07, eta: 0:00:29, time: 0.588, data_time: 0.344, memory: 6509, decode.loss_seg: 0.6338, loss: 0.6338
2026-10-06 02:49:36 - mmseg - INFO - Iter [60/60]\tlr: 1.967e-08, eta: 0:00:00, time: 0.332, data_time: 0.125, memory: 6712, decode.loss_seg: 0.6012, loss: 0.6012
"""


def test_parse_train_log_extracts_progress_speed_memory() -> None:
    stats = repro.parse_train_log(SAMPLE_TRAIN_LOG)
    assert stats["iterations"] == 60
    assert stats["max_iters"] == 60
    assert stats["mean_iter_seconds"] == pytest.approx(0.46)
    assert stats["mean_data_seconds"] == pytest.approx(0.234, abs=1e-3)
    assert stats["peak_memory_mb"] == 6712
    assert stats["final_loss"] == pytest.approx(0.6012)


def test_parse_train_log_tolerates_a_log_without_iter_lines() -> None:
    stats = repro.parse_train_log("Traceback: nothing trained\n")
    assert stats == {"iterations": 0, "max_iters": None}


def test_parse_train_log_handles_cpu_logs_without_memory() -> None:
    # The CPU build never prints `memory:`, and that must not fabricate one.
    line = "Iter [1200/1200]\tlr: 5.000e-08, eta: 0:00:00, time: 3.359, data_time: 0.004, loss: 0.0169\n"
    stats = repro.parse_train_log(line)
    assert stats["iterations"] == 1200
    assert "peak_memory_mb" not in stats
    assert stats["final_loss"] == pytest.approx(0.0169)


def test_sha256_file_matches_hashlib_reference(tmp_path: Path) -> None:
    import hashlib

    payload = b"segformer" * 1000
    target = tmp_path / "blob.bin"
    target.write_bytes(payload)
    assert repro.sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_read_protocol_reads_rendered_and_base_values(tmp_path: Path) -> None:
    # The base-chain fallback needs the benchmark submodule; a bare clone or
    # `git worktree add` without submodules can legitimately lack it.
    if not repro.BASE_CONFIG.is_file():
        pytest.skip("benchmark submodule not checked out")
    config = tmp_path / "config.py"
    config.write_text(
        repro.render_config(
            data_root=tmp_path / "seg",
            max_iters=50_000,
            warmup_iters=3000,
            pretrained=None,
            log_interval=50,
        )
    )
    protocol = repro.read_protocol(config)
    assert protocol["max_iters"] == 50_000
    assert protocol["warmup_iters"] == 3000
    assert protocol["log_interval"] == 50
    # Not set by the render: read from the authors' base config chain instead.
    assert protocol["samples_per_gpu"] == 4
    assert protocol["workers_per_gpu"] == 2
    assert protocol["config_sha256"] == repro.sha256_file(config)
    assert protocol["base_config"] == str(repro.BASE_CONFIG)


def test_archive_md5s_returns_only_labeled_zips(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "dataset,source,id,file,bytes,md5,license,downloaded\n"
        "myceliumseg,zenodo,1,labeled-GL.zip,10,aaaa,cc-by-4.0,2026-10-01\n"
        "myceliumseg,zenodo,1,unlabeled-GL2.zip,10,bbbb,cc-by-4.0,2026-10-01\n"
        "myceliumseg,zenodo,1,labeled-GS_PO_TS.zip,10,cccc,cc-by-4.0,2026-10-01\n"
        "myceliumseg,zenodo,1,labeled-MYG_PDA_TEMP.zip,10,dddd,cc-by-4.0,2026-10-01\n"
    )
    assert repro.archive_md5s(manifest) == {
        "labeled-GL.zip": "aaaa",
        "labeled-GS_PO_TS.zip": "cccc",
        "labeled-MYG_PDA_TEMP.zip": "dddd",
    }
    assert repro.archive_md5s(tmp_path / "missing.csv") == {}


def test_collect_environment_never_raises_and_records_torch_absence() -> None:
    env = repro.collect_environment()
    # Runs in .venv (no torch) and in the shells (torch present) alike.
    for key in ("python", "platform", "torch", "cuda_available", "mmcv", "mmseg", "gpus"):
        assert key in env
    assert isinstance(env["cuda_available"], bool)


def test_collect_evidence_assembles_a_run(tmp_path: Path) -> None:
    work = tmp_path / "full-gpu"
    train = work / "train"
    train.mkdir(parents=True)
    (work / "config.py").write_text(
        repro.render_config(
            data_root=tmp_path / "seg",
            max_iters=50_000,
            warmup_iters=3000,
            pretrained=None,
            log_interval=50,
        )
    )
    (train / "train.log").write_text(SAMPLE_TRAIN_LOG)
    (train / "iter_60.pth").write_bytes(b"checkpoint-bytes")
    (tmp_path / "seg" / "manifest.json").parent.mkdir(parents=True)
    (tmp_path / "seg" / "manifest.json").write_text(
        json.dumps({"splits": {"train": ["a.jpg", "b.jpg"], "test": ["c.jpg"]}})
    )
    (work / "metrics.json").write_text(
        json.dumps(
            {
                "test": {
                    "images": 50,
                    "wall_seconds": 12.5,
                    "headline": {"mycelium_f1": 0.97},
                    "aggregate": {"mIoU": 0.96},
                    "per_class": [{"class": "mycelium"}],
                }
            }
        )
    )

    evidence = repro.collect_evidence(work, tmp_path / "seg")
    assert evidence["protocol"]["max_iters"] == 50_000
    assert evidence["training"]["iterations"] == 60
    assert evidence["training"]["peak_memory_mb"] == 6712
    assert evidence["checkpoint"]["bytes"] == len(b"checkpoint-bytes")
    assert len(evidence["checkpoint"]["sha256"]) == 64
    assert evidence["dataset"]["counts"] == {"train": 2, "test": 1}
    assert evidence["metrics"]["test"]["headline"]["mycelium_f1"] == 0.97
    # per_class stays out of the summary; it lives in metrics.json.
    assert "per_class" not in evidence["metrics"]["test"]
    assert evidence["commands"]


def test_shim_stream_device_coerces_int_ids_for_torch2() -> None:
    """mmcv's int device ids must become torch.device (or None for CPU).

    Regression guard for the GPU-path crash: ``AttributeError: 'int' object
    has no attribute 'type'`` raised inside torch's ``_get_stream`` on the
    first training iteration of every CUDA run.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repro.SHIM_DIR)
    probe = (
        "import sitecustomize, json;"
        "f = lambda *a: ('device',) + a;"
        "print(json.dumps(["
        "    list(sitecustomize._stream_device(0, f)),"
        "    list(sitecustomize._stream_device(3, f)),"
        "    sitecustomize._stream_device(-1, f),"
        "    sitecustomize._stream_device('cpu', f),"
        "]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert json.loads(result.stdout) == [
        ["device", "cuda", 0],
        ["device", "cuda", 3],
        None,
        "cpu",
    ]
