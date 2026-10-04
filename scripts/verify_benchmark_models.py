#!/usr/bin/env python3
"""Smoke-check the benchmark's segmentation models against the devShell stack.

Builds the segmentation models whose configs ship in
external/MyceliumSeg-benchmark from those configs and runs a single forward
pass on a synthetic image, which proves that the mmseg fork, mmcv 1.3.0 and the
pinned torch in flake.nix interoperate. Run from inside `nix develop`:

    nix develop --command python scripts/verify_benchmark_models.py

By default only one config is checked: the SegFormer ``mit_b0`` model, the one
``script_inference.sh`` and ``script_train.sh`` point at. It is the
representative compatibility path because its backbone (``mit_b*``) exists only
in the benchmark's mmseg fork, so it fails if the fork does not shadow the
packaged mmsegmentation, and its attention blocks exercise torch's linear/matmul
kernels on top of mmcv's compiled ops. The other configs are opt-in with
``--all``; they add coverage but also wall-clock time, so they are not part of
the default smoke test.

Models are built with random weights and the trained checkpoints are not in
this repository, so this verifies the environment, not the trained models.

The run is bounded twice over: ``--threads`` pins torch (1 by default, see the
note in ``pin_torch_threads``) and ``--timeout`` aborts the process with exit
code 3 if it overruns, so this script can be used as a gate.

Exit codes: 0 all checks passed, 1 a check failed, 2 the environment is not
usable, 3 the deadline expired.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Self

BENCHMARK = Path(__file__).resolve().parent.parent / "external" / "MyceliumSeg-benchmark"

# The benchmark ships its own fork of mmseg (a 0.11.0-era tree that adds the
# SegFormer `mit_b*` backbones and the EPA_mycelium dataset on top of it). Its
# configs reference those names, so the checkout directory has to come first on
# sys.path for this fork to shadow the mmsegmentation the flake installs. This
# mirrors what happens when the benchmark's own scripts run from its root.
sys.path.insert(0, str(BENCHMARK))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402

import mmseg  # noqa: E402
from mmseg.models import build_segmentor  # noqa: E402

# Config path -> (expected backbone feature strides, expected number of classes)
#
# The strides are the spatial downsampling factors of the features the backbone
# emits, i.e. IMAGE_SIZE / feature_size. encode_decode upsamples back to the
# input resolution, so the stride has to be read off extract_feat instead.
CONFIGS: dict[str, tuple[tuple[int, ...], int]] = {
    # This fork's UNet returns the bottleneck first and then walks the decoder
    # outwards, so the strides run coarse to fine: 16, 8, 4, 2, 1. The heads
    # confirm it - PSPHead takes in_index=4 with 64 channels, which is the
    # stride-1 output.
    "mycelium_model/model/pspnet_unet_s5-d16.py": ((16, 8, 4, 2, 1), 2),
    "mycelium_model/model/deeplabv3_unet_s5-d16.py": ((16, 8, 4, 2, 1), 2),
    # SegFormer: four overlapping-patch stages at strides 4, 8, 16, 32.
    "mycelium_model/model/segformer.b0.1024x1024.mycelium.507.py": ((4, 8, 16, 32), 2),
}

# The single config the default run checks, and the reason it is the one.
REPRESENTATIVE = "mycelium_model/model/segformer.b0.1024x1024.mycelium.507.py"

DEFAULT_IMAGE_SIZE = 128
DEFAULT_THREADS = 1
DEFAULT_TIMEOUT = 300.0


def pin_torch_threads(threads: int) -> int:
    """Limit torch's intra-op threads and return the value now in effect.

    torch 2.5.1 as built by nixpkgs (BLAS_INFO=open, USE_EIGEN_FOR_BLAS=ON)
    deadlocks in ``nn.Linear`` once more than one intra-op thread is in play on
    this path: the calling thread spins at 100% CPU inside an
    ``nn.Linear.forward`` while the OpenMP workers sit in ``futex_wait``, for
    any ``torch.get_num_threads() >= 2`` and any input from 64x64 up. 32x32
    inputs and ``torch.get_num_threads() == 1`` both complete, and a hand-built
    module with the same architecture does not hang, so this is a property of
    this build rather than of the model.

    A smoke test does not need the threads, so pin to 1 by default; pass
    ``--threads 0`` to keep torch's own default (24 here) and expect the hang.
    """
    if threads > 0:
        torch.set_num_threads(threads)
    return torch.get_num_threads()


class Deadline:
    """Abort the whole process after ``seconds``, so a wedged op cannot hang a gate.

    A ``signal.alarm`` handler would not help: the hang sits inside an ATen op
    that never returns to the interpreter, so no handler runs. This thread
    calls ``os._exit`` instead, which is the only thing guaranteed to take a
    process stuck in C code down.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        if self.seconds > 0:
            self._thread = threading.Thread(target=self._fire, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._done.set()

    def _fire(self) -> None:
        if self._done.wait(self.seconds):
            return
        print(
            f"\nTIMEOUT: benchmark model check exceeded {self.seconds:.0f}s and was aborted."
            "\n         The usual cause is the torch threading deadlock described in"
            "\n         pin_torch_threads(); re-run with --threads 1 (the default)."
            "\n         Use --timeout to raise the budget if the box is simply loaded.",
            flush=True,
        )
        os._exit(3)


def disable_pretrained(node: object) -> None:
    """Point every ``pretrained`` key already in ``node`` at None, recursively.

    The configs point ``pretrained`` at checkpoints under the authors' absolute
    paths (./pretrained/mit_b0.pth etc), which are not in this repository, so
    the models must be built with random weights instead.

    Only keys that are already there are touched. Adding ``pretrained`` to a
    submodule that does not accept it - the fork's ``UNet`` backbone, for one -
    makes mmcv's registry pass an argument the constructor rejects.
    """
    if isinstance(node, dict):
        if "pretrained" in node:
            node["pretrained"] = None
        for value in node.values():
            disable_pretrained(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            disable_pretrained(item)


def check(config_path: str, expected_strides: tuple[int, ...], num_classes: int, image_size: int) -> bool:
    path = BENCHMARK / config_path
    started = time.monotonic()
    cfg = Config.fromfile(str(path))
    disable_pretrained(cfg.model)

    # Inference mode: the configs ship `mode='slide'` with crop_size (1024, 1024).
    # Slide inference tiles the padded image into 1024x1024 crops, which on CPU
    # is minutes of work per model for no extra coverage of this check. Force
    # `whole` so each model runs as a single forward pass.
    cfg.model.test_cfg = dict(mode="whole")

    model = build_segmentor(cfg.model)
    model.eval()

    img = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        # A 4D NCHW tensor is what this fork's encode_decode expects: it resizes
        # to `img.shape[2:]` and mix_transformer unpacks `x.shape` into 4 values.
        features = model.extract_feat(img)
        logits = model.encode_decode(
            img=img,
            img_metas=[
                dict(
                    ori_shape=(image_size, image_size, 3),
                    img_shape=(image_size, image_size, 3),
                    pad_shape=(image_size, image_size, 3),
                    scale_factor=np.array([1.0, 1.0, 1.0, 1.0]),
                    flip=False,
                    img_norm_cfg=dict(
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
                    ),
                )
            ],
        )

    # encode_decode may return a list (slide inference) or a tensor.
    out = logits[0] if isinstance(logits, list) else logits

    strides = tuple(image_size // f.shape[-1] for f in features)
    ok = out.shape[1] == num_classes and strides == expected_strides

    print(
        f"{'OK  ' if ok else 'FAIL'} {config_path}  ({time.monotonic() - started:.1f}s)\n"
        f"       backbone={type(model.backbone).__name__} "
        f"decode_head={type(model.decode_head).__name__}\n"
        f"       feature strides={strides} (want {expected_strides}) "
        f"from {[tuple(f.shape[-2:]) for f in features]}\n"
        f"       logits={tuple(out.shape)} classes={out.shape[1]} (want {num_classes})",
        flush=True,
    )
    return ok


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"check every config in the benchmark, not just {REPRESENTATIVE}",
    )
    parser.add_argument(
        "--config",
        action="append",
        metavar="PATH",
        help="config to check, relative to the benchmark root; repeatable",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"square input edge in pixels (default: {DEFAULT_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=(
            "torch intra-op threads; 0 keeps torch's default "
            f"(default: {DEFAULT_THREADS}, see pin_torch_threads)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"abort the whole run after this many seconds, 0 disables (default: {DEFAULT_TIMEOUT:.0f})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the known configs and exit",
    )
    return parser.parse_args(argv)


def selected_configs(args: argparse.Namespace) -> list[str]:
    if args.config:
        return list(args.config)
    if args.all:
        return list(CONFIGS)
    return [REPRESENTATIVE]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.list:
        for path in CONFIGS:
            marker = " (default)" if path == REPRESENTATIVE else ""
            print(f"{path}{marker}")
        return 0

    if not BENCHMARK.is_dir():
        print(f"benchmark checkout not found at {BENCHMARK}", file=sys.stderr)
        return 2

    unknown = [path for path in args.config or [] if path not in CONFIGS]
    if unknown:
        print(f"unknown config(s): {', '.join(unknown)}; try --list", file=sys.stderr)
        return 2

    threads = pin_torch_threads(args.threads)

    print(f"python  {sys.version.split()[0]}")
    print(f"mmseg   {mmseg.__version__} from {Path(mmseg.__file__).parent}")
    print(f"mmcv    {__import__('mmcv').__version__}")
    print(f"torch   {torch.__version__} ({threads} intra-op thread(s))")
    print()

    started = time.monotonic()
    failures = 0
    with Deadline(args.timeout):
        for config_path in selected_configs(args):
            expected_strides, num_classes = CONFIGS[config_path]
            try:
                if not check(config_path, expected_strides, num_classes, args.image_size):
                    failures += 1
            except Exception:
                failures += 1
                print(f"FAIL {config_path}", flush=True)
                traceback.print_exc()

    elapsed = time.monotonic() - started
    if failures:
        print(f"\n{failures} benchmark model check(s) failed in {elapsed:.1f}s")
        return 1
    print(f"\nall benchmark model checks passed in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
