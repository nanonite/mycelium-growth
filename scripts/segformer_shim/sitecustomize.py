"""Compatibility shims for running the pinned benchmark on this machine.

Python imports ``sitecustomize`` automatically at startup whenever this
directory is on ``PYTHONPATH``, which is how ``scripts/segformer_repro.py``
starts the benchmark's ``train.py`` and ``test.py``. Nothing here runs in the
repository's own tests or in a plain ``nix develop`` shell.

The pinned environment (README "Known limitations") differs from the one the
benchmark was written for in ways that break its code paths:

1. torch is built with ``USE_CUDA=OFF``. ``mmseg/apis/train.py`` wraps the
   model with ``MMDataParallel(model.cuda(...))`` unconditionally, which
   raises on a CPU-only build. With no accelerator present, ``Module.cuda``
   becomes a no-op - ``MMDataParallel`` already special-cases an empty device
   list, and every other caller of ``.cuda()`` in the train path wants "the
   model on the only device there is". On a machine that *does* have CUDA
   this patch is not applied.
2. numpy >= 1.24 removed the ``np.float`` alias, but the benchmark's
   evaluation code (``mmseg/core/evaluation/metrics.py``, every metric
   including plain mIoU) still calls it. Restore the alias when missing.
3. yapf >= 0.40 removed the ``verify`` kwarg of ``FormatCode``, which mmcv
   1.3.0's ``Config.pretty_text`` - and therefore train.py's config dump -
   still passes.
4. mmcv 1.3.0's ``MMDataParallel.train_step``/``val_step`` take a broken
   branch when the device list is empty (the CPU case): they forward the
   per-device *lists* returned by ``scatter`` instead of the first device's
   entry, which raises ``TypeError: ... argument after ** must be a mapping``.
5. On a CUDA machine, mmcv 1.3.0's ``Scatter.forward`` calls torch's
   ``_get_stream`` with the plain integer device ids from ``cfg.gpu_ids``,
   but torch 2.x dereferences ``device.type`` and needs a ``torch.device``:
   every GPU training step died with ``AttributeError: 'int' object has no
   attribute 'type'``. Translate the id (mmcv's ``-1`` CPU marker means "no
   background stream"). This patch only applies when CUDA is available.

Because a ``PYTHONPATH`` directory shadows every later ``sitecustomize``,
including Nix's own (which is what puts the environment's site-packages -
torch, mmcv, numpy - on ``sys.path`` in the first place), Nix's module is
chained in explicitly before anything here imports a third-party package.

Editing the pinned submodule to fix either issue would move the gitlink away
from the upstream commit, so the fixes are injected here instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _chain_shadowed_sitecustomize() -> None:
    """Run the first other ``sitecustomize.py`` on ``sys.path``, if any.

    Nix's interpreter ships one that processes ``NIX_PYTHONPATH``; if we
    shadowed it, every package in the dev shell would be missing below.
    """
    own = Path(__file__).resolve()
    for entry in list(sys.path):
        if not entry:
            continue
        try:
            candidate = Path(entry) / "sitecustomize.py"
            if not candidate.is_file() or candidate.resolve() == own:
                continue
        except OSError:
            continue
        spec = importlib.util.spec_from_file_location("_shadowed_sitecustomize", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return


_chain_shadowed_sitecustomize()


def _stream_device(device, make_device):
    """mmcv hands ints to torch's stream helper; torch 2.x wants a device.

    ``make_device`` is ``torch.device`` (injected so the rule stays testable in
    an interpreter without torch). mmcv's ``-1`` marks the CPU, which has no
    dedicated background stream, hence ``None``.
    """
    if isinstance(device, int):
        if device < 0:
            return None
        return make_device("cuda", device)
    return device

import numpy as np  # after the chaining: it may need the env's site-packages

_NP_ALIASES = {"float": float, "int": int, "bool": bool}

for _name, _target in _NP_ALIASES.items():
    # Membership on __dict__: a hasattr() probe warns on np.bool/np.int.
    if _name not in np.__dict__:
        np.__dict__[_name] = _target


def _patch_yapf_format_code() -> None:
    """Drop the ``verify`` kwarg mmcv passes to yapf, which no longer has it.

    ``mmcv.utils.config.Config.pretty_text`` (used by train.py's config dump
    and logger) calls ``FormatCode(text, style_config=..., verify=True)``;
    yapf >= 0.40 removed ``verify``, so every config dump would raise
    TypeError. The flag only asked yapf to re-parse its own output, so it is
    safe to swallow.
    """
    try:
        from yapf import yapf_api
    except ImportError:
        return
    import inspect

    original = yapf_api.FormatCode
    try:
        accepted = set(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        return
    if "verify" in accepted:
        return

    def format_code_without_verify(source, *args, **kwargs):
        kwargs.pop("verify", None)
        return original(source, *args, **kwargs)

    yapf_api.FormatCode = format_code_without_verify


_patch_yapf_format_code()

try:
    import torch
except ImportError:  # numpy-only environments (e.g. the tests' .venv)
    torch = None

if torch is not None and not torch.cuda.is_available():

    def _cuda_no_op(self, *args, **kwargs):
        """``Module.cuda()`` on a CPU-only build: stay where you are."""
        return self

    torch.nn.Module.cuda = _cuda_no_op
    torch.Tensor.cuda = _cuda_no_op

    def _patch_mmcv_parallel_cpu() -> None:
        """Fix mmcv 1.3.0's CPU scatter and step branches.

        Three problems surface only without CUDA, which is why upstream never
        hit them (the GPU paths are fine):

        1. ``Scatter.forward``'s CPU branch unsqueezes every tensor
           (``unsqueeze(0)`` "so the shape matches GPU scatter"), but a
           DataContainer's ``data`` is a *list* holding the stacked batch, so
           the extra dimension survives into the model as a 5-D image.
           Reproduce what the GPU produces instead: the tensor, whole, as the
           single device's entry.
        2. ``MMDataParallel.train_step``/``val_step`` forward the per-device
           *lists* returned by ``scatter`` instead of the first device's
           entry (``*inputs`` where the GPU branch has ``*inputs[0]``).
        3. (``forward`` already indexes ``[0]`` correctly.)
        """
        try:
            from mmcv.parallel import _functions as mmcv_functions
            from mmcv.parallel.data_parallel import MMDataParallel
        except ImportError:
            return

        original_scatter_forward = mmcv_functions.Scatter.forward

        def scatter_forward(target_gpus, input):
            if target_gpus != [-1]:
                return original_scatter_forward(target_gpus, input)
            if isinstance(input, torch.Tensor):
                return (input.contiguous(),)
            if isinstance(input, list):
                return tuple(
                    item.contiguous() if isinstance(item, torch.Tensor) else item
                    for item in input
                )
            return original_scatter_forward(target_gpus, input)

        mmcv_functions.Scatter.forward = staticmethod(scatter_forward)

        for name in ("train_step", "val_step"):
            original = getattr(MMDataParallel, name)

            def _patched(self, *inputs, _original=original, _name=name, **kwargs):
                if self.device_ids:
                    return _original(self, *inputs, **kwargs)
                inputs, kwargs = self.scatter(inputs, kwargs, [-1])
                step = getattr(self.module, _name)
                return step(*inputs[0], **kwargs[0])

            setattr(MMDataParallel, name, _patched)

    _patch_mmcv_parallel_cpu()


if torch is not None and torch.cuda.is_available():

    def _patch_mmcv_scatter_stream_devices() -> None:
        """Coerce mmcv 1.3.0's integer device ids for torch 2.x.

        ``mmcv/parallel/_functions.py`` does
        ``streams = [_get_stream(device) for device in target_gpus]`` with
        ``target_gpus`` holding plain ints (``cfg.gpu_ids``), while torch 2.5's
        ``_get_stream`` immediately reads ``device.type``. Wrap the helper that
        mmcv imported into its own namespace: ints become ``torch.device``s and
        mmcv's ``-1`` CPU marker keeps meaning "no background copy stream",
        which is what torch returns for ``cpu`` anyway.
        """
        try:
            from mmcv.parallel import _functions as mmcv_functions
        except ImportError:
            return

        original_get_stream = mmcv_functions._get_stream

        def get_stream(device):
            device = _stream_device(device, torch.device)
            if device is None:  # mmcv's CPU marker: no background stream
                return None
            return original_get_stream(device)

        mmcv_functions._get_stream = get_stream

    _patch_mmcv_scatter_stream_devices()
