# Reproducing the SegFormer baseline (#28)

Phase 2 step 1 of [PLAN.md](../PLAN.md): run the MyceliumSeg benchmark's
SegFormer-B0 baseline end to end with this repository's pinned stack, so that
the measurement pipeline (issues #29–#31) has a trained, evaluated segmenter to
build on.

Everything below is produced by `scripts/segformer_repro.py` (see "Reproduce"
at the end). Artifacts live in `work_dirs/segformer-b0-repro/` (gitignored):
the generated configs, `train/train.log`, `eval_test.log`, `eval_conditions.log`
and `metrics.json`.

## The published target

Numbers from Table 6 of Yuan et al., *A Mycelium Dataset with Edge-Precise
Annotation for Semantic Segmentation*, Scientific Data 12:2015 (2025),
doi:10.1038/s41597-025-06265-1, SegFormer row, measured on the 50-image GL
testset:

| Metric | Published (SegFormer) |
| --- | --- |
| Mycelium F1-score ↑ (%) | 97.55 |
| Mycelium IoU ↑ (%) | 95.22 |
| Boundary IoU ↑ (%) | 28.60 |
| HD95 ↓ (px) | 75.95 |
| ASSD ↓ (px) | 15.28 |

Published recipe (paper "Implementation details" + the benchmark's
`mycelium_model/model/segformer.b0.1024x1024.mycelium.507.py`):

- SegFormer MiT-B0, ImageNet-pretrained backbone (`./pretrained/mit_b0.pth`,
  **not** part of the release), 50,000 iterations, batch 4 per GPU on four
  RTX 4090s (effective batch 16).
- AdamW, lr 6e-5, weight decay 1e-2, 3,000-iteration linear warmup
  (ratio 1e-6), polynomial decay power 1.0.
- Train on the 457-image *G. lucidum* trainset; report on the 50-image
  testset; the six 10-image condition sets (GS/PO/TS/MYG/PDA/TEMP15) are a
  separate robustness evaluation (their Table 7: F1/IoU > 92%, Boundary IoU
  11.50–24.50%, ASSD 21.04–118.29 px).
- Data pipeline: resize to native 4608×3456, random 512×512 crop, random flip,
  photometric distortion; test at full resolution with slide inference
  (crop 1024, stride 768).
- Metrics: the benchmark's own `test.py` with
  `--eval mIoU mDice assd hd95 biou`; Boundary IoU uses dilation ratio 0.001
  in the shipped code (5 px at this resolution), matching the paper's text.

## The GPU environment (#56)

`flake.nix` exposes a second dev shell, `devShells.gpu`, alongside the
untouched CPU default:

```bash
nix develop .#gpu --command python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()"
nix develop .#gpu --command nvidia-smi
```

Recorded on this host (also captured in every `evidence.json`):

| Fact | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Ti, 12,282 MiB |
| Driver | 595.91.07 |
| torch | 2.5.1, CUDA build 12.4, `is_available() == True` |
| mmcv / mmseg | 1.3.0 (C++ ops, no CUDA kernels) / 0.11.0 vendored fork |
| Shell | `nix develop .#gpu` (`devShells.gpu`); `nix develop` stays CPU-only |

Design points, per the backrooms-workspace flake pattern:

- **Evaluation stays safe.** `nix flake show` / `nix develop` (default) never
  build or load CUDA anything; the GPU closure is a separate `import nixpkgs`
  with `cudaSupport = true` and `cudaCapabilities = [ "8.9" ]`, referenced
  only by the GPU outputs.
- **Host driver at runtime, not build time.** The GPU shell's `shellHook`
  `LD_PRELOAD`s only `/usr/lib/x86_64-linux-gnu/libcuda.so.1` (falling back to
  `/usr64/lib/libcuda.so.1`) rather than the whole host system directory,
  which would drag incompatible host glibc into the shell.
- **Lazy GPU wrapping.** The hook's smoke check (nvidia-smi + a CUDA tensor
  probe) runs only when the GPU shell is *entered*; nothing GPU-related runs
  during evaluation or in the CPU shell.
- **CPU profile retained.** `devShells.default`, `pythonEnv` and all prior
  verification scripts are unchanged, so CI and the repository's own tests
  (`234 passed` in `.venv`) never touch a GPU.

mmcv is deliberately built **without** CUDA kernels (`MMCV_WITH_CUDA=0`):
its 1.3.0 CUDA sources need THC headers removed from torch 2.x. The model
tensors run on CUDA through torch; the ops mmseg actually calls in this
protocol (scatter, deformable-free SegFormer forward, metrics) do not need
mmcv CUDA kernels.

## How this repository runs it

`scripts/segformer_repro.py` layers four subcommands over the benchmark's
entry points (`train.py`, `test.py`) without editing the pinned submodule:

1. **prepare** – unpacks `labeled-GL.zip`, `labeled-GS_PO_TS.zip` and
   `labeled-MYG_PDA_TEMP.zip` into the layout the benchmark's
   `EPA_Mycelium_Dataset` expects, `data/segmentation/{train,test,conditions}/
   {image,mask}`: 457 / 50 / 60 image–mask pairs (the paper's 457 + 50 split,
   with the 60 condition images kept as the robustness hold-out). The split is
   asserted against those counts, pairs must match by filename, and the
   extraction is idempotent via `manifest.json`.
2. **config** – renders `work_dirs/segformer-b0-repro/full/config.py`, whose
   single `_base_` is the authors' config; it overrides only `data_root`,
   `runner.max_iters`, `lr_config.warmup_iters`, checkpoint/evaluation
   intervals, log interval and `model.pretrained`.
3. **train** – the benchmark's `train.py --seed 42 --deterministic
   --no-validate`, single process, single-threaded torch.
4. **evaluate** – the benchmark's `test.py --eval mIoU mDice assd hd95 biou
   --eval-options efficient_test=False`, then parses the printed per-class and
   summary tables back into `metrics.json` (a log without parseable tables is
   an error, never a result).

A `run --smoke` variant (4/2/2 images, 2 iterations) exercises the identical
code path in ~5 minutes; it is what proves the pipeline on a new machine. Both
variants were run on the GPU shell for #57: see "GPU smoke check" below.

### The evidence record (`evidence` subcommand)

`python scripts/segformer_repro.py evidence [--smoke] [--tag NAME]` writes
`<work>/evidence.json` — the hand-off artifact for #59. It collects, without
retyping anything:

- **environment**: Python/platform, torch version and CUDA build,
  `torch.cuda.is_available()`, device name, mmcv/mmseg versions, and the
  `nvidia-smi` rows (GPU name, driver, total/free memory);
- **protocol**: `max_iters`, `warmup_iters`, `log_interval`,
  `samples_per_gpu`, `workers_per_gpu` and the config SHA-256 — read back out
  of the rendered config, falling back to the authors' base config for values
  the render leaves at their defaults;
- **dataset**: manifest SHA-256 and per-split image counts;
- **training**: iterations completed, mean iteration/data seconds, peak GPU
  memory and final loss, parsed from `train/train.log` (the `memory:` field
  only exists when torch sees CUDA, so its presence is itself the GPU signal);
- **checkpoint**: path, size and SHA-256;
- **metrics**: per-split headline/aggregate numbers from `metrics.json`
  (per-class rows stay in `metrics.json`, which is already committed evidence);
- **provenance**: repository commit, benchmark-submodule commit, and the three
  labeled release archives' MD5s from `data/manifest.csv`.

`run` now ends with `evidence`, so a completed run always leaves one behind.

### The compatibility shim (`scripts/segformer_shim/sitecustomize.py`)

The benchmark as published assumes CUDA GPUs, numpy < 1.24 and yapf < 0.40;
this workspace pins CPU-only torch 2.5.1, numpy 1.26.4 and a newer yapf (see
README "Known limitations"). Rather than moving the submodule's gitlink, the
driver starts its subprocesses with `scripts/segformer_shim/` on
`PYTHONPATH`, and Python's automatic `sitecustomize` import applies four
fixes — only inside those subprocesses:

1. `torch.nn.Module.cuda` / `torch.Tensor.cuda` become no-ops when
   `torch.cuda.is_available()` is False, because `mmseg/apis/train.py`
   wraps the model with `MMDataParallel(model.cuda(...))` unconditionally.
2. `np.float` (and `np.int`, `np.bool`) are restored; `metrics.py` uses
   `np.float` in `total_intersect_and_union`, so **every** metric — plain mIoU
   included — raised `AttributeError` on numpy ≥ 1.24.
3. yapf's `FormatCode` is wrapped to drop the removed `verify=` kwarg that
   `mmcv.Config.pretty_text` (and therefore train.py's config dump) passes.
4. mmcv 1.3.0's CPU scatter is corrected: `Scatter.forward` unsqueezes a
   DataContainer's stacked batch into a 5-D tensor (the GPU branch does not),
   and `MMDataParallel.train_step`/`val_step` forward the per-device lists
   instead of the first device's entry. With these, `img` reaches the
   backbone as `(4, 3, 512, 512)`, exactly as on one GPU.
5. On a CUDA machine the shim instead coerces mmcv's integer device ids to
   `torch.device` before torch's stream helper (see "GPU smoke check" below);
   patches 1 and 4 do not apply when `torch.cuda.is_available()` is True.

Because a `PYTHONPATH` entry shadows Nix's own `sitecustomize` (the thing that
puts the dev shell's site-packages on `sys.path`), the shim chains to it
first.

### Pretrained backbone

`config --pretrained auto` (the default) downloads the ImageNet MiT-B0 from
OpenMMLab, then **remaps** it: that file is timm-format (`layers.*`,
fused `[Q;K;V]` `in_proj`), while the benchmark's backbone is SegFormer-style
(`patch_embed*`, `block*`, split `attn.q`/`attn.kv`, Linear Mix-FFN). The
conversion is mechanical (torch documents the fused order as Q, K, V; the 1×1
Mix-FFN convs flatten to Linears), and it is verified by building the
benchmark's own backbone and loading the result with `strict=True` — 176/176
tensors — before training starts. Output: `data/pretrained/mit_b0.converted.pth`.

## GPU smoke check (#57)

The five-minute `run --smoke` path was executed on the GPU shell after the
environment existed:

```bash
nix develop .#gpu --command python scripts/segformer_repro.py run --smoke
```

Recorded in `work_dirs/segformer-b0-repro/smoke/evidence.json`:

| Fact | Value |
| --- | --- |
| Device / driver | RTX 4070 Ti / 595.91.07 (`CUDA available: True` in `train.log`) |
| CUDA / torch / mmcv | CUDA 12.4 build / torch 2.5.1 / mmcv 1.3.0 |
| Batch size | `samples_per_gpu=4`, `workers_per_gpu=2` (authors' defaults) |
| Iterations | 2, `--seed 42 --deterministic`, checkpoint `iter_2.pth` (SHA-256 in `smoke/evidence.json`) |
| Peak GPU memory | 1,683 MiB (`memory:` field in the train log) |
| Iteration time | 0.093 s steady (first iter 0.97 s incl. data warm-up) |
| Metrics (2-image eval) | F1 0.7767, IoU 0.6350, mIoU 0.7801 — pipeline sanity only, not a quality result |

**Smoke found a real bug.** The first GPU training attempt died on every
iteration with `AttributeError: 'int' object has no attribute 'type'`:
mmcv 1.3.0's `Scatter.forward` passes the integer device ids from
`cfg.gpu_ids` to torch 2.5's `_get_stream`, which now immediately reads
`device.type`. The fix is shim item 5 — `_stream_device()` coerces the int to
`torch.device("cuda", n)` (and maps mmcv's `-1` CPU marker to "no stream") —
guarded by `test_shim_stream_device_coerces_int_ids_for_torch2`. After it,
train and evaluate both exit 0 on the GPU shell.

## Baseline protocol on the GPU (#58)

```bash
nix develop .#gpu --command python scripts/segformer_repro.py run \
  --tag gpu --max-iters 50000 --batch-size 16 --workers 8
```

Why these settings:

- **50,000 iterations = the published count.** A 60-iteration probe measured
  0.32–0.45 s/iter (≈ 6.5 GiB peak), so the full run is ~4.6 h on this GPU —
  affordable, so no iteration-count deviation was needed.
- **`--batch-size 16` = the published *effective* batch.** The paper trains
  with batch 4 per GPU on 4 GPUs (16). One 12 GB GPU runs the same global
  batch in a single forward/backward, which is exactly what gradient
  accumulation would have emulated — with no accumulation, no gradient
  scaling and no schedule changes. Effective batch, LR (6e-5), warmup (3,000)
  and iteration count all match the publication.
- **`--workers 8`**: data loading measured 0.12–0.27 s/iter at 8 workers
  (published config uses 2 per GPU × 4 GPUs = 8 feeding a batch of 16, so
  this matches the published decode parallelism too).
- Artifacts go to `work_dirs/segformer-b0-repro/full-gpu/` (`--tag gpu`), so
  the earlier CPU run's directory stays intact for comparison.

### Deviations remaining for the GPU run

| Axis | Published | GPU run | Why |
| --- | --- | --- | --- |
| Hardware | 4 × RTX 4090 | 1 × RTX 4070 Ti (12 GB) | only GPU available; no DDP, single process |
| Backbone init | authors' `./pretrained/mit_b0.pth` (not released) | OpenMMLab MiT-B0, remapped and strict-load-verified | only compatible ImageNet init available |
| Seeds | not published | 42, `--deterministic` | reproducibility |
| Iterations / batch / LR / warmup | 50,000 / 16 eff. / 6e-5 / 3,000 | **identical** | 0.32 s/iter makes the full count affordable |

mmcv 1.3.0 has no gradient-accumulation hook, so the batch match is done the
direct way (one 16-sample step) rather than via accumulation — same effective
batch, same optimizer-step count as the publication.

## Deviations from the published protocol (earlier CPU run)

| Axis | Published | Here | Why |
| --- | --- | --- | --- |
| Hardware | 4 × RTX 4090 | 1 CPU core (24 available) | pinned torch is `USE_CUDA=OFF`; multi-threaded torch deadlocks on this build (README "Known limitations") |
| Iterations | 50,000 | 1,200 | ~3.9 s/iter on CPU → ~80 min; a 50k run would take ~54 h |
| Warmup | 3,000 iters | 240 (`min(3000, max_iters // 5)`) | otherwise the whole short run would be warmup |
| Effective batch | 4 × 4 GPUs = 16 | 4 | single device; lr left at the published 6e-5 |
| Backbone init | authors' `./pretrained/mit_b0.pth` (not released) | OpenMMLab MiT-B0, remapped and strict-load-verified | only compatible ImageNet init available |
| Seeds | not published | 42, `--deterministic` | reproducibility |

Everything else — config base, optimizer, schedule shape, crop/augmentation
pipeline, test-time full-resolution slide inference, metric implementations
(including the 0.001 Boundary-IoU dilation), seed of the data splits — is the
authors' code and settings, run from their own entry points.

## Results

Two runs exist: the **earlier CPU run** (1,200 iterations,
`work_dirs/segformer-b0-repro/full/`) and the **GPU baseline** (#58,
`work_dirs/segformer-b0-repro/full-gpu/`, 50,000 iterations). The CPU run is a
short-run existence proof, not the baseline; the GPU run is the hand-off
checkpoint.

### Test split — GL testset, 50 images (vs. published Table 6)

| Metric | Published | CPU run (1,200 iters) | GPU baseline (50,000 iters) |
| --- | --- | --- | --- |
| Mycelium F1 ↑ (%) | 97.55 | 96.88 | TBD |
| Mycelium IoU ↑ (%) | 95.22 | 93.95 | TBD |
| Boundary IoU ↑ (%) | 28.60 | 19.05 | TBD |
| HD95 ↓ (px) | 75.95 | 51.84 | TBD |
| ASSD ↓ (px) | 15.28 | 20.55 | TBD |

### Condition hold-outs — 60 images (vs. published Table 7 ranges)

| Metric | Published range (GS/PO/TS) | CPU run (all 60) | GPU baseline (50,000 iters) |
| --- | --- | --- | --- |
| Mycelium F1 ↑ (%) | 95.93 – 97.67 | 96.65 | TBD |
| Mycelium IoU ↑ (%) | 92.18 – 95.44 | 93.52 | TBD |
| Boundary IoU ↑ (%) | 11.50 – 24.50 | 17.77 | TBD |
| HD95 ↓ (px) | 51.23 – 512.53 | 233.04 | TBD |
| ASSD ↓ (px) | 21.04 – 118.29 | 80.04 | TBD |

(The published range is per-species over three 10-image sets; ours is one
aggregate over all 60 condition images, so it is indicative, not identical.)

## Failure modes and known limits (#59)

What actually went wrong while producing this baseline, so the next run does
not rediscover it:

| Failure | Symptom | Cause | Resolution |
| --- | --- | --- | --- |
| GPU train step 1 crash | `AttributeError: 'int' object has no attribute 'type'` in `torch.nn.parallel._functions._get_stream` | mmcv 1.3.0 passes int `cfg.gpu_ids` to a torch 2.x helper that reads `device.type` | shim `_stream_device()` coercion (item 5); regression-tested |
| CPU train crash | `AttributeError: module 'numpy' has no attribute 'float'` in every metric | numpy ≥ 1.24 removed `np.float`, benchmark's `metrics.py` still uses it | shim restores the alias (item 2) |
| CPU train hang | forward pass never returns with `torch.get_num_threads() ≥ 2` | nixpkgs CPU torch + OpenMP (README "Known limitations") | `OMP_NUM_THREADS=1` in `child_env()` |
| Eval crash at image ~3 | `FileNotFoundError` in `rebuild_storage_fd` | torch DataLoader worker fd sharing on this stack | `data.workers_per_gpu=0` for `test.py` (decoding is negligible next to slide inference) |
| Config dump crash | `FormatCode() got an unexpected keyword argument 'verify'` | yapf ≥ 0.40 vs mmcv 1.3.0 `Config.pretty_text` | shim wrapper (item 3) |
| Concurrent GPU runs | `torch.OutOfMemoryError` — second trainer gets nothing | one 12 GB device; the baseline alone uses ~7.8 GiB | run one GPU job at a time; the smoke was re-run after the baseline finished |

Metric limitations, independent of the run:

- **Boundary IoU is the weak metric by construction.** Even the published
  number is 28.60%; it is computed on the dilated boundary band, so single-pixel
  shifts move it a lot. Do not treat it as a quality gate.
- **HD95/ASSD are reported in pixels**, not mm; convert with the px→mm
  calibration (`notes/triage_calibration.md`) before any physical claim.
- **The conditions aggregate is not the published per-species table**: ours is
  one 60-image mean, theirs is three 10-image means.
- **The 1,200-iteration CPU run is an existence proof**, not a baseline: its
  LR had barely left warmup-relative scale for part of the run and it saw each
  training image ~10 times rather than the published schedule.

## What these results do and do not support (#59)

Supported by the validated checkpoint (GPU run, `full-gpu/`):

- **Segmentation masks and measurement-error work (#29, #31)**: per-image
  mycelium area/radius/perimeter error on held-out labeled masks, bias and SD
  of morphometrics, duplicate sensitivity. The test split is the same
  50-image GL hold-out the paper reports, so errors are comparable to Table 6.
- **Mask generation for any downstream measurement in this repo**, with the
  metric limitations above carried into the measurement-error model.

Explicitly **not** supported:

- **Biological growth**: MyceliumSeg is cross-sectional — no usable ordered
  frames (`notes/provenance.md`), so no rate, growth curve or time-to-event
  claim comes from these masks.
- **Tip formation / hyphal events**: boundary-level events are below the
  resolution that Boundary IoU ~0.19–0.28 certifies; candidate protrusions
  stay secondary until false-event rates are quantified (#62, #31).
- **Causal temperature effects**: TEMP15 is one condition folder in a
  cross-sectional set, not an intervention; temperature effects need the
  controlled-rig path in PLAN.md Phase 4.
- **Any "exact reproduction" claim**: see the deviations tables — the baseline
  matches the published protocol in iterations, effective batch, LR, warmup,
  data split and metrics, but not in hardware count, backbone checkpoint or
  seed provenance.

Handoff: the validated checkpoint is
`work_dirs/segformer-b0-repro/full-gpu/train/` (final iteration, SHA-256 in
`full-gpu/evidence.json`), config `full-gpu/config.py`. #29 (inference →
masks) and #31 (measurement-error model) consume it from there; do not retrain
for either.

## Reproduce

```bash
# GPU baseline (the hand-off run, #58): ~5 h train + ~40 min two evals
nix develop .#gpu --command python scripts/segformer_repro.py run \
  --tag gpu --max-iters 50000 --batch-size 16 --workers 8

# Five-minute end-to-end smoke check (CPU shell)
nix develop --command python scripts/segformer_repro.py run --smoke
# ... or on the GPU shell, which is what #57 recorded
nix develop .#gpu --command python scripts/segformer_repro.py run --smoke

# The earlier short CPU run (existence proof, ~2.5 h)
nix develop --command python scripts/segformer_repro.py prepare
nix develop --command python scripts/segformer_repro.py config --max-iters 1200
nix develop --command python scripts/segformer_repro.py train
nix develop --command python scripts/segformer_repro.py evaluate --split test
nix develop --command python scripts/segformer_repro.py evaluate --split conditions

# Evidence JSON for either run (#59)
nix develop .#gpu --command python scripts/segformer_repro.py evidence --tag gpu
```

Tests and lint for the driver itself (no GPU needed):

```bash
.venv/bin/python -m pytest tests -q          # includes tests/test_segformer_repro.py
ruff check scripts/segformer_repro.py scripts/segformer_shim tests/test_segformer_repro.py
```
