# Mycelium Growth Causal Modeling

One `nix develop` at the repository root gives both halves of the toolchain: the R/Stan
stack for the causal models, and the Python stack the
[MyceliumSeg benchmark](external/MyceliumSeg-benchmark) needs for masks and shape
metrics. There is a single flake and a single shell - `flake.nix` and `flake.lock` at
the root - so the same pinned versions land on any machine without pip or
install.packages.

Every version below was read out of the shell with the command shown next to it.

## Quick start

Clone with `git clone --recurse-submodules` to check out the pinned source trees in
`external/`. For an existing clone, run `git submodule update --init --recursive`.

```bash
# Interactive shell with both stacks (run from the repository root)
nix develop

# One-shot commands
nix develop --command python -c "import mmcv, mmseg, torch; print(mmcv.__version__, mmseg.__version__, torch.__version__)"
nix develop --command Rscript -e 'library(rethinking); print(packageVersion("rethinking"))'

# Gate both halves (see "Verification" below)
nix develop --command bash scripts/verify_python_stack.sh
nix develop --command python scripts/verify_benchmark_models.py
```

`nix develop` looks for `flake.nix` in the working directory and then upwards, so it
only works from inside this repository. To get the environment while sitting elsewhere,
point it at the flake:

```bash
cd external/MyceliumSeg-benchmark
nix develop ../.. --command python -c "import mmseg; print(mmseg.__version__, mmseg.__file__)"
# 0.11.0 .../external/MyceliumSeg-benchmark/mmseg/__init__.py
```

## What the shell provides

### R 4.4.1

| Package | Version | Source |
| --- | --- | --- |
| rethinking | 2.21 | built from GitHub tag `v2.2.1` |
| cmdstanr | 0.9.0 | built from GitHub tag `v0.9.0` |
| StanHeaders | 2.32.10 | nixpkgs |
| rstan | 2.32.6 | nixpkgs |
| dagitty | 0.3.4 | nixpkgs |
| posterior | 1.6.0 | nixpkgs |
| loo | 2.8.0 | nixpkgs |
| bayesplot | 1.11.1 | nixpkgs |
| tidyverse | 2.0.0 | nixpkgs |
| data.table | 1.15.4 | nixpkgs |
| ggplot2 | 3.5.1 | nixpkgs |
| igraph | 2.0.3 | nixpkgs |

Plus Rcpp, RcppParallel, Matrix, coda, mvtnorm, processx, jsonlite, httr, desc,
pkgbuild, callr, R6, vctrs, checkmate, rstudioapi and V8.

```bash
nix develop --command Rscript -e 'for (p in c("rethinking","cmdstanr","dagitty","posterior")) print(c(p, as.character(packageVersion(p))))'
```

### CmdStan 2.35.0

`$CMDSTAN` points at the CmdStan tree and `cmdstanr` picks it up from there:

```bash
nix develop --command Rscript -e 'print(cmdstanr::cmdstan_version())'   # [1] "2.35.0"
```

`stanc` is a symlink to nixpkgs' `stanc-2.35.0`. Asking it for its version prints an
unexpanded `%%NAME%%3 %%VERSION%% (Unix)` placeholder, so read the number from
`cmdstanr::cmdstan_version()` instead.

### Python 3.12.8

| Package | Version | Notes |
| --- | --- | --- |
| mmcv | 1.3.0 | built from source; CPU ops compiled into `mmcv._ext` |
| mmsegmentation | 0.12.0 | built from source |
| torch | 2.5.1 | CPU only: `torch.cuda.is_available()` is `False` |
| torchvision / torchaudio | 0.20.1a0 / 2.5.1a0 | CPU only |
| scikit-image | 0.24.0 | mask and shape metrics |
| medpy | 0.5.2 | distance metrics |
| scikit-learn | 1.5.2 | |
| numpy | 1.26.4 | |
| scipy | 1.14.1 | |
| pandas | 2.2.3 | |
| matplotlib | 3.9.2 | |
| seaborn | 0.13.2 | |
| opencv (`cv2`) | 4.9.0 | |
| Pillow | 11.0.0 | |
| timm | 1.0.11 | |
| fsspec / safetensors | 2024.3.0 / 0.4.5 | |
| pytest / yapf / isort | 8.3.3 / 0.40.2 / 5.13.2 | |
| IPython | 8.29.0 | |
| exiftool | 13.00 | CLI, on `PATH` |

Plus `tqdm`, `terminaltables`, `addict`, `pyyaml` and `packaging`.

#### Why mmcv and mmsegmentation are built here

nixpkgs 24.11 only ships mmcv 2.x, and the benchmark's mmseg fork asserts
`1.1.4 <= mmcv <= 1.3.0` in `external/MyceliumSeg-benchmark/mmseg/__init__.py`. The
flake therefore builds mmcv-full 1.3.0 and mmsegmentation 0.12.0 from source, with two
patches from `patches/`:

- `mmc-v1.3.0-remove-pytest-runner.patch` drops the `setup_requires=['pytest-runner']`
  that would otherwise be fetched from PyPI during the sandboxed build.
- `mmc-v1.3.0-torch2-ddp-gpu-num.patch` restores `SyncBatchNorm._specify_ddp_gpu_num`,
  which torch 2.x removed and mmcv 1.3.0 calls for every SyncBN layer.

Only the C++ (CPU) kernels are compiled in the default shell
(`MMCV_WITH_OPS=1`, `MMCV_WITH_CUDA=0`), so no CUDA toolkit is needed there.

#### GPU shell

The repository also exposes `devShells.gpu` for the RTX 4070 Ti host used for the
SegFormer work. It keeps the CPU shell as the reproducible default and selects a
CUDA-enabled PyTorch, torchvision, mmcv, and mmsegmentation stack only
when requested:

```bash
# Build the GPU package closure once, then enter the shell
nix build .#gpuPythonEnv --no-link
nix develop .#gpu

# Confirm that PyTorch can see the host driver and GPU
nix develop .#gpu --command python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
```

The GPU shell exposes the host `libcuda.so.1` through `LD_PRELOAD` in its shell hook.
It uses the runtime-library pattern from the `backrooms-workspace` flake while
avoiding the host system library directory, which can override the shell's glibc.
Nix evaluation stays safe on machines without a GPU, while the runtime check is
performed only when the GPU shell is entered.

## Usage

### R and Stan

```bash
nix develop                      # then, inside the shell:
R                                # interactive session
```

```r
library(rethinking)
library(cmdstanr)
library(dagitty)
library(posterior)

cmdstanr::cmdstan_version()      # "2.35.0", picked up from $CMDSTAN
dagitty::dagitty('dag { x -> y }')
```

### Python

```bash
nix develop --command python     # interactive interpreter
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### MMSegmentation benchmark

The benchmark vendors its own mmseg fork - version string `0.11.0`, an upstream
0.11.0-era tree plus the SegFormer `mit_b*` backbones and the `EPA_mycelium` dataset.
When the working directory is the benchmark root, that fork shadows the packaged
mmsegmentation 0.12.0, because the working directory (or the script's own directory) is
`sys.path[0]`. Run the benchmark's entry points from there:

```bash
cd external/MyceliumSeg-benchmark
nix develop ../.. --command python test-infer.py CONFIG CHECKPOINT
nix develop ../.. --command env PYTHONPATH=. python -c "import mmseg; print(mmseg.__file__)"
```

Real inference and training need the trained checkpoints, which are not in this
repository. To check that the stack can build and run the benchmark's models without
them, use the smoke test below.

### SegFormer baseline reproduction

`scripts/segformer_repro.py` runs the benchmark's own `train.py` / `test.py` end
to end: it unpacks the three labeled release zips into the layout the benchmark's
`EPA_mycelium` dataset expects (`data/segmentation/{train,test,conditions}`),
renders a config layered on the authors' `segformer.b0.1024x1024.mycelium.507.py`,
trains SegFormer-B0, and evaluates with the five metrics the paper reports
(mIoU, mDice, ASSD, HD95, Boundary IoU):

```bash
# Minutes: 2 iterations on 4+2+2 images, proves the whole path works
nix develop --command python scripts/segformer_repro.py run --smoke

# The short reproduction run (CPU; --max-iters and friends are configurable)
nix develop --command python scripts/segformer_repro.py run

# The published protocol on the available GPU: 50,000 iterations at the
# published effective batch of 16 (4 GPUs x 4 -> one 12 GB GPU), ~5 h
nix develop .#gpu --command python scripts/segformer_repro.py run \
  --tag gpu --max-iters 50000 --batch-size 16 --workers 8
```

Artifacts land in `work_dirs/segformer-b0-repro/` (configs, logs, the final
checkpoint, `metrics.json`, and `evidence.json` — device, driver, versions,
batch size, peak GPU memory, checkpoint SHA-256 and metrics, assembled by the
`evidence` subcommand that `run` invokes last); `--tag NAME` writes to
`full-NAME`/`smoke-NAME` so runs do not overwrite each other. The results and
their deviations from the published protocol are written up in
[notes/segformer_baseline.md](notes/segformer_baseline.md).
The benchmark's entry points assume numpy < 1.24 and yapf < 0.40, neither of
which this stack has, and (on the CPU shell) a CUDA-less torch, so
`segformer_repro.py` starts them with a compatibility shim from
`scripts/segformer_shim/` on `PYTHONPATH` - only its subprocesses see it.

### scikit-image for mask shape metrics

The labelled masks are binary PNGs holding 0/1 as uint8, so binarise with `> 0` -
a `> 127` habit silently yields an empty mask:

```bash
nix develop --command python - <<'PY'
import numpy as np
from skimage import io, measure

mask = io.imread("path/to/mask.png", as_gray=True) > 0
regions = measure.regionprops(measure.label(mask))
for r in sorted(regions, key=lambda r: -r.area)[:3]:
    print(f"area={r.area:8.0f} perimeter={r.perimeter:9.1f} eccentricity={r.eccentricity:.3f}")
PY
```

On the `labeled-GS_PO_TS` mask `00018428` (3456x4608) that reports one component, area
1214257 px, perimeter 8431.9, eccentricity 0.184.

### exiftool

```bash
nix develop --command exiftool -ver                        # 13.00
nix develop --command exiftool -csv -FileName -DateTimeOriginal -Model \
    -ImageWidth -ImageHeight path/to/extracted/images/
nix develop --command exiftool -s path/to/image.jpg        # every tag, one file
```

`exiftool` reads loose files, not JPEG members inside the download zips - for those use
`scripts/triage/exif_scan.py`, whose findings are written up in
[notes/triage_exif.md](notes/triage_exif.md).

## Verification

Both scripts are meant to be used as gates: they report failed checks and exit non-zero if any check fails.

### `scripts/verify_python_stack.sh`

```bash
nix develop --command bash scripts/verify_python_stack.sh    # ~19 s wall clock, 15 s of checks
```

Prints the resolved versions, then checks `import mmcv`, `import mmcv.ops`,
`import mmseg`, `mmseg.models` / `.datasets` / `.apis`, `skimage` and `cv2`, runs four
compiled CPU kernels out of `mmcv._ext` (`nms`, `RoIAlign`, `point_sample` and a
`SyncBN` layer - `import mmcv` alone succeeds even when the extension is missing), and
checks `exiftool` on `PATH`.

### `scripts/verify_benchmark_models.py`

```bash
nix develop --command python scripts/verify_benchmark_models.py          # ~7 s, one config
nix develop --command python scripts/verify_benchmark_models.py --all     # ~9 s, all three
nix develop --command python scripts/verify_benchmark_models.py --list    # the known configs
```

Each model is built from its config with random weights and run once on a 128x128
synthetic image; the check asserts the backbone's feature strides and the number of
output classes. By default only the SegFormer `mit_b0` config is checked - the one
`script_train.sh` and `script_inference.sh` use, and the one whose backbone exists only
in the benchmark's fork, so it fails if that fork does not shadow the packaged
mmsegmentation. `--all` adds the two UNet configs (`pspnet_unet_s5-d16`,
`deeplabv3_unet_s5-d16`).

Flags: `--config PATH` (repeatable), `--image-size N`, `--threads N`, `--timeout
SECONDS`. Exit codes: `0` all checks passed, `1` a check failed, `2` the environment is
not usable, `3` the `--timeout` budget expired.

## Development workflow

```bash
nix flake update                        # bump the pinned nixpkgs / flake-utils
nix build .#devShell                    # realise the shell without entering it
nix build .#pythonEnv                   # the Python half on its own
nix build .#gpuPythonEnv                # the CUDA-enabled Python half
nix build .#rEnv                        # the R half on its own
nix build .#cmdstan                     # CmdStan on its own
```

`nix build .#devShells.default` does not work: the flake exposes the shell as
`devShells.default` for `nix develop` and as `packages.devShell` for `nix build`.

The repository's own tests run in `.venv`, not in the Nix shell:

```bash
.venv/bin/python -m pytest tests -q      # 245 tests
```

## Layout

```
flake.nix, flake.lock    the one shell: R stack + Python stack + CmdStan
patches/                 patches applied to mmcv 1.3.0 at build time
scripts/                 data fetching, metadata triage, verification scripts
  triage/                EXIF, label, numbering, dish-track, px→mm scale and sequence-provenance triage for the raw downloads
  measurement_contract.py  estimands, units and reporting guards (notes/estimands.md)
external/                vendored third-party checkouts (MyceliumSeg-benchmark, ...)
data/
  raw/                   downloaded archives (gitignored, ~73 GB)
  triage/                triage output CSVs
notes/                   findings and decisions, e.g. notes/gate.md, notes/estimands.md, notes/provenance.md
tests/                   pytest suite for scripts/ (triage, the measurement contract, the sequence inventory)
logs/                    fetch logs
```

The measurement contract — what is estimated, in which unit, over which
population, under which precondition — is
[notes/estimands.md](notes/estimands.md), mirrored in
`scripts/measurement_contract.py` and checked by
`tests/test_measurement_contract.py`.

The sequence provenance inventory — which dataset has ordered frames, specimen
identity, a capture interval, and what each can consequently identify — is
[notes/provenance.md](notes/provenance.md), generated by
`scripts/triage/sequence_inventory.py` into
`data/triage/sequence_inventory.csv` and checked by
`tests/test_sequence_inventory.py`.

The analysis layout in [PLAN.md](PLAN.md) (`R/`, `py/`) is not built yet.

## Known limitations

- **Multi-threaded torch deadlocks on this path.** With
  `torch.get_num_threads() >= 2`, a forward pass of the benchmark's `mit_b0` backbone on
  an input of 64x64 or larger never returns: the calling thread spins at 100% CPU inside
  `nn.Linear.forward` while the OpenMP workers sit in `futex_wait`. 32x32 inputs and
  `torch.set_num_threads(1)` both complete, and a hand-built module with the same
  architecture does not hang, so this looks like a property of nixpkgs' torch 2.5.1
  (`BLAS_INFO=open`, `USE_EIGEN_FOR_BLAS=ON`, `USE_OPENMP=ON`) rather than of the model.
  Until it is understood, run torch work single-threaded;
  `verify_benchmark_models.py` does that by default and explains why in
  `pin_torch_threads()`. **This affects the default (CPU) shell only** - the
  GPU shell's torch is a CUDA build without the issue, and its shell hook pins
  `OMP_NUM_THREADS=1` anyway.
- **CPU only (default shell).** `devShells.default` is built with `USE_CUDA=OFF`, so
  `torch.cuda.is_available()` is `False` there and the benchmark's `dist_train.sh` /
  `script_inference.sh`, which assume several GPUs and `torch.distributed.launch`,
  cannot be used as written. Use `nix develop .#gpu` for CUDA (see "GPU shell"
  above); its mmcv 1.3.0 still has **no CUDA kernels** (the sources need removed THC
  headers), which this protocol does not need.
- **`stanc --version` prints a placeholder** (`%%NAME%%3 %%VERSION%% (Unix)`). Read the
  CmdStan version from `cmdstanr::cmdstan_version()`.
- **Six harmless locale warnings** (`input string ' ±' cannot be translated from
  'ANSI_X3.4-1968' to UTF-8`) appear when the shell hook loads `rethinking`. Exporting
  `LANG`/`LC_ALL` as `C.UTF-8` does not silence them; every package still loads.
- **`nix develop` needs the repository root.** From a subdirectory, name the flake
  (`nix develop ../..`) because Nix refuses to search above the working directory's
  parent boundary.
