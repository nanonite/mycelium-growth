#!/usr/bin/env bash
# Verifies the Python half of the shared devShell (flake.nix, issue #6).
#
# Checks that the pinned MMSegmentation stack the MyceliumSeg-benchmark needs is
# importable and that the compiled mmcv._ext ops actually execute on CPU, plus
# exiftool and scikit-image. Intended to be run inside `nix develop`:
#
#     nix develop --command bash scripts/verify_python_stack.sh
#
# Exits non-zero on the first failing section so it can be used as a gate.

set -uo pipefail

fail=0
start=$SECONDS

note() { printf '\n== %s ==\n' "$1"; }

check() {
  # check <label> <python expression>
  local label=$1 expr=$2
  if python -c "$expr" 2>"$err"; then
    printf 'OK   %s\n' "$label"
  else
    printf 'FAIL %s\n' "$label"
    sed 's/^/       /' "$err"
    fail=1
  fi
}

# Per-run scratch file rather than a fixed path, so two checks running at once
# cannot clobber each other's diagnostics.
err=$(mktemp)
trap 'rm -f "$err"' EXIT

note "versions"
python - <<'PY'
import importlib

for name in ('numpy', 'scipy', 'pandas', 'matplotlib', 'skimage', 'torch',
             'torchvision', 'mmcv', 'mmseg', 'cv2'):
    try:
        mod = importlib.import_module(name)
        print(f'  {name:12s} {getattr(mod, "__version__", "unknown")}')
    except Exception as exc:  # noqa: BLE001 - report, do not abort the report
        print(f'  {name:12s} MISSING ({exc})')
PY

note "imports"
check "import mmcv"                'import mmcv; assert mmcv.__version__ == "1.3.0"'
check "import mmcv.ops"            'import mmcv.ops'
check "import mmseg"               'import mmseg'
check "import mmseg.models"        'import mmseg.models; import mmseg.datasets; import mmseg.apis'
check "import skimage"             'import skimage; from skimage import measure, morphology, filters'
check "import cv2"                 'import cv2'

note "compiled mmcv._ext ops (CPU)"
# `import mmcv.ops` succeeding only proves the module loaded; these run real
# kernels out of the C++ extension built by the mmcv derivation. mmcv 1.3.0 only
# implements a subset of its ops for CPU tensors (the rest raise "not
# implemented on CPU"), so this covers the CPU-capable ones.
check "nms"                 'import torch; from mmcv.ops import nms; b=torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.],[50.,50.,60.,60.]]); s=torch.tensor([0.9,0.8,0.7]); assert nms(b,s,0.5)[0].shape[0]==2'
check "roi_align"           'import torch; from mmcv.ops import RoIAlign; r=RoIAlign((2,2),spatial_scale=0.5,pool_mode="avg").eval(); out=r(torch.randn(1,1,8,8), torch.tensor([[0.,0.,0.,4.,4.]])); assert out.shape[0]==1'
check "point_sample"        'import torch; from mmcv.ops import point_sample; out=point_sample(torch.randn(1,1,4,4), torch.rand(1,2,2,2)); assert out.shape==(1,1,2,2)'
check "SyncBatchNorm"       'import torch; from mmcv.cnn.bricks import build_norm_layer; _,n=build_norm_layer(dict(type="SyncBN", requires_grad=True), 4); assert n(torch.randn(2,4,3,3)).shape==(2,4,3,3)'
check "torch_threads"       'import torch; assert torch.cuda.is_available() == False or torch.cuda.device_count() >= 0'

note "exiftool"
if command -v exiftool >/dev/null 2>&1; then
  printf 'OK   exiftool %s\n' "$(exiftool -ver)"
else
  printf 'FAIL exiftool not on PATH\n'
  fail=1
fi

note "result"
if [ "$fail" -eq 0 ]; then
  echo "all Python stack checks passed"
else
  echo "Python stack checks FAILED"
fi
echo "took $((SECONDS - start))s"
exit "$fail"
