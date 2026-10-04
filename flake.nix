{
  description = "Mycelium growth causal modeling environment - shared R and Python stacks";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; };
        };

        # Build custom R packages NOT available in nixpkgs
        customR = {
          # cmdstanr from GitHub (not on CRAN)
          cmdstanr = pkgs.rPackages.buildRPackage {
            pname = "cmdstanr";
            name = "cmdstanr";
            version = "0.9.0";
            src = pkgs.fetchFromGitHub {
              owner = "stan-dev";
              repo = "cmdstanr";
              rev = "v0.9.0";
              sha256 = "sha256-wXfOxBexnuL83fvCM+6qv6d7UhTLGq+Xhvja3lRfQpI=";
            };
            propagatedBuildInputs = with pkgs.rPackages; [
              posterior
              processx
              rstudioapi
              jsonlite
              httr
              desc
              pkgbuild
              callr
              R6
              data_table
              withr
            ];
            meta = {
              homepage = "https://mc-stan.org/cmdstanr/";
              license = pkgs.lib.licenses.mit;
            };
          };

          # rethinking from GitHub
          rethinking = pkgs.rPackages.buildRPackage {
            pname = "rethinking";
            name = "rethinking";
            version = "2.2.1";
            src = pkgs.fetchFromGitHub {
              owner = "rmcelreath";
              repo = "rethinking";
              rev = "v2.2.1";
              sha256 = "sha256-0f3bBtoRSV9QGjtnbGId00aEd4XTb+9rvL9DS/Lezxo=";
            };
            propagatedBuildInputs = with pkgs.rPackages; [
              mvtnorm
              loo
              coda
              Matrix
              Rcpp
              RcppParallel
              StanHeaders
              rstan
              customR.cmdstanr
              dagitty
              posterior
              shape
            ];
            nativeBuildInputs = [ pkgs.gnumake ];
            meta = {
              homepage = "https://github.com/rmcelreath/rethinking";
              license = pkgs.lib.licenses.gpl3;
            };
          };
        };

        # CmdStan from nixpkgs (provides the CmdStan binaries)
        cmdstan = pkgs.cmdstan;

        # Number of parallel compiler jobs used to build the mmcv C++ ops.
        # Every op translation unit pulls in <torch/extension.h>, so the build is
        # memory bound rather than CPU bound; 6 keeps it inside a modest box.
        mmcvMaxJobs = "6";

        # Build custom Python packages NOT available in nixpkgs.
        #
        # nixpkgs 24.11 only ships mmcv 2.x, which is incompatible with the
        # mmsegmentation 0.12.0 fork used by external/MyceliumSeg-benchmark: that
        # fork asserts 1.1.4 <= mmcv <= 1.3.0 in mmseg/__init__.py. mmcv 1.3.0 is
        # therefore built from source here, together with the matching
        # mmsegmentation 0.12.0.
        customPython = {
          # mmcv 1.3.0 (mmcv-full) from GitHub, required by MyceliumSeg-benchmark.
          # The C++ (CPU) op implementations are compiled into mmcv._ext.
          mmcv = pkgs.python312Packages.buildPythonPackage rec {
            pname = "mmcv";
            version = "1.3.0";
            format = "setuptools";
            src = pkgs.fetchFromGitHub {
              owner = "open-mmlab";
              repo = "mmcv";
              rev = "v${version}";
              sha256 = "sha256-5neLt9bXeINzq+W3AEC0UhDfy9mFlt1gIY6y+4xGkUc=";
            };

            # mmcv 1.x decides what to compile from these variables. Without
            # MMCV_WITH_OPS=1 setup.py returns no ext_modules at all, so the
            # wheel ships pure Python and every mmcv.ops import dies with
            # "No module named 'mmcv._ext'". MMCV_WITH_CUDA=0 / FORCE_CUDA=0
            # keep the build on the C++ kernels so no CUDA toolkit is needed.
            MMCV_WITH_OPS = "1";
            MMCV_WITH_CUDA = "0";
            FORCE_CUDA = "0";
            MAX_JOBS = mmcvMaxJobs;

            patches = [
              # setup_requires=['pytest-runner'] makes setuptools try to fetch
              # pytest-runner from PyPI during the build; there is no network in
              # the sandbox and the dependency is not needed to install the wheel.
              ./patches/mmc-v1.3.0-remove-pytest-runner.patch
              # torch >= 2.0 removed the SyncBatchNorm._specify_ddp_gpu_num hook
              # that mmcv 1.3.0 calls for every SyncBN norm layer.
              ./patches/mmc-v1.3.0-torch2-ddp-gpu-num.patch
            ];

            # torch.cxxdev provides the C++ headers and the libtorch libraries
            # that mmcv/ops/csrc compiles and links against.
            nativeBuildInputs = [
              pkgs.ninja
              pkgs.gnumake
              pkgs.which
              pkgs.python312Packages.cython
              pkgs.python312Packages.torch.cxxdev
            ];
            buildInputs = [ pkgs.python312Packages.torch.cxxdev ];

            # Runtime imports of mmcv 1.3.0: numpy/torch for the ops, plus the
            # small helpers pulled in by mmcv.{arraymisc,image,utils}.
            dependencies = with pkgs.python312Packages; [
              numpy
              torch
              torchvision
              addict
              packaging
              pillow
              pyyaml
              # mmcv.utils.config imports yapf.yapflib at module scope
              yapf
              # mmcv.utils.env imports cv2 at module scope; mmseg also uses it
              # for imread/imwrite
              opencv4
            ];

            # `import mmcv` alone succeeds even without the compiled ops, so
            # import mmcv.ops as well: that is the import that actually needs
            # mmcv._ext.
            pythonImportsCheck = [ "mmcv" "mmcv.ops" ];
            doCheck = false;

            meta = {
              description = "OpenMMLab Computer Vision Foundation (mmcv-full 1.3.0, CPU ops)";
              homepage = "https://github.com/open-mmlab/mmcv";
              license = pkgs.lib.licenses.asl20;
              platforms = pkgs.lib.platforms.linux;
            };
          };

          # mmsegmentation 0.12.0 from GitHub, matching the mmcv 1.3.0 above.
          mmsegmentation = pkgs.python312Packages.buildPythonPackage rec {
            pname = "mmsegmentation";
            version = "0.12.0";
            format = "setuptools";
            src = pkgs.fetchFromGitHub {
              owner = "open-mmlab";
              repo = "mmsegmentation";
              rev = "v${version}";
              sha256 = "sha256-g7lradaW7boiqgObqrFFBtlUkmDyhZF06ux0XvHIyrI=";
            };
            dependencies = with pkgs.python312Packages; [
              (customPython.mmcv.overridePythonAttrs (_: { pname = "mmcv"; }))
              numpy
              torch
              matplotlib
              terminaltables
              addict
              pyyaml
              timm
              fsspec
            ];
            pythonImportsCheck = [ "mmseg" ];
            doCheck = false;
            meta = {
              description = "OpenMMLab semantic segmentation toolkit 0.12.0";
              homepage = "https://github.com/open-mmlab/mmsegmentation";
              license = pkgs.lib.licenses.asl20;
              platforms = pkgs.lib.platforms.linux;
            };
          };
        };

        # List of R packages to include in the environment
        rPackageList = [
          customR.cmdstanr
          customR.rethinking
          # Use nixpkgs versions for everything else
          pkgs.rPackages.dagitty        # 0.3-4
          pkgs.rPackages.posterior      # 1.7.0
          pkgs.rPackages.tidyverse
          pkgs.rPackages.data_table
          pkgs.rPackages.ggplot2
          pkgs.rPackages.bayesplot
          pkgs.rPackages.loo
          pkgs.rPackages.rstan
          pkgs.rPackages.StanHeaders
          pkgs.rPackages.Rcpp
          pkgs.rPackages.RcppParallel
          pkgs.rPackages.Matrix
          pkgs.rPackages.coda
          pkgs.rPackages.mvtnorm
          pkgs.rPackages.processx
          pkgs.rPackages.jsonlite
          pkgs.rPackages.httr
          pkgs.rPackages.desc
          pkgs.rPackages.pkgbuild
          pkgs.rPackages.callr
          pkgs.rPackages.R6
          pkgs.rPackages.vctrs
          pkgs.rPackages.checkmate
          pkgs.rPackages.rstudioapi
          pkgs.rPackages.V8
          pkgs.rPackages.igraph
        ];

        # Create R wrapper with all required packages
        rWithPackages = pkgs.rWrapper.override {
          packages = rPackageList;
        };

        # Python environment for MMSegmentation (MyceliumSeg-benchmark)
        # Using pinned/reproducible versions compatible with the benchmark requirements
        pythonEnv = pkgs.python312.withPackages (ps: with ps; [
          # Core scientific stack
          numpy
          pandas
          scipy
          matplotlib
          scikit-learn
          scikit-image
          seaborn

          # PyTorch ecosystem (CPU variants for portability; CUDA can be enabled via override)
          torch
          torchvision
          torchaudio

          # MMSegmentation dependencies
          customPython.mmcv
          customPython.mmsegmentation

          # Image processing
          opencv4
          pillow

          # Additional utilities from requirements
          addict
          pyyaml
          tqdm
          terminaltables
          timm
          safetensors
          fsspec
          packaging
          medpy
          ipython

          # Testing utilities (optional)
          pytest
          yapf
          isort
        ]);

        # Development shell
        devShell = pkgs.mkShell {
          name = "mycelium-growth-dev";
          buildInputs = [
            rWithPackages
            cmdstan
            pythonEnv
            pkgs.git
            pkgs.exiftool
            pkgs.graphviz
            pkgs.gnumake
            pkgs.cmake
            pkgs.gcc
            pkgs.glibc
            pkgs.zlib
            pkgs.libiconv
            pkgs.ncurses
            pkgs.readline
            pkgs.xz
            pkgs.bzip2
            pkgs.sqlite
            pkgs.openblas
          ];

          # Environment variables for CmdStan
          CMDSTAN = "${cmdstan}/opt/cmdstan";
          CMDSTANPY = "${cmdstan}/opt/cmdstan";

          # Shell hook for convenient setup
          shellHook = ''
            echo "=== Mycelium Growth Development Environment ==="
            echo "R version: $(R --version | head -1)"
            echo "CmdStan: $CMDSTAN"
            echo ""
            echo "Verifying R packages..."
            R --quiet -e "
              pkgs <- c('rethinking', 'cmdstanr', 'dagitty', 'posterior', 'rstan', 'loo', 'bayesplot')
              for (p in pkgs) {
                if (requireNamespace(p, quietly = TRUE)) {
                  cat(sprintf('✓ %s (v%s)\\n', p, packageVersion(p)))
                } else {
                  cat(sprintf('✗ %s NOT FOUND\\n', p))
                }
              }
            "
            echo ""
            echo "Verifying Python stack..."
            python - <<'PY'
            import importlib
            for name in ('numpy', 'torch', 'mmcv', 'mmseg', 'skimage'):
                try:
                    mod = importlib.import_module(name)
                    print(f"✓ {name} (v{getattr(mod, '__version__', '?')})")
                except Exception as exc:
                    print(f"✗ {name} FAILED: {exc}")
            PY
            echo ""
            echo "Verifying CmdStan..."
            if [ -f "$CMDSTAN/bin/stanc" ]; then
              echo "✓ CmdStan stanc compiler found"
              "$CMDSTAN/bin/stanc" --version
            else
              echo "✗ CmdStan not found at $CMDSTAN"
            fi
            echo ""
            echo "Run 'R' to start R session, or 'nix develop' to re-enter"
            echo "For Python: 'python'"
            echo "Full checks: scripts/verify_python_stack.sh, scripts/verify_benchmark_models.py"
          '';
        };

      in
      {
        devShells.default = devShell;

        packages = {
          default = rWithPackages;
          # The R and Python halves are also exposed on their own so other
          # projects can depend on either stack via `inputs.mycelium.packages.*`.
          rEnv = rWithPackages;
          pythonEnv = pythonEnv;
          cmdstan = cmdstan;
          mmcv = customPython.mmcv;
          mmsegmentation = customPython.mmsegmentation;
          devShell = devShell;
        };
      }
    );
}