#!/bin/bash
# Builds the environment for the gs stage and the demo notebook (official graphdeco 3DGS).
# Its three CUDA extensions are compiled here, so this needs, on the machine that will train:
#   - an NVIDIA GPU and driver, visible to nvidia-smi
#   - the CUDA toolkit 12.x with nvcc (on PATH, or under /usr/local/cuda)
#   - a C++ compiler (gcc 11-14) and uv (https://docs.astral.sh/uv/)
# Tested with: RTX PRO 6000 Blackwell (compute capability 12.0), CUDA 12.8, gcc 13, torch 2.7.1+cu128.
#
#   git submodule update --init third_party/gaussian-splatting
#   env/gaussian_splatting.sh            # ~10 min, creates $ENV_DIR (default ~/envs/gaussian_splatting)
#   source env/gaussian_splatting.activate
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../third_party/gaussian-splatting" && pwd)
ENV_DIR=${ENV_DIR:-$HOME/envs/gaussian_splatting}
export PATH="$HOME/.local/bin:$PATH"   # uv's default install location

if ! command -v nvcc >/dev/null; then
  for c in /usr/local/cuda /usr/local/cuda-12*; do
    [ -x "$c/bin/nvcc" ] && export PATH="$c/bin:$PATH" CUDA_HOME="$c" && break
  done
fi
command -v nvcc >/dev/null || { echo "nvcc not found: install the CUDA toolkit"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible"; exit 1; }
command -v uv >/dev/null || { echo "uv not found"; exit 1; }
[ -f "$REPO/train.py" ] || { echo "run: git submodule update --init third_party/gaussian-splatting"; exit 1; }
# build for the GPUs present (set TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" etc. for other machines)
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd';')}
echo "== nvcc $(nvcc --version | grep -oE 'release [0-9.]+'), arch $TORCH_CUDA_ARCH_LIST, env $ENV_DIR"

git -C "$REPO" submodule update --init --recursive \
  submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim

[ -f "$ENV_DIR/bin/activate" ] || uv venv "$ENV_DIR" --python 3.10
source "$ENV_DIR/bin/activate"
uv pip install "numpy<2"
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install ninja wheel setuptools
# the rasterizer uses uint32_t without <cstdint>, which gcc >= 13 no longer includes transitively
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -include cstdint" CFLAGS="${CFLAGS:-} -include cstdint" CXXFLAGS="${CXXFLAGS:-} -include cstdint"
echo "== compiling the CUDA extensions (~10 min)"
uv pip install --no-build-isolation \
  "$REPO/submodules/diff-gaussian-rasterization" "$REPO/submodules/simple-knn" "$REPO/submodules/fused-ssim"
uv pip install plyfile tqdm "opencv-python-headless<5" joblib "numpy<2" jupyterlab ipykernel matplotlib
python -m ipykernel install --user --name gaussian_splatting

# smoke test: the kernels run on the GPU and train.py imports
cd "$REPO"
python - <<'PY'
import torch
from diff_gaussian_rasterization import GaussianRasterizer  # noqa: F401
from simple_knn._C import distCUDA2
from fused_ssim import fused_ssim
x = torch.rand(256, 3, device="cuda")
assert torch.isfinite(distCUDA2(x)).all()
assert torch.isfinite(fused_ssim(torch.rand(1, 3, 64, 64, device="cuda"), torch.rand(1, 3, 64, 64, device="cuda")))
print("kernels ok:", torch.cuda.get_device_name(0), "torch", torch.__version__, "cuda", torch.version.cuda)
PY
python train.py --help >/dev/null && echo "train.py ok"
echo "ENV_OK $ENV_DIR"
