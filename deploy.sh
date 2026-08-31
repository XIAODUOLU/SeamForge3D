#!/usr/bin/env bash
set -euo pipefail

PROFILE="intel"
if [[ "${1:-}" == "--profile" ]]; then PROFILE="${2:-}"; fi
if [[ "${PROFILE}" != "intel" && "${PROFILE}" != "cuda" ]]; then
  echo "Usage: ./deploy.sh --profile intel|cuda" >&2; exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTONIA_COMMIT="da776a0bd3a48c6df83ac2ae0e27b26141cc7e31"
UTONIA_DIR="${PROJECT_DIR}/third_party/Utonia"
command -v conda >/dev/null || { echo "conda is required" >&2; exit 1; }

if [[ "${PROFILE}" == "intel" ]]; then
  ENV_NAME="${SEAMFORGE_ENV_NAME:-seamforge3d-intel}"
  ENV_FILE="${PROJECT_DIR}/environment-intel.yml"
else
  ENV_NAME="${SEAMFORGE_ENV_NAME:-seamforge3d-cuda}"
  ENV_FILE="${PROJECT_DIR}/environment.yml"
  command -v nvidia-smi >/dev/null || { echo "CUDA profile requires an NVIDIA driver/GPU." >&2; exit 2; }
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

if [[ "${PROFILE}" == "cuda" ]]; then
  mkdir -p "${PROJECT_DIR}/third_party" "${PROJECT_DIR}/weights"
  if [[ ! -d "${UTONIA_DIR}/.git" ]]; then git clone https://github.com/Pointcept/Utonia.git "${UTONIA_DIR}"; fi
  git -C "${UTONIA_DIR}" fetch origin "${UTONIA_COMMIT}"
  git -C "${UTONIA_DIR}" checkout --detach "${UTONIA_COMMIT}"
  conda run -n "${ENV_NAME}" python -m pip install -e "${UTONIA_DIR}" --no-deps
  if [[ ! -s "${PROJECT_DIR}/weights/utonia.pth" ]]; then
    conda run -n "${ENV_NAME}" python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Pointcept/Utonia', 'utonia.pth', local_dir='${PROJECT_DIR}/weights')"
  fi
fi

conda run -n "${ENV_NAME}" python -m pip install -e "${PROJECT_DIR}" --no-deps
if [[ "${PROFILE}" == "intel" ]]; then
  conda run -n "${ENV_NAME}" python -c "import torch, open3d, open3d.ml.torch; print('Intel/CPU profile OK:', torch.__version__, open3d.__version__)"
else
  conda run -n "${ENV_NAME}" python -c "import torch, spconv.pytorch, torch_scatter, flash_attn, utonia; assert torch.cuda.is_available(); print('CUDA/Utonia profile OK:', torch.__version__, torch.cuda.get_device_name(0))"
fi
echo "Deployment complete. Activate with: conda activate ${ENV_NAME}"
