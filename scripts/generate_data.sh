#!/usr/bin/env bash
set -euo pipefail

# Generate primitive previews and offline train/val/test NPZ splits.
# CONFIG controls family list, split sizes, view mode, noise, and GT visibility.
# OVERWRITE=1 regenerates existing scene files; keep 0 to resume an interrupted job.
CONFIG="${CONFIG:-configs/large_procedural.yaml}"
VISUALIZATIONS="${VISUALIZATIONS:-8}"
PRIMITIVE_VARIANTS="${PRIMITIVE_VARIANTS:-2}"
OVERWRITE="${OVERWRITE:-0}"
CONDA_ENV="${CONDA_ENV-seamforge3d-intel}"

run_python() {
  if [[ -n "${CONDA_ENV}" ]]; then conda run --no-capture-output -n "${CONDA_ENV}" python "$@"; else python "$@"; fi
}
run_python tools/generate_primitive_library.py --variants "${PRIMITIVE_VARIANTS}"
args=(tools/generate_dataset.py --config "${CONFIG}" --visualizations "${VISUALIZATIONS}")
[[ "${OVERWRITE}" == "1" ]] && args+=(--overwrite)
run_python "${args[@]}"
