#!/usr/bin/env bash
set -euo pipefail

# Final test-set evaluation. Tune the model and threshold with validate.sh first.
# MODEL should be frozen before running this script.
# RUN_UNIT_TESTS=1 additionally runs repository unit/integration tests.
CONFIG="${CONFIG:-configs/large_procedural.yaml}"
MODEL="${MODEL:-outputs/train_procedural/best.pt}"
DEVICE="${DEVICE:-auto}"
INFERENCE_VOTES="${INFERENCE_VOTES:-8}"
PROBABILITY_THRESHOLD="${PROBABILITY_THRESHOLD:-0.30}"
DENSE_REFINE="${DENSE_REFINE:-1}"
VISUALIZATIONS="${VISUALIZATIONS:-16}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/test}"
CONDA_ENV="${CONDA_ENV-seamforge3d-intel}"

run_python() {
  if [[ -n "${CONDA_ENV}" ]]; then conda run --no-capture-output -n "${CONDA_ENV}" python "$@"; else python "$@"; fi
}
[[ "${RUN_UNIT_TESTS}" == "1" ]] && run_python -m pytest -q
args=(tools/evaluate.py --config "${CONFIG}" --model "${MODEL}" --split test --output "${OUTPUT_DIR}" --device "${DEVICE}" --inference-votes "${INFERENCE_VOTES}" --probability-threshold "${PROBABILITY_THRESHOLD}" --visualizations "${VISUALIZATIONS}")
[[ "${DENSE_REFINE}" == "1" ]] && args+=(--dense-refine)
run_python "${args[@]}"
