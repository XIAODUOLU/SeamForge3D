#!/usr/bin/env bash
set -euo pipefail

# Single-scene production/debug inference.
# INPUT: one generated NPZ scene containing network and raw point clouds.
# MODEL: trained checkpoint.
# INFERENCE_VOTES: RandLA-Net coverage/latency trade-off; ignored by Utonia.
# PROBABILITY_THRESHOLD: threshold selected by validate.sh.
# DENSE_REFINE: 1 keeps millimetre raw-cloud surface refinement enabled.
CONFIG="${CONFIG:-configs/large_procedural.yaml}"
MODEL="${MODEL:-outputs/train_procedural/best.pt}"
INPUT="${INPUT:-outputs/procedural_dataset/test/scene_000000.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/inference}"
DEVICE="${DEVICE:-auto}"
INFERENCE_VOTES="${INFERENCE_VOTES:-8}"
PROBABILITY_THRESHOLD="${PROBABILITY_THRESHOLD:-0.30}"
DENSE_REFINE="${DENSE_REFINE:-1}"
CONDA_ENV="${CONDA_ENV-seamforge3d-intel}"

run_python() {
  if [[ -n "${CONDA_ENV}" ]]; then conda run --no-capture-output -n "${CONDA_ENV}" python "$@"; else python "$@"; fi
}
args=(tools/infer.py --config "${CONFIG}" --model "${MODEL}" --input "${INPUT}" --output "${OUTPUT_DIR}" --device "${DEVICE}" --inference-votes "${INFERENCE_VOTES}" --probability-threshold "${PROBABILITY_THRESHOLD}")
[[ "${DENSE_REFINE}" == "0" ]] && args+=(--no-dense-refine)
run_python "${args[@]}"
