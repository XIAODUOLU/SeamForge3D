#!/usr/bin/env bash
set -euo pipefail

# Validation selects thresholds/checkpoints; never use test results for tuning.
# CONFIG: validation dataset/runtime YAML.
# MODEL: checkpoint to evaluate, normally best.pt or last.pt during comparison.
# DEVICE: auto, cpu, or cuda.
# INFERENCE_VOTES: RandLA-Net random-sampling votes; higher is slower but steadier.
# PROBABILITY_THRESHOLD: candidate threshold calibrated on this validation split.
# DENSE_REFINE: 1 enables raw-cloud refinement.
# VISUALIZATIONS: number of scene overlay PNGs.
# OUTPUT_DIR: per_scene.jsonl, summary.json, and PNG destination.
CONFIG="${CONFIG:-configs/large_procedural.yaml}"
MODEL="${MODEL:-outputs/train_procedural/best.pt}"
DEVICE="${DEVICE:-auto}"
INFERENCE_VOTES="${INFERENCE_VOTES:-4}"
PROBABILITY_THRESHOLD="${PROBABILITY_THRESHOLD:-0.30}"
DENSE_REFINE="${DENSE_REFINE:-0}"
VISUALIZATIONS="${VISUALIZATIONS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/validation}"
CONDA_ENV="${CONDA_ENV-seamforge3d-intel}"

run_python() {
  if [[ -n "${CONDA_ENV}" ]]; then conda run --no-capture-output -n "${CONDA_ENV}" python "$@"; else python "$@"; fi
}
args=(tools/evaluate.py --config "${CONFIG}" --model "${MODEL}" --split val --output "${OUTPUT_DIR}" --device "${DEVICE}" --inference-votes "${INFERENCE_VOTES}" --probability-threshold "${PROBABILITY_THRESHOLD}" --visualizations "${VISUALIZATIONS}")
[[ "${DENSE_REFINE}" == "1" ]] && args+=(--dense-refine)
run_python "${args[@]}"
