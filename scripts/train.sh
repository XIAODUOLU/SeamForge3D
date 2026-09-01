#!/usr/bin/env bash
set -euo pipefail

# Training entry point. Every value can be combined through environment variables.
# CONFIG: YAML containing dataset/model/loss settings.
# DEVICE: auto, cpu, or cuda.
# BACKBONE: randlanet (Intel/CPU) or utonia (NVIDIA/CUDA).
# EPOCHS: total epochs; empty means use the YAML value.
# BATCH_SIZE / WORKERS: loader size and worker processes.
# LEARNING_RATE: task-head/main LR; Utonia backbone LR remains separately configured.
# AMP: 1 enables CUDA mixed precision, 0 disables it.
# RESUME: full training checkpoint; empty starts a new optimizer/scheduler.
# INIT_CHECKPOINT: Utonia initialization weight, not a training resume checkpoint.
# OUTPUT_DIR: checkpoints, resolved config, and metrics.jsonl destination.
# CONDA_ENV: conda environment; set empty to use the currently active Python.
CONFIG="${CONFIG:-configs/large_procedural.yaml}"
DEVICE="${DEVICE:-auto}"
BACKBONE="${BACKBONE:-randlanet}"
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
WORKERS="${WORKERS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
AMP="${AMP:-1}"
RESUME="${RESUME:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train_procedural}"
CONDA_ENV="${CONDA_ENV-seamforge3d-intel}"

run_python() {
  if [[ -n "${CONDA_ENV}" ]]; then conda run --no-capture-output -n "${CONDA_ENV}" python "$@"; else python "$@"; fi
}

args=(tools/train.py --config "${CONFIG}" --device "${DEVICE}" --backbone "${BACKBONE}" --output-dir "${OUTPUT_DIR}")
[[ -n "${EPOCHS}" ]] && args+=(--epochs "${EPOCHS}")
[[ -n "${BATCH_SIZE}" ]] && args+=(--batch-size "${BATCH_SIZE}")
[[ -n "${WORKERS}" ]] && args+=(--workers "${WORKERS}")
[[ -n "${LEARNING_RATE}" ]] && args+=(--learning-rate "${LEARNING_RATE}")
[[ -n "${RESUME}" ]] && args+=(--resume "${RESUME}")
[[ -n "${INIT_CHECKPOINT}" ]] && args+=(--checkpoint "${INIT_CHECKPOINT}")
if [[ "${AMP}" == "1" ]]; then args+=(--amp); else args+=(--no-amp); fi
run_python "${args[@]}"
