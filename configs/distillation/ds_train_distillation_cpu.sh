#!/usr/bin/env bash
# ds_train_distillation_cpu.sh
#
# Minimal proof-of-concept distillation run on CPU.
# Exercises a few training iterations and evaluates student vs teacher.
#
# Usage (from repo root):
#   configs/distillation/ds_train_distillation_cpu.sh
#
# Optional environment overrides:
#   SEED=42          – random seed
#   BASE_OUT_DIR=./output – directory for checkpoints

set -euo pipefail

base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED=${SEED:-42}
BASE_OUT_DIR=${BASE_OUT_DIR:-"${base_dir}/output"}
MASTER_PORT=${MASTER_PORT:-29600}
CONFIG=${CONFIG:-"${base_dir}/configs/distillation/mlm_distillation_cpu.json"}
DS_CONFIG=${DS_CONFIG:-"${base_dir}/configs/distillation/deepspeed_cpu.json"}

echo "===== MLM Distillation CPU Smoke-Test ====="
echo "Base dir  : ${base_dir}"
echo "Config    : ${CONFIG}"
echo "DS config : ${DS_CONFIG}"
echo "Output    : ${BASE_OUT_DIR}"
echo "Seed      : ${SEED}"
echo "==========================================="

DS_ACCELERATOR="cpu" deepspeed \
    --num_accelerators 1 \
    --master_port "${MASTER_PORT}" \
    "${base_dir}/deepspeed_train.py" \
    --config-file "${CONFIG}" \
    --output_dir "${BASE_OUT_DIR}" \
    --deepspeed_config "${DS_CONFIG}" \
    --task_type mlm_distillation \
    --seed "${SEED}" \
    --max_seq_length 64 \
    --num_labels 2 \
    --mask_token_id 103 \
    --teacher_model bert-base-uncased \
    --distillation_alpha 0.5 \
    --distillation_temperature 4.0 \
    --ckpt_to_save 0 \
    --tracking_system tensorboard \
    --dict_backend gloo \
    --max_validation_samples 128 \
    "$@"