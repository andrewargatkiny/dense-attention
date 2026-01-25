#!/bin/bash

base_dir=`pwd`

export CONFIG=${CONFIG:-${base_dir}/configs/lra/dense_attn_pathfinder32.json}
export DS_CONFIG=${DS_CONFIG:-${base_dir}/configs/lra/deepspeed_config_pathfinder32.json}
export BASE_JOB_NAME="lra_pathfinder_32"
export BASE_DATA_DIR="${base_dir}/data/lra/pathfinder32/"
export PROJECT_NAME="lra-pathfinder-32"
export TASK_TYPE="sequence_classification"
export SEED=${SEED:-42}
export NODE=${NODE:-0}

export DEEPSPEED=true
export DENSE_ATTENTION=true
export EVAL_TRAIN_DATA=true
export EVAL_TEST_DATA=true
export LOG_ACTIVATIONS=true

export LOG_DIAGNOSTIC_FREQ=${LOG_DIAGNOSTIC_FREQ:-5}
export INPUTS_LOGGING_RATIO=${INPUTS_LOGGING_RATIO:-0.1}
export EVAL_BS_RATIO=${EVAL_BS_RATIO:-2}
export CKPT_TO_SAVE=${CKPT_TO_SAVE:-1}
export KEEP_LAST_CKPTS=1
export MAX_VALIDATION_SAMPLES=20000

source ${base_dir}/configs/base_script.sh
