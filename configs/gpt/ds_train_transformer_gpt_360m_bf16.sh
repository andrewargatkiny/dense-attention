#!/bin/bash

base_dir=`pwd`

export CONFIG=${CONFIG:-${base_dir}/configs/gpt/llama_340m.json}
export DS_CONFIG=${DS_CONFIG:-${base_dir}/configs/gpt/deepspeed_transformer_4k.json}
export BASE_JOB_NAME="gpt_pretraining"
export DATA_PATH_PREFIX="${base_dir}/data/bert_mlm/"
export PROJECT_NAME="gpt_pretraining"
export TASK_TYPE="transformer_gpt_pretraining"
export SEED=${SEED:-42}

export DEEPSPEED=true
export USE_SHARDED_DATASET=true
export ONLY_MLM_TASK=true
export USE_TORCH_COMPILE=true
export LOG_WEIGHT_NORMS=true
export LOG_ACTIVATIONS=true
export EVAL_TEST_DATA=true

export LOG_DIAGNOSTIC_FREQ=${LOG_DIAGNOSTIC_FREQ:-5}
export INPUTS_LOGGING_RATIO=${INPUTS_LOGGING_RATIO:-0.1}
export EVAL_BS_RATIO=${EVAL_BS_RATIO:-2}
export CKPT_TO_SAVE=${CKPT_TO_SAVE:-1}
export KEEP_LAST_CKPTS=${KEEP_LAST_CKPTS:-3}
export KEEP_CKPT_EPOCHS=${KEEP_CKPT_EPOCHS:-'14'}
export MAX_VALIDATION_SAMPLES=${MAX_VALIDATION_SAMPLES:-20000}

source ${base_dir}/configs/base_script.sh
