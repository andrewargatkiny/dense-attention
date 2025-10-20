#!/bin/bash

base_dir=`pwd`
export CONFIG=${CONFIG:-${base_dir}/configs/gpt/llama_340m.json}
export DS_CONFIG=${DS_CONFIG:-${base_dir}/configs/gpt/deepspeed_transformer_4k.json}
export BASE_JOB_NAME="gpt_pretraining"
export DATA_PATH_PREFIX="${base_dir}/data/bert_mlm/"
export PROJECT_NAME="gpt_pretraining"
export TASK_TYPE="transformer_gpt_pretraining"
export DEEPSPEED_ARGS="--use_sharded_dataset --only_mlm_task --use_torch_compile --log_weight_norms --keep_ckpt_epochs '14'"

source ${base_dir}/configs/base_script.sh