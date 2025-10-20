#!/bin/bash

base_dir=`pwd`
export CONFIG=${CONFIG:-${base_dir}/configs/gpt/gpt_360m_relpe_bf16.json}
export DS_CONFIG=${DS_CONFIG:-${base_dir}/configs/gpt/deepspeed_config_4k_bf16.json}
export BASE_JOB_NAME="gpt_pretraining"
export DATA_PATH_PREFIX="${base_dir}/data/bert_mlm/"
export PROJECT_NAME="gpt_pretraining"
export TASK_TYPE="gpt_pretraining"
export DEEPSPEED_ARGS="--use_sharded_dataset --only_mlm_task --use_torch_compile --no_decay_embeddings --log_weight_norms --keep_ckpt_every 5 --keep_ckpt_epochs '28'"

source ${base_dir}/configs/base_script.sh