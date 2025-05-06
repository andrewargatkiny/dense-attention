#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
DATESTAMP=$(date +'%Y-%m-%d_%H-%M')
JOB_NAME=lra_pathfinder_128_${DATESTAMP}
OUTPUT_DIR=${base_dir}/bert_model_dense_attn_adam_outputs

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_dense_attn_lamb_nvidia_data.json \
--max_seq_length 16384 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--dense_attention \
--unpad_inputs \
--log_diagnostic_freq 1 \
--lr_schedule "constant" \
--lr_offset 0.0 \
--log_activations \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz4k_adam_config_seq128.json \
--data_path_prefix "/workspace/bert/data/lra/path-x/" \
--use_nvidia_dataset \
&> ${JOB_NAME}.log
