#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
DATESTAMP=$(date +'%Y-%m-%d_%H-%M')
JOB_NAME=lamb_dense_attn_64k_seq128_${DATESTAMP}
OUTPUT_DIR=${base_dir}/bert_model_dense_attn_outputs

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_dense_attn_lamb_nvidia_data.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--dense_attention \
--log_diagnostic_freq 1024 \
--lr_schedule "EE" \
--lr_offset 10e-4 \
--log_activations \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz64k_lamb_config_seq128.json \
--data_path_prefix /workspace/bert \
--use_nvidia_dataset \
&> ${JOB_NAME}.log
