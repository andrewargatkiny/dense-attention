#!/bin/bash

if [[ -z $1 ]]; then
    LOAD_EPOCH=2
else
    LOAD_EPOCH=$1
fi

if [[ -z $2 ]]; then
    echo "Subdirectory with model weights is not defined, exiting."
    exit 1
else
    SUBDIR=$2
fi
base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=${SUBDIR}
OUTPUT_DIR=${base_dir}/bert_model_dense_attn_adam_outputs

# Assumes job name in previous seq128 run, will resume training from epoch 2 by default
CHECKPOINT_BASE_PATH=${OUTPUT_DIR}/saved_models/${JOB_NAME}
CHECKPOINT_EPOCH_NAME=`basename ${CHECKPOINT_BASE_PATH}/epoch${LOAD_EPOCH}_*`
echo "checkpoint id: $CHECKPOINT_EPOCH_NAME"
mkdir -p $OUTPUT_DIR
JOB_NAME="${JOB_NAME}_from_epoch_${LOAD_EPOCH}"

NCCL_TREE_THRESHOLD=0 deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_dense_attn_lamb_nvidia_data.json \
--max_seq_length 1024 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--dense_attention \
--log_diagnostic_freq 10 \
--lr_schedule "constant" \
--lr_offset 0.0 \
--log_activations \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz4k_adam_config_seq128.json \
--data_path_prefix "/workspace/bert/data/lra/pathfinder/" \
--use_nvidia_dataset \
--load_training_checkpoint ${CHECKPOINT_BASE_PATH} \
--load_checkpoint_id ${CHECKPOINT_EPOCH_NAME} \
&>> ${JOB_NAME}.log
