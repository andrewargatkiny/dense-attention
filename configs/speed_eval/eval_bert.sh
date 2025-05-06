#!/bin/bash

base_dir=`pwd`
OUTPUT_DIR=${base_dir}/bert_model_dense_attn_adam_outputs
BASE_JOB_NAME="bert_pretraining"


# Default values
: "${BASE_DATA_DIR:=${base_dir}/data}"
CHECKPOINT_BASE_PATH=""
CHECKPOINT_EPOCH_NAME=""

SEED=${SEED:-42}
NODE=${NODE:-0}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG=${CONFIG:-${base_dir}/configs/speed_eval/danet_bert.json}
DS_CONFIG=${DS_CONFIG:-${base_dir}/configs/speed_eval/deepspeed_config_bf16.json}

MODEL_CONFIG=${MODEL_CONFIG:-"$CONFIG"}
DATA_CONFIG=${DATA_CONFIG:-"$CONFIG"}
TRAINING_CONFIG=${TRAINING_CONFIG:-"$CONFIG"}
TASK_TYPE=${TASK_TYPE:-"bert_mlm"}

# Check if we're resuming from a checkpoint
if [ "$1" = "--resume" ]; then
    if [ -n "$2" ]; then
        LOAD_EPOCH=$2
    else
        echo "Epoch number for model checkpoint is not defined, exiting."
        echo "Usage: ./your_train_script_name.sh [--resume EPOCH DIR_WITH_TRAIN_ARTIFACTS]"
        exit 1
    fi

    if [ -z "$3" ]; then
        echo "Subdirectory with model weights is not defined, exiting."
        echo "Usage: ./your_train_script_name.sh [--resume EPOCH DIR_WITH_TRAIN_ARTIFACTS]"
        exit 1
    else
        SUBDIR=$3
    fi

    CHECKPOINT_BASE_PATH=${OUTPUT_DIR}/saved_models/${SUBDIR}
    CHECKPOINT_EPOCH_NAME=$(basename ${CHECKPOINT_BASE_PATH}/epoch${LOAD_EPOCH}_*)
    echo "checkpoint id: $CHECKPOINT_EPOCH_NAME"
    DATESTAMP=$(date +'%Y-%m-%d_%H-%M')
    JOB_NAME="${SUBDIR}_from_epoch_${LOAD_EPOCH}_${DATESTAMP}"
else
    # Set up for initial training
    DATESTAMP=$(date +'%Y-%m-%d_%H-%M')
    JOB_NAME=${BASE_JOB_NAME}_${DATESTAMP}
fi


mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed --include localhost:"$NODE" --master_port "$MASTER_PORT" ${base_dir}/deepspeed_train.py \
--cf "$CONFIG" \
--model_config_file "$MODEL_CONFIG" \
--data_config_file "$DATA_CONFIG" \
--train_config_file "$TRAINING_CONFIG" \
--output_dir $OUTPUT_DIR \
--task_type "$TASK_TYPE" \
--only_mlm_task \
--eval_only \
--eval_test_data \
--eval_train_data \
--use_torch_compile \
--deepspeed \
--ckpt_to_save 0 \
--seed "$SEED" \
--job_name $JOB_NAME \
--deepspeed_config "$DS_CONFIG" \
--data_path_prefix "${BASE_DATA_DIR}/bert_mlm/" \
--eval_bs_ratio 1 \
--inputs_logging_ratio 0.1 \
--load_training_checkpoint $CHECKPOINT_BASE_PATH \
--load_checkpoint_id $CHECKPOINT_EPOCH_NAME \
--load_only_weights \
--project_name "bert_evaluation" \
&> ${JOB_NAME}.log
