#!/bin/bash

base_dir=`pwd`
SEED=${SEED:-100}
NODE=${NODE:-0}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG=${CONFIG:-${base_dir}/configs/lra/dense_attn_pathfinder32.json}
DS_CONFIG=${DS_CONFIG:-${base_dir}/configs/lra/deepspeed_config_pathfinder32.json}
TRACKING_SYSTEM=${TRACKING_SYSTEM:-clearml}

OUTPUT_DIR=${base_dir}/bert_model_dense_attn_adam_outputs
BASE_JOB_NAME="lra_pathfinder_32"

# Default values
: "${BASE_DATA_DIR:=${base_dir}/data}"
CHECKPOINT_BASE_PATH=""
CHECKPOINT_EPOCH_NAME=""

JOB_NAME_SUFFIX=${JOB_NAME_SUFFIX-"_$(date +'%Y-%m-%d_%H-%M')"}
OVERRIDE_ARGS=()

# Check if we're resuming from a checkpoint
if [ "${1-}" = "--resume" ]; then
  shift
  # Handle EPOCH or the keyword 'last'
  if echo "${1-}" | grep -qE '^[0-9]+$'; then
    LOAD_EPOCH=$1; shift
  elif [ "${1-}" = "last" ]; then
    LOAD_EPOCH=""; shift
  else
    LOAD_EPOCH=""
  fi

  [ $# -ge 1 ] || {
    echo "Usage: $0 --resume [EPOCH|last] JOB_NAME"
    echo "   or: $0 --resume [EPOCH|last] JOB_NAME --override cf.key=value ds.key=value ..."
    exit 1
  }
  SUBDIR=$1; shift

  CHECKPOINT_BASE_PATH="${OUTPUT_DIR}/saved_models/${SUBDIR}"

  if [ -z "$LOAD_EPOCH" ]; then # auto-detect newest
    LATEST_TAG=$(ls "$CHECKPOINT_BASE_PATH" | grep '^epoch' | sort -V | tail -n1)
    LOAD_EPOCH=$(printf '%s\n' "$LATEST_TAG" | sed -E 's/^epoch([0-9]+).*/\1/')
    CHECKPOINT_EPOCH_NAME="$LATEST_TAG"
  else
    CHECKPOINT_EPOCH_NAME=$(basename "${CHECKPOINT_BASE_PATH}/epoch${LOAD_EPOCH}"_*)
  fi

  echo ">> Resuming from checkpoint: $CHECKPOINT_EPOCH_NAME"


  JOB_NAME="${SUBDIR}_from_epoch_${LOAD_EPOCH}${JOB_NAME_SUFFIX}"
else
  # Set up for initial training
  JOB_NAME="${BASE_JOB_NAME}${JOB_NAME_SUFFIX}"
fi

if [ "${1-}" = "--override" ]; then
  OVERRIDE_ARGS=( "${@:2}" )
fi

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed --include localhost:"$NODE" --master_port "$MASTER_PORT" ${base_dir}/deepspeed_train.py \
--cf "$CONFIG" \
--max_seq_length 1024 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--dense_attention \
--eval_train_data \
--eval_test_data \
--max_validation_samples 20000 \
--log_diagnostic_freq 5 \
--log_activations \
--tracking_system "$TRACKING_SYSTEM" \
--seed "$SEED" \
--job_name $JOB_NAME \
--deepspeed_config "$DS_CONFIG" \
--data_path_prefix "${BASE_DATA_DIR}/lra/pathfinder32/" \
--eval_bs_ratio 2 \
--inputs_logging_ratio 0.1 \
--load_training_checkpoint $CHECKPOINT_BASE_PATH \
--load_checkpoint_id $CHECKPOINT_EPOCH_NAME \
--keep_last_ckpts 3 \
--ckpt_to_save 1 \
--project_name "lra-pathfinder-32" \
--override ${OVERRIDE_ARGS[@]} \
&> ${JOB_NAME}.log
