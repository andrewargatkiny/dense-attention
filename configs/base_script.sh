#!/bin/bash

base_dir=`pwd`
MASTER_PORT=${MASTER_PORT:-29500}
TRACKING_SYSTEM=${TRACKING_SYSTEM:-clearml}

for var in CONFIG DS_CONFIG BASE_JOB_NAME DATA_PATH_PREFIX PROJECT_NAME TASK_TYPE SEED; do
  value=${!var}
  if [ -z "$value" ]; then
    echo "error: required variable $var is not set."
    exit 1
  fi
done

OUTPUT_DIR=${base_dir}/bert_model_dense_attn_adam_outputs

REUSE_JOB_NAME=false
new_args=()
for arg in "$@"; do
  if [ "$arg" != "--reuse_job_name" ]; then
    new_args+=("$arg")
  else
    REUSE_JOB_NAME=true
  fi
done
set -- "${new_args[@]}"

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
    echo "Options:"
    echo "  --reuse_job_name    Reuse the original job name instead of adding timestamp suffix"
    echo "                      (can appear in any position among the arguments)"
    echo ""
    echo "Examples:"
    echo "  $0 --resume last my_job_name"
    echo "  $0 --reuse_job_name --resume last my_job_name"
    echo "  $0 --resume 10 my_job --reuse_job_name --override cf.key=value"
    exit 1
  }
  SUBDIR=$1; shift

  CHECKPOINT_BASE_PATH="${OUTPUT_DIR}/saved_models/${SUBDIR}"

  if [ -z "$LOAD_EPOCH" ]; then # auto-detect newest
    LATEST_TAG=$(ls "$CHECKPOINT_BASE_PATH" 2>/dev/null | grep '^epoch' | sort -V | tail -n1)
    [ -z "$LATEST_TAG" ] && { echo "ERROR: no checkpoints under $CHECKPOINT_BASE_PATH"; exit 1; }
    LOAD_EPOCH=$(printf '%s\n' "$LATEST_TAG" | sed -E 's/^epoch([0-9]+).*/\1/')
    CHECKPOINT_EPOCH_NAME="$LATEST_TAG"
  else
    CHECKPOINT_EPOCH_NAME=$(basename "${CHECKPOINT_BASE_PATH}/epoch${LOAD_EPOCH}"_* 2>/dev/null)
    [ -z "$CHECKPOINT_EPOCH_NAME" ] && { echo "ERROR: checkpoint epoch${LOAD_EPOCH}_* not found under $CHECKPOINT_BASE_PATH"; exit 1; }
  fi

  echo ">> Resuming from checkpoint: $CHECKPOINT_EPOCH_NAME"

  JOB_NAME="${SUBDIR}_from_epoch_${LOAD_EPOCH}${JOB_NAME_SUFFIX}"
  if $REUSE_JOB_NAME; then
    JOB_NAME="$SUBDIR"
  fi
else
  # Set up for initial training
  JOB_NAME="${BASE_JOB_NAME}${JOB_NAME_SUFFIX}"
fi

if [ "${1-}" = "--override" ]; then
  OVERRIDE_ARGS=( "${@:2}" )
fi

mkdir -p "$OUTPUT_DIR"

if [ -n "${NODE:-}" ]; then
  NODE_ARG="--include localhost:$NODE"
else
  NODE_ARG=""
fi

EXTRA_ARGS=()

if [ -n "${CHECKPOINT_BASE_PATH:-}" ]; then
  EXTRA_ARGS+=( --load_training_checkpoint "$CHECKPOINT_BASE_PATH" )
fi
if [ -n "${CHECKPOINT_EPOCH_NAME:-}" ]; then
  EXTRA_ARGS+=( --load_checkpoint_id "$CHECKPOINT_EPOCH_NAME" )
fi

if [ ${#OVERRIDE_ARGS[@]} -gt 0 ]; then
  EXTRA_ARGS+=( --override "${OVERRIDE_ARGS[@]}" )
fi

NCCL_TREE_THRESHOLD=0 deepspeed $NODE_ARG --master_port "$MASTER_PORT" "${base_dir}/deepspeed_train.py" \
  --cf "$CONFIG" \
  --model_config_file "${MODEL_CONFIG:-$CONFIG}" \
  --data_config_file "${DATA_CONFIG:-$CONFIG}" \
  --train_config_file "${TRAINING_CONFIG:-$CONFIG}" \
  --task_type "$TASK_TYPE" \
  --output_dir "$OUTPUT_DIR" \
  ${DEEPSPEED:+--deepspeed} \
  ${DENSE_ATTENTION:+--dense_attention} \
  ${EVAL_TRAIN_DATA:+--eval_train_data} \
  ${EVAL_TEST_DATA:+--eval_test_data} \
  ${NO_EVAL_VAL_DATA:+--no_eval_val_data} \
  ${EVAL_ONLY:+--eval_only} \
  ${ONLY_MLM_TASK:+--only_mlm_task} \
  ${ONLY_CLS_TASK:+--only_cls_task} \
  ${NO_DECAY_EMBEDDINGS:+--no_decay_embeddings} \
  ${NO_DECAY_POOLER:+--no_decay_pooler} \
  ${SCALE_FFN_WEIGHTS:+--scale_ffn_weights} \
  ${MATERIALIZE_FFN_WEIGHTS:+--materialize_ffn_weights} \
  ${LOAD_ONLY_WEIGHTS:+--load_only_weights} \
  ${REWARMUP:+--rewarmup} \
  ${LOG_WEIGHT_NORMS:+--log_weight_norms} \
  ${USE_SHARDED_DATASET:+--use_sharded_dataset} \
  ${LOG_ACTIVATIONS:+--log_activations} \
  ${RESIZE_POSIT_EMBEDS:+--resize_posit_embeds} \
  ${UNPAD_INPUTS:+--unpad_inputs} \
  ${USE_TORCH_COMPILE:+--use_torch_compile} \
  ${VARIABLE_MASK_RATE:+--variable_mask_rate} \
  ${MLM_USE_RTC_TASK:+--mlm_use_rtc_task} \
  ${ZERO_INIT_POOLER:+--zero_init_pooler} \
  --max_validation_samples "${MAX_VALIDATION_SAMPLES:--1}" \
  --log_diagnostic_freq "${LOG_DIAGNOSTIC_FREQ:-100}" \
  --tracking_system "$TRACKING_SYSTEM" \
  --seed "$SEED" \
  --job_name "$JOB_NAME" \
  --deepspeed_config "$DS_CONFIG" \
  --data_path_prefix "$DATA_PATH_PREFIX" \
  --eval_bs_ratio "${EVAL_BS_RATIO:-8}" \
  --inputs_logging_ratio "${INPUTS_LOGGING_RATIO:-1.0}" \
  --keep_last_ckpts "${KEEP_LAST_CKPTS:-3}" \
  --ckpt_to_save "${CKPT_TO_SAVE:-20}" \
  --project_name "$PROJECT_NAME" \
  ${MAX_STEPS:+--max_steps "$MAX_STEPS"} \
  ${MAX_STEPS_PER_EPOCH:+--max_steps_per_epoch "$MAX_STEPS_PER_EPOCH"} \
  ${LOGGING_NORM_TYPE:+--logging_norm_type "$LOGGING_NORM_TYPE"} \
  ${VALIDATION_DATA_PATH_PREFIX:+--validation_data_path_prefix "$VALIDATION_DATA_PATH_PREFIX"} \
  ${THROUGHPUT_LOGGING_SAMPLES:+--throughput_logging_samples "$THROUGHPUT_LOGGING_SAMPLES"} \
  ${KEEP_CKPT_EVERY:+--keep_ckpt_every "$KEEP_CKPT_EVERY"} \
  ${KEEP_CKPT_EPOCHS:+--keep_ckpt_epochs "$KEEP_CKPT_EPOCHS"} \
  ${NUM_LABELS:+--num_labels "$NUM_LABELS"} \
  ${LM_PROB:+--lm_prob "$LM_PROB"} \
  ${MASK_TOKEN_ID:+--mask_token_id "$MASK_TOKEN_ID"} \
  ${DICT_BACKEND:+--dict_backend "$DICT_BACKEND"} \
  ${MAX_PREDICTIONS_PER_SEQ:+--max_predictions_per_seq "$MAX_PREDICTIONS_PER_SEQ"} \
  "${EXTRA_ARGS[@]}" \
  &> "${JOB_NAME}.log"
