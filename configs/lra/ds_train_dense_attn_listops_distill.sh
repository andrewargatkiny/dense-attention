#!/bin/bash

base_dir=`pwd`

SEED=${SEED:-100}
NODE=${NODE:-0}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG=${CONFIG:-${base_dir}/configs/lra/dense_attn_listops_student.json}
DS_CONFIG=${DS_CONFIG:-${base_dir}/configs/lra/deepspeed_config_listops_distill.json}
TEACHER_CONFIG=${TEACHER_CONFIG:-${base_dir}/configs/lra/dense_attn_listops.json}

OUTPUT_DIR=${base_dir}/bert_model_dense_attn_adam_outputs
BASE_JOB_NAME="lra_listops_distill"

# Default values
: "${BASE_DATA_DIR:=${base_dir}/data}"
: "${TEACHER_CHECKPOINT_DIR:=${base_dir}/teacher_checkpoint/saved_models/lra_listops_teacher}"
: "${TEACHER_CHECKPOINT_ID:=epoch41_step30750}"

# Distillation hyperparameters
: "${DISTILL_ALPHA:=0.6}"
: "${DISTILL_BETA:=0.4}"
: "${DISTILL_GAMMA:=0.0}"
: "${DISTILL_T:=5.0}"

DATESTAMP=$(date +'%Y-%m-%d_%H-%M')
JOB_NAME=${BASE_JOB_NAME}_${DATESTAMP}

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed --include localhost:"$NODE" --master_port "$MASTER_PORT" ${base_dir}/deepspeed_train.py \
--cf "$CONFIG" \
--max_seq_length 2000 \
--output_dir $OUTPUT_DIR \
--task_type "sequence_classification_distill" \
--deepspeed \
--dense_attention \
--eval_train_data \
--eval_test_data \
--max_validation_samples 2000 \
--log_diagnostic_freq 5 \
--log_activations \
--seed "$SEED" \
--num_labels 10 \
--job_name $JOB_NAME \
--deepspeed_config "$DS_CONFIG" \
--data_path_prefix "${BASE_DATA_DIR}/lra/listops/" \
--eval_bs_ratio 2 \
--inputs_logging_ratio 0.5 \
--teacher_checkpoint "$TEACHER_CHECKPOINT_DIR" \
--teacher_checkpoint_id "$TEACHER_CHECKPOINT_ID" \
--teacher_config_file "$TEACHER_CONFIG" \
--distill_alpha "$DISTILL_ALPHA" \
--distill_beta "$DISTILL_BETA" \
--distill_gamma "$DISTILL_GAMMA" \
--distill_T "$DISTILL_T" \
--project_name "lra-listops-distill" \
&> ${JOB_NAME}.log
