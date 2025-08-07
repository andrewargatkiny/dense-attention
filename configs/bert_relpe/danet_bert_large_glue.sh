#!/bin/bash

CONFIG=configs/bert_relpe/danet_bert_large_glue_stage_1.json \
  JOB_NAME_SUFFIX="_stage_1" \
  configs/bert_relpe/ds_train_dense_attn_bert_seq128_bf16.sh

CONFIG=configs/bert_relpe/danet_bert_large_glue_stage_1.json \
  TRAINING_CONFIG=configs/bert_relpe/danet_training_glue_stage_2.json \
  JOB_NAME_SUFFIX="_stage_2_1" configs/bert_relpe/ds_train_dense_attn_bert_seq128_bf16.sh \
  --resume 971  "bert_pretraining_stage_1" --override cf.training.num_epochs=101

CONFIG=configs/bert_relpe/danet_bert_large_glue_stage_1.json \
  TRAINING_CONFIG=configs/bert_relpe/danet_training_glue_stage_2.json \
  JOB_NAME_SUFFIX="_stage_2_2" configs/bert_relpe/ds_train_dense_attn_bert_seq128_bf16.sh \
  --resume 101  "bert_pretraining_stage_1_from_epoch_971_stage_2_1" \
  --override cf.training.num_epochs=201 ds.optimizer.params.betas="[0.9,0.99]"

CONFIG=configs/bert_relpe/danet_bert_large_glue_stage_1.json \
  TRAINING_CONFIG=configs/bert_relpe/danet_training_glue_stage_2.json \
  JOB_NAME_SUFFIX="_stage_2_3" configs/bert_relpe/ds_train_dense_attn_bert_seq128_bf16.sh --resume 201 \
  "bert_pretraining_stage_1_from_epoch_971_stage_2_1_from_epoch_101_stage_2_2" \
  --override cf.training.num_epochs=301 cf.training.weight_decay=0.02 \
  ds.optimizer.params.betas="[0.9,0.99]"

CONFIG=configs/bert_relpe/danet_bert_large_glue_stage_1.json \
  TRAINING_CONFIG=configs/bert_relpe/danet_training_glue_stage_2.json \
  JOB_NAME_SUFFIX="_stage_3" configs/bert_relpe/ds_train_dense_attn_bert_seq128_bf16.sh --resume 301 \
  "bert_pretraining_stage_1_from_epoch_971_stage_2_1_from_epoch_101_stage_2_2_from_epoch_201_stage_2_3" \
  --override ds.optimizer.params.betas="[0.9,0.99]"
