#!/bin/bash
# Script for bulk evaluation of the model on dataset with fixed seq length

# Check if file with model checkpoint paths and config paths is provided
if [ -z "$1" ]; then
    echo "Please supply a path to a text file containing model checkpoints and config paths."
    exit 1
else
    MODEL_LIST_FILE="$1"
fi
if [ -n "$2" ]; then
  SEQ_LEN="$2"
fi
if [ -n "$3" ]; then
  BATCH_SIZE="$3"
fi
# Base directory and other variables
base_dir=$(pwd)
: "${BASE_DATA_DIR:=${base_dir}/data}"
: "${BATCH_SIZE:=1024}"
: "${SEQ_LEN:=512}"
DATESTAMP=$(date +'%Y-%m-%d_%H-%M')
JOB_NAME=eval_dense_attn_4k_seq1024_${DATESTAMP}

# Loop through each line in the file
while read -r CHECKPOINT_PATH CONFIG_PATH; do
    # Skip empty lines or lines starting with #
    if [ -z "$CHECKPOINT_PATH" ] || [ "$(echo "$CHECKPOINT_PATH" | cut -c1)" = "#" ]; then
        continue
    fi

    echo "Starting evaluation for checkpoint: $CHECKPOINT_PATH with config: $CONFIG_PATH"

    # Run the evaluation script
    python bert_evals/eval_mlm_acc.py \
      --checkpoint_path "$base_dir/$CHECKPOINT_PATH/mp_rank_00_model_states.pt" \
      --config-file "$base_dir/$CONFIG_PATH" \
      --batch_size $BATCH_SIZE \
      --use_torch_compile \
      --no_eval_hf \
      --data_path "${BASE_DATA_DIR}/bert_mlm/validation_${SEQ_LEN}" | tee -a ${JOB_NAME}.log

done < "$MODEL_LIST_FILE"
