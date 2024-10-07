if [[ -z $1 ]]; then
    echo "Please supply a path to model checkpoint starting from current directory."
    exit 1
else
    CHECKPOINT_PATH=$1
fi

: "${BASE_DATA_DIR:=${base_dir}/data}"
DATESTAMP=$(date +'%Y-%m-%d_%H-%M')
base_dir=`pwd`
JOB_NAME=eval_dense_attn_4k_seq16384_${DATESTAMP}

python eval_mlm_acc.py \
  --checkpoint_path "$base_dir/$CHECKPOINT_PATH" \
  --config-file "${base_dir}"/configs/bert/bert_large_seq16384.json \
  --batch_size 8 \
  --no_eval_hf \
  --max_seq_len_hf 16384 \
  --data_path "${BASE_DATA_DIR}/bert_mlm/validation_16384" | tee -a ${JOB_NAME}.log
