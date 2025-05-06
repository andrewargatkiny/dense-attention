#!/bin/bash

# Prepare c4 dataset for BERT pre-training
# Usage: BASE_DATA_DIR="desired/path/to/data" prepare_c4.sh

BASE_DATA_DIR=${BASE_DATA_DIR:-"$PWD"/data/bert_mlm/}
mkdir -p "$BASE_DATA_DIR" && cd "$BASE_DATA_DIR"

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "en/*"
git lfs pull --include "realnewslike/c4-validation.00000-of-00001.json.gz"

mkdir validation_data
# Extract to .txt and rename file for legacy reasons
zcat realnewslike/c4-validation.00000-of-00001.json.gz >> validation_data/part_000.txt

mv en/c4-validation.00000*.json.gz validation_data/
mv en c4_en

# Named sentence_512 for legacy reasons, but the data can be used for creating
# sequences of any size.
BERT_DATA_DIR="$BASE_DATA_DIR"/sentence_512
mkdir "$BERT_DATA_DIR"
mv c4_en validation_data "$BERT_DATA_DIR"/

cd "$BASE_DATA_DIR"
rm -rf c4

