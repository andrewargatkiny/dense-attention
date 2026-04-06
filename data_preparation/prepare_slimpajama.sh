#!/bin/bash

# Prepare SlimPajama dataset for GPT/LLAMA pre-training
# Usage: BASE_DATA_DIR="desired/path/to/data" prepare_slimpajama.sh

BASE_DATA_DIR=${BASE_DATA_DIR:-"$PWD"}
mkdir -p "$BASE_DATA_DIR" && cd "$BASE_DATA_DIR"

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/gmongaras/SlimPajama-627B_Reupload
mv SlimPajama-627B_Reupload slimpajama
cd slimpajama
git lfs install
git lfs pull --include data/validation-00000-of-00030.parquet
mkdir val
mv data/validation-00000-of-00030.parquet val/
git lfs pull --include data/train-002*.parquet
mkdir train
mv data/train-002*.parquet train/

rm -rf .git

