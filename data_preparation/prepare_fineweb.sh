#!/bin/bash

# Prepare fineweb dataset for GPT/LLAMA pre-training
# Usage: BASE_DATA_DIR="desired/path/to/data" prepare_fineweb.sh

BASE_DATA_DIR=${BASE_DATA_DIR:-"$PWD"}
mkdir -p "$BASE_DATA_DIR" && cd "$BASE_DATA_DIR"

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
cd fineweb-edu
git lfs pull --include sample/100BT/*
mkdir test
mv 100BT/013_00007.parquet test/
rm -rf .git

