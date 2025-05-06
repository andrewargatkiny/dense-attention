#!/bin/bash

base_dir=`pwd`
: "${BASE_DATA_DIR:=${base_dir}/data}"

# Download and extract the data
wget -v https://storage.googleapis.com/long-range-arena/lra_release.gz -P "$BASE_DATA_DIR"
tar -xvzf "${BASE_DATA_DIR}/lra_release.gz" -C "${BASE_DATA_DIR}/"

# Convert datasets to required format

# ListOps dataset
python3 data_preparation/lra/convert_listops.py \
  --input_path "${BASE_DATA_DIR}/lra_release/listops-1000/basic_train.tsv" \
  --include_cls_token \
  --output_base_dir "${BASE_DATA_DIR}/lra/listops" \
  --output_dataset_name "train"
python3 data_preparation/lra/convert_listops.py \
  --input_path "${BASE_DATA_DIR}/lra_release/listops-1000/basic_val.tsv" \
  --include_cls_token \
  --output_base_dir "${BASE_DATA_DIR}/lra/listops" \
  --output_dataset_name "valid"
python3 data_preparation/lra/convert_listops.py \
  --input_path "${BASE_DATA_DIR}/lra_release/listops-1000/basic_test.tsv" \
  --include_cls_token \
  --output_base_dir "${BASE_DATA_DIR}/lra/listops" \
  --output_dataset_name "test"

# Cifar10
python3 data_preparation/lra/convert_cifar.py \
  --output_base_dir "${BASE_DATA_DIR}/lra/cifar10"

# Pathfinder
python3 data_preparation/lra/convert_pathfinder.py \
  --input_root_dir "${BASE_DATA_DIR}/lra_release/lra_release/pathfinder32" \
  --data_subdir curv_contour_length_14 \
  --output_base_dir "${BASE_DATA_DIR}/lra/pathfinder32"

# Pathfinder-X datasets
python3 data_preparation/lra/convert_pathfinder.py \
  --input_root_dir "${BASE_DATA_DIR}/lra_release/lra_release/pathfinder128" \
  --data_subdir curv_baseline \
  --output_base_dir "${BASE_DATA_DIR}/lra/pathfinder128_simple"
python3 data_preparation/lra/convert_pathfinder.py \
  --input_root_dir "${BASE_DATA_DIR}/lra_release/lra_release/pathfinder128" \
  --data_subdir curv_contour_length_14 \
  --output_base_dir "${BASE_DATA_DIR}/lra/path-x"

# Pathfinder-256 datasets
python3 data_preparation/lra/convert_pathfinder.py \
  --input_root_dir "${BASE_DATA_DIR}/lra_release/lra_release/pathfinder256" \
  --data_subdir curv_baseline \
  --output_base_dir "${BASE_DATA_DIR}/lra/pathfinder256_simple"
python3 data_preparation/lra/convert_pathfinder.py \
  --input_root_dir "${BASE_DATA_DIR}/lra_release/lra_release/pathfinder256" \
  --data_subdir curv_contour_length_14 \
  --output_base_dir "${BASE_DATA_DIR}/lra/pathfinder256_hard"

# IMDB reviews (text) dataset
python3 data_preparation/lra/convert_imdb.py \
  --output_base_dir="${BASE_DATA_DIR}/lra/text_classification"

# AAN (matching) dataset
python3 data_preparation/lra/convert_aan.py \
  --data_dir "${BASE_DATA_DIR}/lra_release/lra_release/tsv_data" \
  --include_cls_token \
  --output_base_dir "${BASE_DATA_DIR}/lra/aan"