#!/bin/bash

base_dir=`pwd`
export CONFIG=${CONFIG:-${base_dir}/configs/lra/dense_attn_pathfinder32.json}
export DS_CONFIG=${DS_CONFIG:-${base_dir}/configs/lra/deepspeed_config_pathfinder32.json}
export BASE_JOB_NAME="lra_pathfinder_32"
export DATA_PATH_PREFIX="${base_dir}/data/lra/pathfinder32/"
export PROJECT_NAME="lra-pathfinder-32"
export MAX_SEQ_LENGTH=1024

source ${base_dir}/configs/base_script.sh