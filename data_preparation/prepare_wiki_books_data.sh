#!/bin/bash

ENV_NAME="bert_data"
BASE_DATA_DIR=${BASE_DATA_DIR:-"$PWD"/data/bert_mlm/}
conda create -n $ENV_NAME -y python=3.10
conda activate $ENV_NAME
cd "$BASE_DATA_DIR"
pip install tqdm boto3 requests six ipdb h5py nltk progressbar "numpy<2.0.0" onnxruntime==1.12.0 "tokenizers>=0.7"
pip install git+https://github.com/NVIDIA/dllogger wget
conda config --set channel_priority flexible
conda install conda-forge::libjemalloc -y
conda install conda-forge::openmpi -y
conda install conda-forge::mpi4py=3.1.3 -y

git clone https://github.com/andrewargatkiny/LDDL.git
cd LDDL && pip install -e .
python -m nltk.downloader punkt
export PATH=${PATH}:${PWD}/lddl/dask/bert
cd ..

# Download Wikipedia dump manually from the Internet Archive, for example enwiki-20180120 dump which we used
# https://archive.org/download/enwiki-20180120
wget https://archive.org/download/enwiki-20180120/enwiki-20180120-pages-articles.xml.bz2
mkdir wikipedia_dataset
ln -s $PWD/enwiki-20180120-pages-articles.xml.bz2 $PWD/wikipedia_dataset/wikicorpus-en.xml.bz2
download_wikipedia --outdir wikipedia_dataset/ --no-download

# Assuming that you have books1.tar.gz file in current directory
mkdir bookcorpus
download_books --no-download --outdir=./
mv source bookcorpus/

mkdir -p seqs128 seqs512 seqs1024 seqs16384_books seqs16384_wiki
cp -r bookcorpus wikipedia_dataset seqs128/
cd seqs128
MAX_SEQ_LEN=128 BOOKS_PATH="bookcorpus/source" WIKI_PATH="wikipedia_dataset/source/en" \
 C4_PATH="" N_TRAIN_RUNS=10 N_TEST_RUNS=1 INITIAL_SEED=100 N_WORKERS=112 create_dataset.sh
rm -rf bookcorpus wikipedia_dataset
shuffle_gather_output --input_hdf5=output_train --output_hdf5=final_data
rm -rf intermediate_hdf5_shards
mv raw_train raw_test ../seqs1024
#shuffle_gather_output --input_hdf5="output_test" --output_hdf5="final_test" --masked-lm-ratio=0.12
# For each created dataset, we move the last produced file from the main training directory as it
# contains less rows than others and cannot be used with in distributed training causing errors
# We use it as test dataset.


cd ../seqs1024
MAX_SEQ_LEN=1024 BOOKS_PATH="bookcorpus/source" WIKI_PATH="wikipedia_dataset/source/en" \
 C4_PATH="" N_TRAIN_RUNS=10 N_TEST_RUNS=1 INITIAL_SEED=100 CREATE_BASE_SHARDS=false \
 N_WORKERS=112 create_dataset.sh
shuffle_gather_output --input_hdf5="output_train" --output_hdf5="final_data" --shard_length=65536 --max_seq_length=1024
rm -rf intermediate_hdf5_shards
#shuffle_gather_output --input_hdf5="output_test" --output_hdf5="final_test" --masked-lm-ratio=0.12 --max_seq_length=1024 --shard_length=65536
#rm -rf intermediate_hdf5_shards
mv raw_train raw_test ../seqs16384_books

cd ../seqs16384_books
MAX_SEQ_LEN=16384 BOOKS_PATH="bookcorpus/source" WIKI_PATH="" \
C4_PATH="" INITIAL_SEED=100 CREATE_BASE_SHARDS=false N_WORKERS=112 \
N_TRAIN_RUNS=40 N_TEST_RUNS=1 create_dataset.sh
# After this stage check that all files have approximately the same size, otherwise smaller
# files may have corrupted data.
shuffle_gather_output --input_hdf5="output_train" --output_hdf5="final_data" --shard_length=16384 --max_seq_length=16384
rm -rf intermediate_hdf5_shards
# Test data for context length 16k consists only of books.
shuffle_gather_output --input_hdf5="output_test" --output_hdf5="final_test" --masked-lm-ratio=0.12  --shard_length=16384 --max_seq_length=16384
mv raw_train raw_test ../seqs16384_wiki

# Change "--partition-size" argument in lddl/dask/bert/pretrain.py to "5MB" just for this dataset,
# Otherwise the pipeline will fail.
cd ../seqs16384_wiki
MAX_SEQ_LEN=16384 BOOKS_PATH="" WIKI_PATH="wikipedia_dataset/source/en" \
C4_PATH="" INITIAL_SEED=100 CREATE_BASE_SHARDS=false N_WORKERS=64 \
N_TRAIN_RUNS=1 N_TEST_RUNS=0 PARTITION_SIZE="5MB" create_dataset.sh
shuffle_gather_output --input_hdf5="output_train" --output_hdf5="final_data" --shard_length=16384 --max_seq_length=16384
rm -rf intermediate_hdf5_shards
# After this stage I combined 137 books shards and 63 randomly chosed wikipedia ones to produce dataset of 200 shards.
# Then 312 books shards and 147 wiki shards for the second pass.
mv raw_train raw_test ../seqs512

cd ../seqs512
MAX_SEQ_LEN=512 BOOKS_PATH="bookcorpus/source" WIKI_PATH="wikipedia_dataset/source/en" \
 C4_PATH="" N_TRAIN_RUNS=10 N_TEST_RUNS=1 INITIAL_SEED=100 N_WORKERS=112 CREATE_BASE_SHARDS=false create_dataset.sh
shuffle_gather_output --input_hdf5="output_train" --output_hdf5="final_data"  --max_seq_length=512
rm -rf intermediate_hdf5_shards
#shuffle_gather_output --input_hdf5="output_test" --output_hdf5="final_test" --masked-lm-ratio=0.12  --max_seq_length=512
cd ..

# Download and process C4 realnewslike for test data
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "realnewslike"
cd .. & mkdir eval
cp -r c4/realnewslike eval/
mv eval/realnewslike eval/c4
cd eval && rm c4/c4-validation.00000-of-00001.json.gz
MAX_SEQ_LEN=1024 BOOKS_PATH="" WIKI_PATH="" C4_PATH="c4" \
INITIAL_SEED=100 N_WORKERS=64 \
N_TRAIN_RUNS=0 create_dataset.sh
shuffle_gather_output --input_hdf5="output_test" --output_hdf5="final_test" --masked-lm-ratio=0.12 --max_seq_length=1024 --shard_length=65536
rm -rf intermediate_hdf5_shards
mkdir seqs1024
mv output_test final_test seqs1024/
MAX_SEQ_LEN=512 BOOKS_PATH="" WIKI_PATH="" C4_PATH="c4" \
INITIAL_SEED=100 N_WORKERS=64 CREATE_BASE_SHARDS=false \
N_TRAIN_RUNS=0 create_dataset.sh
shuffle_gather_output --input_hdf5="output_test" --output_hdf5="final_test" --masked-lm-ratio=0.12 --max_seq_length=512 --shard_length=131072
rm -rf intermediate_hdf5_shards
mkdir seqs512 && mv output_test final_test seqs512/

MAX_SEQ_LEN=128 BOOKS_PATH="" WIKI_PATH="" C4_PATH="c4" \
INITIAL_SEED=100 N_WORKERS=64 CREATE_BASE_SHARDS=false \
N_TRAIN_RUNS=0 create_dataset.sh
shuffle_gather_output --input_hdf5="output_test" --output_hdf5="final_test" --masked-lm-ratio=0.12 --max_seq_length=128
rm -rf intermediate_hdf5_shards
mkdir seqs128 && mv output_test final_test seqs128/

# For testing, we choose number of shards to evaluate on in a way so their volumes
# sum up to around 450-500MB.
# But for books, 16k we only have one test file of size 236MB.