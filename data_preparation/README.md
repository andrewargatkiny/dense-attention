# Data Preparation

## [LRA](https://arxiv.org/abs/2011.04006) and PathFinder-256

To load and preprocess data for all benchmarks, launch from repository root:

```commandline
BASE_DATA_DIR="desired/path/to/data" data_preparation/prepare_lra.sh
```
Or, if you want to get processed data for a specific benchmark and assuming you 
have downloaded and extracted [the LRA data](https://storage.googleapis.com/long-range-arena/lra_release.gz.),
you can launch individual pre-processing scripts. e.g.:

```commandline
python3 data_preparation/lra/convert_pathfinder.py \
  --input_root_dir "${BASE_DATA_DIR}/lra_release/lra_release/pathfinder32" \
  --data_subdir curv_contour_length_14 \
  --output_base_dir "${BASE_DATA_DIR}/lra/pathfinder32"
```



## C4

It is recommended to use C4 dataset for all experiments with Masked Language 
Modeling. It is readily available [on HuggingFace](https://huggingface.co/datasets/allenai/c4).
A convenience script, which downloads relevant files and organizes them for immediate 
use in pre-training, can be invoked with:

```commandline
BASE_DATA_DIR="desired/path/to/data" data_preparation/prepare_c4.sh
```

## Legacy BERT data

for older experiments, Wiki + Books dataset similar to data mix [from the original BERT paper](https://arxiv.org/abs/1810.04805)
was used. This dataset is 
relatively small in size, takes a long time to prepare, and it yields subpar 
results in comparison with C4.
If you want to build it: 

```commandline
# Assuming that Wikipedia and BookCorpus data are available
BASE_DATA_DIR="desired/path/to/data" data_preparation/prepare_wiki_books_data.sh
```
After running this script, there will be `sentence_128`, `sentence_512`, etc. directories
for different sequence lengths.