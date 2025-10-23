import os
import json
import pytest
import shutil
import numpy as np

from datasets import Value, Features
from data.dataset_utils import load_hf_datasets, materialize_data



def make_data(num_files, len_file):
    data = []
    os.makedirs("tests/data/dataset", exist_ok=True)
    for i in range(num_files):
        with open(f"tests/data/dataset/file_{i:02d}.jsonl", "w", encoding="utf-8") as f:
            for j in range(1,len_file+1):
                value = len_file * i + j
                data.append(f"{value}")
                
                json.dump({"file": i, "text": value}, f)
                f.write("\n")
    return num_files * len_file, data

def repeated_elements(arrays):
    unique, counts = np.unique(arrays, return_counts=True)
    repeated = unique[counts > 1]
    return repeated

def get_config(world_size, rank, chunk_size, offset=0):
        dataset_config = [
                {
                    "name": "tests/data/dataset",
                    "chunk_size": chunk_size,
                    "dataset_weight": 1,
                    "offset": offset,
                    "world_size": world_size,
                    "global_rank": rank,
                    "features":Features({
                        "file" : Value("string"),
                        "text" : Value("string")
                    })   
                }
            ]
        return dataset_config

@pytest.fixture(autouse=True)
def cleanup_dataset_folder():
    yield
    folder = "tests/data/dataset"
    if os.path.exists(folder):
        shutil.rmtree(folder)


@pytest.mark.parametrize("len_file", [11,19,20,22,27])
@pytest.mark.parametrize("num_files", [1,4,5,10,11])
@pytest.mark.parametrize("shards", [2,3,4,8])
def test_shard(num_files, shards, len_file):
    """Test data sharding across multiple processes for correctness and consistency.
    This test generates a synthetic dataset split into several JSONL files,
    simulating distributed data loading (sharding) across multiple workers.
    Each shard is expected to receive a unique, non-overlapping portion of the dataset.
    """
    arrays = []
    chunk_size, data = make_data(num_files, len_file)

    for shard in range(shards):
        if shard < num_files:
            dataset_config = get_config(shards, shard, chunk_size)
            dataset = load_hf_datasets(dataset_config)
            dataset = materialize_data(dataset)

            arrays.append(dataset)

            print(f"{shard}_shard", dataset)

    print("data", data)
    concatenated = np.concatenate(arrays)
    assert np.array_equal(np.array(data), np.array(concatenated)), "Data mismatch: elements differ"
    assert repeated_elements(concatenated).size == 0, f"Overlapping elements found: {repeated_elements(concatenated)}"


@pytest.mark.parametrize("len_file", [11,19,20,22,27])
@pytest.mark.parametrize("num_files", [1,4,5,10,11])
@pytest.mark.parametrize("chunk_size", [10,15,25,30,51])
@pytest.mark.parametrize("offset", [0,2,7,20,29,57,100])
def test_epochs(chunk_size, offset, num_files, len_file):
    """"Test sequential dataset loading across multiple epochs with different offsets.
    This test generates a synthetic dataset split into several JSONL files,
    simulates iterating over a dataset in batches of a given `chunk_size`,
    starting from a specified `offset`. Each epoch loads a new portion of the data,
    and together they should cover the entire dataset without overlaps or gaps.
    """

    world_size = 1
    rank = 0

    arrays = []
    file_len, data = make_data(num_files,len_file)
    file_len_with_offset = file_len - offset
    epochs = file_len_with_offset // chunk_size + 1
    for epoch in range(epochs):

        cur_offset = chunk_size * epoch + offset
        dataset_config = get_config(world_size, rank, chunk_size, cur_offset)

        dataset = load_hf_datasets(dataset_config)
        dataset = materialize_data(dataset)

        arrays.append(dataset)
        print(f"{epoch}_epoch", dataset)

    print("data", data[offset:])

    if file_len_with_offset > 0:
        concatenated = np.concatenate(arrays)
        assert np.array_equal(np.array(data[offset:]), np.array(concatenated)), "Data mismatch: elements differ"
        assert repeated_elements(concatenated).size == 0, f"Overlapping elements found: {repeated_elements(concatenated)}"