import copy
import os
import io
import gzip
import json
import random
import time
import pyarrow.parquet as pq
import zstandard as zstd
import pandas as pd
from orjson import loads
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Dict
from itertools import islice
from collections import defaultdict

import datasets
import huggingface_hub
import numpy as np
from torch import distributed as dist
from torch.utils.data import DataLoader, RandomSampler


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def create_dataloader(train_data, num_workers,
                      train_batch_size, data_sampler,
                      worker_init=None):
    train_dataloader = DataLoader(train_data,
                                  sampler=data_sampler,
                                  batch_size=train_batch_size,
                                  num_workers=num_workers,
                                  worker_init_fn=worker_init,
                                  pin_memory=True)
    return train_dataloader, len(train_data)


@dataclass
class DatasetParams:
    """
    Configuration for loading a HuggingFace dataset, potentially for interleaving.

    This dataclass holds all parameters needed to load a portion of a single
    HuggingFace dataset. Multiple instances of this class can be used to specify
    several datasets that are then interleaved into one, using the 'dataset_weight'
    parameter of each configuration.

    Attributes:
        name (str): The name of the dataset on the Hub (e.g.
            "HuggingFaceFW/fineweb-edu").
        chunk_size (int): The number of records to take after the offset.
        subset (Optional[str], optional): The subset of the dataset (e.g.
            "sample-100BT"). Defaults to None.
        offset (int, optional): The number of records to skip from the start
            of the sharded dataset. Defaults to 0.
        split (str, optional): The dataset split to use. Defaults to "train".
        trust_remote_code (bool, optional): See description of this option in
            HuggingFace's `datasets.load_dataset`. Defaults to None.
        world_size (int, optional): Splits the dataset into this number of shards,
            which is useful for distributed training. Defaults to 1.
        global_rank (int, optional): The shard of the dataset with this index
            is used. Defaults to 0.
        dataset_weight (float, optional): The weight for interleaving this
            dataset with others. Defaults to 1.0.
        filter_geq_column (Optional[str], optional): Column to apply a "greater
            than or equal to" filter on. If "text", filters by word count.
            Defaults to None.
        filter_geq_value (int, optional): The value for the "geq" filter.
            Defaults to 0.
        filter_leq_column (Optional[str], optional): Column to apply a "less
            than or equal to" filter on. If "text", filters by word count.
            Defaults to None.
        filter_leq_value (int, optional): The value for the "leq" filter.
            Defaults to 2**64.
    """
    # Required parameters
    name: str
    chunk_size: int

    # Optional parameters with default values

    start_file: int = None
    end_file: int = None
    local_offset: int = None
    take_in_file: int = None
    files_len: list = None
    paths_files: list = None
    cumulate_len: list = None

    subset: Optional[str] = None
    offset: int = 0
    split: str = "train"
    trust_remote_code: Optional[bool] = None
    world_size: int = 1
    global_rank: int = 0
    dataset_weight: float = 1.0
    filter_geq_column: Optional[str] = None
    filter_geq_value: int = 0
    filter_leq_column: Optional[str] = None
    filter_leq_value: int = 2**64


def load_hf_datasets(dataset_configs: List[dict],
                     seed: Optional[int] = None) -> datasets.IterableDataset:
    """
    Loads one or possibly several HuggingFace datasets, specified in
    `dataset_configs`, in streaming mode and mixes them into one.

    Args:
    dataset_configs (List[dict]): A list of HG dataset configurations to load
        and interleave.
    seed (Optional[int], optional): An optional seed for the pseudo-random
        interleaving process. Defaults to None.
    """
    dataset_configs = [DatasetParams(**config) for config in dataset_configs]
    if len(dataset_configs) == 1:
        return load_hf_dataset(dataset_configs[0])
    ds_list = []
    ds_weights = []
    for single_config in dataset_configs:
        ds_list.append(load_hf_dataset(single_config))
        ds_weights.append(single_config.dataset_weight)
    ds_weights = np.array(ds_weights)
    assert np.all(ds_weights >= 0), f"Dataset weights {ds_weights} should be non-negative."
    ds_weights = (ds_weights / ds_weights.sum()).tolist()
    return datasets.interleave_datasets(ds_list, seed=seed,
                                        probabilities=ds_weights,
                                        stopping_strategy="all_exhausted")

def load_hf_dataset(dataset_config: DatasetParams) -> datasets.IterableDataset:
    """
    Loads a portion of a single HuggingFace dataset in streaming mode using a
    HFDatasetParams config.

    Args:
        dataset_config (HFDatasetParams): Configuration object for the dataset.
    """
    TEXT_COLUMN = "text"
    ds = datasets.load_dataset(
        dataset_config.name,
        dataset_config.subset,
        split=dataset_config.split,
        trust_remote_code=dataset_config.trust_remote_code,
        streaming=True
    ).shard(
        # Handle distributed training
        dataset_config.world_size,
        dataset_config.global_rank
    ).skip(dataset_config.offset).take(dataset_config.chunk_size)
    for name in ['code', 'page']:
        if name in ds.features:
            ds = ds.rename_column(name, TEXT_COLUMN)
    # Optionally choose only entries where some columns are geq or
    # leq than some values.
    if dataset_config.filter_geq_column is not None:
        column = dataset_config.filter_geq_column
        value = dataset_config.filter_geq_value
        f = lambda example: example[column] >= value
        if column == TEXT_COLUMN:
            # If we have only text column, let's use number of words
            # as a criterion.
            f = lambda example: len(example[column].split()) >= value
        ds = ds.filter(f)
    if dataset_config.filter_leq_column is not None:
        column = dataset_config.filter_leq_column
        value = dataset_config.filter_leq_value
        f = lambda example: example[column] <= value
        if column == TEXT_COLUMN:
            # If we have only text column, let's use number of words
            # as a criterion.
            f = lambda example: len(example[column].split()) <= value
        ds = ds.filter(f)
    ds = ds.select_columns(TEXT_COLUMN)
    return ds

def materialize_data(ds: datasets.IterableDataset) -> List[str]:
    """Materialize iterable dataset into an actual texts, handling possible
    download errors."""
    success_download = False
    seconds_to_sleep = 60
    while not success_download:
        try:
            all_sentences = [example["text"] for example in ds]
            success_download = True
            return all_sentences
        except huggingface_hub.errors.HfHubHTTPError as ex:
            print(ex)
            print(f"Trying again to load the data "
                  f"in {seconds_to_sleep} seconds.")
            time.sleep(seconds_to_sleep)
            seconds_to_sleep = max(300, seconds_to_sleep + 60)





def open_compressed_file(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.endswith(".zst") or path.endswith(".zstd"):
        dctx = zstd.ZstdDecompressor()
        f = open(path, "rb")
        return io.TextIOWrapper(dctx.stream_reader(f), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def get_file_len(path: str) -> int:
    if path.endswith(".parquet"):
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
    else:
        n = sum(1 for _ in open_compressed_file(path))
    return n


def read_chunk_from_file(path: str, take: int,  local_offset: int = 0) -> pd.DataFrame:
    """
    Efficiently read a chunk of data from a file (Parquet or JSONL) starting at a given offset.

    Args:
        path (str): Path to the dataset file (.parquet, .jsonl, .gz, .zst, .txt)
        local_offset (int): Starting row offset within the file
        take (int): Number of rows to read

    Returns:
        pd.DataFrame: Slice of dataset as DataFrame
    """
    result = []
    
    if path.endswith(".parquet"):
        pf = pq.ParquetFile(path)
        row_group_sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
        cumsum = np.cumsum([0] + row_group_sizes)

        start_group = np.searchsorted(cumsum, local_offset, side="right") - 1
        end_group = np.searchsorted(cumsum, local_offset + take, side="left")

        result = []

        for rg in range(start_group, end_group + 1):
            group_start = cumsum[rg]
            group_end = cumsum[rg + 1]

            start_in_group = max(0, local_offset - group_start)
            end_in_group = min(group_end, local_offset + take) - group_start

            if end_in_group <= start_in_group:
                continue

            table = pf.read_row_group(rg)
            result.append(table.slice(start_in_group, end_in_group - start_in_group).to_pandas())

        df = pd.concat(result, ignore_index=True)

    else:
        with open_compressed_file(path) as f:
            print(local_offset, local_offset + take)
            for line in islice(f, local_offset, local_offset + take):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = loads(line)
                except Exception:
                    continue
                result.append(obj)
                
                
        df = pd.DataFrame(result) if result else pd.DataFrame()
    
    df = df[df["value"].notnull()].copy()
    df["value"] = df["value"].astype(str)
    return df


def read_sharded_across_files(files: List[str],
                              local_offset: int,
                              take_in_files: int,
                              start_file: int,
                              end_file: int,
                              files_len: List[int]) -> List[Dict]:
    dfs = []
    if start_file == end_file:
        df = read_chunk_from_file(files[start_file], take_in_files, local_offset)
        return df.reset_index(drop=True)
    
    
    file_len = files_len[start_file] if files_len else get_file_len(files[start_file])
    dfs.append(read_chunk_from_file(files[start_file], file_len - local_offset, local_offset))

    for file in range(start_file + 1, end_file):
        file_len = files_len[file] if files_len else get_file_len(files[file])
        dfs.append(read_chunk_from_file(files[file], files_len[file]))
        
    dfs.append(read_chunk_from_file(files[end_file], take_in_files))

    df = pd.concat(dfs, ignore_index=True)
    return df

def apply_filters(data: List[Dict], params: DatasetParams) -> List[Dict]:
    if params.filter_geq_column:
        col = params.filter_geq_column
        val = params.filter_geq_value
        if col == "text":
            data = [ex for ex in data if len(ex.get("text", "").split()) >= val]
        else:
            data = [ex for ex in data if ex.get(col) is not None and ex.get(col) >= val]

    if params.filter_leq_column:
        col = params.filter_leq_column
        val = params.filter_leq_value
        if col == "text":
            data = [ex for ex in data if len(ex.get("text", "").split()) <= val]
        else:
            data = [ex for ex in data if ex.get(col) is not None and ex.get(col) <= val]

    return data

def load_local_datasets(dataset_configs: List[dict], seed: Optional[int] = None) -> List[Dict]:
    """
    Loads data from multiple sources, applies filters, and combines
    using weighted random sampling. Supports both single file and
    directory inputs.
    Args:
        dataset_configs: List of dataset configuration dictionaries
        seed: Random seed for reproducible dataset mixing (optional)
        
    Returns:
        List[Dict]: Combined dataset with weighted sampling, or single 
        dataset if only one provided 
    """
    params_list = [DatasetParams(**cfg) for cfg in dataset_configs]
    chunks_with_weights = []

    dfs = []
    weights = []
    for params in params_list:
        data_chunk = read_sharded_across_files(
            files = params.paths_files,
            local_offset = params.local_offset,
            take_in_files = params.take_in_file,
            start_file = params.start_file,
            end_file = params.end_file,
            files_len = params.files_len
        )
        data_chunk = apply_filters(data_chunk, params)
        dfs.append(data_chunk)
        weights.append(params.dataset_weight)

    if len(chunks_with_weights) == 1:
        return chunks_with_weights[0][0][:params_list[0].chunk_size]
    if seed is not None:
        random.seed(seed)

    n_dfs = len(dfs)
    positions = np.zeros(n_dfs, dtype=int)
    lengths = np.array([len(df) for df in dfs], dtype=int)
    weights = np.array(weights, dtype=float)
    weights = (weights / weights.sum()).tolist()
    
    combined_rows = []
    while True:
        if np.any(positions >= lengths):
            break
        idx = np.random.choice(n_dfs, p=weights)
        combined_rows.append(dfs[idx].iloc[positions[idx]])
        positions[idx] += 1
    return pd.DataFrame(combined_rows).reset_index(drop=True)

class ShardedDatasetWrapper:
    """For multi-file datasets and distributed training. Each data file should
    have all components necessary for training, e.g. input, mask, label."""
    def __init__(self, base_dir, dataset_config, args):
        self.base_dir = base_dir
        self.dataset_config = dataset_config

        self.args = args
        self.dataset_class = args.task.dataset_type

        self.logger = args.logger
        if args.local_rank == -1:
            self.global_rank = 0
            self.world_size = 1
        else:
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.use_hf_sources = False
        self.use_local_sources = False
        if self.dataset_config.get("hf_sources"):
            self.use_hf_sources = True
        if self.dataset_config.get("local_sources"):
            self.use_local_sources = True

        self.name = "name"
        if self.use_hf_sources or self.use_local_sources:
            self.chunk_sizes = dict()
            self.current_offsets = dict()
            
            if self.use_local_sources:
                self.files_len = dict()
                self.cumulate_len = dict()
                self.paths_files = dict()
                
            for source in dataset_config[self.source]:
                # Number of raw dataset entries to treat as one chunk
                self.chunk_sizes[source[self.name]] = source.get(
                    "chunk_size", 2 ** 20)
                # A pointer which moves `chunk_size` entries over the
                # dataset each epoch.
                self.current_offsets[source[self.name]] = source.get("offset", 0)
                if self.use_local_sources:
                    files = sorted(
                        os.path.join(source[self.name], f)
                        for f in os.listdir(source[self.name])
                        if os.path.isfile(os.path.join(source[self.name], f))
                    )
                    split_files = [arr.tolist() for arr in np.array_split(files, self.world_size)]
                    self.paths_files[source[self.name]] = split_files[self.global_rank]
                    
                    self.cumulate_len[source[self.name]] = source.get("cumulate_len", 0)
                    self.files_len[source[self.name]] = source.get("files_len", [])

        # Initialize dataset files
        self.dataset_path = os.path.join(
            base_dir,
            dataset_config.get("input_files_path", ""))
        self.dataset_files = [
            f for f in os.listdir(self.dataset_path) if
            os.path.isfile(os.path.join(self.dataset_path, f)) #and 'training' in f
        ]
        self.dataset_files.sort()

        random.seed(args.seed)
        random.shuffle(self.dataset_files)
        self.num_files = len(self.dataset_files)

        self.worker_init = WorkerInitObj(args.seed + args.local_rank)
        self.dataset_future = None
        import multiprocessing
        self.pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))

        if self.global_rank == 0:
            self.logger.info(
                f"ShardedDatasetWrapper - Initialization:  num_files = {self.num_files}"
            )

    def dataset_order_info(self):
        if self.use_hf_sources or self.use_local_sources:
            for source in self.dataset_config[self.source]:
                print(f"rank {self.global_rank} "
                      f"dataset name {source[self.name]} "
                      f"subset {source.get('subset')}, "
                      f"offset {self.current_offsets[source[self.name]]} "
                      f"entries {self.chunk_sizes[source[self.name]]}")
            return
        for i in range(0, self.num_files // 4):
            print(f"rank {self.global_rank} {i}-th foursome of files: {self.dataset_files[4 * i:4 * (i + 1)]}")
        print(f"rank {self.global_rank} last files: {self.dataset_files[(self.num_files // 4) * 4:]}")

    def load_dataset(self, dataset_config):
        dataset = self.dataset_class(base_dir=self.dataset_path,
                                     dataset_config=dataset_config,
                                     args=self.args)
        #self.dataset_future = dataset
        return dataset

    def get_num_file(self, source):    
        path = source[self.name]
        files = self.paths_files[path]
        cur = len(self.files_len[path])
        
        while source["offset"] >= self.cumulate_len[path] and cur < len(files):
            length = get_file_len(files[cur])
            self.files_len[path].append(length)
            self.cumulate_len[path] += length
            cur += 1
        
        assert self.cumulate_len[path] >= source["offset"], \
            (f"закончился")
        start_file = max(0, cur - 1)
        len_last_file = self.files_len[path][-1]
        local_offset = len_last_file - (self.cumulate_len[path] - source["offset"])

        while source["offset"] + source["chunk_size"] > self.cumulate_len[path] and cur < len(files):
            length = get_file_len(files[cur])
            self.files_len[path].append(length)
            self.cumulate_len[path] += length
            cur += 1

        end_file = cur - 1
        take_in_file = len_last_file - (self.cumulate_len[path] - source["offset"] - source["chunk_size"])

        return local_offset, take_in_file, start_file, end_file

    def get_dataset_config(self, index: int, tag: str = "next") -> dict:
        dataset_config = copy.deepcopy(self.dataset_config)
        offset_or_datafile = self._get_shard_file(index)
        if self.use_hf_sources or self.use_local_sources:
            for source in dataset_config[self.source]:
                source["offset"] = offset_or_datafile[source[self.name]]
                source["chunk_size"] = self.chunk_sizes[source[self.name]]
                source["world_size"] = self.world_size
                source["global_rank"] = self.global_rank
                
                if self.use_local_sources:
                    source["paths_files"] = self.paths_files[source[self.name]]
                    source["files_len"] = self.files_len[source[self.name]]
                    source["cumulate_len"] = self.cumulate_len[source[self.name]]
                    
                    (source["local_offset"],
                    source["take_in_file"],
                    source["start_file"],
                    source["end_file"]) = self.get_num_file(source)
                    
                self.logger.info(
                    f"ShardedDatasetWrapper - {tag} dataset offsets: "
                    f"{offset_or_datafile}"
                )

        else:
            dataset_config["input_file"] = offset_or_datafile
            self.logger.info(
                f"ShardedDatasetWrapper - {tag} data_file: "
                f"{offset_or_datafile}"
            )
        return dataset_config

    def get_shard(self, index):
        if self.dataset_future is None:
            dataset_config = self.get_dataset_config(index, tag="current")
            shard = self.dataset_class(
                base_dir=self.dataset_path,
                dataset_config=dataset_config,
                args=self.args
            )
        else:
            shard = self.dataset_future.result(timeout=None)

        #self.dataset_future = None
        self.prefetch_shard(index + 1)
        return shard

    def release_shard(self, index):
        pass

    def prefetch_shard(self, index):
        dataset_config = self.get_dataset_config(index, tag="next")
        # Will cause problems with dataset using args
        self.dataset_future = self.pool.submit(
            self.dataset_class,
            base_dir=self.dataset_path, dataset_config=dataset_config,
            args=None)

    def get_batch(self, batch_iter):
        return batch_iter

    def prefetch_batch(self):
        pass

    def _get_shard_file(self, shard_index):
        if self.use_hf_sources or self.use_local_sources:
            for ds_name in self.current_offsets:
                self.current_offsets[ds_name] = (
                        self.chunk_sizes[ds_name] * shard_index)
            return self.current_offsets
        file_index = self._get_shard_file_index(shard_index, self.global_rank)
        return self.dataset_files[file_index % self.num_files]

    def _get_shard_file_index(self, shard_index, global_rank):
        if dist.is_initialized() and self.world_size > self.num_files:
            remainder = self.world_size % self.num_files
            file_index = (shard_index * self.world_size) + global_rank + (
                remainder * shard_index)
        else:
            file_index = shard_index * self.world_size + global_rank

        return file_index % self.num_files