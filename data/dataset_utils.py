import copy
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

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
    Configuration for loading a dataset (from the Hugging Face or local files), 
    potentially for interleaving.

    This dataclass holds all parameters needed to load a portion of a single
    dataset. Multiple instances of this class can be used to specify several
    datasets that are then interleaved into one, using the 'dataset_weight'
    parameter of each configuration.

    Attributes:
        name_or_path (str): The name of the dataset on the Hub (e.g.
            "HuggingFaceFW/fineweb-edu") or a local path to dataset files 
            (e.g. "/data/bert_mlm/fineweb-edu").
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
    name_or_path: str
    chunk_size: int

    # Optional parameters with default values
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


def load_datasets(dataset_configs: List[dict],
                     seed: Optional[int] = None) -> datasets.IterableDataset:
    """
    Loads one or several datasets (from the Hugging Face or local files), 
    specified in `dataset_configs`, in streaming mode, and mixes them into one.

    Args:
        dataset_configs (List[dict]): A list of dataset configurations (each may
            refer to either a HF dataset name or a local path) to load and interleave.
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
    Loads a portion of a single dataset (from the Hugging Face or local files)
    in streaming mode using a DatasetParams config.

    Args:
        dataset_config (DatasetParams): Configuration object for the dataset.
    """
    TEXT_COLUMN = "text"
    ds = datasets.load_dataset(
        dataset_config.name_or_path,
        dataset_config.subset,
        split=dataset_config.split,
        trust_remote_code=dataset_config.trust_remote_code,
        streaming=True
    ).shard(
        # Handle distributed training
        dataset_config.world_size,
        dataset_config.global_rank
    ).skip(dataset_config.offset).take(dataset_config.chunk_size)
    for name in ['code', 'page', 'content']:
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
        if self.dataset_config.get("sources"):
            self.use_hf_sources = True
            self.chunk_sizes = dict()
            self.current_offsets = dict()
            for source in dataset_config["sources"]:
                # Number of raw dataset entries to treat as one chunk
                self.chunk_sizes[source["name_or_path"]] = source.get(
                    "chunk_size", 2 ** 20)
                # A pointer which moves `chunk_size` entries over the
                # dataset each epoch.
                self.current_offsets[source["name_or_path"]] = source.get("offset", 0)
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
        if self.use_hf_sources:
            for source in self.dataset_config["sources"]:
                print(f"rank {self.global_rank} "
                      f"dataset name {source['name_or_path']} "
                      f"subset {source.get('subset')}, "
                      f"offset {self.current_offsets[source['name_or_path']]} "
                      f"entries {self.chunk_sizes[source['name_or_path']]}")
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

    def get_dataset_config(self, index: int, tag: str = "next") -> dict:
        dataset_config = copy.deepcopy(self.dataset_config)
        offset_or_datafile = self._get_shard_file(index)
        if self.use_hf_sources:
            for source in dataset_config["sources"]:
                source["offset"] = offset_or_datafile[source["name_or_path"]]
                source["chunk_size"] = self.chunk_sizes[source["name_or_path"]]
                source["world_size"] = self.world_size
                source["global_rank"] = self.global_rank
                if dataset_config.get("add_base_dir", False):
                    source["name_or_path"] = os.path.join(self.dataset_path, source["name_or_path"])
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
        if self.use_hf_sources:
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