import copy
import os
import random
from concurrent.futures import ProcessPoolExecutor

import datasets
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

def load_hf_dataset(dataset_config: dict) -> datasets.IterableDataset:
    """
    Loads a portion of a HugginFace dataset using `dataset_config` dict.

    Args:
        dataset_config (dict): A dictionary containing configuration parameters.
            Expected keys are:
            - "hf_dataset_name" (str): The name of the dataset on the Hub (e.g.
              HuggingFaceFW/fineweb-edu).
            - "hf_dataset_subset" (str): The subset of the dataset (e.g.
              'sample-100BT').
            - "offset" (int): The number of records to skip from the start of
              the sharded dataset.
            - "hf_chunk_size" (int): The number of records to take after the
              offset.
            - "hf_dataset_split" (str, optional): The dataset split to use.
              Defaults to "train".
            - "world_size" (int, optional): Splits the dataset into this
              number of shards which is useful for distributed training.
              Defaults to 1.
            - "global_rank" (int, optional): The shard of dataset with this
              index is used. Defaults to 0.
            - "filter_geq_column" (str, optional): Column to apply a "greater
              than or equal to" filter on. If "text", filters by word count.
            - "filter_geq_value" (int, optional): The value for the "geq"
              filter. Defaults to 0.
            - "filter_leq_column" (str, optional): Column to apply a "less
              than or equal to" filter on. If "text", filters by word count.
            - "filter_leq_value" (int, optional): The value for the "leq"
              filter. Defaults to 2**64.
    """

    TEXT_COLUMN = "text"
    ds = datasets.load_dataset(
        dataset_config["hf_dataset_name"],
        dataset_config["hf_dataset_subset"],
        split=dataset_config.get("hf_dataset_split", "train"),
        streaming=True
    ).shard(
        # Handle distributed training
        dataset_config.get("world_size", 1),
        dataset_config.get("global_rank", 0)
    ).skip(dataset_config["offset"]).take(dataset_config["hf_chunk_size"])
    if "code" in ds.features:
        ds = ds.rename_column("code", TEXT_COLUMN)
    # Optionally choose only entries where some columns are geq or
    # leq than some values.
    if "filter_geq_column" in dataset_config:
        column = dataset_config["filter_geq_column"]
        value = dataset_config.get("filter_geq_value", 0)
        f = lambda example: example[column] >= value
        if column == TEXT_COLUMN:
            # If we have only text column, let's use number of words
            # as a criterion.
            f = lambda example: len(example[column].split()) >= value
        ds = ds.filter(f)
    if "filter_leq_column" in dataset_config:
        column = dataset_config["filter_leq_column"]
        value = dataset_config.get("filter_leq_value", 2 ** 64)
        f = lambda example: example[column] <= value
        if column == TEXT_COLUMN:
            # If we have only text column, let's use number of words
            # as a criterion.
            f = lambda example: len(example[column].split()) <= value
        ds = ds.filter(f)
    ds = ds.select_columns(TEXT_COLUMN)
    return ds


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

        self.use_hf_dataset = False
        if self.dataset_config.get("hf_dataset_name"):
            self.use_hf_dataset = True
            # Number of raw dataset entries to treat as one chunk
            self.hf_chunk_size = self.dataset_config.get("hf_chunk_size", 2**20)
            # A pointer which moves `hf_chunk_size` entries over the
            # dataset each epoch.
            self.current_offset = 0
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
        if self.use_hf_dataset:
            print(f"rank {self.global_rank} "
                  f"dataset name {self.dataset_config['hf_dataset_name']} "
                  f"subset {self.dataset_config.get('hf_dataset_subset')}, "
                  f"offset {self.current_offset} entries {self.hf_chunk_size}")
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
        if self.use_hf_dataset:
            dataset_config["offset"] = offset_or_datafile
            dataset_config["hf_chunk_size"] = self.hf_chunk_size
            dataset_config["world_size"] = self.world_size
            dataset_config["global_rank"] = self.global_rank
            self.logger.info(
                f"ShardedDatasetWrapper - {tag} dataset offset: "
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
        if self.use_hf_dataset:
            self.current_offset = self.hf_chunk_size * shard_index
            return self.current_offset
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
