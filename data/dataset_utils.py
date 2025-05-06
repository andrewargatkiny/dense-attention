import copy
import os
import random
from concurrent.futures import ProcessPoolExecutor

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

        # Initialize dataset files
        self.dataset_path = os.path.join(
            base_dir,
            dataset_config["input_files_path"])
        self.dataset_files = [
            f for f in os.listdir(self.dataset_path) if
            os.path.isfile(os.path.join(self.dataset_path, f)) #and 'training' in f
        ]
        self.dataset_files.sort()

        random.seed(args.seed)
        random.shuffle(self.dataset_files)
        self.num_files = len(self.dataset_files)
        self.data_sampler = RandomSampler

        self.worker_init = WorkerInitObj(args.seed + args.local_rank)
        self.dataset_future = None
        import multiprocessing
        self.pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))

        if self.global_rank == 0:
            self.logger.info(
                f"ShardedDatasetWrapper - Initialization:  num_files = {self.num_files}"
            )

    def dataset_order_info(self):
        for i in range(0, self.num_files // 4):
            print(f"rank {self.global_rank} {i}-th foursome of files: {self.dataset_files[4 * i:4 * (i + 1)]}")
        print(f"rank {self.global_rank} last files: {self.dataset_files[(self.num_files // 4) * 4:]}")

    def load_dataset(self, dataset_config):
        dataset = self.dataset_class(base_dir=self.dataset_path,
                                     dataset_config=dataset_config,
                                     args=self.args)
        #self.dataset_future = dataset
        return dataset

    def get_shard(self, index):
        if self.dataset_future is None:
            data_file = self._get_shard_file(index)
            dataset_config = copy.deepcopy(self.dataset_config)
            dataset_config["input_file"] = data_file
            self.logger.info(
                f"ShardedDatasetWrapper - current data_file: {data_file}"
            )
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
        data_file = self._get_shard_file(index)
        self.logger.info(
            f"ShardedDatasetWrapper - next data_file: {data_file}"
        )
        dataset_config = copy.deepcopy(self.dataset_config)
        dataset_config["input_file"] = data_file
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
