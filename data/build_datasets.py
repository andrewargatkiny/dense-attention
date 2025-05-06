#!/usr/bin/env python3

# Run with python -m data.build_datasets from project root
import os
from deepspeed_train import construct_arguments

"""Pre-builds and saves to disk datasets for a specific task/ raw data given
a `ds_train*` config."""


if __name__ == "__main__":
    args = construct_arguments()
    config = args.config
    logger = args.logger
    task = args.task

    dataset_config = config["data"]["training"]
    dataset_path = os.path.join(
        args.data_path_prefix,
        dataset_config["input_files_path"])
    dataset_files = [
        f for f in os.listdir(dataset_path) if
        os.path.isfile(os.path.join(dataset_path, f))  # and 'training' in f
    ]
    dataset_files.sort()

    for data_file in dataset_files:
        dataset_config["input_file"] = os.path.join(dataset_path, data_file)
        dataset_config["save"] = True
        dataset = task.dataset_type(args.data_path_prefix, dataset_config, None)
