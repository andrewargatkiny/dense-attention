import argparse
import logging
import os
import json
from pathlib import Path
from time import asctime
from functools import partial

import pandas as pd
import numpy as np
from PIL import Image
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

def main(args):
    dataset = load_dataset("uoft-cs/cifar10")
    dataset = DatasetDict(train=dataset["train"], test=dataset["test"])
    numericalize = lambda example: {
        "ids": np.array(example["img"].convert("L")).flatten()
    }
    dataset = dataset.map(numericalize, remove_columns=["img"])
    # Shuffle if necessary
    if args.shuffle:
        dataset["train"].shuffle(seed=args.seed)

    split_l = int(len(dataset["train"]) * args.val_split)
    dataset["valid"] = dataset["train"][split_l:]
    dataset["train"] = dataset["train"][:split_l]

    for subset in ["train", "valid", "test"]:
        logging.info(f"Started saving labels at {asctime()}")
        labels = np.array(dataset[subset]["label"], dtype=np.int8)
        # Write labels to a file
        labels_dir = os.path.join(args.output_base_dir, "label")
        os.makedirs(labels_dir, exist_ok=True)
        labels_path = os.path.join(labels_dir, f"{subset}.label")
        np.savetxt(labels_path, labels, fmt='%d')
        logging.info(f"Finished saving labels at {asctime()}")

        # Write sequences to a file
        logging.info(f"Started saving sequences at {asctime()}")
        sequences = np.array(dataset[subset]["ids"], dtype=np.int16)
        seqs_dir = os.path.join(args.output_base_dir, "input")
        os.makedirs(seqs_dir, exist_ok=True)
        sequences_path = os.path.join(seqs_dir, f"{subset}.src")
        np.savetxt(sequences_path, sequences, fmt='%d')
        logging.info(f"Finished saving sequences at {asctime()}")

        print(f"Subset {subset}")
        print(f"Processed {len(sequences)} samples")
        print(f"Values and counts of labels: "
              f"{np.unique(labels, return_counts=True)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Preprocess Cifar dataset")

    parser.add_argument(
        "--output_base_dir",
        type=str,
        help="Path to the base output folder"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.9,
        help="Proportion on how to make train and validation split, denotes "
             "how many of data points remain in train part. Should be between "
             "0 and 1."
    )

    parser.add_argument(
        "--shuffle",
        default=False,
        action="store_true",
        help="Whether to shuffle sequences before forming data splits."
             "Default False."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed in case the train part would be shuffled before extracting "
             "validation split."
    )

    args = parser.parse_args()
    logging.info('args: {}'.format(args))

    main(args)