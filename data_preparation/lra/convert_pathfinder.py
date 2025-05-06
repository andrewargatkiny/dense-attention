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
from tqdm import tqdm


def get_samples(base_path: str, data_dir: str):
    # Handle an empty image in one of the datasets
    BAD_IMAGE = "sample_172.png"
    BAD_IMAGE_DIR = "imgs/0"
    blacklisted = False
    if "pathfinder32" in base_path and data_dir == "curv_baseline":
        blacklisted = True

    base_path = Path(base_path).expanduser()
    assert base_path.is_dir(), f"data_dir {str(base_path)} does not exist"
    diff_levels = ['curv_baseline', 'curv_contour_length_9', 'curv_contour_length_14']
    assert data_dir in diff_levels, f"data_dir should be one of the following: {diff_levels}"

    # Collect metadata paths
    path_list = sorted(
        list((base_path / data_dir / "metadata").glob("*.npy")),
        key=lambda path: int(path.stem),
    )
    assert path_list, "No metadata found"
    logging.info("Metadata files: {}".format(path_list))

    # Process each metadata file
    samples = []
    labels = []
    for metadata_file in path_list:
        logging.info("Started processing medata file {}".format(metadata_file))
        with open(metadata_file, "r") as f:
            logging.info(f"Started converting images from "
                         f"{metadata_file} at {asctime()}")
            for metadata in tqdm(f.read().splitlines()):
                metadata = metadata.split()
                image_path = base_path / data_dir / metadata[0] / metadata[1]
                if (blacklisted and metadata[1] == BAD_IMAGE
                    and metadata[0] == BAD_IMAGE_DIR):
                    continue
                with open(image_path, "rb") as img:
                    sample = np.array(Image.open(img).convert("L")).flatten()
                samples.append(sample)
                labels.append(int(metadata[3]))
    samples = np.stack(samples)
    labels = np.array(labels)
    return samples, labels


def main(args):

    assert len(args.split_proportions) == len(args.split_names), \
        "Number of splits should be equal between their names and values"
    assert min(args.split_proportions) > 0, "Split proportions should be non-negative"

    # Main logic is here
    all_sequences, all_labels = get_samples(args.input_root_dir,
                                            args.data_subdir)

    # Prepare split proportions and names
    # Normalize split_proportions
    split_proportions = np.array([0] + args.split_proportions) / sum(args.split_proportions)
    split_bounds = (np.cumsum(split_proportions) * len(all_sequences)).astype(int)
    assert split_bounds[-1] == len(all_sequences), \
        "Rightmost bound isn't equal to dataset length"
    split_names = [None] + args.split_names

    # Shuffle if necessary
    if args.shuffle:
        np.random.seed(args.seed)
        permutation = np.random.permutation(len(all_sequences))
        all_sequences = all_sequences[permutation]
        all_labels = all_labels[permutation]

    # Save datasets according to split boundaries
    for i in range(1, len(split_bounds)):
        l_bound = split_bounds[i - 1]
        r_bound = split_bounds[i]
        dataset_name = split_names[i]
        sequences = all_sequences[l_bound:r_bound]
        labels = all_labels[l_bound:r_bound]

        # Write labels to a file
        labels_dir = os.path.join(args.output_base_dir, "label")
        os.makedirs(labels_dir, exist_ok=True)
        labels_path = os.path.join(labels_dir, f"{dataset_name}.label")
        np.savetxt(labels_path, labels, fmt='%d')
        logging.info(f"Finished saving labels at {asctime()}")

        # Write sequences to a file
        logging.info(f"Started saving sequences at {asctime()}")
        seqs_dir = os.path.join(args.output_base_dir, "input")
        os.makedirs(seqs_dir, exist_ok=True)
        sequences_path = os.path.join(seqs_dir, f"{dataset_name}.src")
        np.savetxt(sequences_path, sequences, fmt='%d')
        logging.info(f"Finished saving sequences at {asctime()}")
        print(f"Processed {len(sequences)} samples")

    print(f"Processed {len(all_sequences)} samples")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Preprocess Pathfinder dataset")
    parser.add_argument(
        "--input_root_dir",
        type=str,
        help="Path to the base input directory (i. e. pathfinder32 or pathfinder128)."
    )
    parser.add_argument(
        "--data_subdir",
        type=str,
        default="curv_contour_length_14",
        help="directory name of a particular dataset variety, "
             "different in difficulty level."
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        help="Path to the base output folder"
    )
    parser.add_argument(
        "--split_proportions",
        type=float,
        default=[0.8, 0.1, 0.1],
        nargs="*",
        help="Proportions of the parts to split the data, e.g. 0.6, 0.2, 0.2."
    )
    parser.add_argument(
        "--split_names",
        type=str,
        nargs="*",
        default=["train", "valid", "test"],
        help="names of the output datasets, after split, in the same order "
             "as --split_proportions e.g 'train', 'valid', 'test'."
    )
    parser.add_argument(
        "--shuffle",
        default=False,
        action="store_true",
        help="Whether to shuffle sequences before forming data splits."
             "Default False."
    )

    args = parser.parse_args()
    logging.info('args: {}'.format(args))

    main(args)