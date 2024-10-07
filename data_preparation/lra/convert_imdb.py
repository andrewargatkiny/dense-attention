import argparse
import json
import logging
import os
from collections import Counter
from itertools import chain
from time import asctime

import numpy as np
from datasets import DatasetDict, load_dataset

def numericalize_sequence(tokens, vocab, pad_id, unk_id, max_len): #i love you
    ids = map(lambda token: vocab.get(token, unk_id), tokens)
    ids = list(ids) + [pad_id] * (max_len - len(tokens))
    return ids


def main(args):
    dataset = load_dataset("imdb")
    dataset = DatasetDict(train=dataset["train"], test=dataset["test"])

    l_max = args.max_seq_len - int(args.include_cls_token)
    logging.info(f"Started tokenizing data at {asctime()}")
    dataset = dataset.map(
        lambda example: {"tokens": list(example["text"])[:l_max]},
        remove_columns=["text"],
    )
    logging.info(f"Started constructing vocab at {asctime()}")
    tokens_counts = Counter(chain(*dataset["train"]["tokens"]))
    logging.info(f"Parsed unique tokens and their counts: {tokens_counts}")
    for key in list(tokens_counts):
        if tokens_counts[key] < args.min_frequency:
            del tokens_counts[key]
    tokens_list = ["<PAD>"] + list(tokens_counts) + ["<UNK>"]
    if args.include_cls_token:
        logging.info(f"Started adding <CLS> tokens  at {asctime()}")
        l_max += 1
        tokens_list += ["<CLS>"]
        add_cls_func = lambda example: {"tokens": ["<CLS>"] + example["tokens"]}
        dataset = dataset.map(add_cls_func)
    logging.info(f"Started building vocab at {asctime()}")
    vocab_ids = range(len(tokens_list))
    vocab = dict(zip(tokens_list, vocab_ids))

    logging.info(f"Started converting tokens to ids at {asctime()}")
    unk_id = vocab["<UNK>"]
    pad_id = vocab["<PAD>"]
    numericalize = lambda example: {
        "ids": numericalize_sequence(example["tokens"], vocab, pad_id, unk_id, l_max),
        "length": len(example["tokens"])
    }
    dataset = dataset.map(numericalize, remove_columns=["tokens"])

    for subset in ["train", "test"]:
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

        # Write length masks to a file
        logging.info(f"Started saving masks at {asctime()}")
        lengths = np.array(dataset[subset]["length"])
        len_refs = np.arange(l_max)
        # Dims: [1, max_length] < [n_seqs, 1]
        # 1s where tokens are meaningful, 0s at PAD tokens.
        masks = (len_refs < lengths[:, np.newaxis]).astype(np.int8)
        masks_path = os.path.join(seqs_dir, f"{subset}.mask")
        np.savetxt(masks_path, masks, fmt='%d')
        logging.info(f"Finished saving masks at {asctime()}")

        print(f"Subset {subset}")
        print(f"Processed {len(sequences)} samples")
        print(f"Maximum sequence length: {max(dataset[subset]['length'])}")
        print(f"Values and counts of labels: "
              f"{np.unique(labels, return_counts=True)}")


    # Save vocabulary
    with open(os.path.join(args.output_base_dir, "vocabulary.json"), 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary size: {len(vocab)}")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Preprocess IMDB dataset")

    parser.add_argument(
        "--output_base_dir",
        default="../data/lra/text_classification",
        type=str,
        help="Path to the base output folder"
    )
    parser.add_argument(
        "--include_cls_token",
        default=False,
        action="store_true",
        help="Include CLS token at the start of each sequence."
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=15,
        help="Minimum number of times a symbol should be detected in train "
             "set for getting added into vocabulary."
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=4000,
        help="Maximum sequence length to truncate texts to."
    )

    args = parser.parse_args()
    logging.info('args: {}'.format(args))

    main(args)