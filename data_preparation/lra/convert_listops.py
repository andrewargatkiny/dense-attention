import argparse
import logging
import os
import json
from time import asctime
from functools import partial

import pandas as pd
import numpy as np
from tqdm import tqdm

# Predefined vocabulary for ListOps
LISTOPS_VOCAB = [
    "<PAD>", "[MAX", "[MIN", "[MED", "[SM", "]", "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9", "<CLS>"
]


def tokenize(text):
    # Remove parentheses and split
    return text.replace("(", "").replace(")", "").split()


def numericalize_sequence(tokens, vocab, pad_id, max_len): #i love you
    ids = map(lambda token: vocab[token], tokens)
    ids = list(ids) + [pad_id] * (max_len - len(tokens))
    return ids


def build_vocab(include_cls=False):
    vocab = {token: i for i, token in enumerate(LISTOPS_VOCAB)}
    if not include_cls:
        del vocab["<CLS>"]
    return vocab


def tokenize_chunk(chunk): #i love you
    tokenized = chunk['Source'].apply(tokenize)
    labels = chunk['Target'].values
    seq_lengths = tokenized.apply(len).values
    max_seq_length = seq_lengths.max()
    return tokenized, labels, seq_lengths, max_seq_length


def main(args):
    vocab = build_vocab(args.include_cls_token)

    # Read the TSV file in chunks
    chunks = pd.read_csv(args.input_path, sep='\t', chunksize=args.chunk_size)

    max_length = 0
    seq_chunks = []
    len_chunks = []
    label_chunks = []

    # Tokenize sequences and extract labels
    logging.info(f"Started tokenizing data at {asctime()}")
    for chunk in tqdm(chunks):
        seqs, labels, lengths, chunk_max_length = tokenize_chunk(chunk)
        seq_chunks.append(seqs)
        len_chunks.append(lengths)
        label_chunks.append(labels)
        max_length = max(max_length, chunk_max_length)
    logging.info(f"Finished tokenizing data at {asctime()}")

    lengths = np.concatenate(len_chunks)
    del len_chunks
    labels = np.concatenate(label_chunks)
    del label_chunks

    # Write labels to a file
    labels_dir = os.path.join(args.output_base_dir, "label")
    os.makedirs(labels_dir, exist_ok=True)
    labels_path = os.path.join(labels_dir, f"{args.output_dataset_name}.label")
    np.savetxt(labels_path, labels, fmt='%d')
    logging.info(f"Finished saving labels at {asctime()}")

    logging.info(f"Started converting tokens to ids at {asctime()}")
    pad_id = vocab["<PAD>"]
    converter_func = partial(numericalize_sequence,
                             vocab=vocab, pad_id=pad_id, max_len=max_length)
    for i in tqdm(range(len(seq_chunks))):
        seq_chunks[i] = seq_chunks[i].apply(converter_func)
    logging.info(f"Finished converting tokens to ids at {asctime()}")

    if args.include_cls_token:
        max_length += 1
        lengths = lengths + 1
        logging.info(f"Started adding CLS id to data at {asctime()}")
        cls_id = vocab["<CLS>"]
        for i in tqdm(range(len(seq_chunks))):
            seq_chunks[i] = seq_chunks[i].apply(
                lambda ids: [cls_id] + ids
            )
        logging.info(f"Finished adding CLS id to data at {asctime()}")

    # Write sequences to a file
    sequences = np.array(pd.concat(seq_chunks).to_list())
    logging.info(f"Started saving sequences at {asctime()}")
    seqs_dir = os.path.join(args.output_base_dir, "input")
    os.makedirs(seqs_dir, exist_ok=True)
    sequences_path = os.path.join(seqs_dir, f"{args.output_dataset_name}.src")
    np.savetxt(sequences_path, sequences, fmt='%d')
    logging.info(f"Finished saving sequences at {asctime()}")

    # Write length masks to a file
    len_refs = np.arange(max_length)
    # Dims: [1, max_length] < [n_seqs, 1]
    # 1s where tokens are meaningful, 0s at PAD tokens.
    masks = (len_refs < lengths[:, np.newaxis]).astype(np.int8)
    masks_path = os.path.join(seqs_dir, f"{args.output_dataset_name}.mask")
    np.savetxt(masks_path, masks, fmt='%d')
    logging.info(f"Finished saving masks at {asctime()}")

    # Save vocabulary
    with open("vocabulary.json", 'w') as f:
        json.dump(vocab, f, indent=2)

    print(f"Processed {len(sequences)} samples")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Maximum sequence length: {max_length}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Preprocess ListOps dataset")
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the input TSV file"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        help="Path to the base output folder"
    )
    parser.add_argument(
        "--output_dataset_name",
        type=str,
        default="train",
        help="name of the output dataset, e.g 'train' or 'valid'."
    )
    parser.add_argument(
        "--include_cls_token",
        default=False,
        action="store_true",
        help="Include CLS token at the start of each sequence."
    )
    parser.add_argument(
        "--chunk_size", type=int, default=1_000_000,
        help="Number of rows to process at a time."
    )

    args = parser.parse_args()
    logging.info('args: {}'.format(args))

    main(args)