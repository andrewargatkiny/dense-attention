import argparse
import json
import logging
import os
from collections import Counter
from itertools import chain
from time import asctime

import numpy as np
from datasets import DatasetDict, load_dataset

def numericalize_sequence(tokens, vocab, pad_id, unk_id, max_len):
    ids = [vocab.get(token, unk_id) for token in tokens]
    ids = ids + [pad_id] * (max_len - len(ids))
    return ids

def main(args):
    # Load the data from TSV files
    data_files = {
        'train': os.path.join(args.data_dir, 'new_aan_pairs.train.tsv'),
        'valid': os.path.join(args.data_dir, 'new_aan_pairs.eval.tsv'),
        'test': os.path.join(args.data_dir, 'new_aan_pairs.test.tsv')
    }
    dataset = load_dataset(
        'csv',
        data_files=data_files,
        delimiter='\t',
        column_names=['label', 'input1_id', 'input2_id', 'text1', 'text2']
    )

    # Remove 'input1_id' and 'input2_id' columns
    dataset = dataset.remove_columns(['input1_id', 'input2_id'])

    # Tokenize 'text1' and 'text2' into 'tokens1' and 'tokens2'
    l_max = args.max_seq_len - int(args.include_cls_token)
    logging.info(f"Started tokenizing data at {asctime()}")

    def tokenize_example(example):
        tokens1 = list(example['text1'])[:l_max]
        tokens2 = list(example['text2'])[:l_max]
        return {'tokens1': tokens1, 'tokens2': tokens2}

    dataset = dataset.map(tokenize_example, remove_columns=['text1', 'text2'])

    # Build vocabulary from 'tokens1' and 'tokens2' in the training set
    logging.info(f"Started constructing vocab at {asctime()}")
    tokens_counts = Counter(chain(
        chain(*dataset['train']['tokens1']),
        chain(*dataset['train']['tokens2'])
    ))
    logging.info(f"Parsed unique tokens and their counts: {tokens_counts}")
    # Remove tokens with counts less than min_frequency
    tokens_counts = {k: v for k, v in tokens_counts.items() if v >= args.min_frequency}
    tokens_list = ["<PAD>"] + list(tokens_counts) + ["<UNK>"]
    if args.include_cls_token:
        logging.info(f"Started adding <CLS> tokens at {asctime()}")
        l_max += 1
        tokens_list += ["<CLS>"]
        def add_cls_func(example):
            example['tokens1'] = ["<CLS>"] + example['tokens1']
            example['tokens2'] = ["<CLS>"] + example['tokens2']
            return example
        dataset = dataset.map(add_cls_func)
    logging.info(f"Started building vocab at {asctime()}")
    vocab_ids = range(len(tokens_list))
    vocab = dict(zip(tokens_list, vocab_ids))

    logging.info(f"Started converting tokens to ids at {asctime()}")
    unk_id = vocab["<UNK>"]
    pad_id = vocab["<PAD>"]

    def numericalize_example(example):
        ids1 = numericalize_sequence(example['tokens1'], vocab, pad_id, unk_id, l_max)
        ids2 = numericalize_sequence(example['tokens2'], vocab, pad_id, unk_id, l_max)
        length1 = len(example['tokens1'])
        length2 = len(example['tokens2'])
        return {'ids1': ids1, 'ids2': ids2, 'length1': length1, 'length2': length2}

    dataset = dataset.map(numericalize_example, remove_columns=['tokens1', 'tokens2'])

    for subset in ['train', 'valid', 'test']:
        for ordinal in ['1', '2']:
            logging.info(f"Processing subset {subset} and {ordinal} of 2 inputs.")
            subdir = os.path.join(args.output_base_dir, f"inputs{ordinal}")

            # Labels are saved both times for compatibility with ordinary datasets
            logging.info(f"Started saving labels at {asctime()}")
            labels = np.array(dataset[subset]['label'], dtype=np.int8)
            labels_dir = os.path.join(subdir, 'label')
            os.makedirs(labels_dir, exist_ok=True)
            labels_path = os.path.join(labels_dir, f"{subset}.label")
            np.savetxt(labels_path, labels, fmt='%d')
            logging.info(f"Finished saving labels at {asctime()}")

            # Save sequences
            logging.info(f"Started saving sequences{ordinal} at {asctime()}")
            sequences = np.array(dataset[subset][f'ids{ordinal}'], dtype=np.int16)
            seqs_dir = os.path.join(subdir, 'input')
            os.makedirs(seqs_dir, exist_ok=True)
            sequences_path = os.path.join(seqs_dir, f"{subset}.src")
            np.savetxt(sequences_path, sequences, fmt='%d')
            logging.info(f"Finished saving sequences{ordinal} at {asctime()}")

            # Save masks
            logging.info(f"Started saving masks{ordinal} at {asctime()}")
            lengths = np.array(dataset[subset][f'length{ordinal}'])
            len_refs = np.arange(l_max)
            masks = (len_refs < lengths[:, np.newaxis]).astype(np.int8)
            masks_path = os.path.join(seqs_dir, f"{subset}.mask")
            np.savetxt(masks_path, masks, fmt='%d')
            logging.info(f"Finished saving masks{ordinal} at {asctime()}")

            print(f"Subset {subset}")
            print(f"Processed {len(labels)} samples")
            print(f"Maximum sequences{ordinal} length: "
                  f"{max(dataset[subset][f'length{ordinal}'])}")
            print(f"Values and counts of labels: "
                  f"{np.unique(labels, return_counts=True)}")

    # Save vocabulary
    with open(os.path.join(args.output_base_dir, 'vocabulary.json'), 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary size: {len(vocab)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Preprocess AAN dataset for LRA")

    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="Path to the data directory containing the AAN tsv files"
    )
    parser.add_argument(
        "--output_base_dir",
        default="../data/lra/aan",
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
        default=1,
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
