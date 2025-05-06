"""Datasets for Language Models pretraining"""

import gc
import json
import os
import random
import time
from itertools import chain

import h5py
import nltk
import pandas as pd
import torch
import numpy as np
from typing import List, Optional
from torch.utils.data import Dataset
from collections import deque, namedtuple

from tqdm import tqdm
from transformers import BertTokenizerFast, AutoTokenizer

def BertPretrainingDatasetFactory(base_dir, dataset_config, args=None):
    input_file = os.path.join(base_dir, dataset_config["input_file"])
    if os.path.splitext(input_file)[-1] == ".hdf5":
        return BertPretrainingDataset(base_dir, dataset_config, args)
    else:
        return BertPretrainingOnlineDataset(base_dir, dataset_config, args)


class BertPretrainingDataset(Dataset):
    """Dataset for MLM and NSP with pre-made static masks """
    def __init__(self, base_dir, dataset_config, args=None):
        self.input_file = os.path.join(base_dir, dataset_config["input_file"])

        f = h5py.File(self.input_file, "r")
        keys = [
            'input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions',
            'masked_lm_ids', 'next_sentence_labels'
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [
            input_ids, input_mask, segment_ids, masked_lm_positions,
            masked_lm_ids, next_sentence_labels
        ] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else
            torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            for indice, input in enumerate(self.inputs)
        ]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        len_masked_ids = masked_lm_ids.shape[-1]
        # Token at position 0 im any sentence is [CLS] which is never masked
        # by dataset construction procedure, so all 0 indices in masked lm
        # positions are [PAD] tokens.
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            len_masked_ids = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:len_masked_ids]] = (
            masked_lm_ids)[:len_masked_ids]

        return dict(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            label=next_sentence_labels,
            masked_lm_labels=masked_lm_labels
        )


class Sentence:
    def __init__(self, tokens):
        self._tokens = tokens

    def __repr__(self):
        return f"Sentence(_tokens={self._tokens})"

    def __len__(self):
        return len(self._tokens)


class Document:
    def __init__(self, doc_id, sentences, dataset_id=None):
        self._id = doc_id
        self._sentences = sentences  # tuple of Sentence
        self.dataset_id = dataset_id

    def __repr__(self):
        return f"Document(_id={self._id}, _sentences={self._sentences})"

    def __len__(self):
        return len(self._sentences)

    def __getitem__(self, idx):
        return self._sentences[idx]


MaskedLmInstance = namedtuple("MaskedLmInstance", ["index", "label"])

def _truncate_seq_pair_old(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        if len(tokens_a) > len(tokens_b):
            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if random.random() < 0.5:
                del tokens_a[0]
            else:
                tokens_a.pop()
        else:
            if random.random() < 0.5:
                del tokens_b[0]
            else:
                tokens_b.pop()


class BertPretrainingOnlineDataset(Dataset):
    """
    Dataset for Bert MLM + NSP pretraining which construct pairs of
    sequences, and masks in online fashion.
    """

    def __init__(self, base_dir, dataset_config, args=None):
        super().__init__()
        print(time.ctime(), f"Started initializing {self.__class__.__name__}")

        self.seed = dataset_config.get("seed", None)
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        file_path = os.path.join(base_dir, dataset_config["input_file"])

        tokenizer_name = dataset_config.get("tokenizer_name", "bert-large-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.vocab_size = self.tokenizer.vocab_size

        self.max_seq_len = dataset_config.get("max_seq_length", 512)
        self.short_seq_prob = dataset_config.get("short_seq_prob", 0.1)

        self.masked_lm_ratio = dataset_config.get("masked_lm_ratio", 0.15)
        self.p_mask_token = dataset_config.get("p_mask_token", 0.8)
        self.true_mask_thr = self.masked_lm_ratio * self.p_mask_token
        # p_mask_token + (1 - p_mask_token / 2) = 0.5 + p_mask_token / 2
        self.random_mask_thr = self.masked_lm_ratio * (0.5 + self.p_mask_token / 2)
        self.same_dataset_pairs = dataset_config.get("same_dataset_pairs", False)
        self.total_samples = dataset_config.get("total_samples", 2**19)

        print(time.ctime(), f"Started Building documents from {file_path}")
        self.documents = self._build_documents(file_path, dataset_id=0)
        random.shuffle(self.documents)


        self.vocab_words = list(self.tokenizer.get_vocab().keys())

        print(time.ctime(), f"Started generating MLM + NLP sentence-pairs.")
        # Generate all pairs from the documents, possibly multiple times,
        # until we have at least self.total_samples.
        all_pairs = []
        n_pairs = 0
        while n_pairs < self.total_samples:
            for doc_idx in tqdm(range(len(self.documents))):
                pairs = self._create_pairs_from_document(self.documents, doc_idx)
                all_pairs.extend(pairs)
                n_pairs += len(pairs)
                if n_pairs >= self.total_samples:
                    break
            print(time.ctime(), f"Created {n_pairs} pairs")

        # Truncate to exactly self.total_samples
        all_pairs = all_pairs[:self.total_samples]

        # Building in-memory tensor dataset
        print(time.ctime(), f"Started Building in-memory tensor dataset.")
        self.input_ids = torch.stack([p["input_ids"] for p in all_pairs], dim=0)
        self.input_mask = torch.stack([p["attention_mask"] for p in all_pairs], dim=0)
        self.segment_ids = torch.stack([p["token_type_ids"] for p in all_pairs], dim=0)
        self.next_sentence_labels = torch.stack([p["label"] for p in all_pairs])
        self.masked_lm_labels = torch.stack([p["masked_lm_labels"] for p in all_pairs], dim=0)
        print(time.ctime(), f"Finished initializing {self.__class__.__name__}")


    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.input_mask[idx],
            "token_type_ids": self.segment_ids[idx],
            "label": self.next_sentence_labels[idx],
            "masked_lm_labels": self.masked_lm_labels[idx],
        }


    def _build_documents(self, file_path: str, dataset_id: int) -> List[Document]:
        df = pd.read_json(file_path, lines=True)
        print(time.ctime(), f"Started basic preprocessing")
        df['text'] = df['text'].str.replace('\n', ' ')

        # Convert to list of lists of sentences and calculate boundaries.
        print(time.ctime(), f"Started sentence splitting")
        df['text'] = df['text'].apply(nltk.tokenize.sent_tokenize)
        doc_boundaries = df['text'].apply(len).tolist()
        all_sentences = df['text'].tolist()
        all_ids = df['url'].tolist()

        # Flatten sentences and tokenize
        flat_sentences = list(chain.from_iterable(all_sentences))
        print(time.ctime(), f"Started tokenization")
        all_tokens = self.tokenizer(
            flat_sentences,
            add_special_tokens=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )["input_ids"]
        # Reconstruct documents using doc_boundaries
        documents = []
        start_idx = 0
        print(time.ctime(), f"Started reconstructing documents")
        for doc_id, num_sents in tqdm(zip(all_ids, doc_boundaries)):
            end_idx = start_idx + num_sents
            doc_tokens = all_tokens[start_idx:end_idx]

            # Filter out empty sentences and create Sentence objects
            sents = [Sentence(tuple(tokens)) for tokens in doc_tokens if tokens]
            documents.append(Document(
                doc_id=doc_id,
                sentences=tuple(sents),
                dataset_id=dataset_id
            ))

            start_idx = end_idx
        return documents


    def _create_pairs_from_document(
        self,
        all_documents: List[Document],
        document_index: int,
    ) -> List[dict]:
        """Create pairs for a single document."""

        document = all_documents[document_index]

        # Gather indices for same-dataset pairs if needed.
        # (Only done if same_dataset_pairs is True and if doc.dataset_id matches.)
        same_ds_indices = []
        if self.same_dataset_pairs:
            dataset_id = document.dataset_id
            for idx, doc in enumerate(all_documents):
                if doc.dataset_id == dataset_id and idx != document_index:
                    same_ds_indices.append(idx)

        # Account for [CLS], [SEP], [SEP]
        max_num_tokens = self.max_seq_len - 3

        # We *usually* want to fill up the entire sequence (max_num_tokens),
        # but we *sometimes* (short_seq_prob) want to use shorter sequences to
        # minimize mismatch between pretraining and fine-tuning.
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)

        instances = []
        sentences_a = []
        current_length = 0
        i = 0
        while i < len(document):
            sentence = document[i]
            sentences_a.append(sentence)
            current_length += len(sentence)

            # If we're at the end of the doc or we've reached target_seq_length, produce an instance
            if (i == len(document) - 1 or current_length >= target_seq_length) and sentences_a:
                # `a_end` is how many sentences from `sentences_a` go into the A (first) sentence.
                a_end = 1
                if len(sentences_a) >= 2:
                    a_end = random.randint(1, len(sentences_a) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(sentences_a[j]._tokens)

                tokens_b = []
                # Random next half of the time
                if (len(all_documents) > 1 and
                    (len(sentences_a) == 1 or random.random() < 0.5)):
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # Pick a random index for another document
                    random_document_index = random.randint(0, len(all_documents) - 2)
                    if random_document_index >= document_index:
                        random_document_index += 1
                    # For same_dataset_pairs logic:
                    if self.same_dataset_pairs and len(same_ds_indices) > 0:
                        random_document_index = random.choice(same_ds_indices)


                    random_document = all_documents[random_document_index]
                    random_start = random.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j]._tokens)
                        if len(tokens_b) >= target_b_length:
                            break

                    # We didn't actually use these sentences so we "put them back" so
                    # they don't go to waste.
                    num_unused_sentences = len(sentences_a) - a_end
                    i -= num_unused_sentences

                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(sentences_a)):
                        tokens_b.extend(sentences_a[j]._tokens)

                # We now truncate to max_num_tokens
                tokens_a, tokens_b = self._truncate_seq_pair(
                    tokens_a, tokens_b, max_num_tokens
                )
                len_a = len(tokens_a)
                len_b = len(tokens_b)
                #assert len(tokens_a) >= 1, "Tokens A must have at least 1 token"
                #assert len(tokens_b) >= 1, "Tokens B must have at least 1 token"
                filled_length = len_a + len_b + 3
                input_ids = ([self.cls_token_id] + tokens_a +
                             [self.sep_token_id] + tokens_b +
                             [self.sep_token_id])
                input_mask = [1] * filled_length
                segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
                next_sentence_label = 1 if is_random_next else 0

                # Efficient masking procedure
                mask_probs = np.random.rand(filled_length)
                # "[CLS]" and "[SEP]" tokens shouldn't be masked
                mask_probs[[0, len_a + 1, -1]] = 1.
                random_ids = np.random.randint(self.vocab_size,
                                               size=filled_length)
                masked_inputs = np.where(
                    mask_probs < self.true_mask_thr,
                    self.mask_token_id,
                    np.where(mask_probs < self.random_mask_thr,
                             random_ids, input_ids)
                )
                masked_lm_positions = (mask_probs < self.masked_lm_ratio
                                       ).nonzero()[0]

                # Build masked_lm_labels array
                mlm_labels = np.full(shape=self.max_seq_len, fill_value=-1)
                mlm_labels[masked_lm_positions] = np.array(input_ids)[masked_lm_positions]

                # Pad up to self.max_seq_len
                if filled_length < self.max_seq_len:
                    to_pad = self.max_seq_len - filled_length
                    masked_inputs = masked_inputs.tolist() + [0] * to_pad
                    input_mask  += [0] * to_pad
                    segment_ids += [0] * to_pad

                # Append final instance
                instance = {
                    "input_ids": torch.tensor(masked_inputs, dtype=torch.long),
                    "attention_mask": torch.tensor(input_mask, dtype=torch.long),
                    "token_type_ids": torch.tensor(segment_ids, dtype=torch.long),
                    "label": torch.tensor(next_sentence_label, dtype=torch.long),
                    "masked_lm_labels": torch.tensor(mlm_labels, dtype=torch.long),
                }
                instances.append(instance)

                # Reset for next chunk
                sentences_a = []
                current_length = 0

            i += 1

        return instances

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
        """Truncates a pair of sequences to a maximum sequence length."""
        len_a = len(tokens_a)
        len_b = len(tokens_b)
        total_length = len(tokens_a) + len(tokens_b)
        truncation = total_length - max_num_tokens
        if truncation <= 0: return tokens_a, tokens_b
        # Amounts of tokens to remove from both sequences.
        n_trunk_a = 0
        n_trunk_b = 0
        # Surplus indicates at least how much will be actually truncated from
        # a sequence with more tokens.
        surplus = abs(len_b - len_a)
        b_greater = len_b > len_a
        if truncation <= surplus:
            surplus = truncation
        if b_greater:
            n_trunk_b = surplus
        else:
            n_trunk_a = surplus
        if surplus < truncation:
            remain_trunk = truncation - surplus
            # Trunkate equal amount from both sequences.
            to_trunk = remain_trunk // 2
            n_trunk_a += to_trunk
            n_trunk_b += to_trunk
            # If remaining truncation amount is odd, truncate an additional token
            # from the tokens_b (consistent with original implementation).
            n_trunk_b += remain_trunk % 2

        # With equal chances tokens should be truncated from both sides of
        # a sequence.
        n_trunk_a_left = random.randint(0, n_trunk_a)
        n_trunk_a_right = n_trunk_a - n_trunk_a_left
        n_trunk_b_left = random.randint(0, n_trunk_b)
        n_trunk_b_right = n_trunk_b - n_trunk_b_left
        tokens_a = tokens_a[n_trunk_a_left:len_a - n_trunk_a_right]
        tokens_b = tokens_b[n_trunk_b_left:len_b - n_trunk_b_right]

        return tokens_a, tokens_b

class BertOnlyMLMDataset(Dataset):
    """
    Dataset for Bert MLM pretraining which constructs sequences, and masks
    in online fashion.
    """

    def __init__(self, base_dir, dataset_config, args=None):
        super().__init__()
        print(time.ctime(), f"Started initializing {self.__class__.__name__}")

        self.seed = dataset_config.get("seed", None)
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        file_path = os.path.join(base_dir, dataset_config["input_file"])

        tokenizer_name = dataset_config.get("tokenizer_name", "bert-large-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.vocab_size = self.tokenizer.vocab_size

        self.max_seq_len = dataset_config.get("max_seq_length", 512)
        self.short_seq_prob = dataset_config.get("short_seq_prob", 0.1)

        self.masked_lm_ratio = dataset_config.get("masked_lm_ratio", 0.15)
        self.p_mask_token = dataset_config.get("p_mask_token", 0.8)
        self.true_mask_thr = self.masked_lm_ratio * self.p_mask_token
        # p_mask_token + (1 - p_mask_token / 2) = 0.5 + p_mask_token / 2
        self.random_mask_thr = self.masked_lm_ratio * (0.5 + self.p_mask_token / 2)
        self.same_dataset_pairs = dataset_config.get("same_dataset_pairs", False)
        self.total_samples = dataset_config.get("total_samples", 2**19)

        print(time.ctime(), f"Started Building documents from {file_path}")
        df = pd.read_json(file_path, lines=True)
        df.sample(frac=1, replace=False)
        print(time.ctime(), f"Started basic preprocessing")
        # df['text'] = df['text'].str.replace('\n', ' ')
        df['text'] = df['text'] + self.tokenizer.sep_token

        all_sentences = df['text'].tolist()
        del df

        print(f"First sequence: {all_sentences[0]}")
        print(time.ctime(), f"Started tokenization")
        all_seqs = []
        # It's better to feed sequences into tokenizer in chunks rather than
        # at once, because at the latter case it consumes all the RAM and
        # crashes the system. 10_000 is just a magic number chosen arbitrarily.
        chunk_size = 10_000
        for i in tqdm(range(0, len(all_sentences), chunk_size)):
            all_seqs.extend(
                self.tokenizer(
                    all_sentences[i: i + chunk_size],
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_overflowing_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False
                )["input_ids"]
            )
        all_ids = []
        del self.tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                       use_fast = True)
        for seq in all_seqs:
            all_ids.extend(seq)
        del all_sentences, all_seqs
        total_len = self.total_samples * (self.max_seq_len - 1)
        assert len(all_ids) >= total_len, \
            (f"There is not enough tokens in the dataset. Present: "
             f"{len(all_ids)}, required: {total_len}.")
        print(f"Initial number of processed sequences is "
              f"{len(all_ids) // (self.max_seq_len - 1)}"
              f" before trimming to {self.total_samples}.")
        print(time.ctime(), f"Started constructing input_ids and masks.")
        all_ids = all_ids[:total_len]
        gc.collect()
        input_ids = (torch.tensor(all_ids, dtype=torch.int32)
                          .view(-1, self.max_seq_len - 1))
        del all_ids
        cls_tokens = torch.full((input_ids.shape[0], 1),
                                fill_value=self.cls_token_id,
                                dtype=torch.int32)
        input_ids = torch.cat([cls_tokens, input_ids], dim=-1)
        mask_probs = torch.rand_like(input_ids, dtype=torch.float)
        random_ids = torch.randint_like(input_ids, self.vocab_size)
        masked_inputs = torch.where(
            mask_probs < self.true_mask_thr,
            self.mask_token_id,
            torch.where(mask_probs < self.random_mask_thr,
                        random_ids, input_ids)
        )
        mlm_labels = torch.where(
            mask_probs < self.masked_lm_ratio,
            input_ids,
            -1
        ).to(torch.long)

        self.input_ids = masked_inputs
        #self.input_mask = torch.stack([p["attention_mask"] for p in all_pairs], dim=0)
        #self.segment_ids = torch.stack([p["token_type_ids"] for p in all_pairs], dim=0)
        #self.next_sentence_labels = torch.stack([p["label"] for p in all_pairs])
        self.masked_lm_labels = mlm_labels
        print(time.ctime(), f"Finished initializing {self.__class__.__name__}")


    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            #"attention_mask": self.input_mask[idx],
            #"token_type_ids": self.segment_ids[idx],
            #"label": self.next_sentence_labels[idx],
            "masked_lm_labels": self.masked_lm_labels[idx],
        }


class GPTPretrainingPaddedDataset(Dataset):
    """
    Dataset for GPT pretraining which constructs sequences and masks in online
    fashion. It holds at maximum one text in one sample and adds padding tokens
     if the text length is less than max_seq_length. An exception will be
     raised if number of produced samples is less than `total_samples`.
    """
    def __init__(self, base_dir, dataset_config, args=None):
        super().__init__()
        print(time.ctime(), f"Started initializing {self.__class__.__name__}")

        self.seed = dataset_config.get("seed", None)
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        file_path = os.path.join(base_dir, dataset_config["input_file"])

        tokenizer_name = dataset_config.get("tokenizer_name",
                                            "openai-community/gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                       use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vocab_size = self.tokenizer.vocab_size

        self.max_seq_len = dataset_config.get("max_seq_length", 512)
        self.short_seq_prob = dataset_config.get("short_seq_prob", 0.1)

        self.masked_lm_ratio = dataset_config.get("masked_lm_ratio", 0.15)
        self.p_mask_token = dataset_config.get("p_mask_token", 0.8)
        self.true_mask_thr = self.masked_lm_ratio * self.p_mask_token
        # p_mask_token + (1 - p_mask_token / 2) = 0.5 + p_mask_token / 2
        self.random_mask_thr = self.masked_lm_ratio * (0.5 + self.p_mask_token / 2)
        self.same_dataset_pairs = dataset_config.get("same_dataset_pairs", False)
        self.total_samples = dataset_config.get("total_samples", 2**19)

        base_name, ext = os.path.splitext(file_path)
        if ext == ".hdf5":
            print(time.ctime(), f"Started loading data from hdf5 file "
                                f"{file_path}")
            with h5py.File(file_path, 'r') as f:
                self.input_ids = torch.tensor(f['input_ids'][:],
                                              dtype=torch.int32)
                self.masked_lm_labels = torch.tensor(f['masked_lm_labels'][:],
                                                     dtype=torch.long)
            print(time.ctime(), f"Finished initializing "
                                f"{self.__class__.__name__}")
            return

        print(time.ctime(), f"Started Building documents from {file_path}")
        df = pd.read_json(file_path, lines=True)
        print(time.ctime(), f"Started basic preprocessing")
        df['text'] = df['text'].str.replace('\n', ' ')
        all_sentences = df['text'].tolist()
        del df
        assert len(all_sentences) >= self.total_samples, \
            (f"There is not enough samples in the dataset. Present: "
             f"{len(all_sentences)}, required: {self.total_samples}.")
        all_sentences = all_sentences[:self.total_samples]
        print(f"First sequence: {all_sentences[0]}")
        print(time.ctime(), f"Started tokenization")
        all_tokens = self.tokenizer(
            all_sentences,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len + 1,
            return_attention_mask=True,
            return_tensors='pt',
            return_token_type_ids=False
        )
        print(time.ctime(), f"Started Building in-memory tensor dataset.")
        self.input_ids: torch.Tensor = all_tokens["input_ids"][:, :-1]
        self.masked_lm_labels: torch.Tensor = all_tokens["input_ids"][:, 1:]
        self.input_mask = all_tokens["attention_mask"][:, :-1]
        self.masked_lm_labels = self.masked_lm_labels.where(
            self.input_ids != self.tokenizer.pad_token_id, -1
        )
        print(f"First sequence's input_ids: {self.input_ids[0]}, "
              f"masked_lm_labels {self.masked_lm_labels[0]}, "
              f"self.input_mask{self.input_mask[0]}")
        print(f"Lengths of inputs are {self.input_ids.shape[-1]}")
        print(time.ctime(), f"Finished initializing {self.__class__.__name__}")


    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.input_mask[idx],
            "masked_lm_labels": self.masked_lm_labels[idx],
        }

class GPTPretrainingDataset(Dataset):
    """
    Dataset for GPT pretraining which constructs sequences and masks in online
    fashion. It can hold several texts in one sample separated by `eos` token.
    If the text length is less than max_seq_length. An exception will be
    raised if number of produced samples is less than `total_samples`.
    """
    def __init__(self, base_dir, dataset_config, args=None):
        super().__init__()
        print(time.ctime(), f"Started initializing {self.__class__.__name__}")

        self.seed = dataset_config.get("seed", None)
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        file_path = os.path.join(base_dir, dataset_config["input_file"])

        tokenizer_name = dataset_config.get("tokenizer_name",
                                            "openai-community/gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                       use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size

        self.max_seq_len = dataset_config.get("max_seq_length", 2048)
        self.total_samples = dataset_config.get("total_samples", 2**19)

        base_name, ext = os.path.splitext(file_path)
        if ext == ".hdf5":
            print(time.ctime(), f"Started loading data from hdf5 file "
                                f"{file_path}")
            with h5py.File(file_path, 'r') as f:
                self.input_ids = torch.tensor(f['input_ids'][:],
                                              dtype=torch.int32)
                self.masked_lm_labels = torch.tensor(f['masked_lm_labels'][:],
                                                     dtype=torch.long)
            print(time.ctime(), f"Finished initializing "
                                f"{self.__class__.__name__}")
            return

        print(time.ctime(), f"Started Building documents from {file_path}")
        df = pd.read_json(file_path, lines=True)
        print(time.ctime(), f"Started basic preprocessing")
        df['text'] = df['text'].str.strip() + self.tokenizer.eos_token #replace('\n', ' ')
        all_sentences = df['text'].tolist()
        del df

        print(f"First sequence: {all_sentences[0]}")
        print(time.ctime(), f"Started tokenization")
        all_seqs = []
        # It's better to feed sequences into tokenizer in chunks rather than
        # at once, because at the latter case it consumes all the RAM and
        # crashes the system. 10_000 is just a magic number chosen arbitrarily.
        chunk_size = 10_000
        for i in tqdm(range(0, len(all_sentences), chunk_size)):
            all_seqs.extend(
                self.tokenizer(
                    all_sentences[i: i + chunk_size],
                    add_special_tokens=True,
                    padding=False,
                    truncation=False,
                    return_overflowing_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False
                )["input_ids"]
            )
        all_ids = []
        del self.tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                       use_fast = True)
        for seq in all_seqs:
            all_ids.extend(seq)
        del all_sentences, all_seqs
        total_len = self.total_samples * self.max_seq_len
        assert len(all_ids) >= total_len, \
            (f"There is not enough tokens in the dataset. Present: "
             f"{len(all_ids)}, required: {total_len}.")
        print(f"Initial number of processed sequences is "
              f"{len(all_ids) // self.max_seq_len}"
              f" before trimming to {self.total_samples}.")
        print(time.ctime(), f"Started Building in-memory tensor dataset.")
        all_ids = all_ids[:total_len]
        gc.collect()
        self.input_ids = (torch.tensor(all_ids, dtype=torch.int32)
                          .view(-1, self.max_seq_len))
        del all_ids
        #self.input_mask = torch.ones_like(self.input_ids)
        self.masked_lm_labels = torch.roll(self.input_ids, -1).to(torch.long)
        self.masked_lm_labels[-1, -1] = -1
        print(f"First sequence's input_ids: {self.input_ids[0]}, "
              f"masked_lm_labels {self.masked_lm_labels[0]}")
        print(f"Lengths of inputs are {self.input_ids.shape[-1]}")
        print(time.ctime(), f"Finished initializing {self.__class__.__name__}")

        if dataset_config.get("save", False):
            with h5py.File(f'{base_name}.hdf5', 'w') as f:
                print(time.ctime(), f"Started saving to disk")
                f.create_dataset('input_ids', data=self.input_ids.numpy(),
                                 dtype='i4', compression='gzip')
                f.create_dataset('masked_lm_labels',
                                 data=self.masked_lm_labels.numpy(),
                                 dtype='i4', compression='gzip')
                print(time.ctime(), f"Finished saving to disk")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            #"attention_mask": self.input_mask[idx],
            "masked_lm_labels": self.masked_lm_labels[idx],
        }