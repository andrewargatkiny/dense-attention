import copy
import os
import random
import time
from typing import Tuple

import h5py

import numpy as np

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


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


class LRADataset(Dataset):
    def __init__(self, base_dir, dataset_config, args=None):
        # Read sequences from file using numpy for fast reading
        print(time.ctime(), "Started loading data")
        sequences_file = os.path.join(base_dir, dataset_config["inputs"])
        sequences = np.loadtxt(sequences_file, dtype=np.int16)
        self.sequences = torch.tensor(sequences, dtype=torch.int)

        # Read labels from file using numpy for fast reading
        print(time.ctime(), "Started loading labels")
        labels_file = os.path.join(base_dir, dataset_config["labels"])
        labels = np.loadtxt(labels_file, dtype=np.int8)
        self.labels = torch.tensor(labels, dtype=torch.long)
        print(f"Number of samples is {len(self)}")

        # Ensure both sequences and labels have the same number of samples
        assert self.sequences.shape[0] == self.labels.shape[0], "Mismatch between sequences and labels"

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        # Get the sequence and label at the given index
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return dict(input_ids=sequence, label=label)

    def __add__(self, other: "LRADataset"):
        total = copy.deepcopy(self)
        total.sequences = torch.cat([self.sequences, other.sequences], dim=0)
        total.labels = torch.cat([self.labels, other.labels], dim=0)
        return total


class LRATextDataset(LRADataset):
    def __init__(self, base_dir, dataset_config, args=None):
        super(LRATextDataset, self).__init__(base_dir, dataset_config, args)
        print(time.ctime(), "Started loading masks")
        masks_file = os.path.join(base_dir, dataset_config["masks"])
        masks = np.loadtxt(masks_file, dtype=np.int8)
        self.masks = torch.tensor(masks, dtype=torch.int)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        mask = self.masks[idx]
        return dict(input_ids=sequence, label=label, attention_mask=mask)

    def __add__(self, other: "LRATextDataset"):
        total = super().__add__(other)
        total.masks = torch.cat([self.masks, other.masks], dim=0)

class AANDataset(Dataset):
    def __init__(self, base_dir, dataset_config, args=None):
        super(AANDataset, self).__init__()
        base_dir1 = os.path.join(base_dir, "inputs1")
        base_dir2 = os.path.join(base_dir, "inputs2")
        self.dataset1 = LRATextDataset(base_dir1, dataset_config["inputs1"], args)
        self.dataset2 = LRATextDataset(base_dir2, dataset_config["inputs2"], args)

    def __len__(self):
        return self.dataset1.sequences.shape[0]

    def __getitem__(self, idx):
        sequence1 = self.dataset1.sequences[idx]
        mask1 = self.dataset1.masks[idx]
        sequence2 = self.dataset2.sequences[idx]
        mask2 = self.dataset1.masks[idx]
        label = self.dataset1.labels[idx]
        # The first pair of (input_ids, attention_mask) should be written without
        # indices for compatibility with train/eval code.
        return dict(input_ids=sequence1, attention_mask=mask1,
                    input_ids2=sequence2, attention_mask2=mask2,
                    label=label)


class DatasetForMLM(Dataset):
    def __init__(self, base_dir, dataset_config, args=None):
        self.core_dataset = self._init_core_dataset(base_dir, dataset_config,
                                                    args)
        self.max_seq_len = self.core_dataset.sequences.shape[-1]
        self.seed = args.seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.lm_prob = args.lm_prob
        self.variable_mask_rate = args.variable_mask_rate
        self.mlm_use_rtc_task = args.mlm_use_rtc_task
        # It's assumed that max_token_id equals vocab length.
        self.mask_token_id = args.mask_token_id

        self.masking_func = self._get_item_mlm
        if self.mlm_use_rtc_task:
            self.masking_func = self._get_item_rtc
        self.get_lm_prob = lambda: self.lm_prob
        if self.variable_mask_rate:
            self.get_lm_prob = lambda: random.random() * self.lm_prob

    def __len__(self):
        return len(self.core_dataset)

    def __getitem__(self, idx):
        sequence, label, mask = self._get_core_item(idx)
        lm_prob = self.get_lm_prob()
        # Make indices for masking
        lm_probs = torch.rand(size=(1, self.max_seq_len),
                              device=sequence.device)
        indices = (lm_probs <= lm_prob)
        indices = (indices * mask).nonzero() # No masking for PAD tokens
        return self.masking_func(sequence, label, mask, indices)

    def _init_core_dataset(self, base_dir, dataset_config, args=None):
        return LRADataset(base_dir, dataset_config, args)

    def _get_core_item(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.core_dataset[idx]
        attention_mask = torch.ones_like(sample["input_ids"])
        return sample["input_ids"], sample["label"], attention_mask

    def _get_item_mlm(self, sequence, label, mask, indices):
        """Process a sequence with respect to Masked Language Modeling task"""
        masked_sequence = sequence.detach().clone()
        masked_sequence[indices] = self.mask_token_id
        mlm_labels = torch.ones_like(sequence, dtype=torch.long) * -1
        mlm_labels[indices] = sequence.long()[indices]
        return dict(input_ids=masked_sequence, label=label,
                    attention_mask=mask, masked_lm_labels=mlm_labels)

    def _get_item_rtc(self, sequence, label, mask, indices):
        """Process a sequence with respect to Replaced Token Correction task"""

        masked_sequence = sequence.detach().clone()
        masked_sequence[indices] = torch.randint_like(
            indices, self.mask_token_id, dtype=masked_sequence.dtype
        )
        # Let all non PAD tokens serve as labels
        mask_bool = mask.bool()
        mlm_labels = torch.ones_like(sequence, dtype=torch.long) * -1
        mlm_labels[mask_bool] = sequence.long()[mask_bool]
        return dict(input_ids=masked_sequence, label=label,
                    attention_mask=mask, masked_lm_labels=mlm_labels)

class TextDatasetForMLM(DatasetForMLM):
    def _init_core_dataset(self, base_dir, dataset_config, args=None):
        return LRATextDataset(base_dir, dataset_config, args)

    def _get_core_item(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.core_dataset[idx]
        return sample["input_ids"], sample["label"], sample["attention_mask"]

class AANDatasetForMLM(TextDatasetForMLM):
    def _init_core_dataset(self, base_dir, dataset_config, args=None):
        """Simply combine 2 individual parts into 1 for MLM."""
        base_dir1 = os.path.join(base_dir, "inputs1")
        base_dir2 = os.path.join(base_dir, "inputs2")
        dataset1 = LRATextDataset(base_dir1, dataset_config["inputs1"], args)
        dataset2 = LRATextDataset(base_dir2, dataset_config["inputs2"], args)
        return dataset1 + dataset2
