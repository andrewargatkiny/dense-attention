from __future__ import absolute_import, division, print_function

import logging
import random

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class WikiMLMDataset(Dataset):
    """Turn raw Wikipedia text into tensors the model can train on"""

    def __init__(self, data_prefix: str, config: dict, args):
        cfg = config if isinstance(config, dict) else {}

        self.seq_length = int(
            cfg.get("seq_length", getattr(args, "max_seq_length", 64))
        )
        self.mask_prob = float(cfg.get("mask_prob", 0.15))
        self.num_samples = int(cfg.get("num_samples", 1000))
        split = cfg.get("split", "train")
        tokenizer_name = cfg.get("tokenizer", "bert-base-uncased")

        # ---- Load tokenizer ----
        from transformers import BertTokenizerFast
        logger.info(f"Loading tokenizer '{tokenizer_name}' ...")
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.mask_token_id = self.tokenizer.mask_token_id   # 103
        self.vocab_size = self.tokenizer.vocab_size          # 30522

        # ---- Load dataset ----
        from datasets import load_dataset
        logger.info(f"Loading wikitext-2-raw-v1 split='{split}' ...")
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # ---- Build samples ----
        self._samples = []
        self._prepare(raw)
        logger.info(
            f"WikiMLMDataset ready: {len(self._samples)} sequences "
            f"(seq_length={self.seq_length}, split={split})"
        )

    def _prepare(self, raw_dataset):
        """Tokenize all text, chunk into seq_length sequences, apply masking."""
        token_buffer = []

        for item in raw_dataset:
            text = item["text"].strip()
            if not text or text.startswith(" ="):   # skip wikitext headers
                continue

            ids = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(ids)

            # Slice complete chunks from the buffer
            while (len(token_buffer) >= self.seq_length
                   and len(self._samples) < self.num_samples):
                chunk = token_buffer[: self.seq_length]
                token_buffer = token_buffer[self.seq_length :]
                self._samples.append(self._make_sample(chunk))

            if len(self._samples) >= self.num_samples:
                break

    def _make_sample(self, token_ids: list) -> dict:
        """Apply MLM masking to a single chunk and return a sample dict."""
        L = self.seq_length
        input_ids = torch.tensor(token_ids[:L], dtype=torch.long)
        attention_mask = torch.ones(L, dtype=torch.long)
        token_type_ids = torch.zeros(L, dtype=torch.long)
        masked_lm_labels = torch.full((L,), fill_value=-1, dtype=torch.long)

        for pos in range(L):
            if random.random() < self.mask_prob:
                masked_lm_labels[pos] = input_ids[pos].item()
                r = random.random()
                if r < 0.8:
                    input_ids[pos] = self.mask_token_id          # [MASK] 80%
                elif r < 0.9:
                    input_ids[pos] = random.randint(5, self.vocab_size - 1)  # random 10%
                # else keep original 10%

        label = torch.tensor(0, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "masked_lm_labels": masked_lm_labels,
            "label": label,
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]