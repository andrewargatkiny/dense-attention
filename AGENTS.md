# DenseAttention Project Reference

## What This Repository Does
This codebase is a research and experimentation framework for **DenseAttention** and **DANet** (Dense Attention Network), 
various Transformer, Linear-Attention-based and HuggingFace model variants. It currently supports:
- BERT-style MLM/NSP pretraining for language and sequence modeling
- GPT-style causal pretraining
- LRA tasks (classification)
- GLUE fine-tuning (classification/ regression)
- Speed/throughput evaluation

Primary entrypoint: `deepspeed_train.py`. Legacy code in `old_code/` and `src/legacy_layers.py` is intentionally out of scope and should not be consulted.

## End-to-End Execution Flow
1. `configs/**/ds_train_*.sh` sets env vars, config paths, resume args, and launches DeepSpeed.
2. `deepspeed_train.py` loads config(s), applies CLI overrides (`--override cf.* / ds.*`), resolves task via `utils/tasks.py`, then initializes distributed + model + optimizer through DeepSpeed.
3. Training loop handles data loading, logging (ClearML/TensorBoard/W&B), validation, and checkpoint lifecycle.
4. Task abstraction chooses dataset class, model class, and eval function per `task_type`.

## Core Modeling Stack
- DANet stack: `src/modeling.py`, `src/danet_layers.py`, `src/dense_attention.py`
- Transformer and other architectures: `src/other_models/modeling.py`, `src/other_models/attention_kernels.py`
- HF adapter path: `src/other_models/hf_modeling.py`

Key DANet method details:
- DenseAttention uses one learned query projection (`W_Q`) with `K,V` from hidden states.
- Kernel switches between linear/quadratic/auto complexity for full attention.
- Causal mode uses chunked parallel computation with context accumulation.
- Local patterns are defined by `local_scheme` (e.g., `l`, `sl`, `sw`, `g`, `softmax`, `dg`).
- Relative positional encoding is configurable by type and injection points (`relpe_scheme`).

## Data & Tasks
- Task registry is centralized in `utils/tasks.py`.
- Classic fixed-format datasets are in `data/dataset.py`.
- LM pretraining datasets (HDF5, JSONL, streamed HF sources) are in `data/dataset_lm.py`.
- Multi-file and streaming shard orchestration is in `data/dataset_utils.py` via `ShardedDatasetWrapper`.

## Active Config Areas
Most current experiment families are in:
- `configs/bert_relpe/`
- `configs/gpt/`
- `configs/lra/` and `configs/lra_mlm/`
- `configs/glue/`
- `configs/speed_eval/`

Some folders contain historical or ablation configs; treat them as reference unless verified.

## Important Intricacies / Gotchas
- Attention masks for DenseAttention are rescaled (`length^(-1/3)` or `window_size^(-1/3)`), not passed as plain binary masks.
- Local-attention DANet layers consume a tuple `(local_mask, global_mask)`.
- `DenseAttention.forward_inference()` is a stub.
- Sliding-window mask helpers currently hardcode specific window assumptions.
- Test coverage is minimal right now (`tests/src/test_positional_embeddings.py`).


## Fast Start Commands
- Install: `pip install -r requirements.txt`
- Run tests: `pytest tests/src -q`
- BERT pretraining example: `configs/bert_relpe/ds_train_dense_attn_bert_seq128_bf16.sh`
- GPT pretraining example: `configs/gpt/ds_train_dense_attn_gpt_360m_bf16.sh`
- LRA example: `SEED=100 configs/lra/ds_train_dense_attn_pathfinder32.sh`
