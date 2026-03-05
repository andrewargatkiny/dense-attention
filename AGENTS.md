# DenseAttention Project Reference & Minimal Engineering Guide

## 1) What this repository does
This codebase is a DeepSpeed-first research/training framework for sequence models with multiple architecture families under one runner:
- DenseAttention / DANet models (`src/dense_attention.py`, `src/danet_layers.py`, `src/modeling.py`)
- Transformer and kernelized variants (`src/other_models/modeling.py`, `src/other_models/attention_kernels.py`)
- HuggingFace adapter variants (`src/other_models/hf_modeling.py`)

Primary supported workloads:
- BERT-style MLM/NSP pretraining for language and sequence modeling
- GPT-style causal LM pretraining
- LRA classification tasks
- GLUE-style classification/regression
- Throughput/speed evaluation

Primary entrypoint: `deepspeed_train.py`.
Legacy paths `old_code/` and `src/legacy_layers.py` are archival and should be ignored.

## 2) Entry points and execution flow
1. Launch is usually from `configs/**/ds_train_*.sh`.
2. Shell scripts call `deepspeed_train.py` with config paths + runtime flags.
3. `deepspeed_train.py`:
   - parses CLI + DeepSpeed args,
   - loads config(s) and applies `--override` for both `cf.*` and `ds.*`,
   - resolves `task_type` through `utils/tasks.py::TaskRegistry`,
   - builds model/config/optimizer + DeepSpeed engine,
   - runs training/eval/checkpoint lifecycle.
4. Task abstraction chooses dataset class, model class, eval function, and optional config class per task.

## 3) Core modeling stack
- DANet stack: `src/modeling.py`, `src/danet_layers.py`, `src/dense_attention.py`
- Transformer and other architectures: `src/other_models/modeling.py`, `src/other_models/attention_kernels.py`
- HF adapter path: `src/other_models/hf_modeling.py`

Key DANet details:
- DenseAttention uses learned query projection(s) with K,V from hidden states.
- Complexity path switches across `linear` / `quadratic` / `auto`.
- Causal mode supports chunked computation and context accumulation.
- Local patterns are controlled by `local_scheme` (e.g., `l`, `sl`, `sw`, `g`, `softmax`, `dg`).
- Relative positional encoding behavior is configured by `relpe_scheme`.

## 4) Task and data routing
- Task registry: `utils/tasks.py` (central architecture/data/eval dispatch)
- Classic fixed-format datasets: `data/dataset.py`
- LM pretraining datasets (HDF5, JSONL, streaming HF): `data/dataset_lm.py`
- Multi-file/streaming shard orchestration: `data/dataset_utils.py` (`ShardedDatasetWrapper`)

## 5) Active config areas
Most current experiment families are in:
- `configs/bert_relpe/`
- `configs/gpt/`
- `configs/lra/` and `configs/lra_mlm/`
- `configs/glue/`
- `configs/speed_eval/`

Task-to-config quick routing:
- MLM/NSP pretraining → `configs/bert_relpe/`
- Causal LM pretraining → `configs/gpt/`
- Long-range classification/MLM → `configs/lra/`, `configs/lra_mlm/`
- Throughput profiling → `configs/speed_eval/`

Some folders contain historical/ablation configs; treat them as reference unless validated.

## 6) Intricacies / gotchas
- DenseAttention masks are rescaled (`length^(-1/3)` / `window_size^(-1/3)`), not plain binary.
- Local-attention DANet layers consume mask tuples `(local_mask, global_mask)`.
- `DenseAttention.forward_inference()` is currently a stub.
- Non attention modules in `src/modeling.py` and `src/other_models/modeling.py` can also differ in subtle ways.
- Sliding-window/local helpers rely on specific window assumptions.
- Local attention modes may require divisibility constraints (e.g., max positions vs window size).
- `local_scheme` semantics can be overloaded in hybrid paths (locality + kernel forcing).

## 7) Truth hierarchy for contributors/agents
When docs, configs, and behavior conflict, trust in this order:
1. Runtime behavior in source code.
2. Training entrypoint/task wiring (`deepspeed_train.py`, `utils/tasks.py`).
3. Active config scripts/JSONs under `configs/`.
4. Narrative guidance in this `AGENTS.md`.

Before acting on non-trivial assumptions, verify with targeted `rg` queries in the files above.

## 8) High-risk change checklist
If editing attention/model paths, verify all of the following:
- Mask scaling assumptions and mask shapes in DANet layers (`src/danet_layers.py`).
- Tuple mask routing (`local_mask` vs `global_mask`) for local/global schemes (`src/danet_layers.py`).
- `local_scheme` code compatibility with implementation tables (`src/danet_layers.py`, `src/modeling.py`).
- `relpe_scheme` compatibility and injection points (`src/dense_attention.py`, `src/model_config.py`).
- Inference-path constraints (causal + sliding-window) and stub behavior (`src/dense_attention.py`).
- Task/config compatibility through `TaskRegistry` (`utils/tasks.py`).

## 9) Fast start commands
- Install: `pip install -r requirements.txt`
- Run tests: `pytest tests/src -q`
- BERT pretraining example: `configs/bert_relpe/ds_train_dense_attn_bert_seq128_bf16.sh`
- GPT pretraining example: `configs/gpt/ds_train_dense_attn_gpt_360m_bf16.sh`
- LRA example: `SEED=100 configs/lra/ds_train_dense_attn_pathfinder32.sh`

## 10) Python style preference
- Prefer PEP-8-style line length of 80 characters.
- Rarely allow a small overflow (about 1-3 characters), but generally wrap to the next line.

## 11) Test environment preference
- Run tests with project-root `venv` first (`venv/bin/python -m pytest ...`).
- If `venv` is missing/broken/incompatible (e.g., cannot import `pytest` or `torch`), use a suitable preinstalled conda env instead.
- Choose the first conda env that can run pytest and import required test packages.
