# Knowledge Distillation for DenseAttention

Proof-of-concept: compress a teacher model into a smaller student via knowledge distillation on LRA ListOps.

## Quick start

```bash
# 1. Place teacher checkpoint at:
#    teacher_checkpoint/saved_models/lra_listops_teacher/<checkpoint_id>/mp_rank_00_model_states.pt

# 2. Run distillation
bash configs/lra/ds_train_dense_attn_listops_distill.sh

# 3. Override hyperparameters via env vars
DISTILL_ALPHA=0.4 DISTILL_BETA=0.6 DISTILL_T=10.0 \
  bash configs/lra/ds_train_dense_attn_listops_distill.sh
```

## CLI arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--distill_alpha` | 0.6 | Cross-entropy weight |
| `--distill_beta` | 0.4 | KL divergence weight |
| `--distill_gamma` | 0.0 | Cosine embedding loss weight |
| `--distill_T` | 5.0 | Temperature |
| `--teacher_checkpoint` | — | Path to teacher checkpoint dir |
| `--teacher_checkpoint_id` | — | Checkpoint ID |
| `--teacher_config_file` | — | Teacher config JSON |

## Files

| File | What it does |
|------|-------------|
| `src/distillation.py` | Student model with distillation loss |
| `deepspeed_train.py` | Teacher loading and inference in training loop |
| `train_arguments.py` | Distillation CLI arguments |
| `utils/tasks.py` | `sequence_classification_distill` task |
| `configs/lra/dense_attn_listops_student.json` | Student config |
| `configs/lra/ds_train_dense_attn_listops_distill.sh` | Launch script |
