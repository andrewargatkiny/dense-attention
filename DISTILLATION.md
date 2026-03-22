# Knowledge Distillation for DenseAttention

## Overview

Knowledge distillation compresses a larger **teacher** model into a smaller **student** model by training the student to mimic the teacher's output distribution alongside the ground-truth labels. This implementation follows Sanh et al. (2020) and uses a composite loss:

```
L = α · L_ce + β · T² · KL(σ(z_s/T) ‖ σ(z_t/T)) + γ · L_cos
```

| Symbol | Description | Default |
|--------|-------------|---------|
| `α` | Weight for cross-entropy with ground-truth labels | 0.6 |
| `β` | Weight for KL divergence between soft student/teacher logits | 0.4 |
| `γ` | Weight for cosine embedding loss on hidden states | 0.0 |
| `T` | Temperature for softening probability distributions | 5.0 |

When `γ = 0` (default), the hidden-state cosine loss is disabled and the teacher only needs to produce logits. When `γ > 0`, teacher hidden states are also used, and a projection layer is automatically created if teacher and student have different `hidden_size`.

## Architecture

- **Teacher**: DenseAttention, hidden=256, 9 layers, 4 heads (~5.4M params)
- **Student**: DenseAttention, hidden=128, 4 layers, 4 heads (~1.4M params)
- Both use CosRelPE positional embeddings with local attention (window_size=20)

## Files

| File | Description |
|------|-------------|
| `src/distillation.py` | `BertForSequenceClassificationDistill` — student model with distillation loss |
| `deepspeed_train.py` | Training loop with teacher inference (lines 195-219) and teacher loading (`prepare_teacher_model`, lines 762-796) |
| `train_arguments.py` | CLI arguments for distillation (lines 396-446) |
| `utils/tasks.py` | Task registration: `sequence_classification_distill` |
| `configs/lra/dense_attn_listops_student.json` | Student model config |
| `configs/lra/dense_attn_listops.json` | Teacher model config (used via `--teacher_config_file`) |
| `configs/lra/deepspeed_config_listops_distill.json` | DeepSpeed config for distillation |
| `configs/lra/ds_train_dense_attn_listops_distill.sh` | Launch script |

## Usage

### Prerequisites

1. A trained teacher checkpoint. The default path expects:
   ```
   teacher_checkpoint/saved_models/lra_listops_teacher/epoch41_step30750/mp_rank_00_model_states.pt
   ```

2. LRA ListOps data at `data/lra/listops/` (or set `BASE_DATA_DIR`).

### Running distillation

```bash
# Basic run with defaults
bash configs/lra/ds_train_dense_attn_listops_distill.sh

# Override hyperparameters
DISTILL_ALPHA=0.5 DISTILL_BETA=0.5 DISTILL_T=3.0 \
  bash configs/lra/ds_train_dense_attn_listops_distill.sh

# Use a different teacher checkpoint
TEACHER_CHECKPOINT_DIR=/path/to/teacher \
TEACHER_CHECKPOINT_ID=epoch30_step22500 \
  bash configs/lra/ds_train_dense_attn_listops_distill.sh

# Specify GPU node
NODE=0,1 bash configs/lra/ds_train_dense_attn_listops_distill.sh
```

### CLI arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--teacher_checkpoint` | str | None | Path to teacher checkpoint directory |
| `--teacher_checkpoint_id` | str | None | Checkpoint ID (e.g. `epoch41_step30750`) |
| `--teacher_config_file` | str | None | Path to teacher config JSON (required when teacher ≠ student architecture) |
| `--distill_alpha` | float | 0.6 | Cross-entropy loss weight |
| `--distill_beta` | float | 0.4 | KL divergence loss weight |
| `--distill_gamma` | float | 0.0 | Cosine embedding loss weight |
| `--distill_T` | float | 5.0 | Distillation temperature |
| `--task_type` | str | — | Must be `sequence_classification_distill` |

### Student config

The student config (`dense_attn_listops_student.json`) trains for 60 epochs with cosine LR schedule, matching the teacher's training regime but with a smaller model:

```json
{
    "model_config": {
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 4
    }
}
```

## How it works

1. **Teacher loading** (`prepare_teacher_model`): The teacher checkpoint is loaded, all parameters are frozen (`requires_grad=False`), and the model is set to eval mode.

2. **Training loop**: For each batch:
   - Teacher produces logits (and optionally hidden states) via forward pass with `torch.no_grad()`
   - Teacher outputs are attached to the batch dict (`teacher_logits`, `teacher_hidden`)
   - Student forward pass computes the composite distillation loss
   - Only student parameters are updated

3. **Evaluation**: During validation/test, the student model is evaluated independently (standard cross-entropy + accuracy), without the teacher.

## CPU compatibility

The codebase includes guards for CPU-only environments:
- `args.no_cuda` is auto-detected via `torch.cuda.is_available()`
- CUDA seed and `pin_memory` are conditional on CUDA availability
- `LOCAL_RANK` defaults to `0` if not set

To run on CPU (e.g. for testing), pass `--dict_backend gloo` and use `deepspeed_config_listops_cpu.json`.
