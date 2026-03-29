# Knowledge Distillation - MLM Task

This file describes the knowledge distillation feature I added to the DenseAttention framework. The goal is to train a smaller, faster DANet student model by having it learn from a frozen pretrained teacher (e.g. BERT), so it benefits from the teacher's rich output distributions rather than only from hard data labels.

---

## Motivation

DANet runs in O(n) time and works on any hardware, but training it from scratch requires enormous amounts of data and compute. Distillation offers a shortcut: a pretrained Transformer teacher has already extracted deep language patterns from billions of tokens. By training the DANet student to match the teacher's output distributions, the student can inherit much of that quality at a fraction of the training cost.

---

## New Files

### `src/distillation.py`

Contains `DistillationMLMModel`, a `nn.Module` wrapper that holds both the student and the frozen teacher and computes the combined distillation loss during training.

**What it does during training:**

The forward pass returns a scalar loss combining two terms:

```
loss = alpha * CE(student_logits, hard_labels)
     + (1 - alpha) * T² * KL( softmax(student/T) || softmax(teacher/T) )
```

- `CE` is standard cross-entropy computed only at masked token positions
- `KL` is KL-divergence between the temperature-scaled student and teacher distributions
- `alpha` controls the balance between hard-label supervision and soft-label distillation
- `T` is the temperature that softens the distributions, higher T reveals more structure in the teacher's second and third choices
- `T²` compensates for the gradient scaling effect of temperature

The teacher is always frozen: `requires_grad=False` on all its parameters, and `.eval()` mode is enforced on every forward call so dropout is disabled and its output is deterministic.

**What it does during evaluation:**

When `masked_lm_labels=None`, the wrapper passes through to the student and returns its raw `(prediction_scores, seq_relationship_score)` tuple, the same format as any other task model, so the existing eval utilities work without modification.

**Framework compatibility:**

The wrapper sets `self.PATH_TO_BACKBONE = "student." + student.PATH_TO_BACKBONE` so that utility functions in `deepspeed_train.py` which navigate to the backbone via `attrgetter` (e.g. `update_weights_scalers`, `prepare_optimizer_parameters`, `report_model_weights`) continue to work without any changes.

**Teacher loading:**

`--teacher_model` accepts three formats:
- `"self"` - a randomly initialised DANet copy, useful for testing the pipeline without downloading anything
- A HuggingFace hub ID like `"bert-base-uncased"` - downloaded and loaded via `AutoModelForMaskedLM.from_pretrained`
- A local directory path containing a saved HuggingFace model

**Vocab alignment:**

BERT has vocab size 30522, but the framework pads the student vocab to the nearest multiple of 8 (30528). The wrapper handles this automatically, it truncates the teacher logits when teacher vocab > student vocab, and pads with `-1e4` (near-zero probability) in the reverse case.

---

### `data/distillation_dataset.py`

Contains `WikiMLMDataset`, which downloads wikitext-2-raw-v1 (~4 MB) on first use via HuggingFace datasets, tokenizes it with the BERT tokenizer, chunks it into fixed-length sequences, and applies standard BERT MLM masking.

**Masking strategy** follows the original BERT paper exactly:
- 15% of tokens are selected for masking
- Of those: 80% replaced with `[MASK]` (id=103), 10% replaced with a random token, 10% kept unchanged

Each sample is a dict with keys `input_ids`, `attention_mask`, `token_type_ids`, `masked_lm_labels`, and `label` , matching the interface that `deepspeed_train.py` expects when it calls `model(**batch)`.

---

### `utils/distillation_tasks.py`

Contains the task descriptor `DistillationMLMTask` and the evaluation function `distillation_mlm_eval`.

**`DistillationMLMTask`** is a dataclass-style descriptor with three attributes that the framework reads:
- `model_type = DistillationMLMModel`
- `dataset_type = WikiMLMDataset`
- `eval_func = distillation_mlm_eval`

**`distillation_mlm_eval`** runs after each training epoch and reports four metrics:

| Metric | What it means |
|--------|---------------|
| `student_mlm_acc` | Fraction of masked tokens correctly predicted by the student |
| `student_ppl` | Student perplexity on masked tokens , lower is better |
| `teacher_mlm_acc` | Teacher accuracy on the same positions , the quality target |
| `mean_kl` | Mean KL(student ‖ teacher) over batches , should decrease as training progresses |

The eval function calls student and teacher separately and handles vocab alignment between them.

---

### `configs/distillation/mlm_distillation_cpu.json`

Bundled config with three sections the framework requires:

- `model_config` , a small student: 2 layers, 64 hidden, 2 attention heads, vocab 30522, learned positional embeddings, no local attention
- `data` , wikitext-2 split, 1000 training sequences, 200 validation sequences, seq length 64, mask prob 0.15
- `training` , 10 epochs, constant LR of 1e-3

---

### `configs/distillation/deepspeed_cpu.json`

DeepSpeed config for CPU runs: Adam optimizer, fp32 precision, ZeRO stage 0 (no sharding), `gloo` distributed backend (nccl requires GPU).

---

### `configs/distillation/ds_train_distillation_cpu.sh`

Launch script that runs a full distillation experiment on CPU. Follows the same structure as all other `ds_train_*.sh` scripts in the repo.

Key arguments passed to `deepspeed_train.py`:

```bash
--task_type mlm_distillation
--teacher_model bert-base-uncased
--distillation_alpha 0.5
--distillation_temperature 4.0
--mask_token_id 103
--max_seq_length 64
--dict_backend gloo
```

---

## Changes to Existing Files

### `utils/tasks.py`

Two lines added at the bottom, following the existing registration pattern:

```python
from utils.distillation_tasks import DistillationMLMTask
TaskRegistry.register_task("mlm_distillation", DistillationMLMTask)
```

No existing registrations were modified.

### `train_arguments.py`

Three arguments added to `get_argument_parser()`, just before the final `return parser`:

```python
parser.add_argument(
    "--teacher_model",
    type=str,
    default="self",
    help="Teacher model source. 'self' = random init copy (testing). "
         "HuggingFace hub ID e.g. 'bert-base-uncased', or a local path.",
)
parser.add_argument(
    "--distillation_alpha",
    type=float,
    default=0.5,
    help="Weight alpha for the hard-label CE term. 1-alpha is used for KD loss.",
)
parser.add_argument(
    "--distillation_temperature",
    type=float,
    default=2.0,
    help="Softmax temperature T for soft teacher/student distributions.",
)
```

These arguments are ignored for all non-distillation task types.

---

## Running the Experiment

Install the additional dependency if not already present:

```bash
pip install datasets
```

Then launch:

```bash
bash configs/distillation/ds_train_distillation_cpu.sh
```

On first run this downloads `bert-base-uncased` (~440 MB) and wikitext-2 (~4 MB), both cached locally afterwards. Each epoch takes 3–5 minutes on a modern CPU.

After each epoch the log will show something like:

```
[Validation] Student MLM accuracy: 0.0669, perplexity: 1571.60
[Validation] Teacher MLM accuracy: 0.5125, mean KL(S||T): 4.3717
```

Teacher accuracy (~51%) is the quality target. Student accuracy starts low because the student is tiny (2 layers, 64 hidden vs BERT's 12 layers, 768 hidden) and the dataset is small. The pipeline is correct , with a larger student and more data the gap closes.

To use a different teacher:

```bash
bash configs/distillation/ds_train_distillation_cpu.sh \
    --override cf.training.num_epochs=20 \
    --teacher_model bert-large-uncased \
    --distillation_alpha 0.4 \
    --distillation_temperature 6.0
```

---

## Design Decisions

**Wrapper instead of modifying `DANetForPreTraining`** , keeping distillation as a separate wrapper means zero changes to existing model code. The wrapper satisfies the same interface contract as any other task model, so the training loop, checkpointing, and logging all work unchanged.

**Logit distillation only** , this is the simplest and most architecture-agnostic form. It works even when teacher and student have completely different internal structures, which is the case here since DANet replaces softmax attention entirely. Hidden-state and attention-map distillation would require both models to have comparable layer structures.

**WikiMLMDataset instead of synthetic data** , random token sequences produce near-uniform teacher distributions, which carry no useful distillation signal. Real text gives BERT structured, peaked distributions that the student can meaningfully learn from.
