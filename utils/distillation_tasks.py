# Distillation task definitions for the DenseAttention framework.
#
# This module is meant to be imported from utils/tasks.py and its tasks
# registered with TaskRegistry.

from __future__ import absolute_import, division, print_function

import logging
import math

import torch
from tqdm import tqdm

from train_utils import master_process

logger = logging.getLogger(__name__)

# Evaluation function
def distillation_mlm_eval(data_batches, model, max_samples, series_name,
                           index, args):
    """Evaluate the student model on MLM accuracy and perplexity."""
    logger.info(f"[Distillation Eval] {series_name} epoch {index + 1}")

    # Unwrap DeepSpeed engine to access the Python module
    underlying = model.module if hasattr(model, "module") else model
    student = underlying.student

    # Teacher access is optional
    has_teacher = hasattr(underlying, "teacher")
    teacher = underlying.teacher if has_teacher else None

    student_correct = 0
    student_total = 0
    student_nll_sum = 0.0
    teacher_correct = 0
    kl_sum = 0.0
    n_batches = 0
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(data_batches, desc=f"Eval {series_name}"):
            if n_samples >= max_samples:
                break

            batch = {k: v.to(args.device) for k, v in batch.items()}
            masked_lm_labels = batch.get("masked_lm_labels")
            if masked_lm_labels is None:
                continue

            # ---- Student predictions ----
            student_out = student(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                token_type_ids=batch.get("token_type_ids"),
                masked_lm_labels=None,
            )
            if isinstance(student_out, (tuple, list)):
                pred_scores = student_out[0]   # (B, T, vocab_size)
            else:
                pred_scores = student_out

            # Masked positions only
            mask = (masked_lm_labels != -1).view(-1)
            if mask.sum() == 0:
                continue

            s_flat = pred_scores.view(-1, pred_scores.size(-1))[mask]
            targets = masked_lm_labels.view(-1)[mask]

            student_correct += (s_flat.argmax(dim=-1) == targets).sum().item()
            student_total += mask.sum().item()
            student_nll_sum += torch.nn.functional.cross_entropy(
                s_flat, targets, reduction="sum"
            ).item()

            # Teacher predictions (optional)
            if has_teacher and teacher is not None:
                try:
                    try:
                        t_out = teacher(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            token_type_ids=batch.get("token_type_ids"),
                        )
                    except TypeError:
                        t_out = teacher(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                        )
                    if hasattr(t_out, "logits"):
                        t_scores = t_out.logits
                    elif isinstance(t_out, (tuple, list)):
                        t_scores = t_out[0]
                    else:
                        t_scores = t_out

                    # Align teacher vocab to student vocab
                    s_vocab = pred_scores.size(-1)
                    t_vocab = t_scores.size(-1)
                    if t_vocab > s_vocab:
                        t_scores = t_scores[..., :s_vocab]
                    elif t_vocab < s_vocab:
                        pad = torch.full(
                            (*t_scores.shape[:-1], s_vocab - t_vocab),
                            fill_value=-1e4,
                            dtype=t_scores.dtype,
                            device=t_scores.device,
                        )
                        t_scores = torch.cat([t_scores, pad], dim=-1)

                    t_flat = t_scores.view(-1, t_scores.size(-1))[mask]
                    teacher_correct += (
                        t_flat.argmax(dim=-1) == targets
                    ).sum().item()

                    # KL(student || teacher) at temperature 1
                    kl = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(s_flat, dim=-1),
                        torch.nn.functional.softmax(t_flat, dim=-1),
                        reduction="batchmean",
                    )
                    kl_sum += kl.item()
                except Exception as exc:
                    logger.warning(f"Teacher eval error: {exc}")

            n_batches += 1
            n_samples += batch["input_ids"].size(0)

    # Report metrics
    if master_process(args) and student_total > 0:
        student_acc = student_correct / student_total
        student_ppl = math.exp(min(student_nll_sum / student_total, 20))

        args.tracker_logger.report_scalar(
            title=f"Distillation / {series_name}: Student MLM Accuracy",
            series="student_mlm_acc",
            value=student_acc,
            iteration=index,
        )
        args.tracker_logger.report_scalar(
            title=f"Distillation / {series_name}: Student MLM Perplexity",
            series="student_ppl",
            value=student_ppl,
            iteration=index,
        )

        logger.info(
            f"[{series_name}] Student MLM accuracy: {student_acc:.4f}, "
            f"perplexity: {student_ppl:.2f}"
        )

        if has_teacher and n_batches > 0:
            teacher_acc = teacher_correct / student_total
            mean_kl = kl_sum / n_batches

            args.tracker_logger.report_scalar(
                title=f"Distillation / {series_name}: Teacher MLM Accuracy",
                series="teacher_mlm_acc",
                value=teacher_acc,
                iteration=index,
            )
            args.tracker_logger.report_scalar(
                title=f"Distillation / {series_name}: Mean KL(Student||Teacher)",
                series="mean_kl",
                value=mean_kl,
                iteration=index,
            )

            logger.info(
                f"[{series_name}] Teacher MLM accuracy: {teacher_acc:.4f}, "
                f"mean KL(S||T): {mean_kl:.4f}"
            )

# Task descriptor
class DistillationMLMTask:
    """Task descriptor for MLM knowledge distillation."""

    from src.distillation import DistillationMLMModel
    from data.distillation_dataset import WikiMLMDataset

    model_type = DistillationMLMModel
    dataset_type = WikiMLMDataset
    eval_func = staticmethod(distillation_mlm_eval)