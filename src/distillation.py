# This module implements knowledge distillation from a teacher model
# into a student DANet model. The student is trained with a combined loss:
#
#   loss = alpha * student_task_loss + (1 - alpha) * kd_loss
#
# where kd_loss is the KL-divergence between teacher and student soft-label
# distributions on masked positions (for MLM), scaled by temperature^2.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from operator import attrgetter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DistillationMLMModel(nn.Module):
    """Wraps a student (DANetForPreTraining) and a frozen teacher model for
    knowledge distillation on the Masked Language Modelling task."""

    def __init__(self, config, args):
        super().__init__()

        # Import student model class (avoids circular imports)
        from src.modeling import DANetForPreTraining

        # Build student
        self.student = DANetForPreTraining(config, args)

        # Distillation hyper-parameters
        self.alpha = getattr(args, "distillation_alpha", 0.5)
        self.temperature = getattr(args, "distillation_temperature", 2.0)

        # Load / build teacher and freeze it
        teacher_name = getattr(args, "teacher_model", "self")
        self.teacher = self._build_teacher(teacher_name, config, args)
        self._freeze(self.teacher)

        # Expose backbone / layer paths so framework utilities work
        # e.g. attrgetter("student.bert")(distillation_model) == student.bert
        self.PATH_TO_BACKBONE = "student." + self.student.PATH_TO_BACKBONE

        logger.info(
            f"DistillationMLMModel ready. "
            f"alpha={self.alpha}, T={self.temperature}, "
            f"teacher_model='{teacher_name}'"
        )

    def _build_teacher(self, teacher_name: str, config, args) -> nn.Module:
        """Return a teacher model based on the teacher_name argument."""

        if teacher_name in ("self", "none", ""):
            logger.info(
                "teacher_model='self': using a randomly initialised copy of "
                "the student architecture as teacher (for testing only)."
            )
            from src.modeling import DANetForPreTraining
            return DANetForPreTraining(config, args)

        try:
            from transformers import AutoModelForMaskedLM
            logger.info(f"Loading teacher from '{teacher_name}' via HuggingFace …")
            teacher = AutoModelForMaskedLM.from_pretrained(teacher_name)
            logger.info("Teacher loaded successfully.")
            return teacher
        except Exception as exc:
            raise RuntimeError(
                f"Could not load teacher model '{teacher_name}'. "
                f"Set --teacher_model to 'self' for a random teacher, "
                f"a HuggingFace model ID, or a local path. "
                f"Original error: {exc}"
            ) from exc

    @staticmethod
    def _freeze(model: nn.Module) -> None:
        """Freeze all parameters of *model* and put it in eval mode."""
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        masked_lm_labels: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        **kwargs,
    ):

        # Inference mode
        if masked_lm_labels is None:
            return self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                masked_lm_labels=None,
                label=label,
            )

        # Training mode
        # (1) Get student full-sequence logits (WITH gradient).
        #     We call the student with masked_lm_labels=None so it returns the
        #     dense (B, T, V) prediction scores.  We will apply the task CE
        #     loss and KD loss ourselves.
        student_pred_scores, student_cls_scores = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            masked_lm_labels=None,   # returns logits
            label=None,
        )
        # student_pred_scores: (B, T, vocab_size)

        # (2) Masked positions for hard-label CE loss
        masked_positions = torch.nonzero(
            (masked_lm_labels + 1).view(-1), as_tuple=False
        ).view(-1)
        hard_targets = torch.index_select(
            masked_lm_labels.view(-1), 0, masked_positions
        )
        student_masked = torch.index_select(
            student_pred_scores.view(-1, student_pred_scores.size(-1)),
            0,
            masked_positions,
        ) 

        task_loss = F.cross_entropy(student_masked, hard_targets)

        # Add classification head loss when labels are present
        if label is not None and student_cls_scores is not None:
            n_labels = student_cls_scores.size(-1)
            task_loss = task_loss + F.cross_entropy(
                student_cls_scores.view(-1, n_labels),
                label.view(-1),
                ignore_index=-1,
            )

        # (3) Teacher forward (no gradient, frozen)
        teacher_logits = self._teacher_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ) 

        # Align vocab dimension if teacher has different vocab size
        teacher_logits = self._align_vocab(
            teacher_logits, student_pred_scores.size(-1), input_ids.device
        )

        teacher_masked = torch.index_select(
            teacher_logits.view(-1, teacher_logits.size(-1)),
            0,
            masked_positions,
        )  # (n_masked, V)

        # (4) KL-divergence soft-label loss (temperature scaling)
        T = self.temperature
        kd_loss = F.kl_div(
            F.log_softmax(student_masked / T, dim=-1),
            F.softmax(teacher_masked / T, dim=-1).detach(),
            reduction="batchmean",
        ) * (T * T)

        total_loss = self.alpha * task_loss + (1.0 - self.alpha) * kd_loss
        return total_loss

    @torch.no_grad()
    def _teacher_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the (frozen) teacher and return its per-token logits."""
        self.teacher.eval()

        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            try:
                out = self.teacher(**kwargs, token_type_ids=token_type_ids)
            except TypeError:
                # Teacher doesn't accept token_type_ids
                out = self.teacher(**kwargs)
        else:
            out = self.teacher(**kwargs)

        # HuggingFace MaskedLMOutput has .logits attribute
        if hasattr(out, "logits"):
            return out.logits
        # DANetForPreTraining with masked_lm_labels=None returns (pred, cls)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    @staticmethod
    def _align_vocab(
        teacher_logits: torch.Tensor, student_vocab: int, device: torch.device
    ) -> torch.Tensor:
        """Pad or truncate teacher logits to match student vocab size."""
        t_vocab = teacher_logits.size(-1)
        if t_vocab == student_vocab:
            return teacher_logits
        if t_vocab > student_vocab:
            return teacher_logits[..., :student_vocab]
        # Pad with -1e4 (effectively zero probability)
        pad = torch.full(
            (*teacher_logits.shape[:-1], student_vocab - t_vocab),
            fill_value=-1e4,
            dtype=teacher_logits.dtype,
            device=device,
        )
        return torch.cat([teacher_logits, pad], dim=-1)