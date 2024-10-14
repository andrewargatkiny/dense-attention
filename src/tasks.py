from dataclasses import dataclass
from typing import Dict, Type

from src.modeling import BertForPreTrainingNewAttention, BertForSequenceClassification, BertForAANMatching
from dataset import (LRADataset, LRATextDataset,
                     DatasetForMLM, TextDatasetForMLM, AANDataset, AANDatasetForMLM, BertPretrainingDataset,
                     )
from utils import eval_classification_task, eval_mlm_classification_task


@dataclass
class SequenceClassification:
    """Task for basic sequence classification which treats all sequences as
    having same length."""
    dataset_type = LRADataset
    model_type = BertForSequenceClassification
    eval_func = eval_classification_task

@dataclass
class TextClassification:
    """Task with generic text-like sequences of possibly different lengths
    which uses information about the lengths."""
    dataset_type = LRATextDataset
    model_type = BertForSequenceClassification
    eval_func = eval_classification_task

@dataclass
class TextsMatching:
    """Task which separately processes 2 text-like sequences of possibly different lengths
    which uses information about the lengths."""
    dataset_type = AANDataset
    model_type = BertForAANMatching
    eval_func = eval_classification_task

@dataclass
class SequenceClassificationMLM:
    """Task for basic sequence classification which treats all sequences as
    having same length with MLM and classification objectives."""
    dataset_type = DatasetForMLM
    model_type = BertForPreTrainingNewAttention
    eval_func = eval_mlm_classification_task

@dataclass
class TextClassificationMLM:
    """Task with generic text-like sequences of possibly different lengths
    with MLM and classification objectives."""
    dataset_type = TextDatasetForMLM
    model_type = BertForPreTrainingNewAttention
    eval_func = eval_mlm_classification_task

@dataclass
class AANTextClassificationMLM:
    """Like `TextClassificationMLM` but uses `AANDatasetForMLM`."""
    dataset_type = AANDatasetForMLM
    model_type = BertForPreTrainingNewAttention
    eval_func = eval_mlm_classification_task

@dataclass
class BertPretraining:
    dataset_type = BertPretrainingDataset
    model_type = BertForPreTrainingNewAttention
    eval_func = eval_mlm_classification_task


class TaskRegistry:
    _registry: Dict[str, Type] = {}

    @classmethod
    def register_task(cls, task_name: str, task_type: Type):
        """Register `task_type` so it can be instantiated with `task name`."""
        cls._registry[task_name] = task_type

    @classmethod
    def get_task(cls, name):
        """Retrieve task class by its name"""
        if name not in cls._registry:
            raise ValueError(
                f"There's no task with name {name} in the TaskRegistry."
            )
        return cls._registry[name]


TaskRegistry.register_task("sequence_classification", SequenceClassification)
TaskRegistry.register_task("text_classification", TextClassification)
TaskRegistry.register_task("texts_matching", TextsMatching)
TaskRegistry.register_task("sequence_classification_mlm", SequenceClassificationMLM)
TaskRegistry.register_task("text_classification_mlm", TextClassificationMLM)
TaskRegistry.register_task("aan_text_classification_mlm", AANTextClassificationMLM)
TaskRegistry.register_task("bert_pretraining", BertPretraining)

