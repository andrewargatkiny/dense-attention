from dataclasses import dataclass
from typing import Dict, Type

from src.other_models import (TransformerForPreTraining,
                              TransformerForSequenceClassification,
                              TransformerForRegression, TransformerConfig)
from src.modeling import DANetForPreTraining, BertForSequenceClassification, BertForAANMatching, \
    BertForRegression
from src.other_models.bert_hf import BertHFForSequenceClassification
from data.dataset import (LRADataset, LRATextDataset,
                          DatasetForMLM, TextDatasetForMLM, AANDataset,
                          AANDatasetForMLM, GlueBertDataset, GlueDatasetForMLM)
from data.dataset_lm import BertPretrainingDatasetFactory, GPTPretrainingDataset, \
    BertOnlyMLMDataset
from train_utils import eval_classification_task, eval_mlm_classification_task, eval_glue_tasks, eval_regression_task


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
    model_type = DANetForPreTraining
    eval_func = eval_mlm_classification_task

@dataclass
class TextClassificationMLM:
    """Task with generic text-like sequences of possibly different lengths
    with MLM and classification objectives."""
    dataset_type = TextDatasetForMLM
    model_type = DANetForPreTraining
    eval_func = eval_mlm_classification_task

@dataclass
class AANTextClassificationMLM:
    """Like `TextClassificationMLM` but uses `AANDatasetForMLM`."""
    dataset_type = AANDatasetForMLM
    model_type = DANetForPreTraining
    eval_func = eval_mlm_classification_task

@dataclass
class BertPretraining:
    dataset_type = BertPretrainingDatasetFactory
    model_type = DANetForPreTraining
    eval_func = eval_mlm_classification_task

@dataclass
class BertMLM:
    dataset_type = BertOnlyMLMDataset
    model_type = DANetForPreTraining
    eval_func = eval_mlm_classification_task


@dataclass
class GptPretraining:
    dataset_type = GPTPretrainingDataset
    model_type = DANetForPreTraining
    eval_func = eval_mlm_classification_task

#___________________________________________________
@dataclass
class TransformerSequenceClassification:
    """Task for basic sequence classification which treats all sequences as
    having same length."""
    dataset_type = LRADataset
    model_type = TransformerForSequenceClassification
    eval_func = eval_classification_task

@dataclass
class TransformerSequenceMLM:
    """Task for basic sequence classification which treats all sequences as
    having same length with MLM and classification objectives."""
    dataset_type = DatasetForMLM
    model_type = TransformerForPreTraining
    eval_func = eval_mlm_classification_task

@dataclass
class TransformerBertPretraining:
    dataset_type = BertPretrainingDatasetFactory
    model_type = TransformerForPreTraining
    eval_func = eval_mlm_classification_task
    config_type = TransformerConfig

@dataclass
class TransformerBertMLM:
    dataset_type = BertOnlyMLMDataset
    model_type = TransformerForPreTraining
    eval_func = eval_mlm_classification_task
    config_type = TransformerConfig

@dataclass
class TransformerGptPretraining:
    dataset_type = GPTPretrainingDataset
    model_type = TransformerForPreTraining
    eval_func = eval_mlm_classification_task
    config_type = TransformerConfig
#__________________________________________________

@dataclass
class GlueWithAccMetrics:
    """Task for fine-tuning on some of the GLUE benchmark with accuracy as
    the evaluation metric"""
    dataset_type = GlueBertDataset
    model_type = BertForSequenceClassification
    eval_func = eval_classification_task

@dataclass
class GlueWithAllMetrics:
    """Task for fine-tuning on some of the GLUE benchmark with accuracy, F1
    and Matthews correlation as the evaluation metric"""
    dataset_type = GlueBertDataset
    model_type = BertForSequenceClassification
    eval_func = eval_glue_tasks

@dataclass
class GlueForRegression:
    """Task for fine-tuning on some of the GLUE benchmark with accuracy as
    the evaluation metric"""
    dataset_type = GlueBertDataset
    model_type = BertForRegression
    eval_func = eval_regression_task

@dataclass
class GlueTransformerWithAccMetrics:
    """Task for fine-tuning on some of the GLUE benchmark with accuracy as
    the evaluation metric"""
    dataset_type = GlueBertDataset
    model_type = TransformerForSequenceClassification
    eval_func = eval_classification_task
    config_type = TransformerConfig

@dataclass
class GlueTransformerWithAllMetrics:
    """Task for fine-tuning on some of the GLUE benchmark with accuracy, F1
    and Matthews correlation as the evaluation metric"""
    dataset_type = GlueBertDataset
    model_type = TransformerForSequenceClassification
    eval_func = eval_glue_tasks
    config_type = TransformerConfig

@dataclass
class GlueTransformerForRegression:
    """Task for fine-tuning on some of the GLUE benchmark with accuracy as
    the evaluation metric"""
    dataset_type = GlueBertDataset
    model_type = TransformerForRegression
    eval_func = eval_regression_task
    config_type = TransformerConfig

@dataclass
class GlueHFWithAccMetrics:
    """Task for fine-tuning on some of the GLUE benchmark with accuracy as
    the evaluation metric. Uses HuggingFace's Bert-Large."""
    dataset_type = GlueBertDataset
    model_type = BertHFForSequenceClassification
    eval_func = eval_classification_task


@dataclass
class GluePretrainingMLM:
    """Task for fine-tuning on some of the GLUE benchmark with accuracy as
    the evaluation metric"""
    dataset_type = GlueDatasetForMLM
    model_type = DANetForPreTraining
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
TaskRegistry.register_task("bert_mlm", BertMLM)
TaskRegistry.register_task("gpt_pretraining", GptPretraining)
TaskRegistry.register_task("transformer_bert_pretraining", TransformerBertPretraining)
TaskRegistry.register_task("transformer_bert_mlm", TransformerBertMLM)
TaskRegistry.register_task("transformer_gpt_pretraining", TransformerGptPretraining)
TaskRegistry.register_task("transformer_sequence_classification", TransformerSequenceClassification)
TaskRegistry.register_task("transformer_sequence_mlm", TransformerSequenceMLM)
TaskRegistry.register_task("glue_with_acc_metrics", GlueWithAccMetrics)
TaskRegistry.register_task("glue_with_all_metrics", GlueWithAllMetrics)
TaskRegistry.register_task("glue_for_regression", GlueForRegression)
TaskRegistry.register_task("glue_pretraining_mlm", GluePretrainingMLM)
TaskRegistry.register_task("glue_transformer_with_acc_metrics", GlueTransformerWithAccMetrics)
TaskRegistry.register_task("glue_hf_with_acc_metrics", GlueHFWithAccMetrics)
TaskRegistry.register_task("glue_transformer_with_all_metrics", GlueTransformerWithAllMetrics)
TaskRegistry.register_task("glue_transformer_for_regression", GlueTransformerForRegression)
