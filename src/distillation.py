import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from src.modeling import DANetModel, DANetPreTrainedModel

class BertForSequenceClassificationDistill(DANetPreTrainedModel):
    """BERT model for classification distillation from Sanh et. al. 2020.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a ModelConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
        `distill_alpha`: alpha parameter in distillation loss. Default = 0.6.
        `distill_beta`: beta parameter in distillation loss. Default = 0.4.
        `distill_gamma`: gamma parameter in distillation loss. Default = 0.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
        `teacher_logits`: torch.Tensor of shape [batch size x num_labels] with logits of teacher model
        on the same batch.
        `teacher_hidden`: torch.Tensor with teacher model encoded_layers (needed only when distill_beta != 0)

    Outputs:
        if `labels` is not `None` and teacher_logits is not `None`:
            Outputs distillation loss defined as 
            alpha * L_ce + beta * L_distill + gamma * L_cos
        if `labels` is not none and teacher_logits is `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the student's classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = ModelConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassificationDistill(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, args):
        super(BertForSequenceClassificationDistill, self).__init__(config)
        self.num_labels = args.num_labels if hasattr(args, "num_labels") else 2
        self.window_size = config.window_size
        self.PATH_TO_BACKBONE = "bert"
        self.bert = DANetModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels,
                                    bias=config.classifier_bias)

        self.apply(self.init_bert_weights)
        if hasattr(args, "zero_init_pooler") and args.zero_init_pooler:
            self.bert.pooler.dense_act.weight.data.zero_()
        self.use_local_attention = config.local_attention

        # Distillation hyperparameters
        self.distill_alpha = getattr(args, "distill_alpha", 0.6)
        self.distill_beta = getattr(args, "distill_beta", 0.4)
        self.distill_gamma = getattr(args, "distill_gamma", 0.0)
        self.distill_T = getattr(args, "distill_T", 5.0)

        # Projection layer for cosine loss when student and teacher have
        # different hidden sizes.
        teacher_hidden_size = getattr(args, "teacher_hidden_size", None)
        if self.distill_gamma > 0 and teacher_hidden_size is not None \
                and teacher_hidden_size != config.hidden_size:
            self.hidden_projection = nn.Linear(
                config.hidden_size, teacher_hidden_size, bias=False)
        else:
            self.hidden_projection = None

    def forward(self,
                input_ids,
                label=None,
                attention_mask=None,
                token_type_ids=None,
                teacher_logits=None,
                teacher_hidden=None,
                checkpoint_activations=False):
        checkpoint_activations = False
        dtype = self.bert.embeddings.word_embeddings.weight.dtype

        if self.distill_gamma > 0 and teacher_hidden is None and teacher_logits is not None:
            raise ValueError(
                "When distill_gamma > 0, teacher_hidden must be provided "
                "alongside teacher_logits.")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_attention_mask = (
            attention_mask /
            attention_mask.sum(axis=-1, keepdim=True).pow(1. / 3)
        ).to(dtype).unsqueeze(-1)
        if self.use_local_attention:
            local_attention_mask = (
                    attention_mask / self.window_size ** (1. / 3)
            ).to(dtype).unsqueeze(-1)
            extended_attention_mask = (
                local_attention_mask,
                extended_attention_mask
            )

        hidden, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask=extended_attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if label is not None:
            ce_fct = CrossEntropyLoss()

            if teacher_logits is not None:
                # cross-entropy with ground truth labels
                L_ce = ce_fct(logits.view(-1, self.num_labels), label.view(-1))

                # KL divergence between soft student and teacher
                L_distill = self.distill_T ** 2 * F.kl_div(
                    F.log_softmax(logits / self.distill_T, dim=-1),
                    F.softmax(teacher_logits / self.distill_T, dim=-1),
                    reduction='batchmean')

                # cosine embedding loss on hidden states 
                if self.distill_gamma > 0 and teacher_hidden is not None:
                    student_h = hidden.view(-1, hidden.size(-1))
                    if self.hidden_projection is not None:
                        student_h = self.hidden_projection(student_h)
                    teacher_h = teacher_hidden.view(-1, teacher_hidden.size(-1))
                    target = torch.ones(student_h.size(0),
                                        device=student_h.device)
                    L_cos = F.cosine_embedding_loss(
                        student_h, teacher_h, target)
                else:
                    L_cos = 0.0

                loss = (self.distill_alpha * L_ce
                        + self.distill_beta * L_distill
                        + self.distill_gamma * L_cos)
            else:
                loss = ce_fct(logits.view(-1, self.num_labels), label.view(-1))

            if not self.training:
                return loss, logits
            return loss
        else:
            return logits
        