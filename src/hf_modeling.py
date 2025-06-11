# src/hf_modeling.py
from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
from src.modeling import DANetPreTrainedModel, ModelConfig, BertPreTrainingHeads

class HFBaseModel(DANetPreTrainedModel):
    """Core adapter that handles input/output conversion"""
    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config.hf_config)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Extract and convert outputs to match the framework's
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]
        
        return sequence_output, pooled_output

# Task-Specific Heads
class HFForPreTraining(DANetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = HFBaseModel(config)
        self.cls = BertPreTrainingHeads(
            config, 
            self.bert.model.embeddings.word_embeddings.weight,
            num_labels=config.num_labels
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                masked_lm_labels=None, label=None, log=True):
        sequence_output, pooled_output = self.bert(
            input_ids, attention_mask, token_type_ids
        )
        
        if masked_lm_labels is None:
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output
            )
            return prediction_scores, seq_relationship_score
            
        masked_token_indexes = torch.nonzero(
            (masked_lm_labels + 1).view(-1)
        ).view(-1)
        
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output, masked_token_indexes
        )
        
        target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)
        masked_lm_loss = self.loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), target
        )
        next_sentence_loss = self.loss_fct(
            seq_relationship_score.view(-1, self.config.num_labels), label.view(-1)
        )
        total_loss = masked_lm_loss + next_sentence_loss
        
        if not self.training:
            return (masked_lm_loss, next_sentence_loss, target,
                    prediction_scores, seq_relationship_score)
        return total_loss

class HFForSequenceClassification(DANetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = HFBaseModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, label=None, attention_mask=None, 
                token_type_ids=None, checkpoint_activations=False):
        _, pooled_output = self.bert(
            input_ids, attention_mask, token_type_ids
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), label.view(-1))
            if not self.training:
                return loss, logits
            return loss
        return logits

class HFForRegression(DANetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = HFBaseModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, label=None, attention_mask=None,
                token_type_ids=None, checkpoint_activations=False):
        _, pooled_output = self.bert(
            input_ids, attention_mask, token_type_ids
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)

        if label is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), label.to(logits.dtype).view(-1))
            if not self.training:
                return loss, logits
            return loss
        return logits

class HFForAANMatching(DANetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = HFBaseModel(config)
        self.dense = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU(approximate='tanh')
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, input_ids2, attention_mask=None,
                attention_mask2=None, label=None, token_type_ids=None,
                checkpoint_activations=False):
        _, pooled1 = self.bert(input_ids, attention_mask, token_type_ids)
        _, pooled2 = self.bert(input_ids2, attention_mask2, token_type_ids)
        
        hidden_states = torch.cat(
            [pooled1, pooled2, pooled1 * pooled2, pooled1 - pooled2],
            dim=-1
        )
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), label.view(-1))
            if not self.training:
                return loss, logits
            return loss
        return logits