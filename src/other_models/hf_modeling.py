from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
from src.modeling import *
from operator import attrgetter

class HFConfig:
    def __init__(self, hf_model_name_or_path, **kwargs):
        self.hf_config, unused_kwargs = AutoConfig.from_pretrained(hf_model_name_or_path, **kwargs, return_unused_kwargs=True)
        for param, value in unused_kwargs.items():
            setattr(self, param, value)
    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            pass
        return getattr(self.hf_config, name)

class HFPretrainedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not isinstance(config, HFConfig):
            raise ValueError(
                f"Parameter config in `{HFConfig}(config)` should \
                    be an instance of class `HFConfig`."
            )
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights."""
        std = self.config.initializer_range
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0, std=std)
        elif isinstance(module, nn.Linear):
            module.weight.data.uniform_(-std, std)
            if module.bias is not None:
                module.bias.data.zero_()

class HFAdapter(nn.Module):
    """Core adapter that handles input/output conversion"""
    def __init__(self, config, args=None):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_config(self.config.hf_config)
        self.PATH_TO_LAYERS = "model.encoder.layer"
        self.PATH_TO_EMBEDDINGS = "model.embeddings"

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        input_ids = input_ids[:, :self.config.max_position_embeddings]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            token_type_ids = token_type_ids[:, :self.config.max_position_embeddings]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask[:, :self.config.max_position_embeddings]
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
class HFForPreTraining(HFPretrainedModel):
    def __init__(self, config, args=None):
        super().__init__(config)
        self.PATH_TO_BACKBONE = "backbone"
        self.backbone = HFAdapter(config)
        self.cls = BertPreTrainingHeads(
            config, 
            attrgetter(self.backbone.PATH_TO_EMBEDDINGS)(self.backbone).word_embeddings.weight,
            num_labels=config.num_labels
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.cls.apply(self.init_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                masked_lm_labels=None, label=None, log=True):
        sequence_output, pooled_output = self.backbone(
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

class HFForSequenceClassification(HFPretrainedModel):
    def __init__(self, config, args=None):
        super().__init__(config)
        self.PATH_TO_BACKBONE = "backbone"
        self.backbone = HFAdapter(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier.apply(self.init_weights)

    def forward(self, input_ids, label=None, attention_mask=None, 
                token_type_ids=None, checkpoint_activations=False):
        _, pooled_output = self.backbone(
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

class HFForRegression(HFPretrainedModel):
    def __init__(self, config, args=None):
        super().__init__(config)
        self.PATH_TO_BACKBONE = "backbone"
        self.backbone = HFAdapter(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.regressor.apply(self.init_weights)

    def forward(self, input_ids, label=None, attention_mask=None,
                token_type_ids=None, checkpoint_activations=False):
        _, pooled_output = self.backbone(
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

class HFForAANMatching(HFPretrainedModel):
    def __init__(self, config, args=None):
        super().__init__(config)
        self.PATH_TO_BACKBONE = "backbone"
        self.backbone = HFAdapter(config)
        self.dense = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU(approximate='tanh')
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier.apply(self.init_weights)
        self.dense.apply(self.init_weights)

    def forward(self, input_ids, input_ids2, attention_mask=None,
                attention_mask2=None, label=None, token_type_ids=None,
                checkpoint_activations=False):
        _, pooled1 = self.backbone(input_ids, attention_mask, token_type_ids)
        _, pooled2 = self.backbone(input_ids2, attention_mask2, token_type_ids)
        
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