import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel


class BertHFForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a ModelConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

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

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = ModelConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, args):
        super(BertHFForSequenceClassification, self).__init__()
        self.config = config
        self.num_labels = args.num_labels
        self.window_size = config.window_size
        self.bert = BertModel.from_pretrained("google-bert/bert-large-uncased")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels,
                                    bias=config.classifier_bias)
        """
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=args.num_labels)
        self.classifier = self.cls.seq_relationship
        """

        self.classifier.apply(self.init_bert_weights)
        self.use_local_attention = config.local_attention

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        std = self.config.initializer_range  # / math.sqrt(3)
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0, std=std)
        # if isinstance(module, nn.Linear):
        elif isinstance(module, nn.Linear):
            module.weight.data.uniform_(-std, std)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self,
                input_ids,
                label=None,
                attention_mask=None,
                token_type_ids=None,
                checkpoint_activations=False):
        _, pooled_output = self.bert(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask,
                                     return_dict=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if label is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
            if not self.training:
                return loss, logits
            return loss
        else:
            return logits
