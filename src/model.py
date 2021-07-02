import torch, json, time
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel, BertLayer, BertConfig, BertForMaskedLM

from transformers import BertForMaskedLM

from config import Config
args = Config()

class baseline(nn.Module):
    def __init__(self, config):
        super(baseline, self).__init__()
        self.num_labels = len(args.id2label) * 2
        self.bert = BertModel.from_pretrained(args.bert_path) # BertModel
        self.classifier = nn.Linear(768, self.num_labels)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.loss_type = 'ce'

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)