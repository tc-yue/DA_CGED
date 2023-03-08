# -*- coding: utf-8 -*-
# @Time    : 2020/08/22 18:24
# @Author  : Tianchiyue
# @File    : recall_model.py
# @Software: PyCharm Community Edition
from collections import OrderedDict
from bert_model import BertModel, BertPreTrainedModel
from torch.nn import MSELoss, CrossEntropyLoss
from torch import nn
import torch
import torch.nn.functional as F
import logging


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_seq_labels = config.num_seq_labels
        self.hidden_size = config.hidden_size
        config.output_hidden_states = True
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.seq_classifier = nn.Linear(self.hidden_size, config.num_seq_labels)
        # if config.multi_task:
        #     self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        if config.cnn:
            self.cnn = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels=self.hidden_size,
                                    out_channels=self.hidden_size,
                                    kernel_size=3,
                                    padding=1)),
                ('relu1', nn.ReLU()),
            ]))

        self.batch_norm = nn.BatchNorm1d(num_features=self.hidden_size,
                                         affine=True)
        self.config = config
        self.init_weights()

        self.smoothing = 0.09
        self.confidence = 1.0 - self.smoothing

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, seq_labels=None,
                weight=None, valid=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_res=self.config.use_res
        )
        sequence_output = outputs[0]  # last hidden states[32,64,768]
        pooled_output = outputs[1]  # pooled through Linear [32,768]
        hidden_states = outputs[2]  # all hidden states 13*[32,64,768]
        # sequence_output = hidden_states[-1]
        sequence_output = self.dropout(sequence_output)
        if self.config.cnn:
            cnn_out = self.cnn(sequence_output.permute(0, 2, 1))
            seq_logits = self.seq_classifier(cnn_out.permute(0, 2, 1))
        else:
            seq_logits = self.seq_classifier(sequence_output)

        loss = None
        if seq_labels is None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = seq_logits.view(-1, self.num_seq_labels)
            log_probs = F.log_softmax(active_logits, dim=-1)
            loss = torch.sum(log_probs * seq_labels, dim=-1)
            sum_loss = torch.where(active_loss, loss.view(-1),
                                   torch.tensor(0).type_as(loss))
            if valid:
                new_loss = torch.sum(sum_loss.view(-1, 64), dim=-1)
                new_attention = torch.sum(attention_mask, dim=-1)
                loss = -1 * torch.div(new_loss, new_attention)
            else:
                sum_loss = torch.sum(sum_loss)
                total = torch.sum(attention_mask)
                loss = -1 * torch.div(sum_loss, total)
        elif seq_labels is not None:
            if self.config.weight_loss:
                weight_CE = torch.FloatTensor([1, 2, 2, 2, 2]).to(
                    attention_mask.device)
                loss_fct = CrossEntropyLoss(weight=weight_CE)
            else:
                # label imbalance not used
                # self.label_list = {"S": 1, "R": 2, "M": 3, "W": 4, "$": 1}
                weight_CE = torch.FloatTensor([1, 1, 1, 1, 1]).to(
                    attention_mask.device)
                loss_fct = CrossEntropyLoss(reduction='mean', weight=weight_CE)
                if valid:
                    loss_fct = CrossEntropyLoss(reduction='none',
                                                weight=weight_CE)
            # Only keep active parts of the loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = seq_logits.view(-1, self.num_seq_labels)
            active_labels = torch.where(
                active_loss, seq_labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(seq_labels)
            )

            loss = loss_fct(active_logits, active_labels)
            if valid:
                new_loss = torch.sum(loss.view(-1, 64), dim=-1)
                new_attention = torch.sum(attention_mask, dim=-1)
                loss = -1 * torch.div(new_loss, new_attention)
        outputs = (loss, seq_logits)
        return outputs
