import torch
from torch import nn

from utils.constants import *


class BertClassifier(nn.Module):

    def __init__(self, pretrained_bert,
                 bert_model,
                 dropout=0.1):

        super(BertClassifier, self).__init__()

        # Download pretrained model
        self._bert = pretrained_bert
        self._dropout = nn.Dropout(dropout)

        bert_output = BERT_OUTPUT[bert_model]
        self._classifier = nn.Linear(bert_output, 1)

    def forward(self, input_id, mask):

        _, pooled_output = self._bert(input_ids=input_id,
                                      attention_mask=mask,
                                      return_dict=False)
        dropout_output = self._dropout(pooled_output)
        classifier_output = self._classifier(dropout_output)
        return torch.sigmoid(classifier_output)
