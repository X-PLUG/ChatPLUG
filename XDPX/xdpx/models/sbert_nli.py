import os
import json
import torch
import torch.nn as nn
from . import register, Model
from .bert import BertForLanguageModeling, LMHead, Bert, ClassificationHead, BertForClassification
import torch.nn.functional as F


@register('sbert_nli')
class SBertNLIModel(BertForClassification):

    def __init__(self, args):
        super().__init__(args)
        # self.classifier = ClassificationHead(
        #     in_features=self.args.hidden_size * 3,
        #     out_features=args.num_classes,
        # )
        self.classifier = torch.nn.Sequential(
            ClassificationHead(self.args.hidden_size * 3, self.args.hidden_size),
            nn.ReLU(),
            ClassificationHead(self.args.hidden_size, args.num_classes))

    def sent2vec(self, *args, **kwargs):

        return self.bert_forward(*args, **kwargs)[1]

    def get_score(self, sentvec1, sentvec2):

        features = torch.cat([sentvec1, sentvec2, torch.abs(sentvec1-sentvec2)], -1)
        features = F.dropout(features, self.args.output_dropout_prob, self.training)
        return self.classifier(features)

    def forward(self, *args, **kwargs):
        # pooled output as default

        sentvec1 = self.sent2vec(kwargs['tokens1'], kwargs['mask1'])
        sentvec2 = self.sent2vec(kwargs['tokens2'], kwargs['mask2'])

        features = torch.cat([sentvec1, sentvec2, torch.abs(sentvec1-sentvec2)], -1)
        features = F.dropout(features, self.args.output_dropout_prob, self.training)
        return self.classifier(features)


