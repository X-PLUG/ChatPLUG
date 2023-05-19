import tempfile
import os
import re
from typing import Tuple, List

import torch
from xdpx.utils import io
from transformers import BertTokenizer, BertConfig, BertModel, AutoConfig
from torch import nn
import torch.nn.functional as F
from xdpx.utils.chat import DEVICE, text_is_question
import jieba
import jieba.analyse
from xdpx.utils.chat import is_special_skill, is_persona_question
from xdpx.utils.chat.base import CHITCHAT_QUERY, BAIKEQA_QUERY, HistoryItem


class BertForClassification(nn.Module):
    def __init__(self, config: BertConfig, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, *args, **kwargs):
        # pooled output as default
        features = self.bert(*args, **kwargs)[1]
        features = self.dropout(features)
        return self.classifier(features)


class QueryClassifier:
    def __init__(self, classifier_model):
        # step 1. init argument and prepare model file
        assert io.isfile(classifier_model)
        model_dir = "/".join(classifier_model.split("/")[:-1])
        temp_dir = tempfile.mkdtemp()
        print(f'| copy vocab.txt, config.json from {model_dir} to {temp_dir}')
        vocab_file_path = os.path.join(model_dir, "vocab.txt")
        io.copy(vocab_file_path, os.path.join(temp_dir, "vocab.txt"))
        config_file_path = os.path.join(model_dir, "config.json")
        io.copy(config_file_path, os.path.join(temp_dir, "config.json"))
        model_dir = temp_dir

        # step 2. pload finetuned model and tokenizer
        print(f'| create tokenzier from pretrained {model_dir}')
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

        print(f'| loading query_classifier model from {classifier_model}')
        with io.open(classifier_model, 'rb') as state_dict:
            self.model = BertForClassification(AutoConfig.from_pretrained(os.path.join(model_dir, "config.json")),
                                               num_labels=2)
            self.model.load_state_dict(torch.load(state_dict, map_location=DEVICE))
        print(f'| loading success.')
        self.model = self.model.to(DEVICE)
        self.model.eval()

    def predict(self, utterance, threshold=0.99):
        input_ids = self.tokenizer(utterance, padding=True, truncation=True, max_length=200,
                                   return_tensors="pt").input_ids
        input_ids = input_ids.to(DEVICE)
        logits = self.model(input_ids)
        logits_softmax = F.softmax(logits, dim=-1).tolist()
        label = CHITCHAT_QUERY if logits_softmax[0][1] > threshold else BAIKEQA_QUERY
        return label


class BaseLearn2Search(object):
    def __init__(self):
        print(f'| skip query_classifier.')    
        self.query_classifier = None

    def need_search(self, query: str) -> Tuple[bool, str]:
        return text_is_question(query) and not is_persona_question(query), CHITCHAT_QUERY
    
    def get_search_query(self, query: str, history: List[HistoryItem]):
        # only use the last query
        return query


class Learn2Search(object):
    def __init__(self, query_classifier_path):
        print(f'| init learn2search module..')
        if query_classifier_path:
            print(f'| create query classifier..')
            self.query_classifier = QueryClassifier(query_classifier_path)
        else:
            print(f'| skip query_classifier.')
            self.query_classifier = None

    def need_search(self, query: str) -> Tuple[bool, str]:
        query_label = self.query_classifier.predict(
            query) if self.query_classifier is not None else None
        need_search = not is_persona_question(query) and (
                is_special_skill(query)
                or self.query_classifier is None
                or query_label == BAIKEQA_QUERY
        )
        return need_search, query_label

    def get_search_query(self, query: str, history: List[HistoryItem]):
        '''

        Args:
            query: query after rewriting
            history: history utterances after rewriting

        Returns:

        '''
        # if history is not None and len(history) > 0:
        #     if text_is_question(query) \
        #             or is_special_skill(query) \
        #             or len(query) < 8:
        #         search_query = query
        #     else:  # TODO:
        #         rewritten_session = [h.rewritten_utterance or h.utterance for h in history[-4:]] + [query]
        #         if len(query) > 10:
        #             keywords = jieba.analyse.extract_tags('\n'.join(rewritten_session[-5:]), topK=3)
        #             search_query = ' '.join(keywords)
        #         else:
        #             keywords = jieba.analyse.extract_tags('\n'.join(rewritten_session[-5:-1]), topK=2)
        #             keywords = [k for k in keywords if k not in query]
        #             search_query = ' '.join(keywords) + query
        # else:
        #     search_query = query
        search_query = query
        # recent_re = re.compile('(最新|最近)')
        # if recent_re.findall(query):
        #     recent_word = recent_re.findall(query)[0]
        #     search_query = query.replace(recent_word, "2022")
        #     search_query = query + "2022"
        return search_query

def learn_to_search(query):
    search_query = query
    recent_re = re.compile('(最新|最近)')
    if recent_re.findall(query):
        recent_word = recent_re.findall(query)[0]
        search_query = query + "2022"
    return search_query

if __name__ == '__main__':
    while True:
        query = input("输入：")
        print(learn_to_search(query))