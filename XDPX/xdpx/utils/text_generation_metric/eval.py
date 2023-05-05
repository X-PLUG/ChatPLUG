# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module computes evaluation metrics for DuReader dataset.
"""

from xdpx.utils.text_generation_metric.bleu_metric.bleu import Bleu
from xdpx.utils.text_generation_metric.rouge_metric.rouge import Rouge
from pytorch_pretrained_bert import BasicTokenizer
import codecs
import json

def normalize(s,tokenizer:BasicTokenizer=None):
    """
    Normalize strings to space joined chars.

    Args:
        s: strings.

    Returns:
        A list of normalized strings.
    """
    if not s:
        return [""]
    if not tokenizer:
        tokens = [c for c in list(s) if len(c.strip()) != 0]
    else:
        tokens = tokenizer.tokenize(s)
    return [' '.join(tokens)]

def compute_bleu_rouge(pred_dict, ref_dict, bleu_order=4, use_tokenizer=True):
    """
    Compute bleu and rouge scores.
    pred_dict: {'key':"a sentence一个句子",......}
    ref_dict: {'key':"a sentence一个句子",......}
    """
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
            "missing keys: {}".format(set(ref_dict.keys()) - set(pred_dict.keys()))
    # normalize
    tokenizer = None
    if use_tokenizer:
        tokenizer = BasicTokenizer()
    norm_pred_dict = {}
    norm_ref_dict = {}
    for key in pred_dict.keys():
        norm_pred_dict[key] = normalize(pred_dict[key], tokenizer)
        norm_ref_dict[key] = normalize(ref_dict[key], tokenizer)

    scores = {}
    bleu_scores, _ = Bleu(bleu_order).compute_score(norm_ref_dict, norm_pred_dict)
    for i, bleu_score in enumerate(bleu_scores):
        scores['Bleu-%d' % (i + 1)] = bleu_score
    rouge_score, _ = Rouge().compute_score(norm_ref_dict, norm_pred_dict)
    scores['Rouge-L'] = rouge_score
    unigram_p,unigram_r,unigram_f1 = compute_unigram_prf(norm_pred_dict, norm_ref_dict)
    scores["Unigram_P"] = unigram_p
    scores["Unigram_R"] = unigram_r
    scores["Unigram_F1"] = unigram_f1

    return scores

def compute_unigram_prf(pred_dict, ref_dict):

    def cal_intersect_count(pred,ref):
        pred_char_count_dict = {}
        for char in pred:
            if char not in pred_char_count_dict:
                pred_char_count_dict[char]=1
            else:
                pred_char_count_dict[char] += 1
        ref_char_count_dict = {}
        for char in ref:
            if char not in ref_char_count_dict:
                ref_char_count_dict[char] = 1
            else:
                ref_char_count_dict[char] += 1
        intersect_count = 0
        for char in pred_char_count_dict:
            if char not in ref:
                continue
            intersect_count+=min(pred_char_count_dict[char],ref_char_count_dict[char])
        return intersect_count

    precision_score = []
    recall_score = []
    f1_score = []
    for key in pred_dict.keys():
        assert type(pred_dict[key])==list
        assert type(ref_dict[key]) == list
        pred = pred_dict[key][0].split(" ")
        ref = ref_dict[key][0].split(" ")
        intersect_count = cal_intersect_count(pred,ref)
        precision = intersect_count/(len(pred)+1e-9)
        recall = intersect_count/(len(ref)+1e-9)
        if precision+recall==0:
            f1=0.0
        else:
            f1 = 2*precision*recall/(precision+recall)
        precision_score.append(precision)
        recall_score.append(recall)
        f1_score.append(f1)
    avg_precision = sum(precision_score)/len(precision_score)
    avg_recall = sum(recall_score)/len(recall_score)
    ave_f1 = sum(f1_score)/len(f1_score)

    return avg_precision,avg_recall,ave_f1


def plug_eval_predict_file(file_name):
    pred_dict = {}
    ref_dict = {}
    with codecs.open(file_name) as f:
        ""
        data = json.load(f)
        for i,item in enumerate(data):
            pred = item["pred"]
            ref = item["gold"]
            pred_dict[str(i)] = pred
            ref_dict[str(i)] = ref
    result = compute_bleu_rouge(pred_dict,ref_dict)
    print(result)
    result = compute_answer_acc(pred_dict,ref_dict)
    print(result)


def compute_answer_acc(pred_dict, ref_dict):
    """
    Compute bleu and rouge scores.
    pred_dict: {'key':"a sentence一个句子",......}
    ref_dict: {'key':"a sentence一个句子",......}
    """
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
            "missing keys: {}".format(set(ref_dict.keys()) - set(pred_dict.keys()))
    # normalize
    all_count = len(pred_dict)
    hit_count = 0
    for key in pred_dict.keys():
        pred = pred_dict[key]
        answer_list = ref_dict[key].split('|')
        for an in answer_list:
            if an in pred:
                hit_count+=1
                break
    score = hit_count/all_count
    return {'answer_acc':score}

if __name__ == '__main__':
    ""
    plug_eval_predict_file("dialog_data/result_common_qa_test_plug128.json")
