# coding=utf-8
#
# Copyright (c) 2022 Alibaba.com, Inc. All Rights Reserved
"""
eval_metric_model.py

Authors: tjf141457 (tjf141457@alibaba-inc.com)
"""

from scipy.stats import pearsonr, spearmanr
from icecream import ic
import logging
import json
import sys
from tqdm import tqdm
from xdpx.options import Options, Argument, Arg
from xdpx.bootstrap import bootstrap
from xdpx.utils.chat_serve import QARankModel, EnsembleModel
from xdpx.utils import io

'''
usage: x-script eval_open_dialog <test_file_path>
'''
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def score_qa(query, response, rerank_func):
    search_results = [{'snippet': response}]
    results = rerank_func(query, search_results)
    score = results[0]['qa_score']
    return score


def pearson_and_spearman(preds, labels):
    pearson_corr = float(pearsonr(preds, labels)[0])
    spearman_corr = float(spearmanr(preds, labels)[0])
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
    }


DEFAULT_SINGLE_TURN_QA_FILE = 'junfeng_single_turn_qa.json'
DEFAULT_SELFCHAT_FILE = 'junfeng_selfchat.json'
DEFAULT_CCONV_FILE = 'human_score.0706.json'


def load_single_turn_qa_data(file=None):
    """ write file to log file (for emsemble)
    data format: list of dict(history, utterance, response, human_score)
    """
    if not file:
        file = DEFAULT_SINGLE_TURN_QA_FILE
    output_data = []
    with io.open(file) as f:
        data = json.load(f)
        for item in data:
            output_data.append({
                'history': [],
                'utterance': item['utterance'],
                'response': item['response'],
                'human_score': item['sensibility']
            })
    return output_data


def load_selfchat_data(file=None):
    if not file:
        file = DEFAULT_SELFCHAT_FILE
    output_data = []
    with io.open(file) as f:
        data = json.load(f)
        for item in data:
            dialog = item['dialog']
            if len(dialog) >= 6: dialog = dialog[:6]
            output_data.append({
                'history': dialog,
                'utterance': dialog[-2],
                'response': dialog[-1],
                'human_score': item['sensibility']
            })
    return output_data


def load_cconv_data(file=None):
    if not file:
        file = DEFAULT_CCONV_FILE
    output_data = []
    with io.open(file) as f:
        data = json.load(f)
        for item in data:
            score = 0
            if item['sensibility'] == '是' and item['specificity'] == '是':
                score = 1
            output_data.append({
                'history': item['context'],
                'utterance': item['utterance'],
                'response': item['response'],
                'human_score': score
            })
    return output_data


def cli_main(argv):
    assert len(argv) <= 3, 'x-script eval_open_dialog <metric_model_dir> <rerank_model_type>'
    metric_model_dir = argv[1] if len(
        argv) > 1 else 'TODO'
    rerank_model_type = argv[2] if len(
        argv) > 2 else 'rank_qa'

    metric_model = QARankModel(metric_model_dir)
    use_ensemble = True
    if use_ensemble:
        metric_model = EnsembleModel()
        rerank_model_type = 'rank_qa'

    data1 = load_single_turn_qa_data()
    data2 = load_selfchat_data()
    data3 = load_cconv_data()

    datas = {'qa':data1, 'selfchat': data2, 'cconv': data3}
    rerank_func =  metric_model.rerank if rerank_model_type == 'rank_qa' else metric_model.rerank2
    for name, data in datas.items():
        for d in tqdm(data):
            context = d['history']
            query = d['utterance']
            response = d['response']
            if name == 'selfchat':
                dialog = d['history']
                scores = []
                for i in range(len(dialog)-1):
                    query, response = dialog[i], dialog[i+1]
                    _score = score_qa(query, response, rerank_func)
                    scores.append(_score)
                d['score'] = sum(scores) / len(scores)
            else:
                d['score'] = score_qa(query, response, rerank_func)
        scores = [d['score'] for d in data]
        final_score = sum(scores) / len(scores)
        print(f'samples cnt: {len(scores)} \t final_score: {final_score}')

        gold_scores = [d['human_score'] for d in data]
        results = pearson_and_spearman(scores, gold_scores)
        ic(results)


if __name__ == '__main__':
    cli_main(sys.argv[1:])