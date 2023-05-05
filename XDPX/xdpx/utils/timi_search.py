from collections import OrderedDict
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient

import os
import traceback
import torch
from xdpx.tasks import tasks
from xdpx.loaders import loaders
from xdpx.utils import io, move_to_cuda, parse_model_path
from xdpx.options import Options, Arg
import logging
from tqdm import tqdm
from typing import List

from xdpx.models.fewshot.mgimn import BertMatchingNetBase
import torch.nn.functional as F

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.WARN)

POC_OFFICIAL_Q_INDEX = 'poc_official_q_index'
POC_OFFICIAL_Q_INDEX_CHILD = 'poc_official_q_index_child'
POC_CUSTOM_Q_INDEX = 'poc_custom_q_index'
POC_CUSTOM_Q_INDEX_CHILD = 'poc_custom_q_index_child'
POC_SOLUTION_INDEX = 'poc_solution_index'

client = Elasticsearch(
    ['es-cn-tl32oudq1000zalhv.public.elasticsearch.aliyuncs.com'],
    http_auth=('elastic', 'virtualHum123'),
    port=9200,
    use_ssl=False
)
ic = IndicesClient(client)
client.info()


def search_official(question, package_names=['行业通用', '美妆行业'], size=5):
    query = {
        "bool": {
            "must": [
                {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "package_name": package_name
                                }
                            } for package_name in package_names
                        ]
                    }
                },
                {
                    "bool": {
                        "should": {
                            "match": {
                                "question": question
                            }
                        }
                    }
                }
            ]
        }
    }
    ret = client.search(query=query, index=POC_OFFICIAL_Q_INDEX, size=100)
    hits = ret['hits']['hits']
    results = OrderedDict()

    for h in hits:
        item = h['_source']
        question_id = item['question_id']
        if question_id in results:
            continue
            if len(results[question_id]['question']) <=3:
                results[question_id]['question'] += [item['question']]
        else:
            item['question'] = [item['question']]
            results[question_id] = item
    results = [v for k, v in results.items()][:size]
    top_n_keys = [t['scene_key'] for t in results]
    return top_n_keys, results


def search_custom(question, seller_nick='百草味旗舰店', size=5):
    query = {
        "bool": {
            "must": [
                {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "seller_nick": seller_nick
                                }
                            }
                        ]
                    }
                },
                {
                    "bool": {
                        "should": {
                            "match": {
                                "question": question
                            }
                        }
                    }
                }
            ]
        }
    }
    ret = client.search(query=query, index=POC_CUSTOM_Q_INDEX, size=100)
    hits = ret['hits']['hits']
    results = OrderedDict()

    for h in hits:
        item = h['_source']
        question_id = item['question_id']
        if question_id in results:
            continue
            if len(results[question_id]['question']) < 3:
                results[question_id]['question'] += [item['question']]
        else:
            item['question'] = [item['question']]
            results[question_id] = item
    results = [v for k, v in results.items()][:size]
    top_n_qids = [t['question_id'] for t in results]
    return top_n_qids, results


def search_solution(question=None, question_ids=None, seller_nick='百草味旗舰店', item_id="0", size=5):
    must_query = [
        {
            "term": {
                "item_id": item_id
            }
        },
        {
            "term": {
                "seller_nick": seller_nick
            }
        },
    ]
    if question_ids is not None:
        must_query.append(
            {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "question_id": question_id
                            }
                        } for question_id in question_ids
                    ]
                }
            }
        )

    query = {
        "bool": {
            "must":  must_query
        }
    }
    if question is not None:
        solution_query = {
            "bool": {
                "should": {
                    "match": {
                        "solution": question
                    }
                }
            }
        }
        query['bool']['must'].append(solution_query)

    ret = client.search(query=query, index=POC_SOLUTION_INDEX, size=size)
    hits = ret['hits']['hits']
    results = [h['_source'] for h in hits]

    return results


class FewShotModel:
    def __init__(self, save_dir, checkpoint=None):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.cuda else torch.device('cpu')

        with io.open(os.path.join(save_dir, 'args.py')) as f:
            args = Options.parse_tree(eval(f.read()))
        try:
            with io.open(os.path.join(args.data_dir, 'args.py')) as f:
                args = Arg().update(Options.parse_tree(eval(f.read()))).update(args)
        except IOError:
            pass
        args.__cmd__ = 'serve'
        args.save_dir = save_dir
        args.temperature = 1.0
        # build the task
        task = tasks[args.task](args)
        model = task.build_model(args)
        model_path = checkpoint if checkpoint else parse_model_path('<best>', args)
        model.load(model_path)

        if self.cuda:
            model = model.cuda()

        self.processor = task.processor
        self.loader = loaders[args.loader](args)
        self.model = model
        self.args = args

    def rerank(self, query, search_results: List):
        if not search_results:
            return []
        dic = {}
        for s in search_results:
            for q in s['question']:
                label = s['question_id']
                questions = dic.get(label, [])
                dic[label] = questions + [q]

        support_labels = list(dic.keys())
        try:
            support_set = [[self.loader.tokenizer.encode(text) for text in dic[label]] for label in support_labels]
            query = [self.loader.tokenizer.encode(query)]
            inputs = {'id': [0], 'support': support_set, 'domain': 'default',
                      'query': query, 'mode': 'retrieval'}
            episode = [self.processor.numerize(inputs)]
            episode = self.processor.collate(episode)  # 1 * K * max_seq
            if self.cuda:
                episode = move_to_cuda(episode)

            self.model.eval()
            with torch.no_grad():
                probs = self.model(**episode['net_input'])[0].tolist()[0]
                label_probs = dict(zip(support_labels, probs))
                for t in search_results:
                    t['qq_score'] = label_probs[t['question_id']]
                results = sorted(search_results, key=lambda x: x['qq_score'], reverse=True)
                return results
        except Exception as e:
            traceback.print_exc()

class QARankModel:
    def __init__(self, save_dir, checkpoint=None):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.cuda else torch.device('cpu')

        with io.open(os.path.join(save_dir, 'args.py')) as f:
            args = Options.parse_tree(eval(f.read()))
        try:
            with io.open(os.path.join(args.data_dir, 'args.py')) as f:
                args = Arg().update(Options.parse_tree(eval(f.read()))).update(args)
        except IOError:
            pass
        args.__cmd__ = 'serve'
        args.save_dir = save_dir
        args.temperature = 1.0
        # build the task
        task = tasks[args.task](args)
        model = task.build_model(args)
        model_path = checkpoint if checkpoint else parse_model_path('<best>', args)
        model.load(model_path)

        if self.cuda:
            model = model.cuda()

        self.processor = task.processor
        self.loader = loaders[args.loader](args)
        self.model = model
        self.args = args

    def rerank(self, query, search_results: List):
        if not search_results:
            return []
        batch = []
        for s in search_results:
            batch.append([query, s['solution']])
        try:
            batch = [self.processor.encode(self.loader, sample) for sample in batch]
            batch = self.processor.collate(batch)

            if self.cuda:
                batch = move_to_cuda(batch)

            self.model.eval()
            with torch.no_grad():
                z1, z2 = self.model(**batch['net_input'])
                cos_sim = F.cosine_similarity(z1, z2, dim=-1, eps=1e-4).tolist() #batch_size
                for i, t in enumerate(search_results):
                    t['qa_score'] = cos_sim[i]
                results = sorted(search_results, key=lambda x: x['qa_score'], reverse=True)
                return results
        except Exception as e:
            traceback.print_exc()

def search_timi(question, item_id='0', seller_nick='百草味旗舰店', package_names=['行业通用', '零食行业'], size=3,
                qq_rerank_model=None,qa_rerank_model=None, min_rerank_score=0.4):
    from icecream import ic as icp
    o_qs = search_official(question, package_names=package_names, size=size)[1]
    c_qs = search_custom(question, seller_nick=seller_nick, size=size)[1]

    questions = o_qs + c_qs
    if qq_rerank_model is not None:
        questions = qq_rerank_model.rerank(question, questions)
        questions = [t for t in questions if t['qq_score'] > min_rerank_score]


    qids = [h['question_id'] for h in questions]
    qid_probs = {h['question_id']: h['qq_score'] for h in questions}

    if item_id == '0':
        solutions = search_solution(question=question, question_ids=qids, seller_nick=seller_nick, item_id=item_id,
                                    size=10)
    else:
        # 优先使用关联item的答案
        solutions_with_specific_item = search_solution(question=question, question_ids=qids, seller_nick=seller_nick,
                                                       item_id=item_id,
                                                       size=10)
        qids_with_specific_solution = [s['question_id'] for s in solutions_with_specific_item]
        qids_with_no_specific_solution = [qid for qid in qids if qid not in qids_with_specific_solution]
        solutions_general = []
        if qids_with_no_specific_solution:
            solutions_general = search_solution(question=question, question_ids=qids_with_no_specific_solution,
                                                seller_nick=seller_nick, item_id='0',
                                                size=10)
        solutions = solutions_with_specific_item + solutions_general

    solutions = sorted(solutions, key=lambda x: qid_probs.get(x['question_id'], 0.0), reverse=True)
    if qa_rerank_model is not None:
        solutions = qa_rerank_model.rerank(question, solutions)
    if solutions:
        solutions = solutions[:1]

    return questions, solutions
