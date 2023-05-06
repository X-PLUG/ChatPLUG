from collections import OrderedDict
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
import requests
import json
import logging

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.WARN)

client = Elasticsearch(
    ['es-cn-tl32oudq1000zalhv.public.elasticsearch.aliyuncs.com'],
    http_auth=('elastic', 'virtualHum123'),
    port=9200,
    use_ssl=False
)
ic = IndicesClient(client)
# client.info()

JILI_FAQ_Q_INDEX = 'jili_faq_q_index'
JILI_FAQ_A_INDEX = 'jili_faq_a_index'
JILI_SHOUCE_INDEX = 'jili_shouce_index'


def search_faq(question, size=5):
    query = {
        "bool": {
            "must": [
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
    ret = client.search(query=query, index=JILI_FAQ_Q_INDEX, size=100)
    hits = ret['hits']['hits']
    results = OrderedDict()

    for h in hits:
        item = h['_source']
        question_id = item['question_id']
        if question_id in results:
            if len(results[question_id]['question']) < 3:
                results[question_id]['question'] += [item['question']]
        else:
            item['question'] = [item['question']]
            results[question_id] = item
    results = [v for k, v in results.items()][:size]
    top_n_qids = [t['question_id'] for t in results]
    return top_n_qids, results


def search_faq_answer(question=None, question_ids=None, size=5):
    must_query = []
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
            "must": [
                {
                    "bool": {
                        "must": must_query
                    }
                }
            ]
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

    ret = client.search(query=query, index=JILI_FAQ_A_INDEX, size=size)
    hits = ret['hits']['hits']
    results = [h['_source'] for h in hits]

    return results


from collections import OrderedDict


def search_shouce(question=None, size=5):
    query = {
        "bool": {
            "must": [
                {
                    "bool": {
                        "should": {
                            "match": {
                                "content": question
                            }
                        }
                    }
                }
            ]
        }
    }
    ret = client.search(query=query, index=JILI_SHOUCE_INDEX, size=size)
    hits = ret['hits']['hits']
    results = [h['_source'] for h in hits]

    return results


def rerank_faq(question, faq_search_results, min_score=0.9):
    param = {
        "query": question,
        "idKnowledgeMap": {}
    }
    for h in faq_search_results:
        question_id = h['question_id']
        questions = h['question']
        for i, q in enumerate(questions):
            key = f'{question_id}_{i}'
            param['idKnowledgeMap'][key] = q

    url = 'http://1129308980554125.cn-shanghai.pai-eas.aliyuncs.com/api/predict/deepqav3_0000_v2'
    headers = {"Authorization": "NzNmZDJiYmMwMTI5ZGJlZDJlNjBiZDk1N2MzYzdlMGY2ZjM0MWMwOA=="}

    try:
        response = requests.post(url, data=json.dumps(param), headers=headers)
        items = json.loads(response.text)['scores']
        scores = {}
        for t in items:
            id = int(t['bizId'].split('_')[0])
            scores[id] = max(t['score'], scores.get(id, 0.0))
        for h in faq_search_results:
            h['q_score'] = scores.get(h['question_id'], 0.0)
        faq_search_results = [h for h in faq_search_results if h['q_score'] > min_score]
    except:
        print(f'| warning: invoke cloud fewshot exception')
    return faq_search_results


def search_geely(question, global_var='星越', size=3):
    results = []
    faq_search_results = search_faq(question, size=size)[1]
    faq_search_results = rerank_faq(question, faq_search_results, min_score=0.8)
    if not faq_search_results:
        return []

    qids = [h['question_id'] for h in faq_search_results]
    qid_question_map = {h['question_id']: h['question'][0] for h in faq_search_results}
    qid_score_map = {h['question_id']: h['q_score'] for h in faq_search_results}

    answers = search_faq_answer(question=None, question_ids=qids, size=3)
    for answer in answers:
        if answer['global_var'] == global_var:
            qid = answer['question_id']
            answer_text = answer['answer']
            passage = qid_question_map.get(qid, '') + '？ ' + answer_text
            results.append({
                'source': answer['source'],
                'snippet': passage,
                'q_score': qid_score_map.get(qid)
            })

    shouce_search_results = search_shouce(question=question, size=3)
    for p in shouce_search_results:
        if p['global_var'] == global_var:
            results.append({
                'source': p['source'],
                'snippet': p['content']
            })

    return results
