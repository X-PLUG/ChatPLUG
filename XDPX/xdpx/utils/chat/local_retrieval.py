'''
local information retrieval
including faq, document, triples, ds intents
'''
from typing import List, Tuple, Optional

from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from elasticsearch import helpers
import requests
import json
import logging
from collections import OrderedDict, Iterable
from dataclasses import dataclass, field

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.WARN)

LOCAL_IR_FAQ_Q_INDEX = 'local_ir_faq_q_index'
LOCAL_IR_FAQ_A_INDEX = 'local_ir_faq_a_index'
LOCAL_IR_DOC_PASSAGE_INDEX = 'local_ir_doc_passage_index'


@dataclass
class FAQ:
    question_id: int
    question: str
    answer: str
    tenant: str = field(default=None)
    q_score: float = field(default=0)
    ontology: Optional[str] = field(default=None)
    category: Optional[str] = field(default=None)
    similar_questions: Optional[List[str]] = field(default_factory=list)


class LocalRetrieval(object):
    def __init__(self, host, user, password, faq_q_threshold=0.95, faq_q_threshold2=0.9):
        self.client = Elasticsearch(
            [host],
            http_auth=(user, password),
            port=9200,
            use_ssl=False
        )
        self.indices_client = IndicesClient(self.client)
        self.faq_q_threshold = faq_q_threshold
        self.faq_q_threshold2 = faq_q_threshold2
        from xdpx.utils.chat.chat_skills import ChatSkills
        self.special_answers = list(ChatSkills.chat_skills.keys())

    def add_faq_to_index(self, local_faqs: List[FAQ]):
        items = []
        for local_faq in local_faqs:
            item = {
                "_index": LOCAL_IR_FAQ_Q_INDEX,
                '_source': {
                    'question_id': local_faq.question_id,
                    'question': local_faq.question,
                    'ontology': local_faq.ontology,
                    'category': local_faq.category,
                    'tenant': local_faq.tenant
                }
            }
            items.append(item)
            for simq in local_faq.similar_questions:
                item = {
                    "_index": LOCAL_IR_FAQ_Q_INDEX,
                    '_source': {
                        'question_id': local_faq.question_id,
                        'question': simq,
                        'ontology': local_faq.ontology,
                        'category': local_faq.category,
                        'tenant': local_faq.tenant
                    }
                }
                items.append(item)
            if len(items) > 1000:
                helpers.bulk(self.client, items)
                items = []
        if items:
            helpers.bulk(self.client, items)

        for local_faq in local_faqs:
            item = {
                "_index": LOCAL_IR_FAQ_A_INDEX,
                '_source': {
                    'question_id': local_faq.question_id,
                    'tenant': local_faq.tenant,
                    'answer': local_faq.answer
                }
            }
            items.append(item)
            if len(items) > 1000:
                helpers.bulk(self.client, items)
                items = []
        if items:
            helpers.bulk(self.client, items)

    def create_faq_indexs(self, remove_if_exists=False):
        ignore_faq_q_index = False
        ignore_faq_a_index = False
        if self.indices_client.exists(index=LOCAL_IR_FAQ_Q_INDEX):
            if remove_if_exists:
                self.indices_client.delete(index=LOCAL_IR_FAQ_Q_INDEX)
            else:
                print(f'| {LOCAL_IR_FAQ_Q_INDEX} existed.')
                ignore_faq_q_index = True
        if self.indices_client.exists(index=LOCAL_IR_FAQ_A_INDEX):
            if remove_if_exists:
                self.indices_client.delete(index=LOCAL_IR_FAQ_A_INDEX)
            else:
                print(f'| {LOCAL_IR_FAQ_A_INDEX} existed.')
                ignore_faq_a_index = True

        if not ignore_faq_q_index:
            self.indices_client.create(index=LOCAL_IR_FAQ_Q_INDEX, body={
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "my_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase"
                                ]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "tenant": {"type": "keyword"},
                        "question_id": {"type": "keyword"},
                        "category": {"type": "keyword"},
                        "ontology": {"type": "keyword"},
                        "question": {
                            "type": "text",
                            "analyzer": "my_analyzer"
                        }
                    }
                }
            })
        if not ignore_faq_a_index:
            self.indices_client.create(index=LOCAL_IR_FAQ_A_INDEX, body={
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "my_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": [
                                    "lowercase"
                                ]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "question_id": {"type": "keyword"},
                        "tenant": {"type": "keyword"},
                        "answer": {
                            "type": "text",
                            "analyzer": "my_analyzer"
                        }
                    }
                }
            })

    def faq_q_retrieval(self, tenant, question, size=5, must_fields=None):
        '''

        Args:
            tenant:
            question:
            size:
            must_fields: extra fields like item_id, ontology_word

        Returns:
            list of faq questions,
            each faq fields include question_id, question, tenant,..
        '''
        query = {
            "bool": {
                "must": [
                    {"match": {"tenant": tenant}},
                    {"match": {"question": question}}
                ]
            }
        }
        if must_fields:
            for k, v in must_fields.items():
                query['bool']['must'].append({"match": {k: v}})
        ret = self.client.search(query=query, index=LOCAL_IR_FAQ_Q_INDEX, size=100)
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
        return results

    def faq_a_retrieval(self, tenant, question_ids=None, size=5, must_fields=None):
        '''

        Args:
            tenant:
            question_ids:
            size:
            must_fields: extra fields like item_id, ontology_word

        Returns:
        '''
        if not isinstance(question_ids, Iterable):
            question_ids = [question_ids]
        query = {
            'bool': {
                'must':
                    [
                        {'terms': {"question_id": question_ids}},
                        {'match': {'tenant': tenant}},
                    ]
            }
        }

        if must_fields:
            for k, v in must_fields.items():
                query['bool']['must'].append({"match": {k: v}})
        ret = self.client.search(query=query, index=LOCAL_IR_FAQ_A_INDEX, size=size)
        hits = ret['hits']['hits']
        results = [h['_source'] for h in hits]
        return results

    def doc_passage_retrieval(self, tenant, question=None, size=5):
        '''

        Args:
            tenant:
            question:
            size:

        Returns:

        '''
        query = {
            "bool": {
                "must": [
                    {"match": {"tenant": tenant}},
                    {"match": {"passage": question}}
                ]
            }
        }
        ret = self.client.search(query=query, index=LOCAL_IR_DOC_PASSAGE_INDEX, size=size)
        hits = ret['hits']['hits']
        results = [h['_source'] for h in hits]
        return results

    def faq_q_rerank_and_filter(self, question, faq_retrieval_qs):
        """ TODO: implement faq rerank and filter process.

        Args:
            question: str
            faq_retrieval_qs: List[str]
            min_score: float

        Returns:
            faq_retrieval_qs: List[str]
        """
        
        return faq_retrieval_qs

    def local_faq_retrieval(self, tenant, question, size=3, must_q_fields=None, must_a_fields=None) -> Tuple[
        Optional[FAQ], List[FAQ]]:
        faq_retrieval_qs = self.faq_q_retrieval(tenant, question, size=size, must_fields=must_q_fields)
        faq_retrieval_qs = self.faq_q_rerank_and_filter(question, faq_retrieval_qs)
        if not faq_retrieval_qs:
            return None, []
        qids = [h['question_id'] for h in faq_retrieval_qs]
        # qid_question_map = {h['question_id']: h['question'][0] for h in faq_retrieval_qs}
        # qid_score_map = {h['question_id']: h['q_score'] for h in faq_retrieval_qs}
        # qid_ontology_map = {h['question_id']: h['ontology'] for h in faq_retrieval_qs}
        # qid_category_map = {h['question_id']: h['category'] for h in faq_retrieval_qs}
        qid_item_map = {h['question_id']: h for h in faq_retrieval_qs}
        answers = self.faq_a_retrieval(tenant, question_ids=qids, size=size, must_fields=must_a_fields)
        answers = sorted(answers, key=lambda ans: qid_item_map.get(ans['question_id'], {}).get('q_score'), reverse=True)
        local_faqs = []
        for answer in answers:
            qid = answer['question_id']
            answer_text = answer['answer']
            item = qid_item_map.get(qid, {})
            sim_questions = item.get('question', None)
            question = sim_questions
            if sim_questions:
                question = sim_questions[0]
            # passage = qid_question_map.get(qid,
            #                               '') + '？ ' + answer_text if answer_text not in self.special_answers else answer_text
            passage = answer_text
            ontology = item.get('ontology','')
            category = item.get('category','')
            faq = FAQ(tenant=answer['tenant'],
                      answer=passage,
                      q_score=item.get('q_score', -1),
                      question=question,
                      question_id=qid,
                      ontology=ontology,
                      category=category,
                      similar_questions=sim_questions)
            local_faqs.append(faq)
        direct_faq = None
        if local_faqs and local_faqs[0].q_score > self.faq_q_threshold:
            direct_faq = local_faqs[0]
        return direct_faq, local_faqs


if __name__ == '__main__':
    host = 'HOST'
    user = 'USER'
    password = 'PASSWD'
    local_retrieval = LocalRetrieval(host=host, user=user, password=password)
    faq,local_faqs = local_retrieval.local_faq_retrieval('geely',question='如何自动泊车')
    print(local_faqs)
