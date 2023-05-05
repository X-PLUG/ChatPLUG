#!/usr/bin/env python
# encoding: utf-8

"""
@author: tanfan.zjh
@contact: tanfan.zjh@alibaba-inc.com
@software: PyCharm
@time: 2022/9/26 2:34 PM
"""
from typing import List, Any
from dataclasses import dataclass, field


@dataclass
class FewshotConfig:
    maas_model_id: str = field(default='damo/nlp_structbert_faq-question-answering_chinese-base')
    max_len: int = field(default=30)


@dataclass
class RankParam:
    query: str
    faq_candidates: List = field(default=None)
    ds_candidates: List = field(default=None)


class MaasModel:
    def __init__(self, config: FewshotConfig):
        self.config = config
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        self.infer_pipeline = pipeline(Tasks.faq_question_answering, config.maas_model_id)

    def invoke(self, query_set, support_set):
        outputs = self.infer_pipeline({'query_set': query_set, 'support_set': support_set}, max_len=self.config.max_len)
        return outputs


class UniversalRank:
    def __init__(self, model_id=None):
        config = FewshotConfig(max_len=50)
        if model_id:
            config.maas_model_id = model_id
        self.infer_pipeline = MaasModel(config)

    def invoke(self, rank_param: RankParam):
        query = rank_param.query
        faq_candidates = rank_param.faq_candidates
        if not faq_candidates:
            faq_candidates = []
        ds_candidates = rank_param.ds_candidates
        if not ds_candidates:
            ds_candidates = []
        query_set = [query]
        support_set = []
        final_result = []
        for item in faq_candidates + ds_candidates:
            sim_questions = item.get('sim_questions', [])
            if item.get('match_type', '').lower() != 'lgf' and sim_questions:
                biz_id = item.get('bizId', -1)
                for q in sim_questions:
                    support_set.append({'text': q, 'label': biz_id})
            else:
                final_result.append({'bizId': item.get('bizId', -1), 'score': 1})
        if len(support_set) > 0:
            result = self.infer_pipeline.invoke(query_set, support_set)
            if 'output' in result and len(result['output']) > 0:
                for item in result['output'][0]:
                    final_result.append({'bizId': item.get('label'), 'score': min(item.get('score', 0), 0.9999999999)})
        final_result = sorted(final_result, key=lambda d: d['score'], reverse=True)
        return final_result


if __name__ == '__main__':
    model = UniversalRank()
    query = '什么时候变黑色'
    faq_candidates = [{'bizId': 1, 'question_id': 324, 'question': '',
                       'sim_questions': ['什么时候变黑色', '什么时候变黑色']},
                      {'bizId': 2, 'question_id': 324, 'question': '',
                       'sim_questions': ['什么时候变黑色', '什么时候变黑色']}]
    ds_candidates = [
        {'bizId': 3, 'intent_id': 324, 'sim_questions': ['什么时候变黑色', '什么时候变黑色'],
         'match_type': 'lgf'}
    ]
    rank_param = RankParam(query, faq_candidates=faq_candidates, ds_candidates=ds_candidates)
    rst = model.invoke(rank_param)
    print(rst)
