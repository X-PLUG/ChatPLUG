from typing import List
from . import register, Loader


@register('fewshot')
class FewshotLoader(Loader):
    '''
    there are two data formats
    1. mode == episode
        [
            {'text':'i want to top up', 'label':'intent0', 'domain':'banking'}
            {'text':'alarm ...' , 'label': 'intent1', 'domain':'general'}
           ....
        ]
    2. mode=='retrieval'
        [
            {
                'query':[
                    {'text':'i want to topup', 'label':0},
                    {'text':'transfer', 'label':1},
                    {'text':'good morning', 'label':-1}
                ]
                'support':[
                    ['top up failed','how to top up', ... ]
                    ['transfer time','when transfer', ...]
                ]
                'domain':'banking'
            }
        ]
    '''
    EPISODE_MODE = 'episode'
    RETRIEVAL_MODE = 'retrieval'

    @property
    def num_sections(self):
        return 4

    @property
    def header(self):
        return ['mode', 'field1', 'field2', 'field3']

    @classmethod
    def parse(cls, contents: List, _id=0) -> dict:
        mode = contents[0]
        if mode == 'episode':
            text = contents[1]  # str
            label = contents[2]  # str
            domain = contents[3]  # str
            return {
                'id': _id,
                'text': cls.tokenize(text),
                'label': label.lower(),
                'domain': domain,
                'mode': mode
            }
        else:
            query_set = contents[1]  # List[{'text':'xxx','label':0}]
            support_set = contents[2]  # List[List[str]]
            support_set_tokens = []
            query_set_tokens = [cls.tokenize(q['text']) for q in query_set]
            query_targets = [int(q['label']) for q in query_set]
            for items in support_set:
                support_set_tokens.append([cls.tokenize(s) for s in items])
            return {
                'id': _id,
                'query': query_set_tokens,
                'support': support_set_tokens,
                'query_targets': query_targets,
                'mode': mode,
                'domain': contents[3],
            }

    def length(self, sample):
        if sample['mode'] == self.EPISODE_MODE:
            return len(sample['text'])
        if sample['mode'] == self.RETRIEVAL_MODE:
            if sample['support']:
                max_s = max([max([len(s) for s in items]) for items in sample['support']])
            if sample['query']:
                max_q = max([max([len(s) for s in items]) for items in sample['query']])
            return max(max_s, max_q)

    def with_targets(self):
        return False
