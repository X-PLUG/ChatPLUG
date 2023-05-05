from typing import List
from . import register, Loader


@register('dpr')
class DPRLoader(Loader):
    @property
    def num_sections(self):
        return 2

    @property
    def header(self):
        return ['question', 'passage']

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        return {
            'id': _id,
            'question': cls.tokenize(contents[0]),
            'passage': cls.tokenize(contents[1]),
        }
    
    def length(self, sample):
        len1 = len(sample['question'])
        len2 = len(sample['passage'])
        if len1 and len2:
            return len1 + len2
        else:
            return 0
    def with_targets(self):
        return False

