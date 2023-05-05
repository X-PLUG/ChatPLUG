from typing import List
from . import register, Loader


@register('pair')
class PairLoader(Loader):
    @property
    def num_sections(self):
        return 3

    @property
    def header(self):
        return ['tokens1', 'tokens2'] + (['target'] if self.with_targets else [])

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        return {
            'id': _id,
            'tokens1': cls.tokenize(contents[0]), 
            'tokens2': cls.tokenize(contents[1]), 
        }
    
    def length(self, sample):
        len1 = len(sample['tokens1'])
        len2 = len(sample['tokens2'])
        if len1 and len2:
            return max(len1, len2)
        else:
            return 0


@register("pair_with_logits")
class PairWithLogitsLoader(PairLoader):
    @property
    def num_sections(self):
        return 4

    @property
    def header(self):
        return 'tokens1 tokens2 logits target'.split()

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        return {
            'id': _id,
            'tokens1': cls.tokenize(contents[0]),
            'tokens2': cls.tokenize(contents[1]),
            'logits': eval(contents[2])
        }


@register('rank')
class RankLoader(PairLoader):
    @property
    def num_sections(self):
        return 4

    @property
    def header(self):
        return ['id', 'tokens1', 'tokens2'] + (['target'] if self.with_targets else [])

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        return {
            'id': f'{contents[0]}-{_id}',
            'tokens1': cls.tokenize(contents[1]),
            'tokens2': cls.tokenize(contents[2]),
        }
