from typing import List
from . import register, Loader


@register('single')
class SingleLoader(Loader):
    @property
    def num_sections(self):
        return 2

    @property
    def header(self):
        return ['tokens'] + (['target'] if self.with_targets else [])

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        return {
            'id': _id,
            'tokens': cls.tokenize(contents[0]),
        }

    def length(self, sample):
        return len(sample['tokens'])


@register("single_with_logits")
class SingleWithLogitsLoader(SingleLoader):
    @property
    def num_sections(self):
        return 3

    @property
    def header(self):
        return ['tokens', 'logits'] + (['target'] if self.with_targets else [])

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        return {
            'id': _id,
            'tokens': cls.tokenize(contents[0]),
            'logits': eval(contents[1])
        }
