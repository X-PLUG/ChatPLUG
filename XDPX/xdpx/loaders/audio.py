from typing import List
from . import register, Loader


@register('single_audio_text')
class SingleAudioLoader(Loader):
    @property
    def num_sections(self):
        return 3

    @property
    def header(self):
        return ['tokens', 'audio'] + (['target'] if self.with_targets else [])

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        return {
            'id': _id,
            'tokens': cls.tokenize(contents[0]),
            'audio': contents[1]
        }

    def length(self, sample):
        return len(sample['tokens'])
