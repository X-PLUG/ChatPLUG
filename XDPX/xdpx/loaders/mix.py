from typing import List
from . import register, MalformedError, Loader
from .single import SingleLoader
from .pair import PairLoader
from .corpus import CorpusLoader
from xdpx.options import Argument
from xdpx.tokenizers import tokenizers
from xdpx.tokenizers.bert import BertTokenizer


class MixLoader(Loader):
    _with_words = False

    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('with_words', default=False),
        )
        options.add_global_constraint(lambda args: issubclass(tokenizers[args.tokenizer], BertTokenizer))

    def __init__(self, args):
        super().__init__(args)
        self.__class__._with_words = args.with_words
        assert hasattr(self.tokenizer, 'clean_text')
        assert hasattr(self.tokenizer, 'word_begin_mask')


@register('single_mix')
class SingleMixLoader(SingleLoader, MixLoader):
    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        text = contents[0]
        text = cls._tokenizer.clean_text(text)
        words = text.split()
        tokens = cls.tokenize(text)
        if not words or not tokens:
            raise MalformedError(f'tokenizer returns an empty string for @{_id}: {contents}')
        mask = cls._tokenizer.word_begin_mask(words, tokens)
        if mask is None:
            raise MalformedError(f'bad tokenization found @{_id}: {contents}')
        sample = {
            'id': _id,
            'tokens': tokens,
            'word_begin_mask': mask,
        }
        if cls._with_words:
            sample['words'] = words
        return sample

    def length(self, sample):
        return len(sample['tokens'])


@register('pair_mix')
class PairMixLoader(PairLoader, MixLoader):
    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        text1 = contents[0]
        text2 = contents[1]
        text1 = cls._tokenizer.clean_text(text1)
        text2 = cls._tokenizer.clean_text(text2)
        words1 = text1.split()
        words2 = text2.split()
        tokens1 = cls.tokenize(text1)
        tokens2 = cls.tokenize(text2)
        if not words1 or not tokens1 or not words2 or not tokens2:
            raise MalformedError(f'tokenizer returns an empty string for @{_id}: {contents}')
        mask1 = cls._tokenizer.word_begin_mask(words1, tokens1)
        mask2 = cls._tokenizer.word_begin_mask(words2, tokens2)
        if mask1 is None or mask2 is None:
            raise MalformedError(f'bad tokenization found @{_id}: {contents}')
        sample = {
            'id': _id,
            'tokens1': tokens1,
            'tokens2': tokens2,
            'mask1': mask1,
            'mask2': mask2,
        }
        if cls._with_words:
            sample.update({
                'words1': words1,
                'words2': words2,
            })
        return sample

    def length(self, sample):
        return len(sample['tokens1']) + len(sample['tokens2']) + 3


@register('corpus_mix')
class CorpusMixLoader(CorpusLoader, MixLoader):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('min_word_len', default=1),
        )

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        text = cls._tokenizer.clean_text(contents[0])
        if not text:
            return {'id': _id, 'content': [], 'word_begin_mask': []}
        tokens = cls.tokenize(text)
        words = text.split()
        mask = cls._tokenizer.word_begin_mask(words, tokens)
        if mask is None:
            return {'id': _id, 'content': [], 'word_begin_mask': []}
        return {
            'id': _id,
            'content': tokens,
            'word_begin_mask': mask,
        }


@register('corpus_mix_prompt')
class CorpusMixPromptLoader(CorpusMixLoader):
    _args = None

    def __init__(self, args):
        super(CorpusMixPromptLoader, self).__init__(args)
        self.__class__._args = args

    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('prompt_length', default=20),
        )

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        result = super().parse(contents, _id)
        result['prompt_tokens'] = ["[unused{}]".format(i + 1) for i in range(cls._args.prompt_length)]

        return result
