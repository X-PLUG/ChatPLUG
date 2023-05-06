import os
from typing import List, Optional

from icecream import ic

from . import register
from .space import SpaceTokenizer
from xdpx.utils import cache_file, io


@register('bert')
class BertTokenizer(SpaceTokenizer):

    def __init__(self, args):
        super().__init__(args)
        from .thirdparty.transformers.tokenization_bert import BertTokenizer
        # 'vocab.txt' can only be accessed from local file due to the limitation of BertTokenizer API
        if hasattr(args, 'vocab_file'):
            path = cache_file(args.vocab_file)
        elif hasattr(args, 'save_dir') and io.exists(os.path.join(args.save_dir, 'vocab.txt')):
            path = cache_file(os.path.join(args.save_dir, 'vocab.txt'))
        else:
            path = cache_file(os.path.join(args.data_dir, 'vocab.txt'))
        self.tokenizer = BertTokenizer.from_pretrained(path)
        # TODO(junfeng): add vocab user_profile
        # ic(self.tokenizer.tokenize("user_profile: 姓名是俊峰"))

    def encode(self, x: str) -> List[str]:
        x = self.clean_text(x)
        return self.tokenizer.tokenize(x)

    def decode(self, x: List[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(x)
    
    # pad_index is assumed to be 0

    @property
    def cls_index(self):
        return self.tokenizer.cls_token_id

    @property
    def sep_index(self):
        return self.tokenizer.sep_token_id
    
    @property
    def mask_index(self):
        return self.tokenizer.mask_token_id
    
    def _autoset_meta(self):
        meta = super()._autoset_meta()
        meta.update({
            'cls_index': self.cls_index,
            'sep_index': self.sep_index,
            'mask_index': self.mask_index,
        })
        return meta
    
    def __len__(self):
        return len(self.tokenizer)
    
    @staticmethod
    def word_begin_mask(words: List[str], tokens: List[str]) -> Optional[List[bool]]:
        "whole word masking based on pretokenized (space-separated) text; 1 indicating the beginning of a word"
        mask = []
        length = 0  # current word length after processing the current "token"
        cursor = 0  # corresponding position in "words" of the current "token"
        for i, token in enumerate(tokens):
            mask.append(length == 0)  # one mask for one token
            token_len = len(token)
            if token.startswith('##'):  # original tags
                token_len -= 2
            length += token_len
            if cursor >= len(words) or length > len(words[cursor]):
                # happends in languages like Korean or other rare cases, which is not useful for us
                # in this case a single character is splitted into multiple tokens
                return None
            if length == len(words[cursor]):
                length = 0
                cursor += 1
                if cursor >= len(words) and i < len(tokens) - 1:
                    return None

        if cursor != len(words):
            # happens when text contains special invisible tokens, emojis or kaomojis
            return None
        return mask


@register('bert_id')
class BertIdTokenizer(BertTokenizer):
    """
    If input data is already converted to ids.
    """
    def encode(self, x: str) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(list(map(int, x.split())))

    def decode(self, x: List[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(x)
    
    @staticmethod
    def word_begin_mask(text: str, tokens: List[str]) -> Optional[List[bool]]:
        # "_" is an indicator of the beginning of a word
        raise NotImplementedError


@register('palm_tokenizer')
class PalmTokenizer(SpaceTokenizer):

    def __init__(self, args):
        super().__init__(args)
        from transformers import BertTokenizer
        # 'vocab.txt' can only be accessed from local file due to the limitation of BertTokenizer API
        if hasattr(args, 'vocab_file'):
            path = cache_file(args.vocab_file)
        elif hasattr(args, 'save_dir') and io.exists(os.path.join(args.save_dir, 'vocab.txt')):
            path = cache_file(os.path.join(args.save_dir, 'vocab.txt'))
        else:
            path = cache_file(os.path.join(args.data_dir, 'vocab.txt'))
        self.tokenizer = BertTokenizer.from_pretrained(path)
        # unused1: </s> SEP \t[SEP]\t
        # unused2: user_profile
        # unused3: knowledge
        # unused4: bot_profile
        # unused5: history
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]",
                                                                         "[unused2]",
                                                                         "[unused3]",
                                                                         "[unused4]",
                                                                         "[unused5]"]})

        ic(self.tokenizer.tokenize("[CLS]一般人我不告诉他_百度经验[SEP]怎么去掉视频上的水[unused1]"))
        # TODO(junfeng): add vocab user_profile
        # ic(self.tokenizer.tokenize("user_profile: 姓名是俊峰"))

    def encode(self, x: str) -> List[str]:
        x = self.clean_text(x)
        x = x.replace("[sep]", "[SEP]")
        x = x.replace("[cls]", "[CLS]")
        return self.tokenizer.tokenize(x)

    def decode(self, x: List[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(x)

    @property
    def cls_index(self):
        return self.tokenizer.cls_token_id

    @property
    def sep_index(self):
        return self.tokenizer.sep_token_id

    @property
    def mask_index(self):
        return self.tokenizer.mask_token_id

    def _autoset_meta(self):
        meta = super()._autoset_meta()
        meta.update({
            'cls_index': self.cls_index,
            'sep_index': self.sep_index,
            'mask_index': self.mask_index,
        })
        return meta

    def __len__(self):
        return len(self.tokenizer)
