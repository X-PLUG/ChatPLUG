from typing import List
from . import register
from .bert import BertTokenizer
from transformers import AutoTokenizer


@register('auto')
class HuggingfaceAutoTokenizer(BertTokenizer):
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.vocab_file)

    def encode(self, x: str) -> List[str]:
        x = self.clean_text(x)
        return self.tokenizer.tokenize(x)

    def decode(self, x: List[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(x)
