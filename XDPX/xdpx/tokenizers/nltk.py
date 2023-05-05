import re
from . import register, Tokenizer


@register('nltk')
class NLTKTokenizer(Tokenizer):

    def __init__(self, args):
        super().__init__(args)
        from nltk.tokenize import TweetTokenizer
        self.tokenizer = TweetTokenizer()
        

    def encode(self, string: str) -> list:
        string = ' '.join(self.tokenizer.tokenize(string))
        string = re.sub(r"[-.#\"/]", " ", string)
        string = re.sub(r"\'(?!(s|m|ve|t|re|d|ll)( |$))", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'m", " \'m", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r"\s+", " ", string)
        return string.strip()

    def decode(self, x: list) -> str:
        raise NotImplementedError
