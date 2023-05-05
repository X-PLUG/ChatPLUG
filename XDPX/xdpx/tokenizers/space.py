
import re
import unicodedata
from . import register, Tokenizer
from xdpx.options import Argument


@register('space')
class SpaceTokenizer(Tokenizer):
    @classmethod
    def register(cls, options):
        options.register(
            Argument('rm_punc', default=False, doc='whether to remove punctuations'),
        )

    def clean_text(self, s):
        # strip_accent
        s = unicodedata.normalize('NFD', s)
        if self.args.rm_punc:
            buffer = []
            for char in s:
                cat = unicodedata.category(char)
                if _is_punctuation(char) or cat == 'So':  # "So" contains emoji and other special symbols
                    continue
                buffer.append(char)
            s = ''.join(buffer)
        s = re.sub(r'\s', ' ', s)  # replace \r \n \t at first, avoid collding with the decision of control characters
        
        buffer = []
        for char in s:
            cat = unicodedata.category(char)
            code = ord(char)
            if cat == 'Mn' or cat.startswith('C') or code == 0 or code == 0xfffd:  # nonspacing marks and control characters
                continue
            elif cat == 'Zs':  # whitespace characters
                char = ' '
            buffer.append(char)
        s = ''.join(buffer)
        
        s = re.sub(r'\s+', ' ', s).strip()  # replace consecutive whitespaces with one space
        if self.args.lower:
            s = s.lower()
        return s

    def encode(self, s: str) -> list:
        s = self.clean_text(s)
        return s.split()

    def decode(self, x: list) -> str:
        return ' '.join(x)


def _is_punctuation(char):
    """
    Reference: https://github.com/google-research/bert/blob/master/tokenization.py#L386
    Checks whether `chars` is a punctuation character.
    """
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
