import re
from . import register
from .space import SpaceTokenizer, _is_punctuation
from xdpx.utils import is_chinese


@register('char_zh')
class ChineseCharacterTokenizer(SpaceTokenizer):
    """
    - Keep English words and Chinese characters
    - Keep the first one of consecutive punctuations
    - Encoder long numbers as special tokens such as "N8"
    """

    def encode(self, s: str) -> list:
        s = self.clean_text(s)
        s = ''.join(s.split())
        tokens = []
        buffer = []  # buffer for English words
        num_buffer = []  # buffer for numbers

        def submit_buffer():
            if buffer:
                tokens.append(''.join(buffer))
                buffer.clear()
            elif num_buffer:
                if len(num_buffer) <= 4:
                    tokens.append(''.join(num_buffer))
                else:
                    tokens.append(f'NUM{len(num_buffer)}')
                num_buffer.clear()

        for c in s:
            if c == ' ':
                submit_buffer()
            elif re.match(r'\d', c):
                if buffer:
                    submit_buffer()
                num_buffer.append(c)
            elif is_chinese(c):
                submit_buffer()
                tokens.append(c)
            elif _is_punctuation(c):
                submit_buffer()
                if not tokens or c != tokens[-1]:
                    tokens.append(c)
            else:
                if num_buffer:
                    submit_buffer()
                buffer.append(c)
        submit_buffer()

        return tokens

    def decode(self, x: list) -> str:
        return ''.join(x)
