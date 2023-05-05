from collections import Counter
from typing import List, Union
from xdpx.utils import io


class Dictionary:
    def __init__(self, pad=None, unk=None, extra_special_symbols=[]):
        self.indices = {}
        self.symbols = []
        self.count = []
        self._pad_index = None
        self._pad_word = None
        self._unk_index = None
        self._unk_word = None
        if pad:
            self.pad_word = pad
            self.add_symbol(self.pad_word)
            self._pad_index = self.indices[self.pad_word]
        if unk:
            self.unk_word = unk
            self.add_symbol(self.unk_word)
            self._unk_index = self.indices[self.unk_word]
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    @property
    def unk_index(self):
        if self._unk_index is None:
            raise LookupError('UNK is not enabled')
        return self._unk_index

    @property
    def pad_index(self):
        if self._pad_index is None:
            raise LookupError('PAD is not enabled')
        return self._pad_index

    def encode(self, tokens: Union[List[str], str]) -> Union[List[int], int]:
        if isinstance(tokens, list):
            return [self.index(token) for token in tokens]
        return self.index(tokens)

    def decode(self, indices: Union[List[int], int]) -> Union[List[str], str]:
        if isinstance(indices, list):
            return [self[index] for index in indices]
        return self[indices]

    def __eq__(self, other):
        return self.indices == other.indices

    def __len__(self):
        return len(self.symbols)

    def __getitem__(self, index):
        if index < len(self.symbols):
            return self.symbols[index]
        if self.unk_index is not None:
            return self.unk_word
        raise LookupError(f'index {index} out of range when UNK is not allowed.')

    def __contains__(self, symbol):
        return symbol in self.indices

    def index(self, symbol: str):
        assert isinstance(symbol, str)
        if symbol in self.indices:
            return self.indices[symbol]
        try:
            return self.unk_index
        except LookupError:
            raise IndexError(f'Unknown symbol "{symbol}" when UNK is not allowed.')

    def add_symbol(self, symbol, n=1):
        if symbol in self.indices:
            idx = self.indices[symbol]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[symbol] = idx
            self.symbols.append(symbol)
            self.count.append(n)
            return idx

    def finalize(self, threshold=-1, nwords=-1, pretrained=[], ignore_in_emb=False):
        pretrained = set(pretrained)
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[:self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[:self.nspecial]
        new_count = self.count[:self.nspecial]

        c = Counter(dict(sorted(zip(self.symbols[self.nspecial:], self.count[self.nspecial:]))))
        omit_symbols = []
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold or (ignore_in_emb and symbol in pretrained):
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                omit_symbols.append(symbol)
        assert len(new_symbols) == len(new_indices)
        if self._unk_index is not None or pretrained:
            new_symbols_set = set(new_symbols)
            unk = c.keys() - new_symbols_set
            total_count = sum(self.count[self.nspecial:])
            known_count = sum(c[symbol] for symbol in new_symbols)
            print(f'| symbols proportion: ')
            if self._unk_index is not None:
                print(f'| --{sum(c[symbol] for symbol in unk) / total_count * 100:.2f}% UNK')
                print(f'| --{known_count / total_count * 100:.2f}% known')
            if pretrained:
                covered = new_symbols_set & pretrained
                covered_rate = sum(c[symbol] for symbol in covered) / known_count
                print(f'|   |--{covered_rate * 100:.2f}% pretrained')
                print(f'|   |--{(1. - covered_rate) * 100:.2f}% random')
                dict_coverage = len(covered) / (len(new_symbols) - self.nspecial)
                print(f'| pretrained symbols occupy {dict_coverage*100:.2f}% of dict size.')
        if omit_symbols:
            print('| UNK symbols:', ' '.join(omit_symbols[:10]), '...' if len(omit_symbols) > 10 else '')

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

    char_map = {  # escape special characters for safe serialization
        '\n': '[NEWLINE]',
    }

    def save(self, file):
        with io.open(file, 'w') as f:
            for symbol in self.symbols:
                symbol = self.char_map.get(symbol, symbol)
                f.write(f'{symbol}\n')

    @classmethod
    def load(cls, file, pad=None, unk=None):
        dictionary = cls(pad=None, unk=None)
        reverse_char_map = {v: k for k, v in cls.char_map.items()}
        with io.open(file) as f:
            for line in f:
                symbol = line.rstrip('\n')
                symbol = reverse_char_map.get(symbol, symbol)
                if symbol in dictionary.indices:
                    raise ValueError(f'Duplicated values found in "{file}": {symbol}')
                dictionary.add_symbol(symbol)
        for symbol in dictionary.symbols:
            if pad and symbol == pad:
                dictionary._pad_index = dictionary.indices[pad]
                dictionary.nspecial += 1
            if unk and symbol == unk:
                dictionary._unk_index = dictionary.indices[unk]
                dictionary.nspecial += 1
        if pad and dictionary._pad_index is None:
            raise RuntimeError(f'pad word "{pad}" not found in vocab file {file}')
        if unk and dictionary._unk_index is None:
            raise RuntimeError(f'unk word "{unk}" not found in vocab file {file}')
        return dictionary

    @classmethod
    def from_pretrained(cls, path, pad=None, unk=None):
        from transformers import AutoTokenizer
        vocab = AutoTokenizer.from_pretrained(path).vocab
        dictionary = cls(pad=None, unk=None)
        for symbol, index in sorted(vocab.items(), key=lambda x: x[1]):
            dictionary.add_symbol(symbol)
        for symbol in dictionary.symbols:
            if pad and symbol == pad:
                dictionary._pad_index = dictionary.indices[pad]
                dictionary.nspecial += 1
            if unk and symbol == unk:
                dictionary._unk_index = dictionary.indices[unk]
                dictionary.nspecial += 1
        if pad and dictionary._pad_index is None:
            raise RuntimeError(f'pad word "{pad}" not found in vocab file {path}')
        if unk and dictionary._unk_index is None:
            raise RuntimeError(f'unk word "{unk}" not found in vocab file {path}')
        return dictionary
