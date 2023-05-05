import os
import math
import importlib
import torch
from collections import Counter, OrderedDict
from functools import partial
from typing import List, Iterator
from tqdm import tqdm
from ..dictionary import Dictionary
from xdpx.utils import register
from xdpx.options import Argument, Options
from xdpx.utils import io

processors = {}
register = partial(register, registry=processors)


class Processor:
    @classmethod
    def register(cls, options):
        options.register(
            Argument('max_len', required=True, type=int),
            Argument('min_len', default=1, validate=lambda value: value > 0, doc='will pad to this minimum length.'),
            Argument('pad_index', type=int),
            Argument('pad_word', default='[PAD]'),
            Argument('unk_word', default='[UNK]'),
            Argument('vocab_size', type=int, doc='auto computed during preprocessing'),
            Argument('num_classes', type=int, doc='auto computed during preprocessing'),
            Argument('target_unk', type=str, doc='class name for the unknown'),
        )
        options.add_global_constraint(lambda args: args.min_len <= args.max_len)

    def __init__(self, args):
        self.args = args
        self.max_len = args.max_len
        self.min_len = args.min_len
        # lazy loading
        self._dictionary = None
        self._target_map = None
        self.epoch = 0
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def encode(self, loader, segments: List[str], _id=0) -> dict:
        """tokenize, length-trim, and numerize data"""
        inputs = loader.parse(segments, _id)
        return self.numerize(inputs)
    
    def decode(self, ids: List[int]) -> List[str]:
        return self.dictionary.decode(ids)

    def numerize_samples(self, samples: List[dict], with_target=True) -> List[dict]:
        """batch numerize samples loaded by 'load_data'"""
        results = []
        for sample in tqdm(samples, desc='numerizing'):
            result = self.numerize(sample)
            if not result:
                continue
            if with_target:
                result.update(self.numerize_target(sample))
            results.append(result)
        return results

    def numerize(self, sample: dict) -> dict:
        """
        1. convert texts and discrete targets to their integer ids
        2. add ids of special tokens if needed
        3. limit the maximum length
        """
        raise NotImplementedError

    def numerize_target(self, sample):
        if 'target' not in sample:
            return {}
        return {
            'target': self.target_map.encode(sample['target'])
        }

    def numerize_tokens(self, tokens: List[str]):
        return self.dictionary.encode(tokens)

    @property
    def resources(self):
        return ['vocab.txt', 'target_map.txt', 'embeddings.pt']

    def inspect(self, samples: List[dict], name: str):
        """
        Input samples are numerized data.
        1. summarize the class distribution (if applicable)
        2. other data-specific inspections
        """
        if 'target' in samples[0]:
            counter = Counter(sample['target'] for sample in samples)
            stats = OrderedDict([(self.target_map.decode(key), val) for key, val in counter.most_common()])
            print(
                f'| target distribution of {name}:\n' + '| ' +
                ' | '.join(f'{key} ({val})' for key, val in stats.items())
            )
            return stats
        return {}

    def load_vocab(self, path=None):
        if path is None:
            if hasattr(self.args, 'save_dir') and io.exists(os.path.join(self.args.save_dir, 'vocab.txt')):
                path = os.path.join(self.args.save_dir, 'vocab.txt')
            else:
                path = os.path.join(self.args.data_dir, 'vocab.txt')
        return Dictionary.load(
            path,
            pad=self.args.pad_word,
            unk=self.args.unk_word,
        )

    def load_target_map(self, path=None):
        if path is None:
            if hasattr(self.args, 'save_dir') and io.exists(os.path.join(self.args.save_dir, 'vocab.txt')):
                path = os.path.join(self.args.save_dir, 'target_map.txt')
            else:
                path = os.path.join(self.args.data_dir, 'target_map.txt')
        return Dictionary.load(
            path,
            unk=getattr(self.args, 'target_unk', None)
        )

    @property
    def dictionary(self) -> Dictionary:
        if not self._dictionary:
            self._dictionary = self.load_vocab()
        return self._dictionary

    @property
    def target_map(self) -> Dictionary:
        if not self._target_map:
            self._target_map = self.load_target_map()
        return self._target_map
    
    # only call the following properties in "meta"
    @property
    def _vocab_size(self):
        return len(self.dictionary)
    
    @property
    def _pad_index(self):
        return self.dictionary.pad_index
    
    @property
    def _num_classes(self):
        return len(self.target_map)

    def _autoset_meta(self):
        return {
            'vocab_size': self._vocab_size ,
            'pad_index': self._pad_index,
            'num_classes': self._num_classes,
        }
    
    def arguments(self):
        options = Options()
        self.__class__.register(options)
        return options.keys()

    def meta(self) -> dict:
        "meta configurations to save in args during preprocessing steps"
        return {**{name: getattr(self.args, name) for name in self.arguments()}, **self._autoset_meta()}

    def text_length(self, sample):
        """
        Get the text length in sample, help to update the max_size parameter in the preprocessing steps. Why we should do this:
        1. In some cases (like when the model has positional embeddings) it is helpful to save resources.
        2. It can help check whether the implementation of processor acutally limits the max length
        Input parameter "sample" is the one after numerization.
        """
        raise NotImplementedError

    @staticmethod
    def token_stream(data: List[dict]) -> Iterator[str]:
        raise NotImplementedError

    @staticmethod
    def target_stream(data: List[dict]) -> Iterator[str]:
        raise NotImplementedError

    @property
    def _dict_for_emb(self):
        return self.dictionary

    @staticmethod
    def read_emb_meta(path):
        with io.open(path) as f:
            start = 0
            dim = len(f.readline().strip().split()) - 1
            if dim == 1:
                # skipping the heading line
                start = f.tell()
                dim = len(f.readline().strip().split()) - 1
            f.seek(start)
            total = 0
            for _ in f:
                total += 1
            f.seek(start)
        return {
            'start': start,
            'dim': dim,
            'total': total,
        }

    def load_embedding_dict(self, path):
        vocab = []
        meta = self.read_emb_meta(path)
        dim, start, total = meta['dim'], meta['start'], meta['total']
        num_malformed = 0
        with io.open(path) as f:
            f.seek(start)
            for line in tqdm(f, total=total, desc='loading emb dict'):
                elem = line.strip().split()
                token = elem[0]
                if len(elem) != dim + 1:
                    if num_malformed < 10:
                        print('| WARNING: Malformed embedding line:', ' '.join(elem[:5]), '...')
                    num_malformed += 1
                    continue
                if self.args.lower:
                    token = token.lower()
                vocab.append(token)
        if num_malformed > 10:
            skip_ratio = num_malformed / total
            if skip_ratio == 1:
                raise RuntimeError(f'Embedding format error: Cannot find {dim}-dimensional embeddings in {path}')
            print(f'| Skipped {num_malformed} malformed embedding lines while loading ({skip_ratio*100:.2f}%).')
        return vocab

    def extract_embeddings(self, src, dst):
        dictionary = self._dict_for_emb
        num_malformed = 0
        meta = self.read_emb_meta(src)
        dim, start, total = meta['dim'], meta['start'], meta['total']
        with io.open(src) as f:
            f.seek(start)
            g = torch.Generator()
            if self.args.seed is not None:
                g.manual_seed(self.args.seed)
            embeddings = torch.randn(len(dictionary), dim, generator=g) * 0.02
            embeddings[:dictionary.nspecial, :] = 0.
            for line in tqdm(f, total=total, desc='extracting embeddings'):
                elem = line.strip().split()
                token = elem[0]
                if self.args.lower:
                    token = token.lower()
                if token in dictionary:
                    if len(elem) != dim + 1:
                        if num_malformed < 10:
                            print('| WARNING: Malformed embedding line:', ' '.join(elem[:5]), '...')
                        num_malformed += 1
                        continue
                    vector = torch.tensor(list(map(float, elem[1:])))
                    embeddings[dictionary.index(token)] = vector
        if num_malformed > 10:
            skip_ratio = num_malformed/(len(embeddings)-dictionary.nspecial)
            if skip_ratio == 1:
                raise RuntimeError(f'Embedding format error: Cannot find {dim}-dimensional embeddings in {src}')
            print(f'| Skipped {num_malformed} malformed embedding lines while extracting ({skip_ratio*100:.2f}%).')
        with io.open(dst, 'wb') as f:
            torch.save(embeddings, f)
        return dim
                
    def collate(self, samples: List[dict])-> dict:
        raise NotImplementedError
    
    def pad(self, samples, pad_index=None):
        if pad_index is None:
            pad_index = self.args.pad_index
        max_len = max(max(map(len, samples)), self.min_len)
        if getattr(self.args, 'fp16', False):
            max_len = int(math.ceil(max_len / 8) * 8)
        return [sample + [pad_index] * (max_len - len(sample)) for sample in samples]

    def pad3d(self, samples, pad_index=None):
        if pad_index is None:
            pad_index = self.args.pad_index
        max_len = max(max(map(len, samples)), self.min_len)
        inner_max_len = max(len(x) for sample in samples for x in sample)
        if getattr(self.args, 'fp16', False):
            max_len = int(math.ceil(max_len / 8) * 8)
            inner_max_len = int(math.ceil(inner_max_len / 8) * 8)
        padded_samples = []
        for sample in samples:
            padded_sample = []
            for x in sample:
                padded_sample.append(x + (inner_max_len - len(x)) * [pad_index])
            padded_sample.extend([[pad_index] * inner_max_len] * (max_len - len(sample)))
            padded_samples.append(padded_sample)
        return padded_samples


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('.' + module_name, __name__)
