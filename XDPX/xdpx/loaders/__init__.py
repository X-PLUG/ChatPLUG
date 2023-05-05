import os
import json
import importlib
from typing import List, Union, Optional
from tqdm import tqdm
from functools import partial
from xdpx.utils import register, io
from xdpx.options import Argument
from ..tokenizers import tokenizers, Tokenizer
from .parsers import parsers, Parser


loaders = {}
register = partial(register, registry=loaders)


class Loader:
    _tokenizer: Tokenizer = None
    _parser: Parser = None
    _skip_bad_lines = False

    @classmethod
    def register(cls, options):
        options.register(
            Argument('lower', default=True),
            Argument('remove_duplicate', default=False),
            Argument('skip_bad_lines', default=False),
        )
        options.register(
            Argument('tokenizer', type=str, required=True, validate=lambda value: value in tokenizers,
                     register=lambda value: tokenizers[value].register),
            domain='tokenizer',
        )
        options.register(
            Argument('parser', default='csv', validate=lambda value: value in parsers,
                     register=lambda value: parsers[value].register),
            domain='parser',
        )
        options.assume_defined('max_len', by='Loader', type=int)

    def __init__(self, args):
        self.args = args
        self.__class__._parser = parsers[args.parser](args)
        self.__class__._skip_bad_lines = args.skip_bad_lines
        self._with_targets = None
        self.tokenizer
        if self.num_sections is not None:
            try:
                assert self.num_sections == len(self.header), '{}\t{}'.format(self.num_sections,self.header)
            except NotImplementedError:
                ...
    
    @property
    def tokenizer(self):
        if self.__class__._tokenizer is None:
            self.__class__._tokenizer = tokenizers[self.args.tokenizer](self.args)
        return self.__class__._tokenizer
    
    def meta(self):
        return self.tokenizer.meta()
    
    @property
    def parser(self):
        return self.__class__._parser

    @classmethod
    def tokenize(cls, text):
        return cls._tokenizer.encode(text)
    
    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        raise NotImplementedError

    def parse_target(self, contents: List[str]) -> dict:
        return {'target': str(contents[-1]).lower()}

    @property
    def header(self):
        raise NotImplementedError

    @classmethod
    def _parse_safe(cls, args):
        try:
            return cls.parse(*args)
        except Exception as e:
            cls.error_handler(e)

    @classmethod
    def error_handler(cls, e):
        if not isinstance(e, MalformedError):
            raise e
        if not cls._skip_bad_lines:
            raise RuntimeError(str(e) + '\nSet "skip_bad_lines" to true to ignore errors.')
        print(e)

    @property
    def num_sections(self) -> Optional[int]:
        # None means variable number of sections
        return None

    @property
    def num_targets(self):
        return 1

    @property
    def with_targets(self):
        # if num_sections is None, assume this kind of data has no targets
        if self._with_targets is None:
            self._with_targets = self.num_sections is not None
        return self._with_targets

    @with_targets.setter
    def with_targets(self, value):
        assert isinstance(value, bool)
        self._with_targets = value

    def length(self, sample: dict) -> int:
        """this method is called after `merge`"""
        raise NotImplementedError

    def merge(self, samples: List[dict]) -> List[dict]:
        return samples

    def load_data(self, path: str, ordered: Union[str, bool]='auto') -> List[dict]:
        """
        1. load data from the given path (supporting with/without targets for supervised data) with a progress bar
        2. support args.skip_bad_lines and args.remove_duplicate, print the number of samples skipped
        3. tokenize text with self.tokenize and preprocess (e.g. lowercase) targets 
        4. summarize the proportion of samples exceeding the maximum length if applicable
        """
        print(f'| loading {path} ({io.md5(path)})')
        samples = []
        seen = set()
        f, total, num_sections = self.parser.open_file(path)

        if self.num_sections:
            if num_sections == self.num_sections - self.num_targets:
                print(f'| loading "{path}" without targets.')
            elif num_sections != self.num_sections:
                raise RuntimeError(f'format of file "{path}" is not compatible with '
                                   f'{self.parser.__class__.__name__} of {self.__class__.__name__}.')
        skip_malform = 0
        skip_duplicate = 0
        parse_with_targets = self.num_sections is not None and num_sections == self.num_sections
        self.with_targets = parse_with_targets
        targets = []

        desc = f'processing ({self.args.workers} worker' + ('s)' if self.args.workers > 1 else ')')
        progress = tqdm(f, total=total, desc=desc) if self.args.workers == 1 else tqdm(f, total=total, desc='submitting')
        for i, line in enumerate(progress):
            sections = self.parser.parse_line(line)
            if not sections:
                continue
            if self.num_sections:
                if len(sections) != num_sections:
                    if not self.args.skip_bad_lines:
                        raise RuntimeError(f'Malformed line found @{i}: {line}')
                    # Malformed lines
                    skip_malform += 1
                    print(f'Number of segments mismatch @{i}: ', line.rstrip('\n'))
                    continue
            if self.args.workers == 1:
                try:
                    sample = self.parse(sections, i)
                    if self.args.remove_duplicate:
                        sample_str = str(sample)
                        if sample_str in seen:
                            skip_duplicate += 1
                            continue
                        seen.add(sample_str)
                    if parse_with_targets:
                        sample.update(self.parse_target(sections))
                    samples.append(sample)
                except Exception as e:
                    self.error_handler(e)
            else:
                samples.append((sections, i))
                if parse_with_targets:
                    targets.append(self.parse_target(sections))
        self.parser.close_file(f)
        if self.args.workers > 1:
            from multiprocessing import cpu_count, get_context
            mp = get_context('fork')
            lock = mp.Lock()
            pool = mp.Pool(min(self.args.workers, cpu_count()), initializer=init, initargs=(lock, self))
            map_func = pool.imap if targets else pool.imap_unordered
            if isinstance(ordered, bool) and ordered:
                map_func = pool.imap
            samples = list(tqdm(map_func(self._parse_safe, samples, chunksize=256), total=total, desc='processing'))
            pool.close()
            if parse_with_targets:
                assert len(targets) == len(samples)
            samples = [(sample, i) for i, sample in enumerate(samples)]
            if self.args.remove_duplicate:
                dedup_samples = []
                dedup_targets = []
                for sample, i in samples:
                    if sample is not None:
                        sample_str = str(sample)
                        if sample_str in seen:
                            skip_duplicate += 1
                            continue
                        seen.add(sample_str)
                    dedup_samples.append((sample, i))
                    if parse_with_targets:
                        dedup_targets.append(targets[i])
                samples = dedup_samples
                targets = dedup_targets
            for sample, i in samples:
                if sample is not None and targets:
                    sample.update(targets[i])
            samples = [sample for sample, i in samples if sample is not None]
        samples = self.merge(samples)
        too_long = sum(self.length(sample) > self.args.max_len for sample in samples)
        assert samples, f'File is empty or in wrong format: {path}'
        if skip_duplicate:
            print(f'| skip {skip_duplicate} duplicated lines in {path}.')
        if skip_malform:
            print(f'| skip {skip_malform} malformed lines in {path}.')
        if too_long:
            print(f'| {too_long} ({too_long / len(samples) * 100:.2f}%) samples exceed max length in {path}.')
        return samples


def init(lock, loader):
    with lock:
        loader.tokenizer


class MalformedError(Exception):
    pass


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('.' + module_name, __name__)
