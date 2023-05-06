from typing import List
from tqdm import tqdm
from xdpx.options import Argument
from . import Loader, register


@register('corpus')
class CorpusLoader(Loader):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument(
                'break_mode', default='complete',
                validate=lambda value: value in 'none complete complete_doc eos none_doc'.split(),
                doc='''
                    'none': break tokens into equally sized blocks (up to block_size)
                    'complete': break tokens into blocks (up to block_size) such that
                        blocks contains complete sentences. A sentence longer than 
                        block size will be truncated
                    'complete_doc': similar to 'complete' mode, but do not
                        cross document boundaries
                    'eos': each block contains one sentence (block_size is ignored)
                    'none_doc': similar to 'none' mode, but do not cross document 
                        boundaries
                '''
            ),
        )
        options.assume_defined('max_len', by=cls.__name__, type=int)
        options.assume_defined('min_len', by=cls.__name__, type=int)

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        return {
            'id': _id,
            'content': cls._tokenizer.encode(contents[0])
        }

    @property
    def header(self):
        return ['content']
    
    def merge(self, samples):
        blocks = []
        if self.args.break_mode == 'eos':
            i = 0
            for sample in tqdm(samples, desc='merging'):
                if self.length(sample) < self.args.min_len:
                    continue
                blocks.append({**sample, 'id': i})
                i += 1
            return blocks
        
        max_len = self.args.max_len - 3
        block = []
        masks = []
        i = 0

        def add_block():
            nonlocal block, i, masks
            if block:
                sample = {'id': i, 'content': block}
                should_submit = self.length(sample) >= self.args.min_len
                if masks:
                    sample['word_begin_mask'] = masks
                    should_submit = should_submit and sum(masks) >= self.args.min_word_len
                    masks = []
                if should_submit:
                    blocks.append(sample)
                    i += 1
                block = []
    
        for sample in tqdm(samples, desc='merging'):
            content = sample['content']
            mask = sample.get('word_begin_mask', None)
            if self.args.break_mode.startswith('none'):
                if len(content) == 0:
                    if self.args.break_mode == 'none_doc':
                        add_block()
                    continue
                while True:
                    fill_len = max_len - len(block)
                    assert fill_len > 0
                    should_submit = len(block) + min(len(content), fill_len) == max_len
                    if mask and len(mask) > fill_len:
                        # do not break word boundaries
                        while not mask[fill_len] and fill_len:
                            fill_len -= 1
                        if not fill_len and not len(block):
                            # a single word exceed max length. Skip it.
                            break

                    block.extend(content[:fill_len])
                    if mask:
                        masks.extend(mask[:fill_len])
                    if should_submit:
                        add_block()
                        content = content[fill_len:]
                        if mask:
                            mask = mask[fill_len:]
                    else:
                        break
            else:
                if len(content) == 0:
                    if self.args.break_mode == 'complete_doc':
                        add_block()
                    continue
                if len(content) > max_len:
                    add_block()
                    sample = {'id': i, 'content': content}
                    if mask:
                        sample['word_begin_mask'] = mask
                    blocks.append(sample)
                    i += 1
                    continue
                if len(content) + len(block) > max_len:
                    add_block()
                    block = content
                    if mask:
                        masks = mask
                else:
                    block.extend(content)
                    if mask:
                        masks.extend(mask)
        add_block()
        return blocks
    
    def length(self, sample):
        return len(sample['content'])
