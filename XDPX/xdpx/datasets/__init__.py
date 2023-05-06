from functools import partial, lru_cache
from xdpx.utils import register
import torch.utils.data

datasets = {}
register = partial(register, registry=datasets)


@register('default')
class Dataset(torch.utils.data.Dataset):
    @classmethod
    def register(cls, options):
        ...

    def __init__(self, data, is_train):
        self.data = data
        self.is_train = is_train

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)
