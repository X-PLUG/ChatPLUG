import math
import hjson
import random
from typing import Dict
from typeguard import check_type
from xdpx.utils import io


class HParam:
    def __init__(self, name, values, **kwargs):
        self.name = name
        self.values = values
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'
    
    def sample(self, r=random.Random()):
        return self._sample(r)

    def _sample(self, r: random.Random):
        raise NotImplementedError


class Categorial(HParam):
    def _sample(self, r):
        return r.choice(self.values)

    def __iter__(self):
        for value in self.values:
            yield value


class Float(HParam):
    def __init__(self, name, values, log=False):
        super().__init__(name, values)
        self.min_value, self.max_value = values
        self.log_scale = log
        if log:
            self.min_value = math.log1p(self.min_value)
            self.max_value = math.log1p(self.max_value)
    
    def _sample(self, r):
        value = r.random() * (self.max_value - self.min_value) + self.min_value
        if self.log_scale:
            value = math.expm1(value)
        return value

    def __iter__(self):
        """binary search"""
        if not self.log_scale:
            min_value, max_value = self.min_value, self.max_value
            yield min_value
            yield max_value
        else:
            min_value = math.expm1(self.min_value)
            max_value = math.expm1(self.max_value)
            yield min_value
            yield max_value
            # transform the scale back to get accurate middle points
            min_value = math.log(min_value)
            max_value = math.log(max_value)
        points = [min_value, max_value]
        for _ in range(5):  # 33 points in total
            snapshot = points.copy()
            pointer = -1
            for i in range(len(snapshot) - 1):
                middle = (snapshot[i] + snapshot[i + 1]) / 2
                if not self.log_scale:
                    yield middle
                else:
                    yield math.exp(middle)
                pointer += 2
                points.insert(pointer, middle)


class Integer(HParam):
    def __init__(self, name, values, step=1, log=False):
        super().__init__(name, values)
        self.min_value, self.max_value = values
        self.step = step
        self.log_scale = log
        check_type('step', step, int)
        if log:
            assert step > 1
            self.diff = int(math.log(self.max_value / self.min_value) // math.log(self.step))
        else:
            self.diff = (self.max_value - self.min_value) // self.step
        assert self.diff >= 1
    
    def _sample(self, r):
        return self._transform_diff(r.randint(0, self.diff))

    def _transform_diff(self, diff):
        if self.log_scale:
            value = self.min_value * self.step ** diff
        else:
            value = self.min_value + diff * self.step
        return value

    def __iter__(self):
        for i in range(self.diff + 1):
            yield self._transform_diff(i)


registry = {
    'categorical': Categorial,
    'float': Float,
    'int': Integer,
}


def load_space(path):
    with io.open(path) as f:
        config = hjson.load(f)
    check_type('space_config', config, Dict[str, dict])
    space = []
    for key, val in config.items():
        assert 'values' in val, f'"values" should be set in space config of param "{key}"'
        assert 'type' in val, f'"type" should be set in space config of param "{key}"'
        param = registry[val.pop('type')](name=key, **val)
        space.append(param)
    return space
