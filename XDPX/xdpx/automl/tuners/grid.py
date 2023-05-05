from itertools import product
from xdpx.options import Argument
from . import register, Tuner, SpaceExhausted
from ..space import Float


@register('grid')
class GridTuner(Tuner):
    @staticmethod
    def register(options):
        options.register(
            Argument('float_samples', default=5),
        )

    def __init__(self, *args):
        super().__init__(*args)
        self.space_iter = iterspace(self.space, float_samples=self.args.float_samples)

    def suggest(self, config):
        try:
            suggestion = dict(next(self.space_iter))
            config.update(suggestion)
            print(f'| suggest a new run with params {dict(suggestion)}')
            return config
        except StopIteration:
            raise SpaceExhausted


def iterspace(space_def, float_samples=5):
    space = []
    for param in space_def:
        if isinstance(param, Float):
            param_iter = iter(param)
            values = [next(param_iter) for _ in range(float_samples)]
        else:
            values = list(param)
        space.append([(param.name, value) for value in values])
    for point in product(*space):
        yield point
