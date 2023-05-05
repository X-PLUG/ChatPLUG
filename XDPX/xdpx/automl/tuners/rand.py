from . import register, Tuner


@register('random')
class RandomTuner(Tuner):
    def suggest(self, config):
        for param in self.space:
            config[param.name] = param.sample(self.random)
        return config
