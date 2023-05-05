import os
import importlib
import torch.nn.functional as F
from functools import partial
from xdpx.utils import register
from torch.nn.modules.loss import _Loss

losses = {}
register = partial(register, registry=losses)


class Loss(_Loss):
    @staticmethod
    def register(options):
        pass
    
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def state_dict(self, **kwargs):
        return {}
    
    def load_state_dict(self, state_dict, **kwargs):
        ...

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError
    
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `aggregate_logging_outputs`. Setting this
        to True will improves distributed training speed.
        """
        return False

    def get_prob(self, logits):
        return F.softmax(logits, dim=1)

    def inference(self, model, sample):
        raise NotImplementedError

    def distill(self, model, sample):
        return model(**sample['net_input']).tolist()


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module('.' + module_name, __name__)
