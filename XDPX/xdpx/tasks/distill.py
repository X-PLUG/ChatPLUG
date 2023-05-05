import torch
from . import register, Task
from xdpx.options import Argument
from xdpx.utils import move_to_cuda


@register('distill')
class DistillTask(Task):
    @classmethod
    def register(cls, options):
        from xdpx.models import models, Model
        from xdpx.losses import losses
        from xdpx.losses.distill import DistillLoss
        with options.with_prefix('teacher_'):
            options.register(
                Argument('model', required=True, validate=lambda value: value in models,
                         register=lambda value: Model.build_model_class(value).register),
                domain='teacher_model',
            )
        with options.with_prefix('student_'):
            options.register(
                Argument('model', required=True, validate=lambda value: value in models,
                         register=lambda value: Model.build_model_class(value).register),
                domain='student_model',
            )
        options.register(
            Argument('loss', required=True, register=lambda value: losses[value].register,
                     validate=lambda value: value in losses and issubclass(losses[value], DistillLoss)),
            domain='loss'
        )
        cls.register_dataset_options(options)
        options.add_global_constraint(lambda args: (args.max_epoch is None,
                                                    'max_epoch is not compatible with distill task'))
    
    def build_model(self, args):
        from xdpx.models import models
        # always load pretrained model for the teacher because it'll be generating data
        teacher_model = models[args.teacher_model].build(args.strip_prefix('teacher_').change(__cmd__='train'))
        student_model = models[args.student_model].build(args.strip_prefix('student_'))
        teacher_model.eval()
        self.teacher = teacher_model
        if args.cuda:
            self.teacher = self.teacher.cuda()
        return student_model
    
    def build_dataset(self, data, is_train):
        data = super().build_dataset(data, is_train)
        data = TeacherDataLoader(data, self.teacher)
        return data


class TeacherDataLoader:
    def __init__(self, data, model):
        super().__init__()
        self.data = data
        self.model = model
        self.cuda = next(model.parameters()).is_cuda

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for sample in self.data:
            net_input = sample['net_input']
            if self.cuda:
                net_input = move_to_cuda(net_input)
            with torch.no_grad():
                outputs = self.model(**net_input)
            yield outputs
