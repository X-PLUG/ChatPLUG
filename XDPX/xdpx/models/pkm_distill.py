import re
import os
import json
import torch
import torch.nn as nn
from functools import partial
from . import register, Model
from .bert import Bert
from typing import List
from xdpx.options import Argument, Options
from xdpx.utils import io, should_save_meta, move_to_cuda, pin_memory
from xdpx.modules.projections.pkm import HashingMemory
from xdpx.modules.projections.lopkm import LocallyOptimizedHashingMemory
from xdpx.losses.distill import register as register_loss, LocalDistillLoss
from xdpx.tasks.distill import register as register_task, DistillTask


@register('bert_teacher_for_pkm')
class BertTeacherPKM(Bert):
    @staticmethod
    def register(options):
        options.register(
            Argument('target_layers', type=List[int], required=True, doc='starting from 0',
                     post_process=lambda value: [value] if isinstance(value, int) else value),
        )
        options.set_default('teacher_loader', 'async')

    def __init__(self, args):
        super().__init__(args)
        self.input_buffer = [None] * len(self.args.target_layers)
        self.output_buffer = [None] * len(self.args.target_layers)

    def build_bert_backend(self):
        self.args.num_hidden_layers = max(self.args.target_layers) + 1
        super().build_bert_backend()

        def save_input(module, input, output, i):
            self.input_buffer[i] = input[0]

        def save_output(module, input, output, i):
            self.output_buffer[i] = output

        for i, layer_id in enumerate(self.args.target_layers):
            self.bert.encoder.layer[layer_id].intermediate.dense.register_forward_hook(partial(save_input, i=i))
            self.bert.encoder.layer[layer_id].output.dense.register_forward_hook(partial(save_output, i=i))

    @torch.no_grad()
    def forward(self, input_ids, **kwargs):
        mask = input_ids.ne(self.args.pad_index)
        self.bert_forward(input_ids, attention_mask=mask.long(), **kwargs)
        row_mask = mask.view(-1)
        buffer = []
        for i in range(len(self.args.target_layers)):
            xi = self.input_buffer[i]
            xo = self.output_buffer[i]
            xi = xi.view(-1, xi.size(2))[row_mask, :]
            xo = xo.view(-1, xo.size(2))[row_mask, :]
            buffer.append((xi, xo))
        self.input_buffer = [None] * len(self.args.target_layers)
        self.output_buffer = [None] * len(self.args.target_layers)
        return buffer

    def _load_and_parse_state_dict(self, path):
        state_dict = super()._load_and_parse_state_dict(path)
        max_target_layer = max(self.args.target_layers)
        re_layer = re.compile(r'bert\.encoder\.layer\.(\d+)')
        obsolete_keys = []
        for key in list(state_dict.keys()):
            m = re_layer.match(key)
            if m and int(m.group(1)) > max_target_layer:
                obsolete_keys.append(key)
            elif key.startswith('cls.') or key.startswith('heads.'):
                obsolete_keys.append(key)
        for key in obsolete_keys:
            del state_dict[key]
        return state_dict


@register_task('pkm_distill')
class PKMDistillTask(DistillTask):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('teacher_batch_size', type=int, post_process=(
                lambda val, args: val if val is not None else max(args.batch_size // 768, 4)
            )),
        )

    def build_dataset(self, data, is_train):
        self.args, args = self.args.change(batch_size=self.args.teacher_batch_size), self.args
        data = super(DistillTask, self).build_dataset(data, is_train)
        self.args = args
        data_size = sum(self.processor.text_length(sample) for sample in data.dataset) * len(self.teacher.args.target_layers)
        data = AsyncTeacherDataLoader(data, self.teacher, self.args.batch_size, drop_last=is_train)
        data.data_size = data_size
        return data


class AsyncTeacherDataLoader:
    def __init__(self, data, model, batch_size, drop_last):
        super().__init__()
        self.data = data
        self.model = model
        self.cuda = next(model.parameters()).is_cuda
        self.target_layers = self.model.args.target_layers
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data_size = None

    def __len__(self):
        if self.data_size is None:
            raise NotImplementedError
        return self.data_size // self.batch_size + bool(not self.drop_last and self.data_size % self.batch_size)

    def __iter__(self):
        bsz = self.batch_size
        layer_data = [[] for _ in range(len(self.target_layers))]
        for sample in self.data:
            net_input = sample['net_input']
            if self.cuda:
                net_input = move_to_cuda(net_input)
            with torch.no_grad():
                for i, (input_tensor, output_tensor) in enumerate(self.model(**net_input)):
                    inputs = torch.split(input_tensor.cpu(), 1)
                    outputs = torch.split(output_tensor.cpu(), 1)
                    for input_t, output_t in zip(inputs, outputs):
                        layer_data[i].append((input_t, output_t))
            if len(layer_data[0]) > bsz:
                for t in range(bsz, len(layer_data[0]), bsz):
                    for i, layer_id in enumerate(self.target_layers):
                        batch = collate(layer_data[i][t - bsz: t], layer_id=layer_id)
                        if self.cuda:
                            batch = pin_memory(batch)
                        yield batch
                layer_data = [data[-(len(data) % bsz):] for data in layer_data]
        if not self.drop_last:
            assert len(layer_data) <= bsz
            for i, layer_id in enumerate(self.target_layers):
                batch = collate(layer_data[i], layer_id=layer_id)
                yield batch


def collate(batch, layer_id):
    inputs = []
    outputs = []
    for input_tensor, output_tensor in batch:
        inputs.append(input_tensor)
        outputs.append(output_tensor)
    return {
        'net_input': {
            'inputs': torch.cat(inputs, 0),
            'layer_id': layer_id,
        },
        'target': torch.cat(outputs, 0),
        'ntokens': len(inputs),
    }


class AbstractPKMStudent(Model):
    @staticmethod
    def register(options):
        options.register(
            Argument('share_memory', default=False),
        )
        options.assume_defined('teacher_target_layers', by='PKMStudent')
        options.add_global_constraint(lambda args: not hasattr(args, 'optimizer') or args.optimizer == 'pair')
        options.add_global_constraint(lambda args: not args.mem_sparse or not hasattr(args, 'optimizer')
                                      or args.second_optimizer == 'sparse_adam')

    def __init__(self, args):
        super().__init__(args)
        self.input_size = self.output_size = self.args.teacher_hidden_size
        self.memories = nn.ModuleList([
            self.build_memory() for _ in range(len(self.args.teacher_target_layers))
        ])
        self.size = self.memories[0].size
        self.heads = self.memories[0].heads
        self.target_layers = {layer_id: i for i, layer_id in enumerate(self.args.teacher_target_layers)}
        self.value_param_postfix = self.memories[0].__class__.MEM_VALUES_PARAMS
        self.save_meta()

    def trainable_parameters(self):
        ordinary = []
        memory = []
        no_weight_decay = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'LayerNorm' in name or 'bias' in name:
                    no_weight_decay.append(param)
                elif name.endswith(self.value_param_postfix):
                    memory.append(param)
                else:
                    ordinary.append(param)
        return [{'params': ordinary}, {'params': no_weight_decay, 'weight_decay': 0.0}], [{'params': memory}]

    def save_meta(self):
        if should_save_meta(self.args):
            options = Options()
            self.register(options)
            meta = {name: getattr(self.args, name) for name in options.keys()}
            meta['mem_positions'] = self.args.teacher_target_layers
            meta['share_memory'] = self.args.share_memory
            with io.open(os.path.join(self.args.save_dir, f'mem_config.json'), 'w') as f:
                json.dump(meta, f, indent=2)

    def forward(self, inputs, layer_id, **kwargs):
        return self.memories[self.target_layers[layer_id]](inputs, **kwargs)


@register('pkm_student')
class PKMStudent(AbstractPKMStudent):
    @staticmethod
    def register(options):
        options.register(
            Argument('mem_k_dim', default=256, doc='Memory keys dimension', validate=lambda value: value % 2 == 0),
            Argument('mem_heads', default=4, doc='Number of memory heads'),
            Argument('mem_knn', default=32, doc='Number of memory slots to read / update - k-NN to the query'),
            Argument('mem_keys', default=512, doc='Number of product keys. Total memory size: n ** 2'),
            Argument('query_norm', validate=lambda value: not value or value in ('batchnorm', 'layernorm', 'groupnorm')),
            Argument('query_net', default='linear', validate=lambda value: value in 'linear mlp'.split()),
            Argument('distance_fn', default='dot', validate=lambda value: value in ('dot', 'euc', 'mah', 'mah_fast')),
            Argument('init_kernel_alpha', default=0.5, doc='default is standard Gaussian kernel'),
            Argument('mem_sparse', default=False, doc='Perform sparse updates for the values'),
            Argument('input_dropout', default=0.),
            Argument('query_dropout', default=0.),
        )

    def build_memory(self):
        args = self.args
        return HashingMemory(
            self.input_size, self.output_size, k_dim=args.mem_k_dim, n_keys=args.mem_keys, query_net=args.query_net,
            heads=args.mem_heads, knn=args.mem_knn, input_dropout=args.input_dropout,
            query_dropout=args.query_dropout, query_norm=args.query_norm, share_values=args.share_memory,
            distance_fn=args.distance_fn, init_kernel_alpha=args.init_kernel_alpha,
        )


@register('lopkm_student')
class LOPKMStudent(AbstractPKMStudent):
    @staticmethod
    def register(options):
        options.register(
            Argument('mem_k_dim', default=256, doc='Memory keys dimension', validate=lambda value: value % 2 == 0),
            Argument('mem_heads', default=4, doc='Number of memory heads'),
            Argument('mem_knn', default=32, doc='Number of memory slots to read / update - k-NN to the query'),
            Argument('mem_keys', default=512, doc='Number of product keys. Total memory size: n ** 2'),
            Argument('mem_sparse', default=False, doc='Perform sparse updates for the values'),
            Argument('input_dropout', default=0.),
        )

    def build_memory(self):
        args = self.args
        return LocallyOptimizedHashingMemory(
            self.input_size, self.output_size, k_dim=args.mem_k_dim, n_keys=args.mem_keys,
            heads=args.mem_heads, knn=args.mem_knn, input_dropout=args.input_dropout,
            share_values=args.share_memory, sparse=args.mem_sparse
        )


@register_loss('pkm_local_distill')
class PKMLocalDistillLoss(LocalDistillLoss):
    @staticmethod
    def register(options):
        pass

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert reduce is True
        logits, indices = model(**sample['net_input'], return_index=True)
        head_index = indices[:, 0] // model.size
        targets = sample['target']
        sample_loss = ((logits - targets) ** 2).mean(1)
        batch_size = targets.size(0)
        loss = 0
        logging_output = {}
        for i in range(model.heads):
            head_loss = sample_loss[head_index == i]
            if not head_loss.numel():
                logging_output.update({f'head_loss_{i}': -1})
                continue
            head_bs = head_loss.numel()
            head_loss = head_loss.sum()
            logging_output.update({f'head_loss_{i}': (head_loss / head_bs).item()})
            loss += head_loss
        loss = loss / batch_size
        logging_output['loss'] = loss.item()
        if len(self.args.teacher_target_layers) > 1:
            logging_output['layer_id'] = sample['net_input']['layer_id']
        return loss, 1, logging_output

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        """Aggregate logging outputs from data parallel training."""
        agg_output = super().aggregate_logging_outputs(logging_outputs, sample_size, max_count)
        head_loss_names = {key for key in logging_outputs[0].keys() if key.startswith('head_loss_')}
        if len(self.args.teacher_target_layers) > 1:
            for layer_id in self.args.teacher_target_layers:
                layer_loss = [log['loss'] for log in logging_outputs if log['layer_id'] == layer_id]
                agg_output[f'loss_layer_{layer_id}'] = sum(layer_loss) / len(layer_loss) if layer_loss else 0
        for name in head_loss_names:
            if len(self.args.teacher_target_layers) > 1:
                for layer_id in self.args.teacher_target_layers:
                    head_loss = [log[name] for log in logging_outputs if log[name] != -1 and log['layer_id'] == layer_id]
                    agg_output[f'L{layer_id}_' + name] = sum(head_loss) / len(head_loss) if head_loss else 0
            else:
                head_loss = [log[name] for log in logging_outputs if log[name] != -1]
                agg_output[name] = sum(head_loss) / len(head_loss) if head_loss else 0
        return agg_output
