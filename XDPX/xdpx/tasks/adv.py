import torch
from . import register, Task
from xdpx.options import Argument


@register('freelb')
class FreeLBTask(Task):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('adv_lr', default=2e-1),
            Argument('adv_steps', default=2),
            Argument('rand_init_mag', default=3.2e-1),
            Argument('max_norm', default=7e-1),
            Argument('adv_begin_iter', default=-1),
            Argument('norm_method', default='l2', validate=lambda value: value in 'l2 linf'.split()),
            Argument('no_sync_dp', default=False),
            Argument('grad_square', default=False),
            Argument('recompute_emb', default=False),
            Argument('orig_loss_proportion', default=0., validate=lambda value: 0 <= value < 1),
            domain='adversarial'
        )
        options.set_default('force_sync', True, strict=False)  # otherwise the "backward" call will not accumulate gradients across GPUs
        options.add_global_constraint(lambda args: not args.fp16)
        options.add_global_constraint(lambda args: not ((args.rand_init_mag == 0 or args.adv_begin_iter > 0) and (args.update_freq > 1 or args.auto_ga)))

    def __init__(self, args):
        super().__init__(args)
        self.rng_states = {}

    def bottom_predict(self, model, sample):
        net_input = sample['net_input']
        input_ids = net_input['input_ids']
        input_mask = input_ids.ne(self.args.pad_index).float()
        embeds_init = model.get_embeddings()(input_ids)
        return embeds_init, input_mask, net_input

    def upper_predict(self, model, embeds, net_input):
        net_input = {**net_input, 'inputs_embeds': embeds}
        self.restore_rng_states()
        return model(**net_input)

    def train_step(self, sample, model, loss, optimizer, num_updates=0):
        loss_module = loss
        model.train()
        total_loss = 0

        embeds_init, input_mask, net_input = self.bottom_predict(model, sample)
        input_lens = input_mask.sum(1)
        self.stash_rng_states()

        adv_proportion = 1 - self.args.orig_loss_proportion
        # freelb adversarial training
        if self.args.rand_init_mag > 0 and num_updates >= self.args.adv_begin_iter:

            if self.args.norm_method == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lens * embeds_init.size(-1)
                mag = self.args.rand_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.args.norm_method == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.args.rand_init_mag, self.args.rand_init_mag) * input_mask.unsqueeze(2)

            delta.requires_grad_()
            logits = self.upper_predict(model, delta + embeds_init, net_input)
            loss, sample_size, logging_output = loss_module(model, sample, logits=logits)
        else:
            delta = 0
            self.restore_rng_states()
            loss, sample_size, logging_output = loss_module(model, sample)

        if num_updates >= self.args.adv_begin_iter:
            loss = loss / (1 + self.args.adv_steps) * adv_proportion

        optimizer.backward(loss, retain_graph=True)
        total_loss += loss.detach()

        if self.args.rand_init_mag > 0 and num_updates >= self.args.adv_begin_iter:
            delta_grad = delta.grad.clone().detach()
        else:
            delta_grad = model.get_embeddings().grad.clone().detach()

        for _ in range(self.args.adv_steps):
            if num_updates < self.args.adv_begin_iter:
                break

            if self.args.norm_method == "l2":
                # doing l2 norm normalization and clipping
                denorm = torch.clamp(torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1), min=1e-10)
                if self.args.grad_square:
                    denorm = denorm ** 2
                delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()
                if self.args.max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).to(embeds_init).detach()
                    exceed_mask = (delta_norm > self.args.max_norm).to(embeds_init)
                    delta = delta * (self.args.max_norm / delta_norm * exceed_mask + (1-exceed_mask)).view(-1, 1, 1).detach()
            elif self.args.norm_method == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()
                if self.args.max_norm > 0:
                    delta = torch.clamp(delta, -self.args.max_norm, self.args.max_norm).detach()

            delta.requires_grad_()
            if self.args.recompute_emb:
                embeds_init_adv, *_ = self.bottom_predict(model, sample)
            else:  # these two are not exactly the same (don't know why)
                embeds_init_adv = embeds_init
            logits = self.upper_predict(model, delta + embeds_init_adv, net_input)
            loss, sample_size, logging_output = loss_module(model, sample, logits=logits)

            loss = loss / (1 + self.args.adv_steps) * adv_proportion
            optimizer.backward(loss, retain_graph=True)
            delta_grad = delta.grad.clone().detach()
            total_loss += loss.detach()

        if self.args.orig_loss_proportion > 0:
            logits = self.upper_predict(model, embeds_init, net_input)
            loss, sample_size, logging_output = loss_module(model, sample, logits=logits)
            loss = loss * self.args.orig_loss_proportion
            optimizer.backward(loss)
            total_loss += loss.detach()

        logging_output['ntokens'] = sample['ntokens']
        return total_loss, sample_size, logging_output

    def stash_rng_states(self):
        # stash the random states, so we can generate exactly the same dropout mask later
        self.rng_states['rng_state'] = torch.get_rng_state()
        if self.args.cuda:
            self.rng_states['cuda_rng_state'] = torch.cuda.get_rng_state()

    def restore_rng_states(self):
        if not self.args.no_sync_dp:
            torch.set_rng_state(self.rng_states['rng_state'])
            if self.args.cuda:
                torch.cuda.set_rng_state(self.rng_states['cuda_rng_state'])


@register('freelb_cnn')
class FreeLBCNNTask(FreeLBTask):
    def bottom_predict(self, model, sample):
        net_input = {**sample['net_input']}
        cnn_net_input = {
            key: net_input.pop(key) for key in 'token_mask words word_mask word_begin_mask'.split()
        }
        _, cls_token, bert_features = model.bert_forward(**net_input)
        cnn_net_input['cls_token'] = cls_token
        bert_features = bert_features[self.args.bert_seq_layer]
        return bert_features, cnn_net_input['token_mask'].float(), cnn_net_input

    def upper_predict(self, model, embeds, net_input):
        return model.cnn_forward(embeds, **net_input)


@register('freelb_cnn_distill')
class FreeLBCNNDistillTask(FreeLBTask):
    def upper_predict(self, model, embeds, net_input):
        mask = net_input['mask'].unsqueeze(2)
        a = model.encoder(embeds, mask)
        a = model.pooling(a, mask)
        a = model.prediction(a)
        return a

    def bottom_predict(self, model, sample):
        net_input = sample['net_input']
        input_ids = net_input['tokens']
        input_mask = input_ids.ne(self.args.pad_index).float()
        embeds_init = model.embedding(input_ids)
        return embeds_init, input_mask, net_input
