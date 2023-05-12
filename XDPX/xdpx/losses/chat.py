import math
from xdpx.models.chat import FIDT5Chat, T5Chat
from . import Loss, register
from xdpx.options import Argument
import torch.nn.functional as F
import torch


@register('chat')
class ChatLoss(Loss):
    @staticmethod
    def register(options):
        options.register(
            Argument('predict_threshold', type=float),
        )

    def forward(self, model, sample, logits=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        padding_index = 0

        input_ids = sample['net_input']['input_ids']
        decoder_input_ids = sample['net_input']['decoder_input_ids']

        outputs = model(**sample['net_input'])
        loss = outputs.loss

        decoder_input_mask = decoder_input_ids.ne(padding_index).long()
        # consider padding with index -100
        decoder_input_mask_ignore = decoder_input_ids.ne(-100).long()
        decoder_input_mask = decoder_input_mask*decoder_input_mask_ignore

        token_size = torch.sum(decoder_input_mask).item()

        logging_output = {}
        logging_output.update({
            # 'loss': loss.detach().cpu(),
            'loss_sum': loss.item()*token_size,
            'token_size': token_size,
        })

        # loss has already averaged, so 1 is used as the sample_size
        return loss, 1, logging_output


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `aggregate_logging_outputs`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def inference(self, model, sample, tokenizer=None):
        assert tokenizer is not None, 'ChatTask inference tokenizer should not be None '
        input_ids = sample['net_input']['input_ids']
        generate_config = {
            'num_beams': 4,
            'num_return_sequences': 4,
            'num_beam_groups': 4,
            'diversity_penalty': 2.0,
            'temperature': 0.8,
            'do_sample': False,
            'early_stopping': True,
            'top_k': 50,
            'top_p': 0.8,
            'repetition_penalty': 1.2,
            'length_penalty': 1.2,
            'min_length': 10,
            'max_length': 80,
            'no_repeat_ngram_size': 2
        }

        return_dict = model.generate(input_ids,
                                     eos_token_id=tokenizer.sep_token_id,
                                     decoder_start_token_id=tokenizer.cls_token_id,
                                     output_scores=True,
                                     return_dict_in_generate=True,
                                     **generate_config)
        hypotheses = return_dict.sequences
        sequences_scores = return_dict.sequences_scores
        responses = []
        for h in hypotheses:
            decoded_hypo = tokenizer.decode(h, skip_special_tokens=True).replace(' ', '')
            responses.append(decoded_hypo)
        return responses, sequences_scores

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        """Aggregate logging outputs from data parallel training."""

        agg_output = {
            'sample_size': sample_size,
        }
        if 'ntokens' in logging_outputs[0]:
            ntokens = sum(log['ntokens'] for log in logging_outputs)
            agg_output['ntokens'] = ntokens

        if max_count:
            sample_size = max(min(max_count, sample_size), 1)

        loss = [log['loss_sum'] for log in logging_outputs]
        token_size = [log['token_size'] for log in logging_outputs]
        # agg_output['loss'] = sum(loss) / sample_size / math.log(2) if sample_size > 0 else 0.
        agg_output['loss'] = sum(loss) / sum(token_size)

        return agg_output


@register('chat_ctr')
class ChatCtrLoss(ChatLoss):
    @staticmethod
    def register(options):
        options.register(
            Argument('normalize', default=True),
            Argument('score_mode', default='log'),
            Argument('length_penalty', default=1.2),
            Argument('pad_token_id', default=-100),
            Argument('margin', default=0.0),
            Argument('gold_margin', default=0.0),
            Argument('gold_weight', default=0.0),
            Argument('rank_weight', default=1.0),
            Argument('mle_weight', default=1.0)
        )

    def ranking_loss(self, score, gold_score=None):
        '''

        Args:
            score:[batch_size, n_cand]
            gold_score: [batch_size]

        Returns:

        '''
        ones = torch.ones_like(score)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(score, score, ones)
        # candidate loss
        n = score.size(1)
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(self.args.margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss

        # gold loss
        pos_score = gold_score.unsqueeze(-1).expand_as(score)
        neg_score = score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones_like(pos_score)
        loss_func = torch.nn.MarginRankingLoss(self.args.gold_margin)
        TotalLoss += self.args.gold_weight * loss_func(pos_score, neg_score, ones)
        return TotalLoss

    def forward(self, model, sample, logits=None, reduce=True):
        input_ids = sample['net_input']['input_ids']
        gold_decoder_input_ids = sample['net_input']['decoder_input_ids']
        outputs = model(**sample['net_input'])
        mle_loss = outputs.loss
        sample_size = input_ids.shape[0]

        gold_logits = outputs.logits  # batch_size, max_length, vocab_size
        cands_input_ids = sample['cands_input_ids']  # batch_size, n_cand, max_length
        n_cand = cands_input_ids.shape[1]
        cands_output_logits = []
        for cand_i in range(n_cand):
            candidate_id = cands_input_ids[:, cand_i].contiguous()  # batch_size, max_length
            cand_i_outputs = model(input_ids=input_ids, decoder_input_ids=candidate_id)
            cand_i_logits = cand_i_outputs.logits  # batch_size, max_length, vocab_size
            cands_output_logits.append(cand_i_logits.unsqueeze(1))
        cands_output_logits = torch.concat(cands_output_logits, dim=1)  # batch_size, n_cand, max_length, vocab_size

        cand_mask = cands_input_ids != self.args.pad_token_id  # batch_size, n_cand, max_length
        cand_mask = cand_mask.float()
        gold_mask = gold_decoder_input_ids != self.args.pad_token_id  # batch, max_length,
        gold_mask = gold_mask.float()
        cands_input_ids = torch.mul(cands_input_ids, cand_mask).to(torch.int64).unsqueeze(-1)
        gold_decoder_input_ids = torch.mul(gold_decoder_input_ids, gold_mask).to(torch.int64).unsqueeze(-1)
        if self.args.normalize:
            if self.args.score_mode == "log":
                _output = F.log_softmax(cands_output_logits, dim=3)
                gold_score = F.log_softmax(gold_logits, dim=2)
            else:
                _output = F.softmax(cands_output_logits, dim=3)
                gold_score = F.softmax(gold_logits, dim=2)
            scores = torch.gather(_output, 3, cands_input_ids).squeeze(-1)  # [batch_size, n_cand, max_length]
            gold_score = torch.gather(gold_score, 2, gold_decoder_input_ids).squeeze(-1)  # [batch_size, max_length]
        else:
            scores = torch.gather(cands_output_logits, 3, cands_input_ids).squeeze(
                -1)  # [batch_size, n_cand, max_length]
            gold_score = torch.gather(gold_logits, 2, gold_decoder_input_ids).squeeze(-1)  # [batch_size, max_length]

        scores = torch.mul(scores, cand_mask).sum(-1) / (
                (cand_mask.sum(-1)) ** self.args.length_penalty)  # [batch_size, cand_num]

        gold_score = torch.mul(gold_score, gold_mask).sum(-1) / (
                (gold_mask.sum(-1)) ** self.args.length_penalty)  # [batch_size]

        ranking_loss = self.ranking_loss(scores, gold_score)
        loss = self.args.rank_weight * ranking_loss + self.args.mle_weight * mle_loss
        logging_output = {}
        logging_output.update({
            'loss': loss.detach().cpu(),
            'ranking_loss': ranking_loss.detach().cpu(),
            'mle_loss': mle_loss.detach().cpu(),
            'sample_size': sample_size,
        })

        return loss, sample_size, logging_output

    def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
        """Aggregate logging outputs from data parallel training."""

        agg_output = {
            'sample_size': sample_size,
        }
        if 'ntokens' in logging_outputs[0]:
            ntokens = sum(log['ntokens'] for log in logging_outputs)
            agg_output['ntokens'] = ntokens

        if max_count:
            sample_size = max(min(max_count, sample_size), 1)
        loss = [log['loss'] for log in logging_outputs]
        agg_output['loss'] = sum(loss) / sample_size / math.log(2) if sample_size > 0 else 0.
        ranking_loss = [log['ranking_loss'] for log in logging_outputs]
        agg_output['ranking_loss'] = sum(ranking_loss) / sample_size / math.log(2) if sample_size > 0 else 0.
        mle_loss = [log['mle_loss'] for log in logging_outputs]
        agg_output['mle_loss'] = sum(mle_loss) / sample_size / math.log(2) if sample_size > 0 else 0.

        return agg_output
