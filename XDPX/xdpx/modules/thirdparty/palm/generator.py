# coding=utf-8
# 
# Copyright (c) 2022 Alibaba.com, Inc. All Rights Reserved
"""
generator.py

Authors: tjf141457 (tjf141457@alibaba-inc.com)
"""
import torch
from icecream import ic
from torch import nn


class TextGenerator:
  """
  Uses a model to translate a batch of sentences.


  Args:
     model (:obj:`onmt.modules.NMTModel`):
        NMT model to use for translation
     fields (dict of Fields): data fields
     beam_size (int): size of beam to use
     n_best (int): number of translations produced
     max_length (int): maximum length output to produce
     global_scores (:obj:`GlobalScorer`):
       object to rescore final translations
     copy_attn (bool): use copy attention during translation
     cuda (bool): use cuda
     beam_trace (bool): trace beam search for debugging
     logger(logging.Logger): logger.
  """

  def __init__(self,
               model,
               vocab,
               symbols,
               beam_size=5,
               min_length=0,
               max_length=64,
               global_scorer=None,
               logger=None,
               dump_beam=""):
    self.alpha = 0.6

    self.logger = logger
    # self.cuda = args.visible_gpus != '-1'
    self.cuda = (torch.cuda.device_count() > 0)

    self.model = model
    # TODO generator
    # self.generator = self.model.generator
    self.vocab = vocab
    self.symbols = symbols
    self.start_token = symbols['[CLS]']  # ['[PAD]']
    self.end_token = symbols['[SEP]']  # '[PAD]']

    self.global_scorer = global_scorer
    self.beam_size = beam_size
    self.min_length = min_length
    self.max_length = max_length

    self.dump_beam = dump_beam

    # for debugging
    self.beam_trace = self.dump_beam != ""
    self.beam_accum = None

    if self.beam_trace:
      self.beam_accum = {
        "predicted_ids": [],
        "beam_parent_ids": [],
        "scores": [],
        "log_probs": []}

  def _build_target_tokens(self, pred):
    # vocab = self.fields["tgt"].vocab
    tokens = []
    for tok in pred:
      tok = int(tok)
      tokens.append(tok)
      if tokens[-1] == self.end_token:
        tokens = tokens[:-1]
        break
    tokens = [t for t in tokens if t < len(self.vocab)]
    tokens = self.vocab.DecodeIds(tokens).split(' ')
    return tokens

  def tile(self, x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
      perm[0], perm[dim] = perm[dim], perm[0]
      x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
      .transpose(0, 1) \
      .repeat(count, 1) \
      .transpose(0, 1) \
      .contiguous() \
      .view(*out_size)
    if dim != 0:
      x = x.permute(perm).contiguous()
    return x

  def translate_batch(self, encoder_inputs, fast=False):
    """
    Translate a batch of sentences.

    Mostly a wrapper around :obj:`Beam`.

    Args:
       batch (:obj:`Batch`): a batch from a dataset object
       data (:obj:`Dataset`): the dataset object
       fast (bool): enables fast beam search (may not support all features)

    Todo:
       Shouldn't need the original dataset.
    """
    with torch.no_grad():
      return self._fast_translate_batch(encoder_inputs, self.max_length, min_length=self.min_length)

  def _fast_translate_batch(self,
                            encoder_inputs,
                            max_length,
                            min_length=0):
    # TODO: faster code path for beam_size == 1.
    # TODO: support these blacklisted features.
    assert not self.dump_beam

    beam_size = self.beam_size
    tokens, types, padding_mask = encoder_inputs
    batch_size = tokens.size(0)
    device = tokens.device
    tmp_alive_seq = torch.full(
      [batch_size, 1],
      self.start_token,
      dtype=torch.long,
      device=device)
    prediction_scores, dec_feat_seq, sequence_output = self.model(tokens, types, padding_mask, tmp_alive_seq, None,
                                                                  None, checkpoint_activations=False, is_infer=True,
                                                                  sequence_output=None)
    # ic(prediction_scores, dec_feat_seq, sequence_output)
    src_features = sequence_output

    # Tile states and memory beam_size times.
    # dec_states.map_batch_fn(
    #     lambda state, dim: tile(state, beam_size, dim=dim))
    src_features = self.tile(src_features, beam_size, dim=0)
    attention_mask = self.tile(padding_mask, beam_size, dim=0)
    # TODO support p_gen ...
    # if self.args.p_gen:
    #     src = tile(batch.src, beam_size, dim=0)
    batch_offset = torch.arange(
      batch_size, dtype=torch.long, device=device)
    beam_offset = torch.arange(
      0,
      batch_size * beam_size,
      step=beam_size,
      dtype=torch.long,
      device=device)
    alive_seq = torch.full(
      [batch_size * beam_size, 1],
      self.start_token,
      dtype=torch.long,
      device=device)

    # Give full probability to the first beam on the first step.
    topk_log_probs = (
      torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                   device=device).repeat(batch_size))

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

    results = {}
    results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
    results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
    results["gold_score"] = [0] * batch_size
    results["batch"] = []
    dec_attn_mask = None
    dec_position_ids = None

    for step in range(max_length):
      tgt_len = alive_seq.size()[1]
      ## modified with tgt_len
      # repeat_attention_mask = attention_mask.repeat([1, 1, tgt_len, 1])
      # dec_attn_mask = _make_causal_mask(alive_seq.shape, attention_mask.dtype, device=alive_seq.device)
      # dec_feat_seq = self.model.decode(self.model.bert.embeddings, src_features, alive_seq,
      #                           enc_attn_mask=repeat_attention_mask, dec_attn_mask=dec_attn_mask)

      prediction_scores, dec_feat_seq, _ = self.model(tokens, types, attention_mask, alive_seq, dec_position_ids,
                                                      dec_attn_mask, checkpoint_activations=False, is_infer=True,
                                                      sequence_output=src_features)

      dec_feat_seq = dec_feat_seq[:, -1, :]
      vocab_size = dec_feat_seq.size(-1)
      log_probs = torch.log(torch.softmax(dec_feat_seq.view(-1, vocab_size), dim=-1))

      if step < min_length:
        log_probs[:, self.end_token] = -1e20
      log_probs += topk_log_probs.view(-1).unsqueeze(1)

      alpha = self.alpha  # global_scorer.alpha
      length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
      curr_scores = log_probs / length_penalty

      curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
      topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)
      topk_log_probs = topk_scores * length_penalty

      # Resolve beam origin and true word ids.

      # topk_beam_index = topk_ids.div(vocab_size, rounding_mode="trunc")
      # topk_beam_index = topk_ids.div(vocab_size)
      topk_beam_index = topk_ids // vocab_size
      topk_ids = topk_ids.fmod(vocab_size)

      # Map beam_index to batch_index in the flat representation.
      batch_index = (
          topk_beam_index
          + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
      select_indices = batch_index.view(-1)
      # ic(select_indices.type())
      # ic(select_indices)

      # Append last prediction.
      alive_seq = torch.cat(
        [alive_seq.index_select(0, select_indices),
         topk_ids.view(-1, 1)], -1)

      is_finished = topk_ids.eq(self.end_token)
      if step + 1 == max_length:
        is_finished.fill_(1)  # self.end_token)
      # End condition is top beam is finished.
      end_condition = is_finished[:, 0].eq(1)  # self.end_token)
      # Save finished hypotheses.
      if is_finished.any():
        predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
        for i in range(is_finished.size(0)):
          b = batch_offset[i]
          if end_condition[i]:
            is_finished[i].fill_(1)  # self.end_token)
          finished_hyp = is_finished[i].nonzero().view(-1)
          # Store finished hypotheses for this batch.
          for j in finished_hyp:
            hypotheses[b].append((
              topk_scores[i, j],
              predictions[i, j, 1:]))
          # If the batch reached the end, save the n_best hypotheses.
          if end_condition[i]:
            best_hyp = sorted(
              hypotheses[b], key=lambda x: x[0], reverse=True)
            # if self.args.dataset == "qg_ranking_test" or (self.args.dataset == 'paraphrase' and (not self.args.sample_topk)):
            #     for each in best_hyp[:beam_size]:
            #         score, pred = each
            #         results["scores"][b].append(score)
            #         results["predictions"][b].append(pred)
            # else:
            score, pred = best_hyp[0]
            results["scores"][b].append(score)
            results["predictions"][b].append(pred)
        non_finished = end_condition.eq(0).nonzero().view(-1)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
          break
        # Remove finished batches for the next step.
        topk_log_probs = topk_log_probs.index_select(0, non_finished)
        batch_index = batch_index.index_select(0, non_finished)
        batch_offset = batch_offset.index_select(0, non_finished)
        alive_seq = predictions.index_select(0, non_finished) \
          .view(-1, alive_seq.size(-1))
      # Reorder states.
      select_indices = batch_index.view(-1)
      src_features = src_features.index_select(0, select_indices)
      attention_mask = attention_mask.index_select(0, select_indices)

    return results
