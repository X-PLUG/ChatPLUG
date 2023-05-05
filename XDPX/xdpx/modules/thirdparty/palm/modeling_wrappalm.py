# coding=utf-8
# 
# Copyright (c) 2022 Alibaba.com, Inc. All Rights Reserved
"""
modeling_wrappalm.py

Authors: tjf141457 (tjf141457@alibaba-inc.com)
"""
from typing import Optional

import torch
import torch.nn.functional as F
from icecream import ic
from transformers.modeling_outputs import Seq2SeqModelOutput, Seq2SeqLMOutput

from xdpx.modules.thirdparty.palm.modeling_palm import PalmPreTrainedModel, PalmEncoderModel, PalmPreTrainingHeads, \
    PalmDecoderModel


class WrapPalmModel(PalmPreTrainedModel):
    """Palm model.
    The palm model consists of an encoder model, a pretrainedhead and a decoder model.

    Params:
        config: a PalmConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `decode_input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `decode_attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `is_infer`: a boolean indicating whether the model is used for inference.

    Outputs:
        if `is_infer` is `True`:
            Tuple of (preduction_scores, seq_relationship_logits, encoder_sequence_output).
        if `is_infer` is `False`:
            Tuple if (preduction_scores, seq_relationship_logits).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = PalmConfig(
      vocab_size_or_config_json_file=21504,
      hidden_size=768,
      num_hidden_layers=12,
      num_attention_heads=12,
      intermediate_size=3072)
    model = PalmModel(config)
    prediction_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super().__init__(config)
        self.bert = PalmEncoderModel(config)
        self.cls = PalmPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.decoder = PalmDecoderModel(config)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                checkpoint_activations=False,
                encoder_outputs=None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ):
        # TODO: check here
        # if decoder_input_ids is None:
        #   decoder_input_ids = shift_tokens_right(
        #     input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        #   )

        # sequence_output is None is for encoder
        if encoder_outputs is None:

            bsz, turnnum, seqlen = input_ids.shape
            input_ids = input_ids.view(bsz * turnnum, -1)
            # ic(input_ids)

            # TODO: attention mask not to None
            assert token_type_ids is None
            assert attention_mask is None

            encoder_outputs, pooled_output = self.bert(input_ids,
                                                       token_type_ids,
                                                       attention_mask,
                                                       output_all_encoded_layers=False,
                                                       checkpoint_activations=checkpoint_activations,
                                                       return_dict=False)
            prediction_scores, seq_relationship_score = self.cls(encoder_outputs, pooled_output)

            # if isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs.view(bsz, turnnum * seqlen, -1)
            # else:
            #  encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.view(bsz, turnnum * seqlen, -1)

        # sequence_output is None is for training
        else:
            encoder_outputs = encoder_outputs.last_hidden_state
            encoder_outputs = encoder_outputs.to(dtype=next(self.decoder.parameters()).dtype)

        decoder_outputs = self.decoder(self.bert.embeddings,
                                       encoder_outputs,
                                       decoder_input_ids,
                                       decoder_attention_mask,
                                       attention_mask,
                                       checkpoint_activations=checkpoint_activations,
                                       return_dict=return_dict)

        if not return_dict:
            # ic(decoder_outputs.shape)
            # ic(encoder_outputs.shape)
            # TODO(junfeng): with T5
            return (decoder_outputs, encoder_outputs)

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs,
        )


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        # need fixed?
        self.n_passages = 1
        self.embeddings = self.bert.embeddings

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                output_all_encoded_layers=True,
                checkpoint_activations=False,
                detach_index=-1,
                return_dict: Optional[bool] = None,
                ):
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)

        if attention_mask is not None:
            attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)

        encoder_outputs = self.bert(input_ids,
                                    token_type_ids,
                                    attention_mask,
                                    output_all_encoded_layers=False,
                                    checkpoint_activations=checkpoint_activations,
                                    return_dict=return_dict)
        if isinstance(encoder_outputs, tuple):
            encoder_outputs = (encoder_outputs[0].view(bsz, self.n_passages * passage_length, -1),) + encoder_outputs[
                                                                                                      1:]
        else:
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.view(bsz,
                                                                                       self.n_passages * passage_length,
                                                                                       -1)

        return encoder_outputs


class WrapPalmForConditionalGeneration(PalmPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = WrapPalmModel(config)
        self.encoder_proxy = None

    def get_encoder(self):
        """get_encoder for generation_utils singleton"""
        if self.encoder_proxy:
            return self.encoder_proxy
        else:
            self.encoder_proxy = EncoderWrapper(self.model.bert)
            return self.encoder_proxy

    def get_decoder(self):
        return self.model.decoder

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # input_ids,
        # attention_mask = None,
        # decoder_input_ids = None,
        # decoder_attention_mask = None,
        # token_type_ids = None,
        # checkpoint_activations = False,
        # encoder_outputs = None,
        # labels = None,

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                token_type_ids=None,
                checkpoint_activations=False,
                encoder_outputs=None,
                labels=None,
                past_key_values=None,
                output_attentions=None,
                output_hidden_states=None,
                dec_loss_mask=None,
                use_cache: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ):
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            # decoder_input_ids = shift_tokens_right(labels)
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

        if return_dict:
            outputs = self.model(input_ids,
                                 token_type_ids,
                                 attention_mask,
                                 decoder_input_ids,
                                 decoder_attention_mask,
                                 checkpoint_activations=checkpoint_activations,
                                 encoder_outputs=encoder_outputs,
                                 return_dict=None)
            lm_logits = F.linear(outputs[0], self.get_encoder().embeddings.word_embeddings.weight)
            # lm_logits = F.linear(outputs.last_hidden_state[0], self.get_encoder().embeddings.word_embeddings.weight)
        else:
            outputs = self.model(input_ids,
                                 token_type_ids,
                                 attention_mask,
                                 decoder_input_ids,
                                 decoder_attention_mask,
                                 checkpoint_activations=checkpoint_activations,
                                 encoder_outputs=encoder_outputs,
                                 return_dict=return_dict)
            lm_logits = F.linear(outputs[0], self.get_encoder().embeddings.word_embeddings.weight)

        masked_lm_loss = None

        if labels is not None:
            # ic(lm_logits.shape)
            # ic(labels.shape)
            masked_lm_loss = F.cross_entropy(lm_logits.view(-1, self.config.vocab_size),
                                             labels.view(-1).contiguous())

        # if not return_dict:
        #   output = (lm_logits,) + outputs[1:]
        #   return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits
            # ,
            # past_key_values=outputs.past_key_values,
            # decoder_hidden_states=outputs.decoder_hidden_states,
            # decoder_attentions=outputs.decoder_attentions,
            # cross_attentions=outputs.cross_attentions,
            # encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            # encoder_hidden_states=outputs.encoder_hidden_states,
            # encoder_attentions=outputs.encoder_attentions,
        )

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix,
                                     keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    @staticmethod
    def palm_batchify(data):
        tmp = {k: list() for k in data[0]}
        for dataset in data:
            for k, v in dataset.items():
                tmp[k].append(torch.tensor(v))
        data = {k: torch.stack(v, 0) for k, v in tmp.items()}

        tokens = data['input_ids'].long()
        types = data['segment_ids'].long()
        padding_mask = data['input_mask'].byte()
        target_ids = data['target_ids'].long()

        target_tokens = target_ids[:, :-1].contiguous()
        target_labels = target_ids[:, 1:].contiguous()

        def get_masks_and_position_ids(data, eod_token):
            _, seq_length = data.size()
            # Attention mask (lower triangular).
            att_mask_batch = 1
            attention_mask = torch.tril(torch.ones(
                (att_mask_batch, seq_length, seq_length), device=data.device)).view(
                att_mask_batch, 1, seq_length, seq_length)
            # Loss mask.
            loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
            loss_mask[data == eod_token] = 0.0
            # Position ids.
            position_ids = torch.arange(seq_length, dtype=torch.long,
                                        device=data.device)
            position_ids = position_ids.unsqueeze(0).expand_as(data).clone()

            return attention_mask, loss_mask, position_ids

            # Get the masks and postition ids.

        attention_mask, dec_loss_mask, position_ids = get_masks_and_position_ids(target_tokens, 0)

        return {"input_tokens": tokens, "token_type_ids": types, "attention_mask": padding_mask,
                "target_tokens": target_tokens, "position_ids": position_ids, "decode_attention_mask": attention_mask,
                "labels": target_labels, "dec_loss_mask": dec_loss_mask}


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
