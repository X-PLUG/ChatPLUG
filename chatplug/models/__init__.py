# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
Tensor = Union['torch.Tensor', 'tf.Tensor']

@dataclass
class TextGenerationModelOutput():
    """The output class for text generation models.

    Args:
        logits (`Tensor`): The logits output of the model. loss (`Tensor`,
        *optional*) The loss of the model, available when training.
        hidden_states (`Tensor`, *optional*) Hidden-states of the model at the
        output of each layer plus the optional initial embedding outputs.
    """

    logits: Tensor = None
    loss: Tensor = None

@dataclass
class TokenGeneratorOutput():
    """
    The output class for generate method of text generation models.


    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`
        is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size*num_return_sequences, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`
        is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(num_return_sequences*batch_size, num_heads, generated_length,
            sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`
        is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(num_return_sequences*batch_size, generated_length, hidden_size)`.
    """

    sequences: Tensor = None
    scores: Optional[Tuple[Tensor]] = None
    attentions: Optional[Tuple[Tuple[Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[Tensor]]] = None