# coding=utf-8

""" Encoder-Decoder model configuration """

import json
import os
import copy
from transformers import PretrainedConfig
from typing import Any, Dict, Union

T5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
  "t5-small": "https://huggingface.co/t5-small/resolve/main/config.json",
  "t5-base": "https://huggingface.co/t5-base/resolve/main/config.json",
  "t5-large": "https://huggingface.co/t5-large/resolve/main/config.json",
  "t5-3b": "https://huggingface.co/t5-3b/resolve/main/config.json",
  "t5-11b": "https://huggingface.co/t5-11b/resolve/main/config.json",
}


class EVAConfig(PretrainedConfig):
  model_type = "eva"
  keys_to_ignore_at_inference = ["past_key_values"]
  attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

  def __init__(
      self,
      d_model=768,
      d_kv=64,
      d_ff=256,
      num_layers=12,
      num_decoder_layers=12,
      num_heads=12,
      relative_attention_num_buckets=32,
      dropout_rate=0.1,
      layer_norm_epsilon=1e-6,
      initializer_factor=1.0,
      feed_forward_proj="relu",
      use_cache=True,
      use_scaled_init_for_output_weights=True,
      init_method_std=0.02,
      max_position_embeddings=1024,
      do_dim_trick=False,
      attn_scale=False,
      **kwargs
  ):
    super().__init__()
    self.d_model = d_model
    self.d_kv = d_kv
    self.d_ff = d_ff
    self.num_layers = num_layers
    self.num_decoder_layers = (
      num_decoder_layers if num_decoder_layers is not None else self.num_layers
    )  # default = symmetry
    self.num_heads = num_heads
    self.relative_attention_num_buckets = relative_attention_num_buckets
    self.dropout_rate = dropout_rate
    self.layer_norm_epsilon = layer_norm_epsilon
    self.initializer_factor = initializer_factor
    self.feed_forward_proj = feed_forward_proj
    self.use_cache = use_cache
    self.use_scaled_init_for_output_weights = use_scaled_init_for_output_weights
    self.init_method_std = init_method_std
    self.max_position_embeddings = max_position_embeddings
    self.vocab_size = None
    self.do_dim_trick = do_dim_trick
    self.attn_scale = attn_scale

  @classmethod
  def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "EVAConfig":
    config_dict = cls._dict_from_json_file(json_file)
    return cls(**config_dict)

  @classmethod
  def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
    with open(json_file, "r", encoding="utf-8") as reader:
      text = reader.read()
    return json.loads(text)

  def to_json_file(self, json_file_path: Union[str, os.PathLike]):
    with open(json_file_path, "w", encoding="utf-8") as writer:
      writer.write(self.to_json_string())

  def to_dict(self) -> Dict[str, Any]:
    output = copy.deepcopy(self.__dict__)
    if hasattr(self.__class__, "model_type"):
      output["model_type"] = self.__class__.model_type
    return output

  def to_json_string(self) -> str:
    config_dict = self.to_dict()
    return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
