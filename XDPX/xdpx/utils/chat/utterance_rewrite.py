from transformers import BertTokenizer
import jieba
from typing import List

from xdpx.utils import cache_file
import os
from icecream import ic
from xdpx.utils import io
import torch
from xdpx.utils.chat import DEVICE
from xdpx.utils.chat.base import HistoryItem
from dataclasses import dataclass

@dataclass
class RewriteConfig:
    utterance_rewriter_save_dir: str
    utterance_rewriter_is_onnx: bool
    utterance_rewriter_quantized: bool
    utterance_rewriter_provider: str


class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pre_tokenizer(self, x):
        return jieba.cut(x, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        text = text.replace("[", '').replace("]", '')
        split_tokens = []
        jieba_cut_list = [each for each in self.pre_tokenizer(text)]
        for text in jieba_cut_list:
            if text in self.vocab:
                split_tokens.append(text)
            elif "["+text+"]" in self.vocab:
                split_tokens.append("["+text+"]")
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens

class RewriteModel:
    def __init__(self, save_dir, is_onnx=False, quantized=False, provider='cuda'):
        # step 1. init argument and prepare model file
        if is_onnx:
            from xdpx.utils.thirdparty.onnx_transformers.models.t5.onnx_model import OnnxT5
            from transformers import AutoConfig
            model_name = 'summary_model'
            if save_dir.startswith("oss://"):
                assert not quantized
                quantized_str = '-quantized' if quantized else ''
                encoder_path = cache_file(
                    os.path.join(save_dir, '{}_encoder{}.onnx'.format(model_name, quantized_str)))
                ic(encoder_path)
                decoder_path = cache_file(
                    os.path.join(save_dir, '{}_decoder{}.onnx'.format(model_name, quantized_str)))
                ic(decoder_path)
                init_decoder_path = cache_file(
                    os.path.join(save_dir, '{}_decoder_init{}.onnx'.format(model_name, quantized_str)))
                ic(init_decoder_path)
                _ = cache_file(os.path.join(save_dir, 'config.json'))
                _ = cache_file(os.path.join(save_dir, 'vocab.txt'))
                save_dir = os.path.dirname(encoder_path)

            self.tokenizer = T5PegasusTokenizer.from_pretrained(save_dir)
            ic('rewrite start get_onnx_model...')
            config = AutoConfig.from_pretrained(save_dir)
            self.model = OnnxT5(model_name, save_dir, provider, config)
            self.model = self.model.to(DEVICE)
            ic('rewrite end get_onnx_model .')
        else:
            if save_dir.startswith("oss://"):
                _ = cache_file(os.path.join(save_dir, 'vocab.txt'))
                _ = cache_file(os.path.join(save_dir, 'summary_model'))
                config_file = cache_file(os.path.join(save_dir, 'config.json'))
                save_dir = os.path.dirname(config_file)

            # step 2. pload finetuned model and tokenizer
            self.tokenizer = T5PegasusTokenizer.from_pretrained(save_dir)
            with io.open(os.path.join(save_dir, 'summary_model'), 'rb') as state_dict:
                self.model = torch.load(state_dict, map_location=DEVICE)
        self.cuda = torch.cuda.is_available()
        self.model.eval()

    def rewrite(self, query, history: List[HistoryItem]) -> tuple:
        '''

        Args:
            query: before rewriing
            history: dialog context

        Returns:
            query after rewriting
        '''
        if not history:
            return query, []

        utterances = [t.rewritten_utterance or t.utterance for t in history[-4:]] + [query]
        processed_utterances = [each for each in utterances]
        sentence = '[SEP]'.join(processed_utterances)
        input_ids = self.tokenizer(sentence, padding=True, truncation=True, max_length=200,
                                   return_tensors="pt").input_ids
        input_ids = input_ids.to(DEVICE)
        hypotheses = self.model.generate(input_ids, eos_token_id=self.tokenizer.sep_token_id,
                                         decoder_start_token_id=self.tokenizer.cls_token_id,
                                         max_length=80)  # 若不设置最大长度默认为20会导致截断
        if self.cuda:
            hypotheses = hypotheses.detach().cpu().tolist()

        result = self.tokenizer.decode(hypotheses[0], skip_special_tokens=True).replace(' ', '')
        return result, utterances
