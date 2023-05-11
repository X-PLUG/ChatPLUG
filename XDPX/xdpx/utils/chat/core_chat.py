'''
core chat model class
'''
from typing import List, Tuple, Optional

import torch
import os
import time
import random
from xdpx.tasks import tasks
from xdpx.loaders import loaders
from xdpx.options import Options, Argument, Arg
from xdpx.utils import cache_file
from xdpx.utils import io
from xdpx.models.chat import FIDT5Chat
from transformers import AutoTokenizer

from icecream import ic
from . import DEVICE
import re


def convert_to_local_cfg_vocab(args):
    config = {
        'google/mt5-base': 'tests/sample_data/mt5-base/config.json'
    }
    vocab = {
        'google/mt5-base': 'tests/sample_data/mt5-base'
    }
    args.auto_model = config.get(args.auto_model, args.auto_model)
    args.vocab_file = vocab.get(args.vocab_file, args.vocab_file)
    return args


def process_context(context_list):
    subject = "我"
    for i in range(len(context_list) - 1, -1, -1):
        if len(context_list[i]) > 0 and context_list[i][
            -1] not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~、。，？！；：“”（）【】《》〈〉……':
            context_list[i] = context_list[i] + "。"
        context_list[i] = subject + "：" + context_list[i]
        subject = "你" if subject == "我" else "我"
    return "".join(context_list)


def process_history(history_list):
    subject = "你"
    for i in range(len(history_list) - 1, -1, -1):
        if len(history_list[i]) > 0 and history_list[i][
            -1] not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~、。，？！；：“”（）【】《》〈〉……':
            history_list[i] = history_list[i] + "。"
        history_list[i] = subject + "：" + history_list[i]
        subject = "你" if subject == "我" else "我"
    return "".join(history_list)


context_template = "假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}"
history_template = "假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}#以下是在此之前我们的对话内容，可作为回复时的参考。{history}"
knowledge_template = "假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}#以下是和对话相关的知识，请你参考该知识进行回复。{knowledge}"
user_profile_template = "假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}#假设以下是你对我所了解的信息，请你参考该信息并避免你的回复和该信息矛盾，信息如下：{user_profile}"
bot_profile_template = "假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}#假设以下是你的人物设定，请你参考该信息并避免你的回复和该信息矛盾，信息如下：{bot_profile}"


class CoreChat(object):
    def __init__(self, save_dir,
                 checkpoint=None,
                 is_onnx=False,
                 pretrained_version='google/mt5-base',
                 quantized=False,
                 provider='cuda',
                 allspark_gpu_speed_up=False,
                 allspark_gen_cfg=None,
                 max_encoder_length=300,
                 bad_words=None,
                 no_repeat_session_ngrams=4,
                 use_instruction=False
                 ):
        self.allspark_gpu_speed_up = allspark_gpu_speed_up
        self.max_encoder_length = max_encoder_length
        self.bad_words = bad_words.split('|') if bad_words else []
        self.no_repeat_session_ngrams = no_repeat_session_ngrams
        self.use_instruction = use_instruction

        if is_onnx:  # onnx t5 model
            from xdpx.utils.thirdparty.onnx_transformers.models.t5.onnx_model import OnnxT5
            model_name = pretrained_version.split('/')[1]
            assert not quantized
            quantized_str = '-quantized' if quantized else ''
            encoder_path = cache_file(os.path.join(save_dir, '{}_encoder{}.onnx'.format(model_name, quantized_str)))
            ic(encoder_path)
            decoder_path = cache_file(os.path.join(save_dir, '{}_decoder{}.onnx'.format(model_name, quantized_str)))
            ic(decoder_path)
            init_decoder_path = cache_file(
                os.path.join(save_dir, '{}_decoder_init{}.onnx'.format(model_name, quantized_str)))
            ic(init_decoder_path)

            save_dir = os.path.dirname(encoder_path)
            ic('start get_onnx_model...')
            backbone = OnnxT5(pretrained_version, save_dir, provider)
            ic('end get_onnx_model.')
            options = Options()
            options.register(
                Argument('gradient_checkpointing', default=False),
                Argument('auto_model', default=None)
            )
            args = options.parse_dict({
                'gradient_checkpointing': False,
                'auto_model': None
            })
            model = FIDT5Chat(args, backbone)
            self.model = model.to(DEVICE)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_version)
        else:
            with io.open(os.path.join(save_dir, 'args.py')) as f:
                args = Options.parse_tree(eval(f.read()))
            try:
                with io.open(os.path.join(args.data_dir, 'args.py')) as f:
                    args = Arg().update(Options.parse_tree(eval(f.read()))).update(args)
            except IOError:
                pass
            args = convert_to_local_cfg_vocab(args)
            args.__cmd__ = 'serve'
            args.save_dir = save_dir
            args.strict_size = True
            # build the task
            task = tasks[args.task](args)
            model = task.build_model(args)
            model.load(checkpoint)

            self.model = model.cuda() if torch.cuda.is_available() else model
            if allspark_gpu_speed_up:
                self.model.load_allspark(allspark_gen_cfg)
            self.tokenizer = loaders[args.loader](args).tokenizer.tokenizer

        self.is_t5 = isinstance(self.model, FIDT5Chat)
        from threading import Lock
        self.lock = Lock()

    def build_model_input(self, utterance: str, context: list, history: list, user_profile: list,
                          bot_profile: list,
                          knowledge_list: list) -> Tuple[List[str], str]:
        '''
       Args:
           utterance:
           context:
           history:
           user_profile:
           bot_profile:
           knowledge_list:
       '''
        if self.is_t5:
            if self.use_instruction:
                return self.build_model_input_for_t5_instruction(utterance, context, history, user_profile, bot_profile,
                                                                 knowledge_list)
            else:
                return self.build_model_input_for_t5(utterance, context, history, user_profile, bot_profile, knowledge_list)
        else:
            if self.use_instruction:
                return self.build_model_input_for_plug_instruction(utterance, context, history, user_profile,
                                                                   bot_profile,
                                                                   knowledge_list)
            else:
                return self.build_model_input_for_plug(utterance, context, history, user_profile, bot_profile,
                                                   knowledge_list)

    def build_model_input_for_t5_instruction(self, utterance: str, context: list, history: list, user_profile: list,
                                             bot_profile: list,
                                             knowledge_list: list) -> Tuple[List[str], str]:

        model_input = []

        if context:
            context = context + [utterance]
        else:
            context = [utterance]
        context = process_context(context)

        if history and len(history) > 0:
            history = process_history(history)
            model_input.append(history_template.format(context=context, history=history))
        if knowledge_list and len(knowledge_list) > 0:
            for knowledge in knowledge_list:
                model_input.append(knowledge_template.format(context=context, knowledge=knowledge))
        if user_profile and len(user_profile) > 0:
            for profile in user_profile:
                model_input.append(user_profile_template.format(context=context, user_profile=profile))
        if bot_profile:
            for profile in bot_profile:
                model_input.append(bot_profile_template.format(context=context, bot_profile=profile))

        if not model_input:
            model_input.append(context_template.format(context=context))

        return model_input, context

    def build_model_input_for_t5(self, utterance: str, context: list, history: list, user_profile: list,
                                 bot_profile: list,
                                 knowledge_list: list) -> Tuple[List[str], str]:

        model_input = []
        SEP_TOKEN = '</s>'
        CONTEXT_PREFIX = 'context: '
        UTTERANCE_PREFIX = 'utterance: '

        if context:
            context = f'{CONTEXT_PREFIX}{SEP_TOKEN.join(context)} {SEP_TOKEN} {UTTERANCE_PREFIX} {utterance}'
        else:
            context = f'{UTTERANCE_PREFIX}{utterance}'

        if history:
            history = SEP_TOKEN.join(history)
            model_input.append(f'{context}{SEP_TOKEN} history:{history}')
        if knowledge_list:
            for knowledge in knowledge_list:
                model_input.append(f'{context}{SEP_TOKEN} knowledge:{knowledge}')
        if user_profile:
            for profile in user_profile:
                model_input.append(f'{context}{SEP_TOKEN} user_profile:{profile}')
        if bot_profile:
            for profile in bot_profile:
                model_input.append(f'{context}{SEP_TOKEN} bot_profile:{profile}')

        if not model_input:
            model_input.append(f'{context}')

        return model_input, context

    def build_model_input_for_plug_instruction(self, utterance: str, context: list, history: list, user_profile: list,
                                               bot_profile: list,
                                               knowledge_list: list) -> Tuple[List[str], str]:

        model_input = []

        if context:
            context = context + [utterance]
        else:
            context = [utterance]
        context = process_context(context)

        if history and len(history) > 0:
            history = process_history(history)
            model_input.append(history_template.format(context=context, history=history))
        if knowledge_list and len(knowledge_list) > 0:
            for knowledge in knowledge_list:
                model_input.append(knowledge_template.format(context=context, knowledge=knowledge))
        if user_profile and len(user_profile) > 0:
            for profile in user_profile:
                model_input.append(user_profile_template.format(context=context, user_profile=profile))
        if bot_profile:
            for profile in bot_profile:
                model_input.append(bot_profile_template.format(context=context, bot_profile=profile))

        if not model_input:
            model_input.append(context_template.format(context=context))

        for i in range(len(model_input)):
            model_input[i] = model_input[i].replace('</s>', "[SEP]")

        return model_input, context

    def build_model_input_for_plug(self, utterance: str, context: list, history: list, user_profile: list,
                                   bot_profile: list,
                                   knowledge_list: list) -> Tuple[List[str], str]:
        model_input = []
        SEP_TOKEN = '[SEP]'

        if context:
            context = f'{SEP_TOKEN.join(context)}{SEP_TOKEN}{utterance}'
        else:
            context = utterance

        if history:
            history = SEP_TOKEN.join(history)
            model_input.append(f'{context}{SEP_TOKEN}history: {history}')
        if knowledge_list:
            for knowledge in knowledge_list:
                model_input.append(f'{context}{SEP_TOKEN}knowledge: {knowledge}')
        if user_profile:
            for profile in user_profile:
                model_input.append(f'{context}{SEP_TOKEN}user_profile: {profile}')
        if bot_profile:
            for profile in bot_profile:
                model_input.append(f'{context}{SEP_TOKEN}bot_profile: {profile}')

        if not model_input:
            model_input.append(f'{context}')

        for i in range(len(model_input)):
            model_input[i] = re.sub('[ \t]+', "▂", model_input[i])
            model_input[i] = model_input[i].replace('</s>', "[SEP]")

        return model_input, context

    def build_token_type_ids(self, fid_model_input_passages, context_prefix, input_ids):
        token_type_id_dict = {
            "utterance": 1,
            "context": 2,
            "history": 3,
            "knowledge": 4,
            "user_profile": 5,
            "bot_profile": 6
        }
        context_prefix_token_ids = self.tokenizer(context_prefix, truncation=True, max_length=300).input_ids
        context_prefix_ids_length = len(context_prefix_token_ids)
        max_passage_length = input_ids.shape[2]

        context_end_index = 0
        sep_id = 102
        if sep_id in context_prefix_token_ids:
            for i in range(len(context_prefix_token_ids) - 1):
                if context_prefix_token_ids[i] == sep_id:
                    context_end_index = i
        context_type_id = [token_type_id_dict['context']] * len(context_prefix_token_ids[:context_end_index + 1]) + \
                          [token_type_id_dict['utterance']] * len(context_prefix_token_ids[context_end_index + 1:])

        token_type_ids = []
        for passage in fid_model_input_passages:
            hit_type = None
            for type in ['history', 'user_profile', 'bot_profile', 'knowledge']:
                if type in passage:
                    type_id = token_type_id_dict[type]
                    token_type_ids.append(
                        torch.tensor(context_type_id + [type_id] * (max_passage_length - context_prefix_ids_length)))
                    hit_type = type
                    break
            if not hit_type:
                token_type_ids.append(context_type_id + [1] * (max_passage_length - context_prefix_ids_length))

        token_type_ids = torch.stack(token_type_ids, dim=0).unsqueeze(0).to(DEVICE)
        token_type_ids = token_type_ids * torch.gt(input_ids, 0)
        return token_type_ids

    def build_bad_words_ids(self, no_repeat_session):
        TOKEN_START = 1 if self.is_t5 else 0

        bad_words_ids = []
        with self.lock:
            forbidden_utterances_ids = self.tokenizer(self.bad_words, add_special_tokens=False).input_ids
        bad_words_ids.extend([tuple(t[TOKEN_START:]) for t in forbidden_utterances_ids])

        if no_repeat_session:
            with self.lock:
                no_repeat_session_tokens = self.tokenizer(no_repeat_session, add_special_tokens=False).input_ids
            for utt_tokens in no_repeat_session_tokens:
                utt_tokens = utt_tokens[TOKEN_START:]
                bad_words_ids.extend(
                    list(zip(*[utt_tokens[i:] for i in range(self.no_repeat_session_ngrams)])))

        bad_words_ids = [list(t) for t in set(bad_words_ids) if len(t) > 0]
        return bad_words_ids

    def respond(self, utterance: str,
                context: List[str],
                history: List[str],
                user_profile: List[str],
                bot_profile: List[str],
                knowledge_list: List[str],
                no_repeat_session: List[str],
                generate_config: dict):
        '''

        Args:
            utterance:
            context:
            history:
            user_profile:
            bot_profile:
            knowledge_list:
            no_repeat_session:
            generate_config:

        Returns:
            tuple< responses, cost_time >

        '''
        model_input, context_prefix = self.build_model_input(utterance, context, history, user_profile, bot_profile,
                                                             knowledge_list)

        with self.lock:
            input_ids = self.tokenizer(model_input,
                                       padding=True,
                                       truncation=True,
                                       max_length=self.max_encoder_length,
                                       return_tensors="pt").input_ids.unsqueeze(0).to(DEVICE)  # batch_size= 1

        bad_words_ids = self.build_bad_words_ids(no_repeat_session)
        if not self.is_t5:
            token_type_ids = self.build_token_type_ids(model_input, context_prefix, input_ids)
        else:
            token_type_ids = None
        start_generate = time.time()

        if self.allspark_gpu_speed_up:
            input_dict = {
                "input_ids": input_ids.to(torch.int64).cuda(),
                "attention_mask": (input_ids != 0).to(torch.int64).cuda(),
                "token_type_ids": torch.zeros(input_ids.shape).to(torch.int64).cuda()
            }
            hypotheses = self.model.generate_allspark(input_dict,
                                                      max_length=generate_config['max_length'],
                                                      bad_words_ids=bad_words_ids)
        else:
            if token_type_ids is None:
                hypotheses = self.model.generate(input_ids,
                                                 bad_words_ids=bad_words_ids,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 decoder_start_token_id=self.tokenizer.cls_token_id,
                                                 **generate_config)
            else:
                hypotheses = self.model.generate(input_ids,
                                                 bad_words_ids=bad_words_ids,
                                                 token_type_ids=token_type_ids,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 decoder_start_token_id=self.tokenizer.cls_token_id,
                                                 **generate_config)

        if torch.cuda.is_available():
            hypotheses = hypotheses.detach().cpu().tolist()

        results = []
        for h in hypotheses:
            decoded_hypo = self.tokenizer.decode(h, skip_special_tokens=True).replace(' ', '')
            results.append(decoded_hypo)

        generate_time = time.time() - start_generate

        return results, generate_time, model_input
