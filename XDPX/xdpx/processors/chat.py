import torch

from xdpx.options import Argument
from . import register, Processor
from typing import List
from tqdm import tqdm
import random


@register('chat')
class ChatProcessor(Processor):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('max_encoder_length', default=480),
            Argument('max_decoder_length', default=80),
        )

    def numerize(self, inputs: dict):
        results = {
            'id': inputs['id'],
            'context': self.numerize_tokens(inputs['context'])[-self.args.max_encoder_length:],
            'response': self.numerize_tokens(inputs['response'])[:self.args.max_decoder_length],
        }
        return results

    def numerize_samples(self, samples: List[dict], with_target=True) -> List[dict]:
        """batch numerize samples loaded by 'load_data'"""
        results = []
        for sample in tqdm(samples, desc='numerizing'):
            result = self.numerize(sample)
            if not result:
                continue
            results.append(result)
        return results

    def text_length(self, sample):
        return max(len(sample['context']), len(sample['response']))

    def collate(self, samples):

        # fake padding test
        # input_ids = torch.tensor(self.pad([sample['context']+(300-len(sample['context']))*[0] for sample in samples]), dtype=torch.long)
        # decoder_input_ids = torch.tensor(self.pad([sample['response']+(80-len(sample['response']))*[0] for sample in samples], pad_index=-100),
        #                                  dtype=torch.long)
        
        input_ids = torch.tensor(self.pad([sample['context'] for sample in samples]), dtype=torch.long)
        decoder_input_ids = torch.tensor(self.pad([sample['response'] for sample in samples], pad_index=-100),
                                         dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,
                'decoder_input_ids': decoder_input_ids  # labels
            },
            'ntokens': input_ids.numel() + decoder_input_ids.numel(),
        }
        return batch

    def sanity_check(self, inputs):
        batch = self.collate(inputs)

        outputs = []
        input_ids = batch['net_input']['input_ids'].tolist()
        decoder_input_ids = batch['net_input']['decoder_input_ids'].tolist()

        for context, response in zip(input_ids, decoder_input_ids):
            context = ' '.join(self.decode(context))
            context = context.replace(' ' + self.args.pad_word, '')
            response = [max(0, id) for id in response]
            response = ' '.join(self.decode(response)).replace(self.args.pad_word, '_')

            sample = {
                'context': context,
                'response': response
            }

            outputs.append(sample)
        return outputs

@register('plugchat')
class PlugChatProcessor(ChatProcessor):

    def collate(self, samples):
        cls_id = 101
        sep_id = 102

        # !!!note: context add cls_id and sep_id
        input_cls_id = [cls_id] if samples[0]['context'][0]!=cls_id else []
        input_sep_id = [sep_id] if samples[0]['context'][-1]!=sep_id else []
        dec_input_sep_id = [sep_id] if samples[0]['response'][-1]!=sep_id else []

        input_ids = torch.tensor(self.pad([input_cls_id+sample['context']+input_sep_id for sample in samples]), dtype=torch.long)

        decoder_input_ids = torch.tensor(self.pad([sample['response']+dec_input_sep_id for sample in samples],pad_index=-100), dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,
                'decoder_input_ids': decoder_input_ids
            },
            'ntokens': input_ids.numel() + decoder_input_ids.numel(),
        }
        return batch

@register('plugv2chat')
class PlugV2ChatProcessor(Processor):

    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('max_encoder_length', default=480),
            Argument('max_decoder_length', default=80),
        )

    def numerize(self, inputs: dict):
        results = {
            'id': inputs['id'],
            'context': self.numerize_tokens(inputs['context'])[:self.args.max_encoder_length],
            'context_type_ids': inputs['context_type_ids'][:self.args.max_encoder_length],
            'response': self.numerize_tokens(inputs['response'])[:self.args.max_decoder_length],
        }
        return results

    def numerize_samples(self, samples: List[dict], with_target=True) -> List[dict]:
        """batch numerize samples loaded by 'load_data'"""
        results = []
        for sample in tqdm(samples, desc='numerizing'):
            result = self.numerize(sample)
            if not result:
                continue
            results.append(result)
        return results

    def text_length(self, sample):
        return max(len(sample['context']), len(sample['response']))

    def collate(self, samples):

        input_ids = torch.tensor(self.pad([sample['context'] for sample in samples],pad_index=0), dtype=torch.long)
        token_type_ids = torch.tensor(self.pad([sample['context_type_ids'] for sample in samples]), dtype=torch.long)
        decoder_input_ids = torch.tensor(self.pad([sample['response'] for sample in samples],pad_index=0), dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'decoder_input_ids': decoder_input_ids
            },
            'ntokens': input_ids.numel() + decoder_input_ids.numel(),
        }
        return batch

    def sanity_check(self, inputs):
        batch = self.collate(inputs)

        outputs = []
        input_ids = batch['net_input']['input_ids'].tolist()
        input_type_ids = batch['net_input']['input_type_ids'].tolist()
        decoder_input_ids = batch['net_input']['decoder_input_ids'].tolist()

        for context,context_type_ids,response in zip(input_ids, input_type_ids, decoder_input_ids):
            context = ' '.join(self.decode(context))
            context = context.replace(' ' + self.args.pad_word, '')
            response = ' '.join(self.decode(response)).replace(self.args.pad_word, '_')

            sample = {
                'context': context,
                'context_type_ids': context_type_ids,
                'response': response
            }

            outputs.append(sample)
        return outputs


@register('fidchat')
class FIDChatProcessor(ChatProcessor):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('max_n_passage', default=20),
        )

    def numerize(self, inputs: dict):
        passages = [self.numerize_tokens(p) for p in inputs['passages'][:self.args.max_n_passage]]
        context = self.numerize_tokens(inputs['context'])
        if passages:
            context_passages = [(context + p)[:self.args.max_encoder_length] for p in passages]
        else:
            context_passages = [context]

        results = {
            'id': inputs['id'],
            'context': context_passages,
            'response': self.numerize_tokens(inputs['response'])[:self.args.max_decoder_length],
        }
        return results

    # used for batch by len
    def text_length(self, sample):
        return max([len(p) for p in sample['context']])

    def collate(self, samples):
        input_ids = torch.tensor(self.pad3d([sample['context'] for sample in samples]), dtype=torch.long)
        decoder_input_ids = torch.tensor(self.pad([sample['response'] for sample in samples],pad_index=-100), dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,  # batch_size, n_passage, max_length
                'decoder_input_ids': decoder_input_ids
            },
            'ntokens': input_ids.numel() + decoder_input_ids.numel(),
        }
        return batch

    def sanity_check(self, inputs):
        batch = self.collate(inputs)

        outputs = []
        input_ids = batch['net_input']['input_ids'].tolist()
        decoder_input_ids = batch['net_input']['decoder_input_ids'].tolist()

        for context, response in zip(input_ids, decoder_input_ids):
            context = '\n> '.join([' '.join(self.decode(p)) for p in context])
            context = context.replace(' ' + self.args.pad_word, '')
            response = ' '.join(self.decode(response)).replace(self.args.pad_word, '_')

            sample = {
                'context': context,
                'response': response
            }

            outputs.append(sample)
        return outputs

@register('plug_fidchat')
class PlugFIDChatProcessor(FIDChatProcessor):

    def collate(self, samples):
        input_ids = torch.tensor(self.pad3d([sample['context'] for sample in samples]), dtype=torch.long)
        decoder_input_ids = torch.tensor(self.pad([sample['response'] for sample in samples],pad_index=-100), dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,  # batch_size, n_passage, max_length
                'decoder_input_ids': decoder_input_ids
            },
            'ntokens': input_ids.numel() + decoder_input_ids.numel(),
        }
        return batch


@register('plugv2_fidchat')
class PlugV2FIDChatProcessor(Processor):

    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('max_encoder_length', default=480),
            Argument('max_decoder_length', default=80),
            Argument('max_n_passage', default=20),
        )

    def numerize(self, inputs: dict):
        passages = [self.numerize_tokens(p) for p in inputs['passages'][:self.args.max_n_passage]]
        context = self.numerize_tokens(inputs['context'])

        passages_type_id = inputs['passages_type_id'][:self.args.max_n_passage]
        context_type_id = inputs['context_type_id']

        if passages:
            context_passages = [(context + p)[:self.args.max_encoder_length] for p in passages]
            context_passages_type_id = [(context_type_id + p)[:self.args.max_encoder_length] for p in passages_type_id]
        else:
            context_passages = [context]
            context_passages_type_id = [context_type_id]

        results = {
            'id': inputs['id'],
            'context': context_passages,
            'response': self.numerize_tokens(inputs['response'])[:self.args.max_decoder_length],
            "context_type_id": context_passages_type_id
        }
        return results

    def numerize_samples(self, samples: List[dict], with_target=True) -> List[dict]:
        """batch numerize samples loaded by 'load_data'"""
        results = []
        for sample in tqdm(samples, desc='numerizing'):
            result = self.numerize(sample)
            if not result:
                continue
            results.append(result)
        return results

    def collate(self, samples):
        input_ids = torch.tensor(self.pad3d([sample['context'] for sample in samples]), dtype=torch.long)
        decoder_input_ids = torch.tensor(self.pad([sample['response'] for sample in samples],pad_index=0), dtype=torch.long)

        token_type_ids = torch.tensor(self.pad3d([sample['context_type_id'] for sample in samples]), dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,  # batch_size, n_passage, max_length
                'decoder_input_ids': decoder_input_ids,
                'token_type_ids': token_type_ids
            },
            'ntokens': input_ids.numel() + decoder_input_ids.numel(),
        }
        return batch

    def sanity_check(self, inputs):
        batch = self.collate(inputs)

        outputs = []
        input_ids = batch['net_input']['input_ids'].tolist()
        decoder_input_ids = batch['net_input']['decoder_input_ids'].tolist()

        token_type_ids = batch['net_input']['token_type_ids'].tolist()

        for context, response, token_type in zip(input_ids, decoder_input_ids, token_type_ids):
            context = '\n> '.join([' '.join(self.decode(p)) for p in context])
            context = context.replace(' ' + self.args.pad_word, '')
            response = ' '.join(self.decode(response)).replace(self.args.pad_word, '_')

            token_type = '\n> '.join([' '.join([str(each) for each in p]) for p in token_type])

            sample = {
                'context': context,
                'response': response,
                'token_type': token_type
            }

            outputs.append(sample)
        return outputs

    # used for batch by len
    def text_length(self, sample):
        return max([len(p) for p in sample['context']])

@register('palmchat')
class PalmChatProcessor(FIDChatProcessor):
    pass



@register('plugv2chat_instruction')
class PlugV2ChatInstructionProcessor(Processor):

    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('max_encoder_length', default=380),
            Argument('max_decoder_length', default=80),
        )

    def numerize(self, inputs: dict):
        results = {
            'id': inputs['id'],
            'context': self.numerize_tokens(inputs['context'])[:self.args.max_encoder_length],
            'response': self.numerize_tokens(inputs['response'])[:self.args.max_decoder_length],
        }
        return results

    def numerize_samples(self, samples: List[dict], with_target=True) -> List[dict]:
        """batch numerize samples loaded by 'load_data'"""
        results = []
        for sample in tqdm(samples, desc='numerizing'):
            result = self.numerize(sample)
            if not result:
                continue
            results.append(result)
        return results

    def text_length(self, sample):
        return max(len(sample['context']), len(sample['response']))

    def collate(self, samples):

        input_ids = torch.tensor(self.pad([sample['context'] for sample in samples],pad_index=0), dtype=torch.long)
        decoder_input_ids = torch.tensor(self.pad([sample['response'] for sample in samples],pad_index=0), dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,
                'decoder_input_ids': decoder_input_ids
            },
            'ntokens': input_ids.numel() + decoder_input_ids.numel(),
        }
        return batch

    def sanity_check(self, inputs):
        batch = self.collate(inputs)

        outputs = []
        input_ids = batch['net_input']['input_ids'].tolist()
        decoder_input_ids = batch['net_input']['decoder_input_ids'].tolist()

        for context,response in zip(input_ids, decoder_input_ids):
            context = ' '.join(self.decode(context))
            context = context.replace(' ' + self.args.pad_word, '')
            response = ' '.join(self.decode(response)).replace(self.args.pad_word, '_')

            sample = {
                'context': context,
                'response': response
            }

            outputs.append(sample)
        return outputs





@register('plugv2_fidchat_instruction')
class PlugV2FIDChatInstructionProcessor(Processor):

    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('max_encoder_length', default=380),
            Argument('max_decoder_length', default=80),
            Argument('max_n_passage', default=20),
        )

    def numerize(self, inputs: dict):
        query = [self.numerize_tokens(p)[:self.args.max_encoder_length] for p in inputs['query'][:self.args.max_n_passage]]
        response = self.numerize_tokens(inputs['response'])[:self.args.max_decoder_length]

        results = {
            'id': inputs['id'],
            'context': query,
            'response': response
        }
        return results

    def numerize_samples(self, samples: List[dict], with_target=True) -> List[dict]:
        """batch numerize samples loaded by 'load_data'"""
        results = []
        for sample in tqdm(samples, desc='numerizing'):
            result = self.numerize(sample)
            if not result:
                continue
            results.append(result)
        return results

    def collate(self, samples):
        input_ids = torch.tensor(self.pad3d([sample['context'] for sample in samples]), dtype=torch.long)
        decoder_input_ids = torch.tensor(self.pad([sample['response'] for sample in samples]), dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,  # batch_size, n_passage, max_length
                'decoder_input_ids': decoder_input_ids
            },
            'ntokens': input_ids.numel() + decoder_input_ids.numel(),
        }
        return batch

    def sanity_check(self, inputs):
        batch = self.collate(inputs)

        outputs = []
        input_ids = batch['net_input']['input_ids'].tolist()
        decoder_input_ids = batch['net_input']['decoder_input_ids'].tolist()

        for context, response in zip(input_ids, decoder_input_ids):
            context = '\n> '.join([' '.join(self.decode(p)) for p in context])
            context = context.replace(' ' + self.args.pad_word, '')
            response = ' '.join(self.decode(response)).replace(self.args.pad_word, '_')

            sample = {
                'context': context,
                'response': response
            }

            outputs.append(sample)
        return outputs

    # used for batch by len
    def text_length(self, sample):
        return max([len(p) for p in sample['context']])


@register('t5_chat')
class T5ChatProcessor(ChatProcessor):
    def numerize(self, inputs: dict):
        context = self.numerize_tokens(inputs['context'])[-self.args.max_encoder_length:]
        response = self.numerize_tokens(inputs['response'])[:self.args.max_decoder_length]
        if not context or not response:
            return None
        EOS_TOKEN = 1
        if context[-1] != EOS_TOKEN:
            context.append(EOS_TOKEN)
        if response[-1] != EOS_TOKEN:
            response.append(EOS_TOKEN)
        results = {
            'id': inputs['id'],
            'context': context,
            'response': response,
        }
        return results


@register('t5_fidchat')
class T5FIDChatProcessor(FIDChatProcessor):
    def numerize(self, inputs: dict):
        passages = [self.numerize_tokens(p) for p in inputs['passages'][:self.args.max_n_passage]]
        context = self.numerize_tokens(inputs['context'])
        if passages:
            context_passages = [(context + p)[:self.args.max_encoder_length] for p in passages]
        else:
            context_passages = [context]
        response = self.numerize_tokens(inputs['response'])[:self.args.max_decoder_length - 1]
        if not context_passages or not response:
            return None
        EOS_TOKEN = 1
        for p in context_passages:
            if p[-1] != EOS_TOKEN:
                p.append(EOS_TOKEN)
        if response[-1] != EOS_TOKEN:
            response.append(EOS_TOKEN)

        results = {
            'id': inputs['id'],
            'context': context_passages,
            'response': response,
        }
        return results


@register('t5_chat_instruction')
class T5ChatInstructionProcessor(ChatProcessor):
    def numerize(self, inputs: dict):
        results = {
            'id': inputs['id'],
            'context': self.numerize_tokens(inputs['context'])[:self.args.max_encoder_length],
            'response': self.numerize_tokens(inputs['response'])[:self.args.max_decoder_length],
        }
        return results


@register('t5_fidchat_instruction')
class T5FIDChatInstructionProcessor(FIDChatProcessor):

    def numerize(self, inputs: dict):
        query = [self.numerize_tokens(p)[:self.args.max_encoder_length] for p in
                 inputs['query'][:self.args.max_n_passage]]
        response = self.numerize_tokens(inputs['response'])[:self.args.max_decoder_length]

        results = {
            'id': inputs['id'],
            'context': query,
            'response': response
        }
        return results


@register('t5_fidchat_ctr')
class T5FIDChatCtrProcessor(T5FIDChatProcessor):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('n_cands', default=4),
        )

    def numerize(self, inputs: dict):
        candidates = [self.numerize_tokens(p)[:self.args.max_decoder_length - 1] for p in
                      inputs['candidates'][:self.args.n_cands]]
        EOS_TOKEN = 1
        for c in candidates:
            if c[-1] != EOS_TOKEN:
                c.append(EOS_TOKEN)
        results = super().numerize(inputs)
        results['candidates'] = candidates
        return results

    def collate(self, samples):
        input_ids = torch.tensor(self.pad3d([sample['context'] for sample in samples]), dtype=torch.long)
        decoder_input_ids = torch.tensor(self.pad([sample['response'] for sample in samples], pad_index=-100),
                                         dtype=torch.long)
        cands_input_ids = torch.tensor(self.pad3d([sample['candidates'] for sample in samples], pad_index=-100),
                                       dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,  # batch_size, n_passage, max_length
                'decoder_input_ids': decoder_input_ids,  # batch_size, max_length
            },
            'cands_input_ids': cands_input_ids,  # batch_size, n_cands, max_length
            'ntokens': input_ids.numel() + decoder_input_ids.numel(),
        }
        return batch

    def sanity_check(self, inputs):
        batch = self.collate(inputs)

        outputs = []
        input_ids = batch['net_input']['input_ids'].tolist()
        decoder_input_ids = batch['net_input']['decoder_input_ids'].tolist()
        cands_input_ids = batch['cands_input_ids'].tolist()

        for context, response, candidates in zip(input_ids, decoder_input_ids, cands_input_ids):
            context = '\n> '.join([' '.join(self.decode(p)) for p in context])
            context = context.replace(' ' + self.args.pad_word, '')
            response = [max(0, id) for id in response]
            response = ' '.join(self.decode(response)).replace(self.args.pad_word, '_')
            cand_responses = []
            for cand in candidates:
                cand = [max(0, id) for id in cand]
                cand = ' '.join(self.decode(cand)).replace(self.args.pad_word, '_')
                cand_responses.append(cand)
            cand_responses = '\n> '.join(cand_responses)

            sample = {
                'context': context,
                'response': response,
                'candidates': cand_responses
            }

            outputs.append(sample)
        return outputs


@register('common_text')
class CommonTextProcessor(Processor):
    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('max_encoder_length', default=512)
        )

    def numerize(self, inputs: dict):
        text_token_id = self.numerize_tokens(inputs['text'])[:self.args.max_encoder_length]
        results = {
            'id': inputs['id'],
            'context': text_token_id,
            'context_type_ids': [1 for _ in range(len(text_token_id))],
            'response': [],
        }
        return results

    def numerize_samples(self, samples: List[dict], with_target=True) -> List[dict]:
        """batch numerize samples loaded by 'load_data'"""
        results = []
        for sample in tqdm(samples, desc='numerizing'):
            result = self.numerize(sample)
            if not result:
                continue
            results.append(result)
        return results

    # for batch by len
    def text_length(self, sample):
        return len(sample['context'])

    def sanity_check(self, inputs):

        return [{"dummy":"dummy"}]


@register('chat_mix_common')
class ChatMixCommonProcessor(Processor):

    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('bos_id', default=101),
            Argument('eos_id', default=102),
            Argument('unsup_task', default="denoising"),  # denoising or prefix
            Argument('corrupt_span_ratio', default=0.15),
            Argument('corrupt_span_avg_len', default=3),
            Argument('prefix_cut_max_len', default=64)
        )

    # for batch by len
    def text_length(self, sample):
        return len(sample['context'])

    def gen_corrupt_spans(self, text_id_list):
        end_symbol_id = 1
        mark_symbol_id = 2
        corrupt_span_ratio = self.args.corrupt_span_ratio / self.args.corrupt_span_avg_len
        corrup_span_min_len = 1
        corrup_span_max_len = 2 * self.args.corrupt_span_avg_len - 1

        input_id = []
        output_id = []
        currupt_len = 0
        for id in text_id_list:
            if currupt_len > 0:
                currupt_len -= 1
                output_id.append(id)
                continue
            if random.random() >= corrupt_span_ratio:
                input_id.append(id)
                continue
            currupt_len = random.randint(corrup_span_min_len, corrup_span_max_len)
            input_id.append(mark_symbol_id)
            output_id.append(mark_symbol_id)
            output_id.append(id)
            mark_symbol_id += 1
            currupt_len -= 1
        output_id.append(end_symbol_id)
        return input_id, output_id

    def gen_prefix_text(self, text_id_list):
        output_len = self.args.prefix_cut_max_len
        if len(text_id_list) < self.args.prefix_cut_max_len * 2:
            output_len = len(text_id_list) // 2
        input_id = text_id_list[:-output_len]
        output_id = text_id_list[-output_len:]
        return input_id, output_id

    def collate(self, samples):
        input_id_list = []
        ouput_id_list = []
        context_type_ids_list = []
        for samp in samples:
            input_id = samp['context']
            output_id = samp['response']
            context_type_ids = samp["context_type_ids"]
            if len(output_id)==0:
                if self.args.unsup_task == "denoising":
                    input_id, output_id = self.gen_corrupt_spans(input_id)
                elif self.args.unsup_task == "prefix":
                    input_id, output_id = self.gen_prefix_text(input_id)
                else:
                    raise Exception("args unsup_task is illegal!!")
                input_id = [self.args.bos_id] + input_id + [self.args.eos_id]
                output_id = [self.args.bos_id] + output_id + [self.args.eos_id]
                context_type_ids = [2]+[1]*(len(input_id)-1)
            input_id_list.append(input_id)
            ouput_id_list.append(output_id)
            context_type_ids_list.append(context_type_ids)
            if samp['id']<5:
                print(f'| id:{id}')
                print(f'| input_id: {input_id}')
                print(f'| output_id: {output_id}')
                print(f'| context_type_ids:{context_type_ids}')
                print("| "+">>"*30)

        input_ids = torch.tensor(self.pad(input_id_list, pad_index=0), dtype=torch.long)
        token_type_ids = torch.tensor(self.pad(context_type_ids_list, pad_index=0), dtype=torch.long)
        decoder_input_ids = torch.tensor(self.pad(ouput_id_list, pad_index=0),dtype=torch.long)

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'decoder_input_ids': decoder_input_ids
            },
            'ntokens': input_ids.numel() + decoder_input_ids.numel(),
        }
        return batch


@register('fid_chat_mix_common')
class FidChatMixCommonProcessor(ChatMixCommonProcessor):

    @classmethod
    def register(cls, options):
        super().register(options)
        options.register(
            Argument('max_encoder_length', default=480),
            Argument('max_decoder_length', default=80),
            Argument('max_n_passage', default=20),
        )

    # for batch by len
    def text_length(self, sample):
        context = sample['context']
        if type(context[0])!=list:
            context = [context]
        return len(context) * max([len(p) for p in context])

    def collate(self, samples):
        input_id_list = []
        ouput_id_list = []
        context_type_ids_list = []
        for samp in samples:
            input_id = samp['context']
            output_id = samp['response']
            if "context_type_ids" in samp:
                context_type_ids = samp["context_type_ids"]
            else:
                context_type_ids = samp["context_type_id"]
            if len(output_id)==0:
                if self.args.unsup_task == "denoising":
                    input_id, output_id = self.gen_corrupt_spans(input_id)
                elif self.args.unsup_task == "prefix":
                    input_id, output_id = self.gen_prefix_text(input_id)
                else:
                    raise Exception("args unsup_task is illegal!!")
                input_id = [self.args.bos_id] + input_id + [self.args.eos_id]
                output_id = [self.args.bos_id] + output_id + [self.args.eos_id]
                context_type_ids = [2]+[1]*(len(input_id)-1)
                input_id = [input_id]
                context_type_ids = [context_type_ids]
            # for i,j in zip(input_id,context_type_ids):
            #     assert len(i)==len(j),f'{len(i)}!={len(j)}!!!{input_id} {context_type_ids}'
            input_id_list.append(input_id)
            ouput_id_list.append(output_id)
            context_type_ids_list.append(context_type_ids)
            if samp['id']<5:
                print(f'| id:{id}')
                print(f'| input_id: {input_id}')
                print(f'| output_id: {output_id}')
                print(f'| context_type_ids:{context_type_ids}')
                print("| "+">>"*30)

        input_ids = torch.tensor(self.pad3d(input_id_list, pad_index=0), dtype=torch.long)
        token_type_ids = torch.tensor(self.pad3d(context_type_ids_list, pad_index=0), dtype=torch.long)
        decoder_input_ids = torch.tensor(self.pad(ouput_id_list, pad_index=0),dtype=torch.long)

        # for i, j in zip(input_ids.size(), token_type_ids.size()):
        #     assert i == j, f'input_ids size: {input_ids.size()} should be equal to input_ids size: {token_type_ids.size()}...'

        batch = {
            'id': [sample['id'] for sample in samples],
            'net_input': {
                'input_ids': input_ids, # batch_size, n_passage, max_length
                'token_type_ids': token_type_ids,
                'decoder_input_ids': decoder_input_ids
            },
            'ntokens': input_ids.numel() + decoder_input_ids.numel(),
        }
        return batch