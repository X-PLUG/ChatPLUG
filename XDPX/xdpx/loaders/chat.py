from typing import List

from . import Loader, register
import re
from xdpx.options import Argument
import random


# fid instruction setting ************************************************************************************************************************************************************************************

context_template = ["假设我和你正在进行对话，请你给我得体、有信息量且生动的回复。以下是我们的对话内容。{context}"]
history_template = ["假设我和你正在进行对话，请你给我得体、有信息量且生动的回复。以下是我们的对话内容。{context}#请注意，在此之前我们还进行了以下对话内容。{history}"]
knowledge_template = ["假设我和你正在进行对话，请你给我得体、有信息量且生动的回复。以下是我们的对话内容。{context}#请注意，你同时也知道以下知识。{knowledge}"]
user_profile_template = ["假设我和你正在进行对话，请你给我得体、有信息量且生动的回复。以下是我们的对话内容。{context}#请注意，你同时也知道我的信息，以下是你对我了解的内容。{user_profile}"]
bot_profile_template = ["假设我和你正在进行对话，请你给我得体、有信息量且生动的回复。以下是我们的对话内容。{context}#请注意，你同时也有一个人物设定，以下是你的人物设定。{bot_profile}"]

def process_context(context, sep):
    if len(context)==0:
        return ""
    context_list = context.split(sep)
    subject = "我"
    for i in range(len(context_list)-1,-1,-1):
        if len(context_list[i])>0 and context_list[i][-1] not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~、。，？！；：“”（）【】《》〈〉……':
            context_list[i] = context_list[i]+"。"
        context_list[i] = subject+"："+context_list[i]
        subject = "你" if subject=="我" else "我"
    return "".join(context_list)

def process_history(history, sep):
    if len(history)==0:
        return ""
    history_list = history.split(sep)
    subject = "你"
    for i in range(len(history_list) - 1, -1, -1):
        if len(history_list[i])>0 and history_list[i][-1] not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~、。，？！；：“”（）【】《》〈〉……':
            history_list[i] = history_list[i]+"。"
        history_list[i] = subject + "：" + history_list[i]
        subject = "你" if subject == "我" else "我"
    return "".join(history_list)

# fid instruction setting ************************************************************************************************************************************************************************************



@register('chat')
class ChatLoader(Loader):
    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:

        return {
            'id': _id,
            'context': cls._tokenizer.encode(contents[0]),
            'response': cls._tokenizer.encode(contents[1])
        }

    @property
    def header(self):
        return ['context', 'response']

    def length(self, sample):
        len1 = len(sample['context'])
        len2 = len(sample['response'])
        if len1 and len2:
            return max(len1, len2)
        else:
            return 0

    @property
    def num_sections(self):
        return 2

    def with_targets(self):
        return False


@register('fidchat')
class FIDChatLoader(Loader):
    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        passages = contents[2].split(';;;')
        return {
            'id': _id,
            'context': cls._tokenizer.encode(contents[0]),
            'response': cls._tokenizer.encode(contents[1]),
            'passages': [cls._tokenizer.encode(passage) for passage in passages]
        }

    @property
    def header(self):
        return ['context', 'response', 'passages']

    def length(self, sample):
        len1 = len(sample['context'])
        len2 = len(sample['response'])
        max_plen = max([len(p) for p in sample['passages']])
        if len1 and len2:
            return max(len1, len2, max_plen)
        else:
            return 0

    @property
    def num_sections(self):
        return 3

    def with_targets(self):
        return False


PLUG_SPACE_TOKEN = '▂'
token_type_id_dict = {
    "utterance": 1,
    "context": 2,
    "history": 3,
    "knowledge": 4,
    "user_profile": 5,
    "bot_profile": 6
}
def plug_token_process(text):
    text = text.replace('</s>', '[SEP]')
    text = text.replace(' \t ', '[SEP]')
    text = re.sub('[ \t]+', PLUG_SPACE_TOKEN, text)
    return text

def t5_token_process(func):

    def wrapper(cls, contents: List[str],*args,**kw):
        for i in range(len(contents)):
            contents[i] = contents[i].replace('\n', '▁<extra_id_22>')
            contents[i] = contents[i].replace('\t', '▁<extra_id_33>')
            contents[i] = contents[i].replace('  ', '▁<extra_id_23>')
        return func(cls, contents,*args,**kw)

    return wrapper


def plug_token_process(func):

    def wrapper(cls, contents: List[str],*args,**kw):
        for i in range(len(contents)):
            contents[i] = contents[i].replace('</s>', '[SEP]')
            # raw data may contain special tokens of t5
            contents[i] = contents[i].replace('▁<extra_id_22>','\n')
            contents[i] = contents[i].replace('▁<extra_id_33>','\t')
            contents[i] = contents[i].replace('▁<extra_id_23>','  ')

            contents[i] = contents[i].replace('\n', '[unused22]')
            contents[i] = contents[i].replace('\t', '[unused33]')
            contents[i] = contents[i].replace('  ', '[unused23]')
        return func(cls, contents,*args,**kw)

    return wrapper


@register('t5_chat')
class T5ChatLoader(ChatLoader):
    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        context = contents[0].strip().strip('</s>').split('</s>')
        if len(context) > 1:
            context = 'context: ' + '</s>'.join(context[:-1]) + ' </s> utterance: ' + context[-1] + '</s>'
        else:
            context = 'utterance: ' + context[0] + '</s>'
        response = contents[1]
        if not response.endswith('</s>'):
            response = response + '</s>'

        return {
            'id': _id,
            'context': cls._tokenizer.encode(context),
            'response': cls._tokenizer.encode(response)
        }


@register('t5_fidchat')
class T5FidChatLoader(FIDChatLoader):
    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        context = contents[0].strip().strip('</s>').split('</s>')
        if len(context) > 1:
            context = 'context: ' + '</s>'.join(context[:-1]) + ' </s> utterance: ' + context[-1] + '</s>'
        else:
            context = 'utterance: ' + context[0] + '</s>'

        response = contents[1]
        if not response.endswith('</s>'):
            response = response + '</s>'
        passages = contents[2].split(';;;')
        passages = [p.replace('\t', '</s>') if 'history' in p else p for p in passages]

        return {
            'id': _id,
            'context': cls._tokenizer.encode(context),
            'response': cls._tokenizer.encode(response),
            'passages': [cls._tokenizer.encode(passage) for passage in passages]
        }


@register('t5_chat_instruction')
class T5ChatInstructionLoader(ChatLoader):

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        SEP = '</s>'

        context = contents[0].strip().strip('</s>')
        context = context.replace('role1:', '').replace('role2:', '')
        response = contents[1].strip().strip('</s>')

        # process context
        context_list = context.split('</s>')
        context = '</s>'.join(context_list[-3:])
        context = process_context(context,'</s>')
        history = '</s>'.join(context_list[:-3])
        history = process_history(history,'</s>')

        if len(history)==0:
            query = context_template[random.randint(0, len(context_template) - 1)].format(context=context)
        else:
            query = history_template[random.randint(0,len(history_template)-1)].format(context=context,history=history)
        query = query+SEP
        query_token = cls._tokenizer.encode(query)

        # process response
        response = response + SEP
        response_token = cls._tokenizer.encode(response)

        return {
            'id': _id,
            'context': query_token,
            'response': response_token
        }


@register('t5_fidchat_instruction')
class T5FidChatInstructionLoader(FIDChatLoader):

    @classmethod
    @t5_token_process
    def parse(cls, contents: List[str], _id=0) -> dict:

        SEP = '</s>'

        contents[0] = contents[0].strip().strip(SEP)
        contents[1] = contents[1].strip().strip(SEP)
        contents[2] = contents[2].strip().strip(SEP)

        context = contents[0]
        response = contents[1]
        passages = contents[2].split(';;;')

        context = process_context(context, SEP)
        query_list = []
        for p in passages:
            p = p.strip()
            if len(p) == 0:
                continue
            p = p.strip(";")
            p = p.strip(SEP)
            if p.startswith("history"):
                p = p[len("history:"):]
                p = process_history(p, SEP)
                p = history_template[random.randint(0, len(history_template) - 1)].format(context=context, history=p)
            elif p.startswith("user_profile"):
                p = p[len("user_profile:"):]
                p = user_profile_template[random.randint(0, len(user_profile_template) - 1)].format(context=context,
                                                                                                    user_profile=p)
            elif p.startswith("bot_profile"):
                p = p[len("bot_profile:"):]
                p = bot_profile_template[random.randint(0, len(bot_profile_template) - 1)].format(context=context,
                                                                                                  bot_profile=p)
            else:
                p = p[len("knowledge:"):]
                p = knowledge_template[random.randint(0, len(knowledge_template) - 1)].format(context=context,
                                                                                              knowledge=p)
            p = p + SEP
            query_list.append(p)
        if len(query_list) == 0:
            p = context_template[random.randint(0, len(context_template) - 1)].format(context=context)
            p = p + SEP
            query_list.append(p)

        response = response + SEP

        query_token_list = [cls._tokenizer.encode(q) for q in query_list]
        response_token = cls._tokenizer.encode(response)

        return {
            'id': _id,
            'query': query_token_list,
            'response': response_token,
        }

    def length(self, sample):
        max_plen = max([len(p) for p in sample['query']])
        return max_plen



@register('plugv2_chat_instruction')
class PlugV2ChatInstructionLoader(ChatLoader):

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        CLS = '[CLS]'
        SEP = '[SEP]'

        context = contents[0].strip().strip('</s>')
        context = context.replace('role1:', '').replace('role2:', '')
        response = contents[1].strip().strip('</s>')

        # process context
        context_list = context.split('</s>')
        context = '</s>'.join(context_list[-3:])
        context = process_context(context,'</s>')
        history = '</s>'.join(context_list[:-3])
        history = process_history(history,'</s>')

        if len(history)==0:
            query = context_template[random.randint(0, len(context_template) - 1)].format(context=context)
        else:
            query = history_template[random.randint(0,len(history_template)-1)].format(context=context,history=history)
        query = CLS+query+SEP
        query_token = cls._tokenizer.encode(query)

        # process response
        response = CLS + response + SEP
        response_token = cls._tokenizer.encode(response)

        return {
            'id': _id,
            'context': query_token,
            'response': response_token
        }



@register('plugv2_fidchat_instruction')
class PlugV2FidChatInstructionLoader(FIDChatLoader):

    @classmethod
    @plug_token_process
    def parse(cls, contents: List[str], _id=0) -> dict:
        CLS = '[CLS]'
        SEP = '[SEP]'


        context = contents[0]
        response = contents[1]
        passages = contents[2].split(';;;')

        context = process_context(context, SEP)
        query_list = []
        for p in passages:
            p = p.strip(PLUG_SPACE_TOKEN)
            if len(p) == 0:
                continue
            p = p.strip(";")
            p = p.strip(SEP)
            if p.startswith("history"):
                p = p[len("history:"):]
                p = process_history(p,SEP)
                p = history_template[random.randint(0,len(history_template)-1)].format(context=context,history=p)
            elif p.startswith("user_profile"):
                p = p[len("user_profile:"):]
                p = user_profile_template[random.randint(0, len(user_profile_template) - 1)].format(context=context,user_profile=p)
            elif p.startswith("bot_profile"):
                p = p[len("bot_profile:"):]
                p = bot_profile_template[random.randint(0, len(bot_profile_template) - 1)].format(context=context,bot_profile=p)
            else:
                p = p[len("knowledge:"):]
                p = knowledge_template[random.randint(0, len(knowledge_template) - 1)].format(context=context, knowledge=p)
            p = CLS+p+SEP
            query_list.append(p)
        if len(query_list)==0:
            p = context_template[random.randint(0, len(context_template) - 1)].format(context=context)
            p = CLS + p + SEP
            query_list.append(p)

        response = CLS+response+SEP

        query_token_list = [cls._tokenizer.encode(q) for q in query_list]
        response_token = cls._tokenizer.encode(response)

        return {
            'id': _id,
            'query': query_token_list,
            'response': response_token,
        }

    def length(self, sample):
        max_plen = max([len(p) for p in sample['query']])
        return max_plen


@register('plugv2_chat')
class PlugV2ChatLoader(ChatLoader):
    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        context = contents[0].strip()
        response = contents[1].strip()
        context = context.replace('</s>', '[SEP]').replace('role1:', '').replace('role2:', '')
        response = response.replace('</s>', '[SEP]')
        # remove [SEP] in start and end
        if context.startswith('[SEP]'):
            context = context[5:]
        if context.endswith('[SEP]'):
            context = context[:-5]
        if response.startswith('[SEP]'):
            response = response[5:]
        if response.endswith('[SEP]'):
            response = response[:-5]
        # process context
        context_list = context.split('[SEP]')
        utterance = context_list[-1]
        context = '[SEP]'.join(context_list[-3:-1])
        history = '[SEP]'.join(context_list[:-3])
        utterance = utterance+'[SEP]'
        if len(context)>0:
            context = context+'[SEP]'
        context = '[CLS]'+context
        if len(history)>0:
            history = 'history: '+history+'[SEP]'
        context_token = cls._tokenizer.encode(context)
        utterance_token = cls._tokenizer.encode(utterance)
        history_token = cls._tokenizer.encode(history)
        passage_token = context_token+utterance_token+history_token
        passage_type_id = [token_type_id_dict['context']]*len(context_token)+[token_type_id_dict['utterance']]*len(utterance_token)+[token_type_id_dict['history']]*len(history_token)
        # process response
        response = '[CLS]'+response+'[SEP]'
        response_token = cls._tokenizer.encode(response)

        return {
            'id': _id,
            'context': passage_token,
            'context_type_ids': passage_type_id,
            'response': response_token
        }


@register('plugv2_fidchat')
class PlugV2FidChatLoader(FIDChatLoader):

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        CLS = '[CLS]'
        SEP = '[SEP]'

        contents[0] = plug_token_process(contents[0])
        contents[1] = plug_token_process(contents[1])
        contents[2] = plug_token_process(contents[2])

        context = contents[0]
        context = CLS + context
        if not context.endswith(SEP):
            context = context + SEP

        response = contents[1]
        response = CLS + response
        if not response.endswith(SEP):
            response = response + SEP

        passages = contents[2].split(';;;')
        new_passages = []
        passages_type_id_order = []
        type_id = token_type_id_dict['knowledge']
        for p in passages:
            p = p.strip(PLUG_SPACE_TOKEN)
            if len(p)==0:
                continue
            if not p.endswith(SEP):
                p = p+SEP
            key = p.split(":")[0].strip(PLUG_SPACE_TOKEN+';')
            if key in token_type_id_dict:
                type_id = token_type_id_dict[key]
            else:
                print(f'error: {p} is invalid passage!!')
                print(f'will use type id: {type_id}')
            passages_type_id_order.append(type_id)
            new_passages.append(p)
        passages = new_passages

        context_token = cls._tokenizer.encode(context)
        response_token = cls._tokenizer.encode(response)
        passages_token = [cls._tokenizer.encode(passage) for passage in passages]

        # context type id
        context_end_index = 0
        assert context_token[-1]==SEP
        if SEP in context_token[:-1]:
            for i in range(len(context_token)-1):
                if context_token[i]==SEP:
                    context_end_index = i
        context_type_id = [token_type_id_dict['context']]*len(context_token[:context_end_index+1]) + [token_type_id_dict['utterance']]*len(context_token[context_end_index+1:])

        # passages type id
        passages_type_id = []
        for type_id,pass_token in zip(passages_type_id_order,passages_token):
            passages_type_id.append([type_id]*len(pass_token))

        return {
            'id': _id,
            'context': context_token,
            'response': response_token,
            'passages': passages_token,
            "context_type_id": context_type_id,
            "passages_type_id": passages_type_id
        }

@register('t5_fidchat_ctr')
class T5FidChatCtrLoader(T5FidChatLoader):
    @property
    def num_sections(self):
        return 4

    @property
    def header(self):
        return ['context', 'response', 'passages', 'candidates']

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        context = contents[0].strip().strip('</s>').split('</s>')
        if len(context) > 1:
            context = 'context: ' + '</s>'.join(context[:-1]) + ' </s> utterance: ' + context[-1] + '</s>'
        else:
            context = 'utterance: ' + context[0] + '</s>'

        response = contents[1]
        if not response.endswith('</s>'):
            response = response + '</s>'
        passages = contents[2].split(';;;')
        passages = [p.replace('\t', '</s>') if 'history' in p else p for p in passages]

        gold_response = response.strip().replace('</s>', '')
        candidates = contents[3].split(';;;')
        candidates = [c for c in candidates if c != gold_response]

        data = {
            'id': _id,
            'context': cls._tokenizer.encode(context),
            'response': cls._tokenizer.encode(response),
            'passages': [cls._tokenizer.encode(passage) for passage in passages],
            'candidates': [cls._tokenizer.encode(cand) for cand in candidates]
        }

        return data


@register('plug_chat')
class PlugChatLoader(ChatLoader):
    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        contents[0] = plug_token_process(contents[0])
        contents[1] = plug_token_process(contents[1])

        context = contents[0].replace('role1:', '').replace('role2:', '')
        context = '[CLS]' + context + '[SEP]'
        response = contents[1]
        if not response.endswith('[SEP]'):
            response = response + '[SEP]'
        return {
            'id': _id,
            'context': cls._tokenizer.encode(context),
            'response': cls._tokenizer.encode(response)
        }


@register('plug_fidchat')
class PlugFidChatLoader(FIDChatLoader):

    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:
        contents[0] = plug_token_process(contents[0])
        contents[1] = plug_token_process(contents[1])
        contents[2] = plug_token_process(contents[2])

        token_type_id_dict = {
            "context": 1,
            "history": 2,
            "knowledge": 3,
            "user_profile": 4,
            "bot_profile":5
        }

        context = contents[0]
        context = '[CLS]' + context
        if not context.endswith('[SEP]'):
            context = context + '[SEP]'

        response = contents[1]
        if not response.endswith('[SEP]'):
            response = response + '[SEP]'

        passages = contents[2].split(';;;')
        new_passages = []
        passages_type_id_order = []
        for p in passages:
            p = p.strip('▂')
            if len(p)==0:
                continue
            if not p.endswith('[SEP]'):
                p = p+'[SEP]'
            key = p.split(":")[0].strip('▂;')
            if key in token_type_id_dict:
                type_id = token_type_id_dict[key]
            else:
                type_id = 3
                print(f'error: {p} is invalid passage!!')
            passages_type_id_order.append(type_id)
            new_passages.append(p)
        passages = new_passages

        context_token = cls._tokenizer.encode(context)
        response_token = cls._tokenizer.encode(response)
        passages_token = [cls._tokenizer.encode(passage) for passage in passages]

        # context type id
        context_type_id = [token_type_id_dict['context']]*len(context_token)
        passages_type_id = []
        for type_id,pass_token in zip(passages_type_id_order,passages_token):
            passages_type_id.append([type_id]*len(pass_token))

        return {
            'id': _id,
            'context': context_token,
            'response': response_token,
            'passages': passages_token,
            "context_type_id": context_type_id,
            "passages_type_id": passages_type_id
        }


@register('session_chat')
class SessionChatLoader(Loader):
    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:

        return {
            'id': _id,
            'session': [cls._tokenizer.encode(utterance) for utterance in contents[0]],
            'source': contents[1]
        }

    @property
    def header(self):
        return ['session','source']

    def length(self, sample):
        return max([len(utterance) for utterance in sample['session']])

    @property
    def num_sections(self):
        return 2

    def with_targets(self):
        return False


@register('common_text')
class CommonTextLoader(Loader):
    @classmethod
    def parse(cls, contents: List[str], _id=0) -> dict:

        text = contents[0]

        return {
            'id': _id,
            'text': cls._tokenizer.encode(text)
        }

    @property
    def header(self):
        return ['text']

    def length(self, sample):
        return len(sample['text'])

    @property
    def num_sections(self):
        return 1

    def with_targets(self):
        return False
