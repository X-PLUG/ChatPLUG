import re
from typing import List, Tuple, Optional

from xdpx.utils.chat.base import ChatInput, ChatOutput, HistoryItem, Evidence
from rouge import Rouge
from dataclasses import dataclass, field
from xdpx.utils.chat import text_is_bye, is_persona_question, is_special_skill

import random
import itertools
import math
from xdpx.utils.chat.learn2search import Learn2Search
from xdpx.utils.chat.post_rerank import PostReranker
from xdpx.utils.chat.core_chat import CoreChat
from xdpx.utils.chat.openkg_retrieval import Triple
from xdpx.utils.chat.openweb_search import Snippet
from xdpx.utils.chat.local_retrieval import FAQ
from xdpx.utils.chat.base import CHITCHAT_QUERY, ChatConfig
from xdpx.utils.chat.persona_utils import extract_persona
from xdpx.utils.chat.rule_control import RuleControl
from xdpx.utils import io
import os

@dataclass
class PipelineConfig:
    rule_control_path: str
    core_chat_save_dir: str
    core_chat_checkpoint: str
    core_chat_pretrained_version: str
    core_chat_quantized: bool
    core_chat_is_onnx: bool
    core_chat_provider: str
    core_chat_allspark_gpu_speed_up: bool
    core_chat_allspark_gen_cfg: dict
    core_chat_max_encoder_length: int

    core_chat_generate_config: dict
    core_chat_max_context_turns: int
    core_chat_max_history_turns: int
    core_chat_max_no_repeat_session: int
    core_chat_max_no_repeat_session_ngrams: int
    core_chat_max_knowledge_length: int
    core_chat_bad_words: str

    learn2search_query_classifier_path: Optional[str]

    post_rerank_save_dir: Optional[str]
    post_rerank_checkpoint: Optional[str]


TOPIC_TRIGGER_PREFIXS = [
    '我们换个话题聊聊怎样',
    '换个话题吧',
    '我们聊个别的',
    '聊个别的呗',
    '我们换个话题吧',
    '尴尬，我们换个别的聊吧',
    '好喜欢和你聊天，我们聊个别的吧'
]


# UNSAFE_RESPONSES = ['我们来聊点别的吧~', '我们聊聊其他话题吧~', '我们换个话题吧~']


@dataclass
class ChatInput(object):
    query: str = ''
    utterence: str = ''
    local_faqs: List[FAQ] = None
    openweb_search_results: List[Snippet] = None
    openkg_triples: List[Triple] = None
    dialog_state: dict = field(default=dict)
    history: List[HistoryItem] = None
    bot_profile: str = field(default='')
    user_profile: str = field(default='')
    chat_config: ChatConfig = field(default=ChatConfig())

    def history_as_string_list(self):
        if self.history:
            return [h.utterance for h in self.history]
        return []

    def history_as_string_list2(self):
        if self.history:
            return [h.rewritten_utterance or h.utterance for h in self.history]
        return []

    def remove_bye_in_history(self):
        if self.history:
            for h in self.history:
                if text_is_bye(h.utterance):
                    h.utterance = h.rewritten_utterance = '#'

    def history_filter_as_user_said(self):
        if self.history:
            return [h.rewritten_utterance or h.utterance for h in self.history if h.role != 'bot']
        return []

    def history_filter_as_bot_said(self):
        if self.history:
            return [h.rewritten_utterance or h.utterance for h in self.history if h.role == 'bot']
        return []


class ChatPipeline(object):
    """
    TODO
    1. Learn2Search
        a. QueryClassifier
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        # assert config.utterance_rewriter_save_dir is not None
        print(f'| initialize pipeline..')
        print(f'| creating core chat..')
        self.core_chat = CoreChat(
            config.core_chat_save_dir,
            config.core_chat_checkpoint,
            config.core_chat_is_onnx,
            config.core_chat_pretrained_version,
            config.core_chat_quantized,
            config.core_chat_provider,
            config.core_chat_allspark_gpu_speed_up,
            config.core_chat_allspark_gen_cfg,
            config.core_chat_max_encoder_length,
            config.core_chat_bad_words,
            config.core_chat_max_no_repeat_session_ngrams
        )
        # print(f'| creating utterance rewriter..')
        # self.utterance_rewriter = RewriteModel(config.utterance_rewriter_save_dir,
        #                                       config.utterance_rewriter_is_onnx,
        #                                       config.utterance_rewriter_quantized,
        #                                       config.utterance_rewriter_provider)
        print(f'| initialize learn2search module..')
        self.learn2search = Learn2Search(config.learn2search_query_classifier_path)
        # print(f'| initialize chat skills..')
        # self.chat_skills = ChatSkills()
        print(f'| initialize rule control..')
        self.rule_control = RuleControl(config.rule_control_path)
        # print(f'| initialize safety filter..')
        # self.safty_filter = SaftyFilter()

        if config.post_rerank_save_dir or config.post_rerank_checkpoint:
            print(f'| initilize post reranker..')
            self.post_reranker = PostReranker(config.post_rerank_save_dir, config.post_rerank_checkpoint)
        else:
            print(f'| skip post reranker.')
        dirpath = os.path.dirname(config.rule_control_path)
        path = os.path.join(dirpath, 'topic_trigger_responses.txt')
        self.TOPIC_TRIGGER_RESPONSES = [l.strip() for l in io.open(path).readlines() if l.strip()]

    def refresh_rule_control(self):
        self.rule_control = RuleControl(self.config.rule_control_path)

    def split_chunks(self, text):
        result = []
        if not text:
            return []
        if len(text) > self.config.core_chat_max_knowledge_length:
            chunks = math.ceil(len(text) / float(self.config.core_chat_max_knowledge_length))
            for chunk in range(chunks):
                window_size = 10
                start = max(chunk * self.config.core_chat_max_knowledge_length - window_size, 0)
                end = (chunk + 1) * self.config.core_chat_max_knowledge_length
                result.append(text[start: end])
        else:
            result.append(text)
        return result

    def chat(self, chat_input: ChatInput):
        dialog_state = chat_input.dialog_state if chat_input.dialog_state else {}
        chat_output = ChatOutput(dialog_state=dialog_state, debug_info={})
        debug_info = chat_output.debug_info

        chat_input.remove_bye_in_history()
        # TODO 工程传入
        # query, history_for_rewrite = self.utterance_rewriter.rewrite(chat_input.query, chat_input.history)
        # debug_info['query_rewritten'] = query
        # debug_info['history_for_rewrite'] = history_for_rewrite
        query = chat_input.query

        q_is_persona_question = is_persona_question(query) or is_persona_question(chat_input.query)
        debug_info.update({'q_is_persona_question': q_is_persona_question})

        knowledge_chunks = []
        openweb_search_results = None
        local_faqs = None
        openkg_triples = []
        # rule_skill_invoke = False
        # faq_skill_invoke = False
        # reform_search_query = None

        # TODO 工程调用 【前置】
        # if self.safty_filter is not None and self.safty_filter.is_unsafe(query):
        #     debug_info.update({
        #         'safty_filter': 'unsafe_query',
        #     })
        #     chat_output.response = random.choice(UNSAFE_RESPONSES)
        #     return chat_output
        # TODO 工程实现，暂时保留
        # hit_rule = self.rule_control.call(query)
        # if hit_rule is not None:
        #     debug_info.update({
        #         'hit_rule': hit_rule.asdict(),
        #     })
        #     if hit_rule.response:
        #         chat_output.response = random.choice(hit_rule.response)
        #         return chat_output
        #     if hit_rule.reform_query:
        #         reform_search_query = hit_rule.reform_query  # 只修改 search query
        #     if hit_rule.append_knowledge:
        #         knowledge_chunks.append(hit_rule.append_knowledge)
        #     if hit_rule.invoke_skill:
        #         rule_skill_invoke = True

        # if self.local_retrieval and not chat_config.local_retrieval_use:
        #     print('|[warning] set local_retrieval_host but not open local_retrieval_use in chat config')
        # if self.local_retrieval and chat_config.local_retrieval_use:
        if chat_input.local_faqs:
            # TODO 工程传入
            # faq, local_faqs = self.local_retrieval.local_faq_retrieval(self.config.local_retrieval_tenant,
            #                                                           chat_input.query,
            #                                                           size=self.config.local_retrieval_max_faq)
            faq = chat_input.local_faqs[0]
            local_faqs = chat_input.local_faqs
            debug_info.update({
                'faq': faq,
                'local_faqs': local_faqs
            })
            # TODO 由工程判断是否调用ds 以及 是否直出
            # if faq:
            #     chat_output.response = faq.answer
            #     if faq.answer in self.local_retrieval.special_answers:  # 触发技能
            #         faq_skill_invoke = True
            #     else:
            #         chat_output.grounding_evidence = faq.__dict__
            #         return chat_output
            for faq in local_faqs:
                knowledge_chunks.append(faq.answer)

        debug_info.update({
            # 'rule_skill_invoke': rule_skill_invoke,
            # 'faq_skill_invoke': faq_skill_invoke
        })
        # TODO 技能模块由工程调用DS
        # if rule_skill_invoke or faq_skill_invoke or dialog_state.get('ds_session_id') is not None:
        #     skill_response, skill_name = self.chat_skills.call(query, dialog_state=dialog_state)
        #     if skill_response is not None:
        #         debug_info.update({
        #             'skill_response': skill_response,
        #             'skill_name': skill_name
        #         })
        #         chat_output.response = skill_response
        #         return chat_output

        # TODO 是否需要搜索，应当在工程调用神马搜索前进行判断，暂时保留
        need_search, query_label = self.learn2search.need_search(query)
        debug_info.update({
            'need_search': need_search,
            'query_label': query_label
        })

        if need_search and chat_input.openweb_search_results:  # self.openweb is not None:
            # if not reform_search_query:
            #     search_query = self.learn2search.get_search_query(query, chat_input.history)
            # else:
            #     search_query = reform_search_query  # 默认优先采用 reform search query
            openweb_search_results = chat_input.openweb_search_results  # self.openweb.search(search_query)
            debug_info.update({
                'openweb_search_results': openweb_search_results
            })

            for snippet in openweb_search_results:
                chunks = self.split_chunks(snippet.snippet)
                knowledge_chunks.extend(chunks)

        # TODO 工程调用ner 和 openkg查询
        # if self.ner_tool is not None :
        if chat_input.openkg_triples:
            # ner_entities = self.ner_tool.recognize(query)
            # debug_info['ner_entities'] = ner_entities
            # if ner_entities and self.openkg is not None:
            # subject = ner_entities[0].entity
            openkg_triples = chat_input.openkg_triples  # self.openkg.search(subject)
            debug_info.update({
                'openkg_triples': openkg_triples
            })
            triples_string = ', '.join([f'{t.subject} {t.predicate} {t.object}' for t in openkg_triples])
            chunks = self.split_chunks(triples_string)[:5]
            knowledge_chunks.extend(chunks)
        context, history = [], []
        utterance = query
        if chat_input.history:
            need_concat_context = not q_is_persona_question \
                                  and not is_special_skill(query) \
                                  and (
                                          self.learn2search.query_classifier is None
                                          or
                                          query_label == CHITCHAT_QUERY
                                  )
            if need_concat_context:
                utterance = chat_input.query
                context = [h.utterance for h in chat_input.history[-self.config.core_chat_max_context_turns:]]
                history = [h.utterance for h in chat_input.history[
                                                -self.config.core_chat_max_history_turns:-self.config.core_chat_max_context_turns]]
            else:
                context = []
                history = []

        if q_is_persona_question:
            user_said = chat_input.history_filter_as_user_said()
            bot_said = chat_input.history_filter_as_bot_said()
            dynamic_user_profile = list(itertools.chain(*[extract_persona(u) for u in user_said]))
            dynamic_bot_profile = list(itertools.chain(*[extract_persona(u) for u in bot_said]))
            if chat_input.user_profile:
                dynamic_user_profile.insert(0, chat_input.user_profile)
            if chat_input.bot_profile:
                dynamic_bot_profile.insert(0, chat_input.bot_profile)

            user_profile = ';'.join(dynamic_user_profile).replace('我', '你')
            bot_profile = ';'.join(dynamic_bot_profile)
        else:
            user_profile, bot_profile = chat_input.user_profile.replace('我', '你'), chat_input.bot_profile

        user_profile_chunks = self.split_chunks(user_profile)
        bot_profile_chunks = self.split_chunks(bot_profile)

        no_repeat_session = [chat_input.query]
        if not q_is_persona_question and chat_input.history:
            no_repeat_session = [h.utterance for h in chat_input.history[-self.config.core_chat_max_no_repeat_session:]] \
                                + [chat_input.query]
        no_repeat_session = [re.sub(r'[0-9]+', ' ', u) for u in no_repeat_session]

        responses, generate_time, model_input = self.core_chat.respond(utterance=utterance,
                                                                       context=context,
                                                                       history=history,
                                                                       user_profile=user_profile_chunks,
                                                                       bot_profile=bot_profile_chunks,
                                                                       knowledge_list=knowledge_chunks,
                                                                       no_repeat_session=no_repeat_session,
                                                                       generate_config=self.config.core_chat_generate_config
                                                                       )

        debug_info.update({
            'query' : query,
            'fid_model_input': model_input,
            'no_repeat_session': no_repeat_session,
            'responses': responses,
            'generate_time': generate_time
        })
        response = responses[0]
        if len(responses) > 1 and self.post_reranker is not None:
            try:
                reranked_responses = self.post_reranker.rerank(query, responses)
                response = reranked_responses[0][0]
                debug_info.update({
                    'reranked_responses': reranked_responses
                })
            except:
                pass
        debug_info['response'] = response
        openkg_triples_tuple = [(item.subject, item.predicate, item.object) for item in openkg_triples]

        chat_output.grounding_evidence = self.find_grounding_evidence(response,
                                                                      local_faqs=local_faqs,
                                                                      openweb_search_results=openweb_search_results,
                                                                      openkg_triples=openkg_triples_tuple)

        response = self.post_process(response, chat_input.history)
        chat_output.response = response
        return chat_output

    def post_process(self, response, history: List[HistoryItem]):
        # TODO 由工程进行过滤
        # if self.safty_filter is not None and self.safty_filter.is_unsafe(response):
        #    response = random.choice(UNSAFE_RESPONSES)
        #    return response
        if text_is_bye(response) \
                and history is not None \
                and len(history) > 0 \
                and history[-1].utterance == '#':
            response = random.choice(TOPIC_TRIGGER_PREFIXS) + random.choice(self.TOPIC_TRIGGER_RESPONSES)
        repeat_response = response == history[-1].utterance if history else False
        repeat_prefix = ['怎么了? ', '重要的事情再说一次，', '可能您刚没听清，我重新说一下， ', '重要的事情讲三遍, ',
                         '不好意思，您刚才没听清吗，我再说一次， ']
        if repeat_response:
            response = repeat_prefix[len(history) % len(repeat_prefix)] + response
        return response

    def find_grounding_evidence(self, response, local_faqs, openweb_search_results: List[Snippet],
                                openkg_triples: List[Tuple]):
        rouge = Rouge()
        evidences = []
        if openweb_search_results:
            for snippet in openweb_search_results:
                rouge_score = rouge.get_scores(' '.join(response), ' '.join(snippet.snippet))[0]
                rouge_p = rouge_score['rouge-l']['p']
                evidences.append(Evidence(obj=snippet, score=rouge_p, source='openweb'))
        if local_faqs:
            for faq in local_faqs:
                rouge_score = rouge.get_scores(' '.join(response), ' '.join(faq.answer))[0]
                rouge_p = rouge_score['rouge-l']['p']
                evidences.append(Evidence(obj=faq, score=rouge_p, source='local_faq'))
        if openkg_triples:
            for triple in openkg_triples:
                triple = f'{triple[0]} {triple[1]} {triple[2]}'
                rouge_score = rouge.get_scores(' '.join(response), ' '.join(triple))[0]
                rouge_p = rouge_score['rouge-l']['p']
                evidences.append(Evidence(obj=triple, score=rouge_p, source='openkg'))
        evidences = sorted(evidences, key=lambda x: x.score, reverse=True)
        return evidences[0] if len(evidences) > 0 and evidences[0].score > 0.7 else None
