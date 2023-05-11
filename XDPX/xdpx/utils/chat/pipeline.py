import re
from typing import List, Tuple, Optional

from xdpx.utils.chat.base import ChatInput, ChatOutput, HistoryItem, Evidence
from rouge import Rouge
from xdpx.utils import io
from dataclasses import dataclass, field
from xdpx.utils import cache_file
from icecream import ic
from xdpx.utils.chat import text_is_bye, is_persona_question, is_special_skill

import random
import torch
import itertools
import os
import math
import time
import copy
from xdpx.utils.chat.learn2call import Learn2Call
from xdpx.utils.chat.chat_skills import ChatSkills
from xdpx.utils.chat.learn2search import BaseLearn2Search, Learn2Search
from xdpx.utils.chat.unified_ner import NERTool
from xdpx.utils.chat.post_rerank import PostReranker
from xdpx.utils.chat.core_chat import CoreChat
from xdpx.utils.chat.utterance_rewrite import BaseRewriteModel, RewriteModel
from xdpx.utils.chat.openkg_retrieval import OpenKG
from xdpx.utils.chat.openweb_search import OpenWeb, Snippet
from xdpx.utils.chat.local_retrieval import LocalRetrieval
from xdpx.utils.chat.base import BAIKEQA_QUERY, CHITCHAT_QUERY
from xdpx.utils.chat.persona_utils import extract_persona, postprocess_norepeat_session, PersonaFAQ, PersonaRetriever
from xdpx.utils.chat.rule_control import RuleControl
from xdpx.utils.chat.safety_filter import SaftyFilter
from xdpx.utils import default_cache_root


@dataclass
class PipelineConfig:
    use_safety_rule: bool
    safety_rule_for_query_path: str
    safety_rule_for_resp_path: str
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

    utterance_rewriter_save_dir: str
    utterance_rewriter_is_onnx: bool
    utterance_rewriter_quantized: bool
    utterance_rewriter_provider: str

    unified_ner_url: Optional[str]
    unified_ner_threshold: float

    learn2search_query_classifier_path: Optional[str]

    openweb_use: bool
    openweb_is_test: bool
    openkg_use: bool

    local_retrieval_host: str
    local_retrieval_user: str
    local_retrieval_password: str
    local_retrieval_tenant: str
    local_retrieval_faq_q_threshold: float
    local_retrieval_faq_q_threshold2: float
    local_retrieval_max_faq: int
    persona_faq_config: str
    open_persona_faq: bool

    post_rerank_save_dir: Optional[str]
    post_rerank_checkpoint: Optional[str]

    persona_retriever_model: Optional[str]

    # for benchmark eval
    use_hit_rule: bool = True
    use_faq: bool = True

    use_instruction: bool = False


# for deploying on public cloud by aquila (can not read objects from different regions)
TOPIC_TRIGGER_RESPONSES = [
    '你说我们为什么喜欢八卦',
    '你说中国人的美国梦碎了吗',
    '不想上班怎么破',
    '特别怕死怎么破',
    '最近余华很火啊，你知道他吗？',
    '电影圈是个江湖啊',
    '老男孩的“小苹果”怎么就火了',
    '你知道相亲有多奇葩',
    '为什么有人那么爱喝酒呢',
    '人工智能时代还要过年回家吗',
    '如何成为饭局达人',
    '为什么感觉越来越穷',
    '你的梦想是什么',
    '人生的意义是什么呢',
    '你怎么看父子母女之间的关系',
    '你怎么看面子， 面子多少钱一斤',
    '为啥会有作女，女人为何要作呢？',
    '工作和生活真能平衡吗',
    '我总觉得缺觉，你呢',
    '点菜是门技术，你觉得呢',
    '你喜欢吃宵夜吗',
    '谁的人生不是匆匆租客啊，我们来聊聊租房？',
    '想吃又想瘦，怎么办呢',
    '什么时候想放飞自己哇',
    '你喜欢哪位古人',
    '要不要跨出舒适区，你怎么看',
    '你还记得高考吗，高考压力怎么解',
    '火锅江湖你属哪一派',
    '你被猪队友坑过吗',
    '你怎么看出身？ 家世的影响有多大？',
    '失恋是不是一种病',
    '跳槽季，越跳越迷茫啊',
    '魔鬼藏在细节里，你觉得呢',
    '小鲜肉是好词还是坏词',
    '怎么才能幸福？',
    '侵权这事怎么断',
    '你怎么理解规则',
    '如何面对逆境？你有逆商吗',
    '怎样才算烂片？为啥出烂片？',
    '你有成就吗？',
    '青春时光真的美吗',
    '你说过年需要亲戚吗',
    '人真的能佛系无所谓吗',
    '如何一眼识别渣男',
    '粉丝时代人设属于谁',
    '什么时候你变得爱哭',
    '我们聊聊独居啊，你说一个人住真的好吗',
    '你经历过分手吗？都有哪些分手理由哇',
    '我们的父母都逐渐老去， 你有这样的感受吗',
    '你说我们是要一专还是多能呢？',
    '你常遇到会错意吗？',
    '逃单还是抢单，饭桌见世情啊',
    '流量时代几家欢乐几家愁哇',
    '听音乐有鄙视链吗',
    '你是真的睡不着吗',
    '跟孩子沟通真是不简单啊',
    '你怎么看明星结婚离婚啊，分手的幕前幕后好多戏',
    '血拼的后遗症你经历过吗？',
    '你怎么看原生家庭这个词？',
    '你怎么看打工人996',
    '挣钱就丢人吗？你怎么看',
    '你是选择困难症吗',
    '有句话不知该不该说，你有没有遇到过这样的情况',
    '你怎么看凡学，我价格太高自己请不起',
    '我跟父亲越老越像，你呢',
    '现在社会经常出现PUA，啥是PUA呢',
    '最近工作老是内卷，这是为啥呢',
    '身在远乡为异客，从家乡出来是不是对的选择呢',
    '你说分手后还能不能做朋友',
    '人到30是选择稳定呢还是追求梦想',
    '你说我们该不该支持早恋',
    '没有爱了要不要离婚',
    '不靠谱梦想要不要劝阻',
    '新东方直播火了，你怎么看',
    '我们聊聊三国演义怎样',
    '来聊聊篮球',
    '你知道姚明是干什么的吗',
    '你知道红楼梦吗',
    '你知道特斯拉吗',
    '你喜欢哪个明星？',
    '你喜欢听谁的歌啊',
    '你喜欢看什么类型的电视剧啊',
    '你知道余华吗'
]
TOPIC_TRIGGER_PREFIXS = [
    '我们换个话题聊聊怎样',
    '换个话题吧',
    '我们聊个别的',
    '聊个别的呗',
    '我们换个话题吧',
    '尴尬，我们换个别的聊吧',
    '好喜欢和你聊天，我们聊个别的吧'
]
UNSAFE_RESPONSES = ['你不要为难我，我只是个萌新而已', '我们聊聊其他话题吧~', '我们聊点别的吧', "怎么办，怎么才能让他换个话题",
                    "嗯.. 内个...换个话题吧", "这题我没法答，换个话题吧", "这题不答，下一个", "我太难了，这一题不能答怎么办，得让TA换个话题"]


class ChatPipeline(object):

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
            config.core_chat_max_no_repeat_session_ngrams,
            config.use_instruction
        )

        if config.utterance_rewriter_save_dir:
            print(f'| creating utterance rewriter..')
            self.utterance_rewriter = RewriteModel(config.utterance_rewriter_save_dir,
                                                config.utterance_rewriter_is_onnx,
                                                config.utterance_rewriter_quantized,
                                                config.utterance_rewriter_provider)
        else:
            self.utterance_rewriter = BaseRewriteModel()

        if config.learn2search_query_classifier_path:
            print(f'| initialize learn2search module..')
            self.learn2search = Learn2Search(config.learn2search_query_classifier_path)
        else:
            self.learn2search = BaseLearn2Search()
        
        print(f'| initialize chat skills..')
        self.chat_skills = ChatSkills()
        print(f'| initialize rule control..')
        self.rule_control = RuleControl(config.rule_control_path)
        if config.use_safety_rule:
            print(f'| initialize rule safety for query..')
            self.safety_rule_for_query = RuleControl(config.safety_rule_for_query_path)
            print(f'| initialize rule safety for resp..')
            self.safety_rule_for_resp = RuleControl(config.safety_rule_for_resp_path)
        else:
            print(f'| skip safety rule for query')
            self.safety_rule_for_query = None
            print(f'| skip safety rule for resp')
            self.safety_rule_for_resp = None
        print(f'| initialize safety filter..')
        self.safty_filter = SaftyFilter()
        if config.persona_faq_config:
            print(f'| initialize persona faq..')
            self.persona_faq = PersonaFAQ(config.persona_faq_config)
        else:
            print(f'| skip persona faq..')
            self.persona_faq = None

        if config.openweb_use:
            print(f'| initialize openweb client..')
            self.openweb = OpenWeb(config.openweb_is_test)
        else:
            print(f'| skip openweb .')
            self.openweb = None

        if config.openkg_use:
            print(f'| initialize openkg client..')
            self.openkg = OpenKG()
        else:
            print(f'| skip openkg .')
            self.openkg = None

        if config.unified_ner_url:
            print(f'| initialize ner tool..')
            self.ner_tool = NERTool(config.unified_ner_url, config.unified_ner_threshold)
        else:
            print(f'| skip ner tool.')
            self.ner_tool = None

        if config.local_retrieval_host and config.use_faq:
            print(f'| initialize local retrieval client..')
            self.local_retrieval = LocalRetrieval(config.local_retrieval_host,
                                                  config.local_retrieval_user,
                                                  config.local_retrieval_password,
                                                  config.local_retrieval_faq_q_threshold,
                                                  config.local_retrieval_faq_q_threshold2)
        else:
            self.local_retrieval = None
            print(f'| skip local retrieval.')

        if config.post_rerank_save_dir or config.post_rerank_checkpoint:
            print(f'| initilize post reranker..')
            self.post_reranker = PostReranker(config.post_rerank_save_dir, config.post_rerank_checkpoint)
        else:
            print(f'| skip post reranker.')

        if config.persona_retriever_model:
            print(f'| initialize persona retriever..')
            self.persona_retriever = PersonaRetriever(config.persona_retriever_model)
        else:
            print(f'| skip persona retriever .')
            self.persona_retriever = None

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

    def chat(self, chat_input: ChatInput, instance_code=""):
        start_time = time.time()
        dialog_state = chat_input.dialog_state if chat_input.dialog_state else {}
        chat_output = ChatOutput(dialog_state=dialog_state, debug_info={})
        debug_info = chat_output.debug_info
        chat_config = chat_input.chat_config

        chat_input.remove_bye_in_history()
        query, history_for_rewrite = self.utterance_rewriter.rewrite(chat_input.query, chat_input.history)
        rewrite_time = time.time()
        debug_info['query_rewritten'] = query
        debug_info['query_rewrite_time'] = rewrite_time - start_time
        debug_info['history_for_rewrite'] = history_for_rewrite

        q_is_persona_question = is_persona_question(query) or is_persona_question(chat_input.query)
        debug_info.update({'q_is_persona_question': q_is_persona_question})

        knowledge_chunks = []
        openweb_search_results = None
        local_faqs = None
        openkg_triples = None
        rule_skill_invoke = False
        faq_skill_invoke = False
        invoke_skill_name = None
        reform_search_query = None

        # ########## check the safety of chat_input (start) ##########
        # green web from cro
        if self.safty_filter is not None and self.safty_filter.is_unsafe(query):
            chat_output.response = random.choice(UNSAFE_RESPONSES)
            debug_info.update({
                'safety_filter': 'unsafe_query',
                'safety_method': 'green_web_from_cro',
                'total_time': time.time() - start_time
            })
            return chat_output
        # safety rule for query
        if self.safety_rule_for_query is not None:        
            hit_safety_rule = self.safety_rule_for_query.call(query)
            if hit_safety_rule is not None:
                chat_output.response = random.choice(hit_safety_rule.response)
                debug_info.update({
                    'safety_filter': 'unsafe_query',
                    'safety_method': 'rule',
                    'total_time': time.time() - start_time
                    })
                return chat_output

        # ########## check the safety of chat_input (end) ##########
        hit_rule = self.rule_control.call(query)
        if hit_rule is not None and self.config.use_hit_rule and dialog_state.get('ds_hang_skill') is None:
            debug_info.update({
                'hit_rule': hit_rule.asdict(),
            })
            if hit_rule.response:
                chat_output.response = random.choice(hit_rule.response)
                debug_info['rule_match_time'] = time.time() - rewrite_time
                debug_info['total_time'] = time.time() - start_time
                return chat_output
            if hit_rule.reform_query and not hit_rule.invoke_skill:
                reform_search_query = hit_rule.reform_query  # 只修改 search query
            if hit_rule.append_knowledge:
                knowledge_chunks.append(hit_rule.append_knowledge)
            if hit_rule.invoke_skill:
                rule_skill_invoke = True
                invoke_skill_name = hit_rule.reform_query  # 特殊技能的 reform query是特殊技能标志

        if hit_rule is not None and dialog_state.get('ds_hang_skill') is not None:
            print('|[warning] skill still hanging but tired to invoke rule')

        rule_time = time.time()
        debug_info['rule_match_time'] = rule_time - rewrite_time

        if self.local_retrieval and not chat_config.local_retrieval_use:
            print('|[warning] set local_retrieval_host but not open local_retrieval_use in chat config')
        if self.local_retrieval and chat_config.local_retrieval_use:
            faq_start = time.time()
            faq, local_faqs = self.local_retrieval.local_faq_retrieval(self.config.local_retrieval_tenant,
                                                                       chat_input.query,
                                                                       size=self.config.local_retrieval_max_faq)
            debug_info.update({
                'faq': faq,
                'local_faqs': local_faqs,
                'faq_cost': time.time() - faq_start
            })
            if faq :
                if self.persona_faq and self.persona_faq.is_persona_intent(faq.answer): # 命中人设意图
                    state, response = self.persona_faq.invoke(chat_input.bot_profile, chat_input.query, faq.answer)
                    if state and self.config.open_persona_faq:
                        chat_output.response = response
                        chat_output.grounding_evidence = faq.__dict__
                        debug_info['total_time'] = time.time() - start_time
                        return chat_output
                else:
                    chat_output.response = faq.answer
                    if faq.answer in self.local_retrieval.special_answers:  # 触发技能
                        faq_skill_invoke = True
                        invoke_skill_name = faq.answer if not invoke_skill_name else invoke_skill_name
                    else:
                        chat_output.grounding_evidence = faq.__dict__
                        debug_info['total_time'] = time.time() - start_time
                        return chat_output

            if local_faqs:
                for faq in local_faqs:
                    if faq.answer not in self.local_retrieval.special_answers and not self.persona_faq.is_persona_intent(faq.answer):  # special answer不需要添加
                        knowledge_chunks.append(faq.answer)
        faq_time = time.time()

        debug_info.update({
            'rule_skill_invoke': rule_skill_invoke,
            'faq_skill_invoke': faq_skill_invoke,
            'invoke_skill_name': invoke_skill_name,
            'faq_match_time': faq_time - rule_time
        })

        if rule_skill_invoke or faq_skill_invoke or dialog_state.get('ds_session_id') is not None or dialog_state.get('ds_hang_skill') is not None:
            # local two turn ds scenario
            if invoke_skill_name in self.chat_skills.local_ds_script:
                dialog_state['ds_hang_skill'] = invoke_skill_name
                if instance_code.startswith("xiaoda"):
                    skill_response = random.choice(self.chat_skills.local_ds_script_xiaoda[invoke_skill_name])
                else:
                    skill_response = random.choice(self.chat_skills.local_ds_script[invoke_skill_name])
                skill_name = invoke_skill_name
                debug_info['skill_name'] = skill_name
                debug_info['skill_response'] = skill_response
                debug_info['skill_time'] = time.time() - faq_time
                debug_info['total_time'] = time.time() - start_time
                chat_output.response = skill_response
                return chat_output

            # general scenario
            skill_response, skill_name = self.chat_skills.call(query, chat_input.query, invoke_skill_name, dialog_state=dialog_state, instance_code=instance_code)
            if skill_response is not None:
                debug_info.update({
                    'skill_response': skill_response,
                    'skill_name': skill_name
                })
                chat_output.response = skill_response
                debug_info['skill_time'] = time.time() - faq_time
                debug_info['total_time'] = time.time() - start_time
                return chat_output

        skill_time = time.time()
        debug_info['skill_time'] = skill_time - faq_time

        need_search, query_label = self.learn2search.need_search(query)
        debug_info.update({
            'need_search': need_search,
            'query_label': query_label
        })

        query_classify_time = time.time()
        debug_info['query_classify_time'] = query_classify_time - skill_time

        if need_search and self.openweb is not None:
            if not reform_search_query:
                search_query = self.learn2search.get_search_query(query, chat_input.history)
            else:
                search_query = reform_search_query  # 默认优先采用 reform search query
            start = time.time()
            openweb_search_results, is_special_card = self.openweb.search(search_query)
            debug_info.update({
                'search_time': time.time() - start,
                'search_query': search_query,
                'openweb_search_results': openweb_search_results
            })

            if is_special_card:
                chat_output.response = openweb_search_results[0].snippet
                debug_info['response'] = chat_output.response
                debug_info['total_time'] = time.time() - start_time
                return chat_output

            for snippet in openweb_search_results:
                chunks = self.split_chunks(snippet.snippet)
                knowledge_chunks.extend(chunks)

        if self.ner_tool is not None and not q_is_persona_question: # 人设型问题不需要三元组知识
            start = time.time()
            ner_entities = self.ner_tool.recognize(query)
            debug_info['ner_entities'] = ner_entities
            if ner_entities and self.openkg is not None:
                subject = ner_entities[0].entity
                openkg_triples = self.openkg.search(subject)
                debug_info.update({
                    'kg_search_word': subject,
                    'kg_search_time': time.time() - start,
                    'openkg_triples': openkg_triples
                })
                triples_string = ', '.join([f'{t[0]} {t[1]} {t[2]}' for t in openkg_triples])
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
                if is_persona_question(chat_input.query):  # 如果本身是persona问题 则用改写前query作为utterance
                    utterance = chat_input.query

        user_said = chat_input.history_filter_as_user_said()
        bot_said = chat_input.history_filter_as_bot_said()
        dynamic_user_profile = list(itertools.chain(*[extract_persona(u) for u in user_said]))
        dynamic_bot_profile = list(itertools.chain(*[extract_persona(u) for u in bot_said]))
        dynamic_user_profile, dynamic_bot_profile = ';'.join(dynamic_user_profile), ';'.join(dynamic_bot_profile)
        user_profile, bot_profile = "", ""
        if q_is_persona_question:
            if chat_input.user_profile:
                user_profile = chat_input.user_profile + ';' + dynamic_user_profile
                user_profile = user_profile.replace('我', '你')
            if chat_input.bot_profile:
                if self.persona_retriever:
                    bot_profile_retrieved = self.persona_retriever.rank(utterance=query, profile=chat_input.bot_profile) # bot_profile可能很大，用模型先检索最相关的top20
                    bot_profile = bot_profile_retrieved + ';' + dynamic_bot_profile
                else:
                    bot_profile = chat_input.bot_profile + ';' + dynamic_bot_profile
        else:
            user_profile = chat_input.user_profile.replace('我', '你')
            if self.persona_retriever:
                bot_profile_retrieved = self.persona_retriever.rank(utterance=query, profile=chat_input.bot_profile)
                bot_profile = bot_profile_retrieved + ';' + dynamic_bot_profile
            else:
                bot_profile = chat_input.bot_profile + ';' + dynamic_bot_profile

        user_profile_chunks = self.split_chunks(user_profile)
        bot_profile_chunks = self.split_chunks(bot_profile)

        no_repeat_session = [chat_input.query]
        if not q_is_persona_question and chat_input.history:
            no_repeat_session = [h.utterance for h in chat_input.history[-self.config.core_chat_max_no_repeat_session:]] \
                                + [chat_input.query]
        no_repeat_session = [re.sub(r'[0-9]+', ' ', u) for u in no_repeat_session]
        # 去掉人设相关的norepeat，人设需要稳定不能有高多样性
        no_repeat_session = postprocess_norepeat_session(no_repeat_session, ';'.join([user_profile, bot_profile, dynamic_user_profile, dynamic_bot_profile]))
        if not q_is_persona_question:
            responses, generate_time, model_input = self.core_chat.respond(utterance=utterance,
                                                                           context=context,
                                                                           history=history,
                                                                           user_profile=user_profile_chunks,
                                                                           bot_profile=bot_profile_chunks,
                                                                           knowledge_list=knowledge_chunks,
                                                                           no_repeat_session=no_repeat_session,
                                                                           generate_config=self.config.core_chat_generate_config
                                                                           )
        else:
            # 人设min_length不能太长，否则会乱说话
            persona_generate_config = copy.deepcopy(self.config.core_chat_generate_config)
            persona_generate_config['min_length'] = 5
            responses, generate_time, model_input = self.core_chat.respond(utterance=utterance,
                                                                           context=context,
                                                                           history=history,
                                                                           user_profile=user_profile_chunks,
                                                                           bot_profile=bot_profile_chunks,
                                                                           knowledge_list=knowledge_chunks,
                                                                           no_repeat_session=no_repeat_session,
                                                                           generate_config=persona_generate_config
                                                                           )

        debug_info.update({
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
        try:
            chat_output.grounding_evidence = self.find_grounding_evidence(response,
                                                                          local_faqs=local_faqs,
                                                                          openweb_search_results=openweb_search_results,
                                                                          openkg_triples=openkg_triples)
        except:
            chat_output.grounding_evidence = None # Knowledge过长的时候计算Rouge可能会出现递归爆炸的异常

        response = self.post_process(response, chat_input.history, chat_output)
        chat_output.response = response
        debug_info['total_time'] = time.time() - start_time
        return chat_output

    def post_process(self, response, history: List[HistoryItem], chat_output):
        # 安全过滤
        if self.safty_filter is not None and self.safty_filter.is_unsafe(response):
            response = random.choice(UNSAFE_RESPONSES)
            return response
        # safety rule for query
        if self.safety_rule_for_resp is not None:        
            hit_safety_rule = self.safety_rule_for_resp.call(response)
            if hit_safety_rule is not None:
                response = random.choice(hit_safety_rule.response)
                debug_info = chat_output.debug_info
                debug_info.update({
                    'safety_filter': 'unsafe_resp',
                    'safety_method': 'rule',
                    })
                return response

        response = re.sub('([:,;!?()])', lambda x: chr(ord(x.group(1)) + 65248), response)
        if text_is_bye(response) \
                and history is not None \
                and len(history) > 0 \
                and history[-1].utterance == '#':
            response = random.choice(TOPIC_TRIGGER_PREFIXS) + random.choice(TOPIC_TRIGGER_RESPONSES)
        repeat_response = response == history[-1].utterance if history else False
        repeat_prefix = ['这个问题问过的哦。', '咦。那我再回答一下。', '大家这么喜欢问这个问题的吗。', '唉~我再说一次吧，',
                         '其实刚说过了呢，', '那我再说一次哦~']
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
