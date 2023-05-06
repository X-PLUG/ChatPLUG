import random
import re, json, os
from xdpx.utils.chat import text_is_question
from jieba.analyse import extract_tags
from xdpx.utils import io, cache_file
import jieba
from fasttext import FastText
import numpy as np
from functools import lru_cache

def extract_persona(utterance):
    line_split = re.split(r'[。！；？，,;.!\?]', utterance.strip())
    ss = [t for t in line_split if '我' in t and len(t) > 2 and not text_is_question(t)]
    return ss

def postprocess_norepeat_session(session, profiles):

    session = '</s>'.join(session)
    for p in profiles.split(';'):
        tags = extract_tags(p)
        for t in tags[:3]:
            session = session.replace(t, '<>').replace(t[:2], '<>')
    session = session.split('</s>')
    return session


class PersonaRetriever:

    def __init__(self, model):
        if model.startswith("oss://"):
            model_path = cache_file(model, dry=True)
            if not io.isfile(model_path):
                io.copy(model, model_path)
            self.model = FastText.load_model(model_path)
        else:
            self.model = FastText.load_model(model)

    @lru_cache(maxsize=50000)
    def get_vector(self, query):

        words = ' '.join(list(jieba.cut(query)))
        vec = self.model.get_sentence_vector(words)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def rank(self, utterance, topk=20, profile=''):

        qv = self.get_vector(utterance)
        profile = profile.split(';')
        if not profile:
            return ''
        if len(profile) <= topk:
            return ';'.join(profile)

        profile_vec = [self.get_vector(s) for s in profile]
        similarity = np.matmul(qv.reshape(1, -1), np.array(profile_vec).transpose(1, 0))[0]
        argindex = np.argsort(-similarity)[:topk]

        profile_retrieved = []
        for i in argindex:
            profile_retrieved.append(profile[i])
        return ';'.join(profile_retrieved)

class PersonaFAQ:

    def __init__(self, persona_faq_config):

        with io.open(persona_faq_config) as inf:
            setting = json.load(inf)

        self.template = setting['template']
        self.bots = []
        for key in setting:
            if key != 'template':
                self.bots.append(setting[key])

    def is_persona_intent(self, answer):
        for type_ in self.template:
            if type_ in answer or '$$$我的' + type_.lstrip('$$$') in answer:
                return True
        return False

    def invoke(self, bot_profile, query, answer):

        # 过滤掉反问user #我叫什么#，保留#自我介绍#
        if '我' in query and '你' not in query and '自我' not in query:
            return False, answer

        for type_ in self.template:
            if type_ in answer:
                answer = type_
        if answer not in self.template:
            return False, answer

        bot = None
        for bot_ in self.bots:
            if bot_.get('$$$姓名$$$')[0] in bot_profile:
                bot = bot_
                break
        if bot is None:
            return False, answer

        if answer in bot:
            return True, random.choice(self.template[answer]).format(random.choice(bot[answer]))

        return False, answer








