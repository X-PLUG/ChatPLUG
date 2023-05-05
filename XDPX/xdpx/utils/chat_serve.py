import torch
import os
import time
import jieba
import unicodedata
import tempfile
import jieba.analyse
import re
import math
import traceback
import random
import requests
import json
import itertools
from typing import List
from xdpx.tasks import tasks
from xdpx.loaders import loaders
from xdpx.tokenizers import tokenizers
from xdpx.options import Options, Argument, Arg
from xdpx.utils import search_shenma, search_baidu, search_google, search_sogou, cache_file, search_shenma_html
from xdpx.utils.geely_search import search_geely

from transformers import AutoTokenizer, BertTokenizer, BertModel, AutoConfig, BertConfig, BertForQuestionAnswering
from xdpx.utils import io, move_to_cuda, parse_model_path

from xdpx.models.chat import FIDT5Chat, T5Chat
from xdpx.utils.chat_skills import call_chat_skills

from icecream import ic
from elasticsearch import Elasticsearch
from torch import nn
import numpy as np
import torch.nn.functional as F

from rank_bm25 import BM25Okapi

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

MAX_PASSAGE_LENGTH = 300
question_regex = [
    re.compile(r'(？|\?|吗|呢|什么|怎么|怎样|咋|啥|如何|为什么|哪|几|谁|多少|多大|多高|是不是|有没有|是否|多久|可不可以|能不能|行不行)'),
    re.compile('(是).+(还是)')]
persona_regex = [
    re.compile('(我|你)+(是|是不是|也是|叫啥|叫什么|几岁|多少岁|毕业|多大|哪里|经常|一般|平时|平常|谁|会|还会|工作|名字|姓名|小名|大名|全名|年龄|年纪|工作|职业|干什么)+'),
    re.compile('(我的|你的)+(名字|姓名|昵称|名称|全名|大名|年纪|年龄|工作|职业|学校|宠物|猫|狗|爱好|大学)+'),
    re.compile('(我|你)+(的)*(父母|爸|妈|男朋友|女朋友|哥|姐|妹|弟|老公|老婆|孩子|女儿|儿子)+'),
    re.compile('(我|你)+(是)*(男的|女的|男生|女生|男孩|女孩|性别)+')
]

bye_response_keywords = ['再见', '拜拜', '不跟你聊', '去忙', '先忙', '不聊', '下次聊', '再聊', '打扰你', '下次再']
bye_regex = re.compile('|'.join(bye_response_keywords))

def text_is_bye(query):
    if bye_regex.findall(query):
        return True
    else:
        return False


def text_is_question(query):
    q = query.replace(' ', '')
    for r in question_regex:
        if r.findall(q):
            return True
    return False


def is_persona_question(q):
    if text_is_question(q):
        for r in persona_regex:
            if r.findall(q):
                return True
    return False


TOPIC_TRIGGER_RESPONSES = [l.strip() for l in io.open(
    'oss://xdp-expriment/gaoxing.gx/chat/benchmark/topic_trigger_responses.txt').readlines() if l.strip()]
TOPIC_TRIGGER_PREFIXS = [
    '我们换个话题聊聊怎样',
    '换个话题吧',
    '我们聊个别的',
    '聊个别的呗',
    '我们换个话题吧',
    '尴尬，我们换个别的聊吧',
    '好喜欢和你聊天，我们聊个别的吧'
]


def random_topic_trigger_response():
    return random.choice(TOPIC_TRIGGER_PREFIXS) + random.choice(TOPIC_TRIGGER_RESPONSES)


def split_chunks(text):
    result = []
    if len(text) > MAX_PASSAGE_LENGTH:
        chunks = math.ceil(len(text) / float(MAX_PASSAGE_LENGTH))
        for chunk in range(chunks):
            window_size = 10
            start = max(chunk * MAX_PASSAGE_LENGTH - window_size, 0)
            end = chunk * MAX_PASSAGE_LENGTH + MAX_PASSAGE_LENGTH
            result.append(text[start: end])
    else:
        result.append(text)
    return result


##TODO:
def extract_persona_i(utterance):
    line_split = re.split(r'[。！；？，,;.!\?]', utterance.strip())
    ss = [t for t in line_split if '我' in t and len(t) > 2 and not text_is_question(t)]
    return ss


def extract_persona_qa(utterance, response):
    question = '，'.join([q for q in re.split('[，。！]', utterance.strip()) if '你' in q and text_is_question(q)])
    persona = '，'.join(
        [q for q in re.split('[，。！]', response.strip()) if q and '你' not in q and not text_is_question(q)])

    if question and persona:
        return '[SEP]'.join([question, persona])
    else:
        return ''

def extract_persona(turn):

    persona = []
    utterance = turn[-2] if len(turn) > 1 else ""
    response = turn[-1]

    qa = extract_persona_qa(utterance, response)
    if qa:
        persona.append(qa)

    for p in extract_persona_i(response):
        if persona and p and p in persona[0]:
            continue
        persona.append(p)

    return persona


def search(engine, search_query, search_cache):
    search_results = []
    if search_query:
        if engine == 'shenma':
            search_results = search_shenma(search_query)
        elif engine == 'shenma_test':
            search_results = search_shenma(search_query, test_url=True)
        elif engine == 'google':
            search_results = search_google(search_query)
        elif engine == 'baidu':
            search_results = search_baidu(search_query)
        elif engine == 'sogou':
            search_results = search_sogou(search_query)
        elif engine == 'shenma_html':
            search_results = search_shenma_html(search_query)
        elif engine == 'shenma_cache':
            search_results = search_cache.get(search_query)
            if not search_results:
                search_results = search_shenma(search_query)
        elif engine == 'geely':
            local_search_results = search_geely(search_query, global_var='星越')
            search_results.extend(local_search_results)
            if not local_search_results or local_search_results[0].get('q_score', 0.0) < 0.99:
                shenma_search_results = search_shenma(search_query)
                search_results.extend(shenma_search_results)
        elif engine == 'geely_jihe':
            local_search_results = search_geely(search_query, global_var='几何')
            search_results.extend(local_search_results)
            if not local_search_results or local_search_results[0].get('q_score', 0.0) < 0.99:
                shenma_search_results = search_shenma(search_query)
                search_results.extend(shenma_search_results)
    return search_results


class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pre_tokenizer(self, x):
        return jieba.cut(x, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


class BertForClassification(nn.Module):
    def __init__(self, config: BertConfig, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, *args, **kwargs):
        # pooled output as default
        features = self.bert(*args, **kwargs)[1]
        features = self.dropout(features)
        return self.classifier(features)


class QueryClassifier:
    BAIKE_QUERY = 'Baike Query'
    CHAT_QUERY = 'Chat Query'

    def __init__(self, classifier_model):
        # step 1. init argument and prepare model file
        if classifier_model.endswith(".pt"):
            model_dir = "/".join(classifier_model.split("/")[:-1])
        else:
            model_dir = classifier_model
        temp_dir = tempfile.mkdtemp()
        print("Building temp query classify config path %s" % model_dir)
        vocab_file_path = os.path.join(model_dir, "vocab.txt")
        io.copy(vocab_file_path, os.path.join(temp_dir, "vocab.txt"))
        config_file_path = os.path.join(model_dir, "config.json")
        io.copy(config_file_path, os.path.join(temp_dir, "config.json"))
        model_dir = temp_dir

        # step 2. pload finetuned model and tokenizer
        self.id2label = [self.BAIKE_QUERY, self.CHAT_QUERY]
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        with io.open(classifier_model, 'rb') as state_dict:
            self.model = BertForClassification(AutoConfig.from_pretrained(os.path.join(model_dir, "config.json")),
                                               num_labels=2)
            self.model.load_state_dict(torch.load(state_dict, map_location=DEVICE))
        self.model = self.model.to(DEVICE)
        self.model.eval()

    def predict(self, utterance, threshold=0.99):
        input_ids = self.tokenizer(utterance, padding=True, truncation=True, max_length=200,
                                   return_tensors="pt").input_ids
        input_ids = input_ids.to(DEVICE)
        logits = self.model(input_ids)
        logits = logits.view(-1).tolist()
        logits_softmax = self.softmax(logits).tolist()
        if logits_softmax[0][1] > threshold:
            label_id = 1
        else:
            label_id = 0
        label = self.id2label[label_id]
        return label

    def softmax(self, logits):
        t = np.exp(logits)
        a = np.exp(logits) / np.sum(t).reshape(-1, 1)
        return a


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

    def rewrite(self, utterances):
        self.model.eval()
        processed_utterances = [each.replace("。", ' ') for each in utterances]  # 去除句子原本带有的句号，否则会被误识别为分隔符
        sentence = '。'.join(processed_utterances)
        input_ids = self.tokenizer(sentence, padding=True, truncation=True, max_length=200,
                                   return_tensors="pt").input_ids
        input_ids = input_ids.to(DEVICE)
        hypotheses = self.model.generate(input_ids, eos_token_id=self.tokenizer.sep_token_id,
                                         decoder_start_token_id=self.tokenizer.cls_token_id,
                                         max_length=80)  # 若不设置最大长度默认为20会导致截断
        if self.cuda:
            hypotheses = hypotheses.detach().cpu().tolist()

        result = self.tokenizer.decode(hypotheses[0], skip_special_tokens=True).replace(' ', '')
        return result


class NER:
    def __init__(self, ner_url):
        self.url = ner_url
        self.header = {'Content-Type': 'application/json', 'Accept-Encoding': 'utf-8'}
        print("Using NER client: %s" % ner_url)

    def predict(self, query):
        param = {"text": query}
        res = requests.post(self.url, data=json.dumps(param), headers=self.header)
        try:
            res_data = json.loads(res.text)
            result = res_data['data']
            result = json.loads(result)
            target_labels = ["Person", "Work", "Location", "Game", "Software", "Medicine", "Food", "Website",
                             "Disease&Symptom"]  # 仅仅链接这几类实体
            result = self.extract_entity(result, target_labels, 0.85)  # 抽取关键实体
        except:
            print("NER Error: " + res.text)
            result = None
        return result

    def extract_entity(self, ner_results, target_labels, score_threshold):
        result_data = []
        if ner_results:
            for each_result in ner_results:
                each_label = each_result["label"]
                each_entity = each_result["span"]
                each_score = float(each_result["score"])
                if each_label in target_labels and each_score > score_threshold:  # 过滤掉其余类别和低置信度样例
                    label_id = target_labels.index(each_label)
                    result_data.append((each_entity, label_id))
        result_data = list(set(result_data))
        result_data = sorted(result_data, key=lambda data: data[1])  # 更重要的实体更靠前
        return result_data  # [(entity, label_id),...]


class KGSearcher:
    def __init__(self, max_length=200, max_passage=5):
        self.host = "es-cn-751b36x4z601u5403.kibana.elasticsearch.aliyuncs.com:5601"
        self.index_name = "deepqa_ownthink_kg_whole"
        self.url = 'https://{}/api/console/proxy?path=%2F{}%2F_search&method=POST'.format(self.host, self.index_name)
        self.cookie = 'cna=4d3AGTDwzWICASp4Smc4yzO7; sid=Fe26.2**ef7c651c27d4f13934c0214c3758145ded7e515e4a8141c03a6d8dea9c4b3d56*82d3kUZJSMgO-DF77bSjTw*r_cfa_NnGejyztjgzqhPa6AXR4wUGrijkRze_GdDzWOB' \
                      'scfUTXLWXgPdK1_ThfrrYj_WsewveDWKQDsWc1HLS1gh2OMFAkB1mPKwM6ya53azfxKPMzMTDI9nYIS9WHNDfoQughF7HjNdl2zUr5nlOeghCiz56ApRbEXt_4lUxf0**3258d964a0af5c76b39ab65fd' \
                      '92d9c94458ee131215c432c9770b65382f1c9c7*LBHEbjznhwc1Sz0-IgXuaoD8hoVBy6ry3Qx89W4j0nc'
        self.header = {
            'Referer': "https://es-cn-751b36x4z601u5403.kibana.elasticsearch.aliyuncs.com:5601/app/kibana",
            "kbn-version": "6.7.0",
            "Content-Type": "application/json"
        }
        self.MAX_PASSAGE_LENGTH = max_length
        self.MAX_PASSAGE_NUM = max_passage

    def search_es_as_passage(self, query):
        param = {"size": 1000, "query": {
            "term": {"subject": query}
        }}
        try:
            res = requests.post(self.url, data=json.dumps(param), cookies={"cookie": self.cookie}, headers=self.header)
            res_data = json.loads(res.text)
        except Exception as e:
            print(e)
            return []
        hit_num = res_data["hits"]["total"]
        if hit_num == 0:
            return []
        else:
            result_list = [""]
            for each_data in res_data["hits"]["hits"]:
                subject = each_data["_source"]["subject"]
                predicate = each_data["_source"]["predicate"]
                object = each_data["_source"]["object"]
                try:
                    cur_record = " ".join([subject, predicate, object]) + '</s>'
                    result_list[-1] += cur_record
                    if len(result_list[-1]) > self.MAX_PASSAGE_LENGTH:
                        result_list.append('')
                except Exception as e:
                    print(e)
                    continue
        return result_list[:self.MAX_PASSAGE_NUM]


class KGSearcher2:
    def __init__(self, max_length=200, max_passage=5):
        self.new_host = "es-cn-751b36x4z601u5403.elasticsearch.aliyuncs.com"
        self.index_name = "deepqa_ownthink_kg_whole"
        self.es = Elasticsearch(hosts=[self.new_host], http_auth=('elastic', 'flashsearchES123'), port=9200,
                                use_ssl=False, timeout=1)
        self.MAX_PASSAGE_LENGTH = max_length
        self.MAX_PASSAGE_NUM = max_passage

    def search_es_as_passage(self, query):
        param = {"size": 1000, "query": {
            "term": {"subject": query}
        }}
        try:
            res_data = self.es.search(index=self.index_name, body=param)
        except Exception as e:
            print(e)
            return []
        hit_num = res_data["hits"]["total"]
        if hit_num == 0:
            return []
        else:
            result_list = [""]
            for each_data in res_data["hits"]["hits"]:
                subject = each_data["_source"]["subject"]
                predicate = each_data["_source"]["predicate"]
                object = each_data["_source"]["object"]
                try:
                    cur_record = " ".join([subject, predicate, object]) + '</s>'
                    result_list[-1] += cur_record
                    if len(result_list[-1]) > self.MAX_PASSAGE_LENGTH:
                        result_list.append('')
                except Exception as e:
                    print(e)
                    continue
        return result_list[:self.MAX_PASSAGE_NUM]


class KnowledgeIntervention:
    def __init__(self, regex_path):

        self.regex_path = cache_file(regex_path, dry=True)
        io.copy(regex_path, self.regex_path)
        self.init_regex()

    def init_regex(self):
        self.regexes = []
        for line in open(self.regex_path):
            content, regex, regex_banned, search_rewrited, knowledge = line.strip().split('\t')
            knowledge = knowledge.split(';;;')
            if regex_banned == '-1':
                self.regexes.append([content, re.compile(regex), None, search_rewrited, knowledge])
            self.regexes.append([content, re.compile(regex), re.compile(regex_banned), search_rewrited, knowledge])

    def rewrite(self, query):

        results = {'query': query, 'query_rewrited': query, 'hit': False, 'knowledge': ''}
        for regex in self.regexes:
            query_n = query.lower().replace(' ', '')
            if re.search(regex[1], query_n) is not None:
                if regex[2] is not None and re.search(regex[2], query_n) is not None:
                    continue
                results['query_rewrited'] = regex[3]
                results['hit'] = True

                knowledge = random.choice(regex[4])
                results['knowledge'] = knowledge if knowledge != '-1' else '-1'

        return results

class QARankModel:
    def __init__(self, save_dir, checkpoint=None):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.cuda else torch.device('cpu')

        with io.open(os.path.join(save_dir, 'args.py')) as f:
            args = Options.parse_tree(eval(f.read()))
        try:
            with io.open(os.path.join(args.data_dir, 'args.py')) as f:
                args = Arg().update(Options.parse_tree(eval(f.read()))).update(args)
        except IOError:
            pass
        args.__cmd__ = 'serve'
        args.save_dir = save_dir
        args.temperature = 1.0
        # build the task
        task = tasks[args.task](args)
        model = task.build_model(args)
        loss = task.build_loss(args)
        model_path = checkpoint if checkpoint else parse_model_path('<best>', args)
        model.load(model_path)

        if self.cuda:
            model = model.cuda()

        self.processor = task.processor
        self.loader = loaders[args.loader](args)
        self.task = task
        self.model = model
        self.loss = loss
        self.args = args

    def rerank(self, query, search_results: List, return_sorted=True):
        if not search_results:
            return []
        batch = []
        for s in search_results:
            batch.append([query, s['snippet']])
        try:
            batch = [self.processor.encode(self.loader, sample) for sample in batch]
            batch = self.processor.collate(batch)

            if self.cuda:
                batch = move_to_cuda(batch)

            self.model.eval()
            with torch.no_grad():
                z1, z2 = self.model(**batch['net_input'])
                cos_sim = F.cosine_similarity(z1, z2, dim=-1, eps=1e-4).tolist()  # batch_size
                for i, t in enumerate(search_results):
                    t['qa_score'] = cos_sim[i]
                if return_sorted:
                    results = sorted(search_results, key=lambda x: x['qa_score'], reverse=True)
                else:
                    results = search_results
                return results
        except Exception as e:
            traceback.print_exc()

    def rerank2(self, query, search_results: List, return_sorted=True):
        if not search_results:
            return []
        batch = []
        for s in search_results:
            batch.append([query, s['snippet']])
        try:
            batch = [self.processor.encode(self.loader, sample) for sample in batch]
            batch = self.processor.collate(batch)

            if self.cuda:
                batch = move_to_cuda(batch)
            self.model.eval()
            with torch.no_grad():
                pred, prob, target_prob = self.task.inference_step(batch, self.model, self.loss)
                for i, t in enumerate(search_results):
                    t['qa_score'] = prob[i][1]
                if return_sorted:
                    results = sorted(search_results, key=lambda x: x['qa_score'], reverse=True)
                else:
                    results = search_results
                return results
        except Exception as e:
            traceback.print_exc()


class EnsembleModel(object):
    def __init__(self):
        self.rerank_model = QARankModel('oss://xdp-expriment/gaoxing.gx/chat/training/rerank/v0.9.1.rerank/1024_0.0001')
        self.metric_model = QARankModel('oss://xdp-expriment/gaoxing.gx/chat/training/metric/1001/64_2e-05_5_True_sbert-base')

    def rerank(self, query, search_results: List):
        rerank_results = self.rerank_model.rerank(query, search_results, return_sorted=False)
        metric_results = self.metric_model.rerank2(query, search_results, return_sorted=False)
        for t1, t2 in zip(rerank_results, metric_results):
            t1['qa_score'] = (t1['qa_score'] + t2['qa_score']) / 2
        results = sorted(rerank_results, key=lambda x: x['qa_score'], reverse=True)
        return results


class MrcModel:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-pert-base-mrc')
        self.model = BertForQuestionAnswering.from_pretrained('hfl/chinese-pert-base-mrc').cuda()

    def process_batch(self, utterance, passages):
        batch = []
        for passage in passages:
            batch.append((utterance, passage))
        inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(torch.device('cuda'))
        with torch.no_grad():
            outputs = self.model(**inputs)
        new_passages = []
        for index, (batch_i, start_score, end_score, input_id) in enumerate(
                zip(batch, outputs.start_logits, outputs.end_logits, inputs.input_ids)):
            max_startscore = torch.argmax(start_score)
            max_endscore = torch.argmax(end_score)
            ans_tokens = inputs.input_ids[index][max_startscore: max_endscore + 1]
            answer_tokens = self.tokenizer.convert_ids_to_tokens(ans_tokens,
                                                                 skip_special_tokens=True)
            answer_span = self.tokenizer.convert_tokens_to_string(answer_tokens).replace(' ', '')

            if answer_span:
                new_passages.append(batch_i[1].replace(answer_span, f'<em>{answer_span}</em>'))
            else:
                new_passages.append(batch_i[1])

        return new_passages


class ConsistencyModel:
    def __init__(self, save_dir, checkpoint=None):

        # from fasttext import FastText
        # model_dir = 'oss://xdp-expriment/wenshen.xws/opendialogue/LTMP/resource/FastText/model.bin'
        # if model_dir.startswith("oss://"):
        #     model_path = cache_file(os.path.join(model_dir, 'model.bin'), dry=True)
        #     if not io.isfile(model_path):
        #         io.copy(os.path.join(model_dir, 'model.bin'), model_path)
        #     self.model = FastText.load_model(model_path)
        # else:
        #     self.model = FastText.load_model(os.path.join(model_dir, "model.bin"))


        self.cuda = torch.cuda.is_available()
        with io.open(os.path.join(save_dir, 'args.py')) as f:
            args = Options.parse_tree(eval(f.read()))
        try:
            with io.open(os.path.join(args.data_dir, 'args.py')) as f:
                args = Arg().update(Options.parse_tree(eval(f.read()))).update(args)
        except IOError:
            pass
        args.__cmd__ = 'serve'
        args.save_dir = save_dir
        args.temperature = 1.0
        # build the task
        task = tasks[args.task](args)
        model = task.build_model(args)
        loss = task.build_loss(args)
        model_path = checkpoint if checkpoint else parse_model_path('<best>', args)
        model.load(model_path)

        if self.cuda:
            model = model.cuda()

        with io.open(os.path.join(args.data_dir, 'args.py')) as f:
            data_args = Options.parse_tree(eval(f.read()))
            data_args.vocab_size = args.vocab_size
            data_args.vocab_file = os.path.join(args.data_dir, 'vocab.txt')
        args.vocab_file = os.path.join(args.data_dir, 'vocab.txt')
        self.tokenizer = tokenizers[data_args.tokenizer](data_args).tokenizer
        self.processor = task.processor
        self.loader = loaders[args.loader](args)
        self.task = task
        self.model = model
        self.loss = loss
        self.args = args

        from cacheout import LRUCache
        self.cache = LRUCache(50000)

    def bm25_rank(self, premises, hypothesis):

        bm25 = BM25Okapi(premises)
        premises_top, premises_index = [], []
        for h in hypothesis:
            score = bm25.get_scores(h)
            argsort = np.argsort(-score)
            premises_top.append([])
            premises_index.append([])
            for i in argsort[:3]:
                premises_top[-1].append(premises[i])
                premises_index[-1].append(i)

        return premises_top, premises_index

    def collate(self, texts):

        tokens = []
        for text in texts:
            tokens.append(self.tokenizer.tokenize(text))

        input_ids = []
        for token in tokens:
            input_ids.append(self.processor.clip(self.processor.numerize_tokens(token)))
        input_ids = torch.LongTensor(self.processor.pad(input_ids))
        mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        if self.cuda:
            input_ids = input_ids.cuda()
            mask = mask.cuda()
        return {
            'input_ids': input_ids,
            'attention_mask': mask
        }

    def sentencode(self, texts):

        inputs = self.collate(texts)
        with torch.no_grad():
            vectors = self.model.sent2vec(**inputs)
        return vectors


    def get_from_cache(self, text):

        if isinstance(text, list):
            text = ''.join(list)
        return self.cache.get(text, [])

    def set_to_cache(self, sentvec: dict):

        for sent, vec in sentvec.items():
            if isinstance(sent, list):
                sent = ''.join(sent)
            self.cache.set(sent, vec)

    def get_vector(self, texts):

        sent_vec = {}
        missed_texts, missed_vectors, = [], []
        for i, text in enumerate(texts):
            v = self.get_from_cache(text)
            if len(v):
                sent_vec[text] = v
            else:
                missed_texts.append(text)

        if len(missed_texts):
            missed_vectors = self.sentencode(missed_texts)
            for i, text in enumerate(missed_texts):
                sent_vec[text] = missed_vectors[i].unsqueeze(0)

        vectors = [0 for i in range(len(texts))]
        for i, text in enumerate(texts):
            vectors[i] = sent_vec[text]

        self.set_to_cache(sent_vec)

        vectors = torch.cat(vectors, 0)
        return vectors


    def rank(self, premises, hypothesis):


        with torch.no_grad():
            texts = premises + hypothesis
            vectors = self.get_vector(texts)
            if self.cuda:
                vectors = vectors.cuda()

            assert vectors.size(0) == len(texts)

            premises_vector = vectors[:len(premises)]
            hypothesis_vector = vectors[-len(hypothesis):]

            premises_vector = premises_vector.unsqueeze(1).repeat(1, len(hypothesis), 1)
            hypothesis_vector = hypothesis_vector.unsqueeze(0).repeat(len(premises), 1, 1)

            score = torch.softmax(self.model.get_score(hypothesis_vector, premises_vector), -1)
            contradiction_score = score.transpose(1, 0)[:, :, 0]
            natural_score = score.transpose(1, 0)[:, :, 1]
            entailment_score = score.transpose(1, 0)[:, :, 2]

            # for i in range(contradiction_score.size(0)):
            #     tmp_score = contradiction_score[i]
            #     tmp_s, tmp_i = torch.sort(tmp_score)
            #     for j in tmp_i:
            #         print (hypothesis[i])
            #         print (premises[j], tmp_score[j].tolist(), score.transpose(1, 0)[i][j].tolist())
            #     for j in tmp_i[-1:]:
            #         print (hypothesis[i], premises[j], tmp_score[j].tolist())
            #     print ()

            contradiction_score_reduce, contradiction_index = contradiction_score.max(dim=1)
            score_sorted, index_sorted = torch.sort(contradiction_score_reduce)

        results, scores = [], []
        for i in index_sorted.tolist():
            results.append(hypothesis[i])
            scores.append(contradiction_score_reduce[i].tolist())

        return results, scores

    def rank2(self, premises, hypothesis):

        premises_top, premises_top_index = self.bm25_rank(premises, hypothesis)

        with torch.no_grad():
            texts = premises + hypothesis
            vectors = self.get_vector(texts)
            if self.cuda:
                vectors = vectors.cuda()

            assert vectors.size(0) == len(texts)

            premises_vector = vectors[:len(premises)]
            hypothesis_vector = vectors[-len(hypothesis):]

            premises_top_index_pt = torch.LongTensor(premises_top_index).to(premises_vector.device)
            premises_vector_unsqueeze = torch.index_select(premises_vector, 0, premises_top_index_pt.view(-1))
            premises_vector_unsqueeze = premises_vector_unsqueeze.view(premises_top_index_pt.size(0), premises_top_index_pt.size(1), -1)
            hypothesis_vector_unsqueeze = hypothesis_vector.unsqueeze(1).repeat(1, premises_vector_unsqueeze.size(1), 1)

            score = torch.softmax(self.model.get_score(hypothesis_vector_unsqueeze, premises_vector_unsqueeze), -1)
            contradiction_score = score[:, :, 0]
            netural_score = score[:, :, 1]
            entailment_score = score[:, :, 2]

            # for i in range(contradiction_score.size(0)):
            #     tmp_score = contradiction_score[i]
            #     tmp_s, tmp_i = torch.sort(-tmp_score)
            #     for j in tmp_i:
            #         print (hypothesis[i])
            #         print (premises_top[i][j], tmp_score[j].tolist(), score[i][j].tolist())
            #     print ()
            #     # for j in tmp_i[-1:]:
            #     #     print (hypothesis[i], premises_top[i][j], tmp_score[j].tolist())
            #     # print ()

            contradiction_score_reduce, contradiction_index = contradiction_score.max(dim=1)
            score_sorted, index_sorted = torch.sort(contradiction_score_reduce)

        contradiction_score = contradiction_score.tolist()
        entailment_score = entailment_score.tolist()
        netural_score = netural_score.tolist()
        results = []
        for i in index_sorted.tolist():
            results.append({})
            results[-1]['snippet'] = hypothesis[i]
            results[-1]['contradiction_score'] = contradiction_score_reduce[i].tolist()
            results[-1]['profile'] = ';'.join(premises_top[i])
            results[-1]['candicate'] = [{'porfile': premises_top[i][j], 'score':contradiction_score[i][j], 'entailment':entailment_score[i][j]} for j in range(len(premises_top[i]))]

        return results


    def test(self):
        premises = '我是男的；我叫莫方; 我今年28岁;我是金牛座; 我在阿里巴巴工作; 我是个工程师；我是贵州人；我喜欢运动; 我比较擅长足球和篮球;我喜欢科比；我喜欢梅西；我从武汉大学计算机专业毕业; 我现在杭州阿里巴巴工作；我喜欢拍照，出版社还买过我的照片；我有一只小狗；我的小狗是泰迪犬；我养了一只泰迪;'
        premises = premises.replace('；', ';').replace(' ', '').strip(';').split(';')
        hypothesis = '我是女的;我今年21岁;我在上海读大学;我喜欢小狗，还养了一只小狗;我不喜欢小狗;我没有小狗;我不是21岁;我没有工作;我不在阿里巴巴工作;我不是男的;我叫小婵;我今年不是28岁;我是水瓶座的;我不在阿里巴巴工作;我在腾讯工作;我是个会计;我不喜欢篮球明星;对比梅西，我更喜欢C罗;我在上海交通大学毕业;我毕业了;我从不拍照;我养了一只拉布拉多;梦到前男友'
        hypothesis = hypothesis.split(';')
        results = self.rank2(premises, hypothesis)

        return results



class Model:
    def __init__(self, save_dir, checkpoint=None, is_onnx=False, pretrained_version='google/mt5-base', quantized=False,
                 provider='cuda', allspark_gpu_speed_up=False, allspark_gen_cfg=None):
        self.cuda = torch.cuda.is_available()
        self.rewrite_model = None
        self.knowledge_model = None
        self.ner_tool = None
        self.kg_searcher = KGSearcher()  # 云上服务器只能采用http+cookie的方式访问弹内ES
        self.query_classifier = None
        self.rerank_model = None
        self.mrc_model = None

        self.search_engine = 'shenma'
        self.search_cache = {}
        self.version = 10.3
        self.CONCAT_HISTORY_INTO_PASSAGE = True
        self.CONCAT_CONTEXT_TURNS = 2
        self.NO_REPEAT_SESSION_SIZE = 20
        self.NO_REPEAT_NGRAM_SIZE = 4
        self.NO_REPEAT_NGRAM_SIZE_FOR_Q = 4

        self.allspark_gpu_speed_up = allspark_gpu_speed_up
        self.is_onnx = is_onnx
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
            args.__cmd__ = 'serve'
            args.save_dir = save_dir
            args.strict_size = True
            # build the task
            task = tasks[args.task](args)
            model = task.build_model(args)
            model.load(checkpoint)

            self.model = model.cuda() if self.cuda else model
            if allspark_gpu_speed_up:
                self.model.load_allspark(allspark_gen_cfg)

            self.tokenizer = loaders[args.loader](args).tokenizer.tokenizer

    def search(self, query, search_query, need_search, query_ner, history, q_is_persona_question, debug_info):
        #### 阿里相关知识干预
        ali_knowledge_results = None
        if self.knowledge_model is not None:
            ali_knowledge_results = self.knowledge_model.rewrite(query)
            debug_info['ali_knowledge_results'] = ali_knowledge_results
            if ali_knowledge_results['hit']:
                search_query = ali_knowledge_results['query_rewrited']

        #### 开放域内容检索
        debug_info['search_query'] = search_query

        search_results = []
        now_search_results = []
        has_search_card = False
        if need_search:
            start = time.time()
            now_search_results = search(self.search_engine, search_query, self.search_cache)
            if self.is_special_skill(query) and now_search_results:
                filtered_search_results = []
                for sr in now_search_results:
                    if sr.get('sc_name') in ('weather_new_huake', 'finance_stock_new', 'covid_19'):
                        filtered_search_results.append(sr)
                if filtered_search_results:
                    now_search_results = filtered_search_results
                    has_search_card = True

            debug_info['search_time'] = time.time() - start

        all_search_results = []
        all_search_results.extend(now_search_results)
        debug_info['search_results'] = all_search_results
        for t in all_search_results:
            snippet = t['snippet']
            chunks = split_chunks(snippet)
            search_results.extend(chunks)

        # 知识图谱三元组知识检索
        if not has_search_card and not text_is_question(
                query) and self.ner_tool is not None and query_ner is not None:  # 仅仅在有NER链接的时候才生效
            query_keywords = query_ner
            start_el = time.time()
            if query_keywords:
                kg_results = self.kg_searcher.search_es_as_passage(query_keywords[0][0])
                debug_info['kg_knowledge'] = kg_results
                if kg_results:
                    search_results.extend(kg_results)
            debug_info['kg_search_time'] = time.time() - start_el

        # ali 知识干预
        if not has_search_card and ali_knowledge_results is not None and ali_knowledge_results['hit']:
            search_results_n = []
            # u
            if ali_knowledge_results['knowledge'] != '-1':
                if '[TOOUTPUT]' in ali_knowledge_results['knowledge']:  # 直接干预输出不prompt
                    search_results_n.append(ali_knowledge_results['knowledge'])
                else:
                    search_results_n.append(ali_knowledge_results['query'] + '：' + ali_knowledge_results['knowledge'])
            else:
                if '头条新闻' in search_results[0]:
                    random.shuffle(search_results)
                    summaries, titles = [], []
                    for res in search_results[:]:
                        if ';;;' not in res:
                            continue
                        title, summary = res.replace('头条新闻:', '').replace('新闻摘要:', '').split(';;;')
                        if len(summary.strip()):
                            summaries.append(summary.strip())
                        else:
                            titles.append(title.strip())
                    if len(summaries):
                        search_results_n.append('[TOOUTPUT]' + summaries[0])
                    else:
                        search_results_n.append('[TOOUTPUT]' + titles[0])
                else:
                    # 只取第一个结果
                    for res in search_results[:1]:
                        search_results_n.append(ali_knowledge_results['query'] + '：' + res)
            search_results = search_results_n
        return search_results, has_search_card

    def rewrite_query(self, utterance, history, debug_info):
        if history is not None and len(history) > 0:
            history_for_rewrite = []
            for t in history[-3:]:
                ru = t.get('rewritten_utterance')
                if ru is None:
                    ru = t.get('utterance')
                history_for_rewrite.append(ru)
            history_for_rewrite.append(utterance)
            debug_info['history_for_rewrite'] = history_for_rewrite
            start = time.time()
            query_rewritten = self.rewrite_model.rewrite(history_for_rewrite)

            debug_info['rewrite_time'] = time.time() - start
            debug_info['query_rewritten'] = query_rewritten
            return query_rewritten
        else:
            return utterance

    def get_rewrite_utterance(self, item):
        ru = item.get('rewritten_utterance')
        if ru is None:
            ru = item.get('utterance')
        return ru

    def is_special_skill(self, query):
        if '新闻' in query \
                or '股价' in query \
                or '疫情' in query \
                or '天气' in query \
                or '市值' in query:
            return True
        else:
            return False

    def get_search_query(self, query, history, debug_info):
        if history is not None and len(history) > 0:
            if text_is_question(query) \
                    or self.is_special_skill(query) \
                    or len(query) < 8:
                search_query = query
            else:  # TODO:
                rewritten_session = [self.get_rewrite_utterance(h) for h in history[-4:]] + [query]
                if len(query) > 10:
                    keywords = jieba.analyse.extract_tags('\n'.join(rewritten_session[-5:]), topK=3)
                    search_query = ' '.join(keywords)
                else:
                    keywords = jieba.analyse.extract_tags('\n'.join(rewritten_session[-5:-1]), topK=2)
                    keywords = [k for k in keywords if k not in query]
                    search_query = ' '.join(keywords) + query
        else:
            search_query = query
        return search_query

    def get_dynamic_profile(self, utterance, history, debug_info):
        bot_said = []
        user_said = []

        dynamic_user_profile, dynamic_bot_profile = [], []
        for i, h in enumerate(history):
            ru = self.get_rewrite_utterance(h)
            if h.get('role') == 'bot':
                bot_said.append(ru)
                dynamic_bot_profile.extend(extract_persona([h.get('utterance') for h in history[:i+1]]))
            else:
                user_said.append(ru)
                dynamic_user_profile.extend(extract_persona([h.get('utterance') for h in history[:i+1]]))
        user_said.append(utterance)

        # dynamic_user_profile = list(itertools.chain(*[extract_persona(u) for u in user_said]))
        # dynamic_bot_profile = list(itertools.chain(*[extract_persona(u) for u in bot_said]))

        dynamic_bot_profile = list(itertools.chain(dynamic_bot_profile))
        dynamic_user_profile = list(itertools.chain(dynamic_user_profile))

        dynamic_user_profile = ';'.join(set(dynamic_user_profile))
        dynamic_bot_profile = ';'.join(set(dynamic_bot_profile))

        debug_info.update({'dynamic_user_profile': dynamic_user_profile, 'dynamic_bot_profile': dynamic_bot_profile})
        return dynamic_user_profile, dynamic_bot_profile

    def chat(self, utterance, history, user_profile, bot_profile, generate_config):
        if history:
            for h in history:
                if text_is_bye(h.get('utterance', '')):
                    h['utterance'] = '#'
                    h['rewritten_utterance'] = '#'

                rewritten_utterance = h.get('rewritten_utterance')
                if rewritten_utterance:
                    segments = rewritten_utterance.split('@@@')
                    if len(segments) == 3:
                        h['type'] = segments[-1]
                        h['ds_session_id'] = segments[1]
                        h['rewritten_utterance'] = segments[0]

        start = time.time()

        SEP_TOKEN = "</s>" if isinstance(self.model, FIDT5Chat) else "[SEP]"
        SPACE_TOKEN = " " if isinstance(self.model, FIDT5Chat) else "▂"

        if generate_config.get('num_beam_groups') == 1:
            del generate_config['num_beam_groups']
        use_type_id = False if isinstance(self.model, FIDT5Chat) else True
        response, debug_info = self.fid_chat(utterance, history, user_profile, bot_profile, generate_config, SEP_TOKEN, SPACE_TOKEN,use_type_id=use_type_id)
        debug_info['total_cost'] = time.time() - start
        ## post_response
        prefixs = [
            '<kg_qa>', '<kg_chat>', '<emotion>', '<persona>', '<kg_fake>'
        ]
        for p in prefixs:
            response = response.replace(p, '')
        if text_is_bye(response) and len(history) > 0 and history[-1]['utterance'] == '#':
            response = random_topic_trigger_response()
        repeat_response = response == history[-1].get('utterance') if history else False
        repeat_prefix = ['怎么了? ', '重要的事情再说一次，', '可能您刚没听清，我重新说一下， ', '重要的事情讲三遍, ', '不好意思，您刚才没听清吗，我再说一次， ']
        if repeat_response:
            response = repeat_prefix[len(history) % len(repeat_prefix)] + response

        return response, debug_info

    def fid_chat(self, utterance, history, user_profile, bot_profile, generate_config, SEP_TOKEN, SPACE_TOKEN, use_type_id=False):
        debug_info = {'user_profile': user_profile, 'bot_profile': bot_profile,
                      'search_engine': self.search_engine, "utterance": utterance
                      }

        token_type_id_dict = {
            "utterance": 1,
            "context": 2,
            "history": 3,
            "knowledge": 4,
            "user_profile": 5,
            "bot_profile": 6
        }

        search_query = None
        if '@@@' in utterance and len(utterance.split('@@@')) == 2:
            utterance, search_query = utterance.split('@@@')

        all_utterances = [h.get('utterance') for h in history] + [utterance]
        debug_info['session'] = all_utterances

        ## step1: rewrite to get search query
        query = self.rewrite_query(utterance, history, debug_info)
        q_is_persona_question = is_persona_question(query) or is_persona_question(utterance)
        debug_info.update({'q_is_persona_question': q_is_persona_question})

        response, ds_session_id = call_chat_skills(query, history)
        if response is not None:
            if ds_session_id is not None:
                query_rewritten = f'{query}@@@{ds_session_id}@@@ds'
            else:
                query_rewritten = f'{query}@@@no_ds@@@no_ds'
            debug_info['query_rewritten'] = query_rewritten
            return response, debug_info

        if search_query is None:
            search_query = self.get_search_query(query, history, debug_info)
            # search_query = query

        ## step1.5: get the global NER results
        query_ner = None
        if self.ner_tool is not None:
            start_ner_time = time.time()
            query_ner = self.ner_tool.predict(query)
            debug_info['query_ner'] = query_ner
            debug_info['NER_time'] = time.time() - start_ner_time

        ## step1.8: need concat context
        if q_is_persona_question:
            need_concat_context = False
            need_search = False
        elif self.is_special_skill(query):
            need_concat_context = False
            need_search = True
        elif self.query_classifier is None:
            need_concat_context = True
            need_search = True
        else:  # 如果预设好了二分类器
            start = time.time()
            query_label = self.query_classifier.predict(query)
            debug_info['query_classify_label'] = query_label
            debug_info['query_classify_time'] = time.time() - start
            need_concat_context = query_label == QueryClassifier.CHAT_QUERY
            need_search = True  # query_label == QueryClassifier.BAIKE_QUERY
        debug_info['need_concat_context'] = need_concat_context
        debug_info['need_search'] = need_search

        ## step2: search knowledge
        debug_info['search_time'] = 0.0
        search_results, has_search_card = self.search(query, search_query, need_search, query_ner, history,
                                                      q_is_persona_question,
                                                      debug_info)

        if debug_info['search_results'] and debug_info['search_results'][0].get('q_score', 0.0) > 0.95:
            response = debug_info['search_results'][0].get('snippet')
            return response, debug_info

        ## step3: history, knowledge
        context_passages = []
        context_passages_type = []
        if q_is_persona_question or has_search_card:
            context = utterance
        elif not need_concat_context:
            context = query
        else:
            history_session = [h.get('utterance') for h in history[-10:]]
            if self.CONCAT_CONTEXT_TURNS > 0:
                context = SEP_TOKEN.join(history_session[-self.CONCAT_CONTEXT_TURNS:] + [utterance])
            else:
                context = query

            if self.CONCAT_HISTORY_INTO_PASSAGE and len(history) > self.CONCAT_CONTEXT_TURNS:
                if SEP_TOKEN == '[SEP]':
                    history_passage = SEP_TOKEN.join(history_session[:-self.CONCAT_CONTEXT_TURNS if self.CONCAT_CONTEXT_TURNS>0 else None])[-MAX_PASSAGE_LENGTH:]
                    context_passages.append(context + f'{SEP_TOKEN}history: ' + history_passage)
                    context_passages_type.append(token_type_id_dict['history'])
                else:
                    if self.CONCAT_CONTEXT_TURNS > 0:
                        history_passage = SEP_TOKEN.join(history_session[:-self.CONCAT_CONTEXT_TURNS])[
                                          -MAX_PASSAGE_LENGTH:]
                    else:
                        history_passage = SEP_TOKEN.join(history_session)[-MAX_PASSAGE_LENGTH:]
                    context_passages.append(('history', history_passage))

        if search_results:
            for chunk in search_results:
                if SEP_TOKEN == '[SEP]':
                    chunk = chunk.replace('</s>', SEP_TOKEN)
                    context_passages.append(context + f'{SEP_TOKEN}knowledge: ' + chunk)
                    context_passages_type.append(token_type_id_dict['knowledge'])
                else:
                    context_passages.append(('knowledge', chunk))

        ## step4: build profiles as long-term memory
        if q_is_persona_question:
            dynamic_user_profile, dynamic_bot_profile = self.get_dynamic_profile(query, history, debug_info)
            user_profile = f'{user_profile};{dynamic_user_profile}'.strip(';')
            bot_profile = f'{bot_profile};{dynamic_bot_profile}'.strip(';')

        if user_profile:
            # user_profile = user_profile.replace('我', '你')
            user_profile_chunks = split_chunks(user_profile)
            for chunk in user_profile_chunks:
                if SEP_TOKEN == '[SEP]':
                    context_passages.append(context + f'{SEP_TOKEN}user_profile: ' + chunk)
                    context_passages_type.append(token_type_id_dict['user_profile'])
                else:
                    context_passages.append(('user_profile', chunk))

        if bot_profile:
            bot_profile_chunks = split_chunks(bot_profile)
            for chunk in bot_profile_chunks:
                if SEP_TOKEN == '[SEP]':
                    context_passages.append(context + f'{SEP_TOKEN}bot_profile: ' + chunk)
                    context_passages_type.append(token_type_id_dict['bot_profile'])
                else:
                    context_passages.append(('bot_profile', chunk))

        if self.version > 11.0 and SEP_TOKEN != '[SEP]':
            context = context.strip().strip(SEP_TOKEN).split(SEP_TOKEN)
            if len(context) > 1:
                context = 'context: ' + SEP_TOKEN.join(context[:-1]) + f' {SEP_TOKEN} utterance: ' + context[-1]
            else:
                context = 'utterance: ' + context[0]


        if not context_passages:
            context_passages.append(context)
            if SEP_TOKEN == '[SEP]':
                context_passages_type.append(token_type_id_dict['context'])
        elif SEP_TOKEN != '[SEP]':
            if self.mrc_model is None:
                context_passages = [f'{context}{SEP_TOKEN} {item[0]}: {item[1]}' for item in context_passages]
            else:
                types = [item[0] for item in context_passages]
                new_passages = self.mrc_model.process_batch(query, [item[1] for item in context_passages])
                context_passages = [f'{context}{SEP_TOKEN} {type}: {passage}' for type, passage in
                                    zip(types, new_passages)]

        debug_info['context'] = context
        debug_info['context_passages'] = context_passages

        for i in range(len(context_passages)):
            context_passages[i] = re.sub('[ \t]+', SPACE_TOKEN, context_passages[i])

        ## step5: tokenize
        input_ids = self.tokenizer(context_passages, padding=True, truncation=True, max_length=300,
                                   return_tensors="pt").input_ids

        token_type_ids = None
        if use_type_id:
            sep_id = 102
            input_ids_length = input_ids.shape[1]
            # return list
            context_ids = self.tokenizer(context, truncation=True, max_length=300).input_ids
            context_ids_length = len(context_ids)
            # context type id
            context_end_index = 0
            assert context_ids[-1] == sep_id
            if sep_id in context_ids[:-1]:
                for i in range(len(context_ids) - 1):
                    if context_ids[i] == sep_id:
                        context_end_index = i
            context_type_id = [token_type_id_dict['context']] * len(context_ids[:context_end_index + 1]) + [token_type_id_dict['utterance']] * len(context_ids[context_end_index + 1:])

            token_type_ids = []
            for type in context_passages_type:
                token_type_ids.append(torch.tensor(context_type_id+[type]*(input_ids_length-context_ids_length)))

            token_type_ids = torch.stack(token_type_ids,dim=0)
            token_type_ids = token_type_ids*torch.gt(input_ids,0)
            token_type_ids = token_type_ids.unsqueeze(0).to(DEVICE)

        input_ids = input_ids.unsqueeze(0).to(DEVICE)  # batch_size= 1

        ## step6: generate
        bad_words_ids = None
        occured_ngrams = []
        forbidden_utterances = ['不知道', '什么工作', '你告诉我一下', '你可以说一下', '你呢']
        if random.random() < 1.2:
            forbidden_utterances.extend(
                ['？', '?', '吗', '你呢', '什么', '怎么', '怎样', '咋', '啥', '如何', '为什么', '哪', '几', '谁', '多少', '多大', '多高', '是不是',
                 '有没有', '是否', '多久', '可不可以', '能不能', '行不行', '干嘛'])
        forbidden_utterances_ids = self.tokenizer(forbidden_utterances, add_special_tokens=False).input_ids
        occured_ngrams.extend([tuple(t[1:] if SEP_TOKEN=="</s>" else t[:]) for t in forbidden_utterances_ids])

        if not debug_info.get('q_is_persona_question'):
            if text_is_question(query):
                no_repeat_ngram_size = self.NO_REPEAT_NGRAM_SIZE_FOR_Q
            else:
                no_repeat_ngram_size = self.NO_REPEAT_NGRAM_SIZE
            no_repeat_session = [h.get('utterance') for h in history] + [utterance]
            no_repeat_session = no_repeat_session[len(no_repeat_session) - self.NO_REPEAT_SESSION_SIZE:]
            no_repeat_session_tokens = []
            if len(no_repeat_session)>0:
                no_repeat_session_tokens = self.tokenizer(no_repeat_session, add_special_tokens=False).input_ids
            for utt_tokens in no_repeat_session_tokens:
                if SEP_TOKEN == "</s>":
                    utt_tokens = utt_tokens[1:]
                # generate ngram in utt_tokens
                occured_ngrams.extend(list(zip(*[utt_tokens[i:] for i in range(no_repeat_ngram_size)])))

        occured_ngrams = list(set(occured_ngrams))

        if occured_ngrams:
            bad_words_ids = [list(t) for t in occured_ngrams]

        start_generate = time.time()

        if self.allspark_gpu_speed_up:
            input_dict = {
                "input_ids": input_ids.to(torch.int64).cuda(),
                "attention_mask": (input_ids != 0).to(torch.int64).cuda(),
                "token_type_ids": torch.zeros(input_ids.shape).to(torch.int64).cuda() if token_type_ids is None else token_type_ids.to(torch.int64).cuda()
            }
            hypotheses = self.model.generate_allspark(input_dict,
                                                      max_length=generate_config['max_length'],
                                                      bad_words_ids=bad_words_ids)
        else:
            if use_type_id:
                hypotheses = self.model.generate(input_ids,
                                             token_type_ids=token_type_ids,
                                             bad_words_ids=bad_words_ids,
                                             eos_token_id=self.tokenizer.sep_token_id,
                                             decoder_start_token_id=self.tokenizer.cls_token_id,
                                             **generate_config)
            else:
                hypotheses = self.model.generate(input_ids,
                                                 bad_words_ids=bad_words_ids,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 decoder_start_token_id=self.tokenizer.cls_token_id,
                                                 **generate_config)

        if self.cuda:
            hypotheses = hypotheses.detach().cpu().tolist()

        results = []
        for h in hypotheses:
            if SEP_TOKEN == '</s>':  # T5
                decoded_hypo = self.tokenizer.decode(h, skip_special_tokens=True).replace(' ', '')
            else:
                decoded_hypo = self.tokenizer.decode(h, skip_special_tokens=True).replace(' ', '')
            results.append(decoded_hypo)

        generate_time = time.time() - start_generate
        debug_info['all_results'] = results
        debug_info['generate_time'] = generate_time

        ## step7: post process step
        if self.rerank_model is not None:
            try:
                candidate_responses = [{'snippet': t} for t in results]
                reranked_results = self.rerank_model.rerank(utterance, candidate_responses)
                results = [t['snippet'] for t in reranked_results]
                debug_info['all_results'] = results
                response = results[0]
            except:
                response = results[0]
        else:
            response = results[0]

        if len(search_results) and '[TOOUTPUT]' in search_results[0]:
            response = search_results[0].replace('[TOOUTPUT]', '')
        if has_search_card and search_results:
            response = search_results[0]

        grounding_evidence = find_grounding_evidence(response, debug_info['search_results'])
        debug_info['grouding_evidence'] = grounding_evidence
        return response, debug_info


from rouge import Rouge


def find_grounding_evidence(response, search_results):
    rouge = Rouge()
    for search_result in search_results:
        rouge_score = rouge.get_scores(' '.join(response), ' '.join(search_result['snippet']))[0]
        rouge_l = rouge_score['rouge-l']['f']
        rouge_p = rouge_score['rouge-l']['p']
        search_result['score'] = rouge_p
    search_results = sorted(search_results, key=lambda x: x['score'], reverse=True)
    return search_results[0] if len(search_results) > 0 and search_results[0]['score'] > 0.7 else None
