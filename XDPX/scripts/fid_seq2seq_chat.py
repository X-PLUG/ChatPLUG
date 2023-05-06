import os
import sys
import torch
import traceback
from xdpx.options import Arg, Options, Argument
from xdpx.tasks import tasks
from xdpx.loaders import loaders
from xdpx.utils import io, parse_model_path
from xdpx.models.chat import PalmChat
from transformers import BertTokenizer
from xdpx.utils import cache_file
from xdpx.models.chat import PlugChat
import json,time
from xdpx.utils.text_generation_metric.eval import compute_bleu_rouge
import json
import time
from xdpx.utils.chat.openweb_search import OpenWeb
import re

class Model:
    def __init__(self, save_dir, strict=True, notnull=False, checkpoint=None):
        self.cuda = torch.cuda.is_available()
        self.strict = strict
        self.notnull = notnull

        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

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

        if checkpoint:
            model_path = checkpoint
        else:
            model_path = parse_model_path('<last>', args)
        model.load(model_path)

        if self.cuda:
            model = model.cuda()
        self.task = task
        self.processor = task.processor
        self.loader = loaders[args.loader](args)
        self.model = model
        self.tokenizer = self.loader.tokenizer.tokenizer

        self.openweb = OpenWeb()

        self.token_type_id_dict = {
            "utterance": 1,
            "context": 2,
            "history": 3,
            "knowledge": 4,
            "user_profile": 5,
            "bot_profile": 6
        }
        self.SEP_TOKEN = '[SEP]'
        self.SEP_ID = 102
        self.SPACE_TOKEN = "▂"

    def chat(self, utterance, history, user_profile, bot_profile, generate_config, max_length = 512):

        user_profile = user_profile.replace('我', '你')

        try:
            search_result = self.openweb.search(utterance)
            knowledge = [s.snippet for s in search_result[0]]
        except Exception as e:
            print(str(e))
            knowledge = []
        context = self.SEP_TOKEN.join(history[-2:]+[utterance])
        if len(history)>2:
            history = self.SEP_TOKEN.join(history[:-2])[:max_length]
        else:
            history = ""

        # 构建passage
        context_passages = []
        context_passages_type = []
        if len(history)>0:
            context_passages.append(context + f'{self.SEP_TOKEN}history: ' + history)
            context_passages_type.append(self.token_type_id_dict['history'])
        if len(knowledge)>0:
            for know in knowledge:
                context_passages.append(context + f'{self.SEP_TOKEN}knowledge: ' + know)
                context_passages_type.append(self.token_type_id_dict['knowledge'])
        if len(user_profile)>0:
            context_passages.append(context + f'{self.SEP_TOKEN}user_profile: ' + user_profile)
            context_passages_type.append(self.token_type_id_dict['user_profile'])
        if len(bot_profile)>0:
            context_passages.append(context + f'{self.SEP_TOKEN}bot_profile: ' + bot_profile)
            context_passages_type.append(self.token_type_id_dict['bot_profile'])

        if len(context_passages)==0:
            context_passages.append(context)
            context_passages_type.append(self.token_type_id_dict['context'])

        for i in range(len(context_passages)):
            context_passages[i] = re.sub('[ \t]+', self.SPACE_TOKEN, context_passages[i])

        ## tokenize
        input_ids = self.tokenizer(context_passages, padding=True, truncation=True, max_length=max_length,
                                   return_tensors="pt").input_ids

        token_type_ids = None
        input_ids_length = input_ids.shape[1]
        # return list
        context_ids = self.tokenizer(context, truncation=True, max_length=max_length).input_ids
        context_ids_length = len(context_ids)
        # context type id
        context_end_index = 0
        assert context_ids[-1] == self.SEP_ID
        if self.SEP_ID in context_ids[:-1]:
            for i in range(len(context_ids) - 1):
                if context_ids[i] == self.SEP_ID:
                    context_end_index = i
        context_type_id = [self.token_type_id_dict['context']] * len(context_ids[:context_end_index + 1]) + [
            self.token_type_id_dict['utterance']] * len(context_ids[context_end_index + 1:])

        token_type_ids = []
        for type in context_passages_type:
            token_type_ids.append(torch.tensor(context_type_id + [type] * (input_ids_length - context_ids_length)))

        token_type_ids = torch.stack(token_type_ids, dim=0)
        token_type_ids = token_type_ids * torch.gt(input_ids, 0)

        input_ids = input_ids.unsqueeze(0).to(self.device)  # batch_size= 1
        token_type_ids = token_type_ids.unsqueeze(0).to(self.device)

        hypotheses = self.model.generate(input_ids, token_type_ids=token_type_ids, eos_token_id=self.tokenizer.sep_token_id,
                                         decoder_start_token_id=self.tokenizer.cls_token_id, **generate_config)
        if self.cuda:
            hypotheses = hypotheses.detach().cpu().tolist()


        response = self.tokenizer.decode(hypotheses[0], skip_special_tokens=True)
        response = response.replace(" ", "")
        # print('Response: {}'.format(response))
        return response, knowledge


DEFAULT_TEST_FILE_DIR = 'benchmark/'
DEFAULT_TEST_FILE = 'pangu.test.json'


def get_test_data(file):
    if not file:
        file = f'{DEFAULT_TEST_FILE_DIR}/{DEFAULT_TEST_FILE}'
    else:
        file = f'{DEFAULT_TEST_FILE_DIR}/{file}'
    with io.open(file) as f:
        if file.endswith('txt'):
            data = [{'utterance': t} for t in f.read().strip().split('\n')]
        elif file.endswith('json'):
            data = json.load(f)
    return data


def cli_main(argv):
    """for interactive testing of model behaviour"""
    assert len(argv) == 2
    save_dir = argv[1]
    checkpoint = None
    if io.isfile(save_dir):
        checkpoint = save_dir
        save_dir = os.path.dirname(checkpoint)
    model = Model(save_dir, strict=False, notnull=False, checkpoint=checkpoint)

    ## configs
    generate_config = {
        'num_beams': 3,
        'num_return_sequences': 1,
        # num_beam_groups: 1,
        # diversity_penalty: 1.2,
        'temperature': 0.8,
        'do_sample': False,
        'early_stopping': True,
        'top_k': 50,
        'top_p': 0.8,
        'repetition_penalty': 1200,
        'length_penalty': 1.2,
        'min_length': 10,
        'max_length': 80,
        'no_repeat_ngram_size': 4
    }

    session = []

    while True:
        try:
            utterance = input('INPUT:  ')
            if utterance.lower() == '#new':
                session = []
                continue
            if utterance.lower() == '#show':
                print(generate_config)
                continue
            elif utterance.lower() == '#exit':
                break
            elif utterance.lower().startswith('#test_file='):
                test_file = utterance.strip().split('=')[1]
                DEFAULT_TEST_FILE_DIR = 'test_file'
                out_file = f'{DEFAULT_TEST_FILE_DIR}/results/test_file.{time.time()}.jsonl'
                print(f'out file is {out_file}')
                with io.open(test_file) as f,io.open(out_file, 'w') as outf:
                    config_json = {
                        'generate_config': generate_config
                    }
                    outf.write(json.dumps({"save_dir":save_dir,"checkpoint":checkpoint}, ensure_ascii=False) + '\n')
                    outf.write(json.dumps(config_json, ensure_ascii=False) + '\n')
                    file_data = json.load(f)
                    for id, d in enumerate(file_data):
                        utterance = d['utterance']
                        history = d['history']
                        user_profile = d['utterance_profile']
                        user_profile = user_profile.replace('我','你')
                        bot_profile = d['response_profile']
                        response,search_result = model.chat(utterance,history,user_profile,bot_profile,generate_config)
                        d['response'] = response
                        d['search_result'] = search_result
                        print('{} | {} | {} \t > \t{}'.format(id, history, utterance,response))
                        for per_search_result in search_result:
                            print(per_search_result)
                        print("*"*100)
                        outf.write(json.dumps(d, ensure_ascii=False) + '\n')
                print(f'out file is {out_file}')
                continue
            elif utterance.lower() == '#test_entity_knowledge':
                test_file = 'entity_knowledge_test.json'
                entity_qa_items = json.loads(io.open(test_file).read())
                DEFAULT_TEST_FILE_DIR = 'entity_knowledge_test'
                out_file = f'{DEFAULT_TEST_FILE_DIR}/results/test_entity_knowledge.{time.time()}.jsonl'
                debug_infos = []
                with io.open(out_file, 'w') as outf:
                    config_json = {
                        'generate_config': generate_config
                    }
                    outf.write(json.dumps({"save_dir": save_dir, "checkpoint": checkpoint}, ensure_ascii=False) + '\n')
                    outf.write(json.dumps(config_json, ensure_ascii=False) + '\n')
                    for id, d in enumerate(entity_qa_items):
                        context = d.get('question')
                        ground_truth = d.get('answer')
                        response = model.chat([context], generate_config)
                        print(f'{id} | {context} => {response}\t GT:{ground_truth}')
                        debug_info = {}
                        debug_info['_id'] = id
                        debug_info['q'] = context
                        debug_info['ground_truth'] = ground_truth
                        debug_info['response'] = response
                        debug_info['is_right'] = False
                        ground_truth = str(ground_truth).lower().split('|')
                        for t in ground_truth:
                            if t in response.lower():
                                debug_info['is_right'] = True
                                break
                        debug_infos.append(debug_info)
                        outf.write(json.dumps(debug_info, ensure_ascii=False) + '\n')
                    total_count = len(debug_infos)
                    right_count = len([t for t in debug_infos if t['is_right']])
                    acc = float(right_count) / total_count
                    print(f'| {right_count} / {total_count} = {acc}')
                outf.write(json.dumps({"acc":acc}, ensure_ascii=False) + '\n')
                print(f'out file is {out_file}')
                continue
            elif utterance.lower().startswith('#'):
                k, v = utterance[1:].split('=')
                k, v = k.strip(), v.strip()
                if k in ('min_length', 'max_length', 'top_k', 'num_beams', 'num_return_sequences'):
                    generate_config[k] = int(v)
                elif k in ('top_p', 'temperature', 'repetition_penalty', 'length_penalty'):
                    generate_config[k] = float(v)

                elif k in ('early_stopping', 'do_sample'):
                    generate_config[k] = '1' == v
                continue

            else:
                session.append(utterance)

            context = session[-5:]
            # print(context)
            response = model.chat(context, generate_config)
            session.append(response)
            print(f"\t>\t{response}")

        except KeyboardInterrupt as e:
            continue
        except Exception as e:
            print(e)
            print(traceback.format_exc())

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: x-script fid_seq2seq_chat $save_dir')
        exit()
    cli_main(sys.argv)
