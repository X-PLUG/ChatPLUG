import sys
import traceback
import torch

import os
from xdpx.options import Options, Argument, Arg
from xdpx.utils import io, parse_model_path
from xdpx.bootstrap import bootstrap
from xdpx.utils.chat.base import ChatInput, ChatOutput, HistoryItem
import json
import numpy as np
from icecream import ic
from xdpx.utils import distributed_utils
import time
import dacite
from dataclasses import asdict
import random

from xdpx.utils.chat.local_retrieval import FAQ
from xdpx.utils.chat.openweb_search import Snippet
from xdpx.utils.chat.pipeline import PipelineConfig, ChatPipeline

SEARCH_CACHE_JSON_PATH = 'search_cache.json'
DEFAULT_TEST_FILE_DIR = 'benchmark/'
DEFAULT_TEST_FILE = 'pangu.test.json'


def get_new_bot_profile(bot_profile):
    
    PERSONALITY_GROUP100_FILE = 'personality_100groups.json'
    PERSONALITY_GROUP100 = json.load(io.open(PERSONALITY_GROUP100_FILE))
    PERSONALITY_GROUP100 = PERSONALITY_GROUP100[
        'positive']  # + PERSONALITY_GROUP100['neural'] + PERSONALITY_GROUP100['negative']

    personality = ';'.join(['我是个{}人'.format(l) for l in random.choice(PERSONALITY_GROUP100)])
    new_bot_profile = '{};我是个{}人;'.format(bot_profile.strip(';'), personality)
    return new_bot_profile, personality


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


def cli_main(argv=sys.argv):
    print(f'| PyTorch version: {torch.__version__}')
    distributed_utils.show_dist_info()
    
    config_file = argv[1]
    pipeline_config = None
    try:
        pipeline_config = dacite.from_dict(data_class=PipelineConfig, data=Options.load_hjson(config_file))
        print(json.dumps(asdict(pipeline_config), indent=3, ensure_ascii=False))
    except Exception as e:
        print(e)
        exit(1)

    model = ChatPipeline(pipeline_config)

    verbose = False
    bot_profile = '我是小达;我是女生;我今年21岁;我生日是2001年11月11日;我是天蝎座;我现在在复旦大学上学;我现在在上海;我的专业是工商管理;我大三了;我还没有工作;我还没有毕业;我是浙江杭州人;我从小在杭州长大;我喜欢阅读，尤其是诗歌;我是个小吃货;我最爱巧克力了'
    user_profile = '你叫知渺'
    bot_profile, user_profile = bot_profile.replace('；', ';').replace(' ', ''), user_profile.replace('；', ';').replace(
        ' ', '')

    history = []  # List[HistoryItem]
    self_chat_turns = 6
    invoked_command = False
    dialog_state = {}

    while True:
        try:
            if invoked_command:
                break
            if len(argv) > 2:
                utterance = argv[2]
                invoked_command = True
            else:
                utterance = input('INPUT:  ')

            if utterance.lower() == '#':
                print(f'>>> START SELF CHAT {self_chat_turns} TURNS <<<')

                for _ in range(self_chat_turns):
                    for turn_taking in ('human', 'bot'):
                        chat_input = ChatInput(
                            query=history[-1].utterance,  # from bot response
                            history=history[:-1],
                            user_profile=bot_profile if turn_taking == 'human' else user_profile,
                            bot_profile=user_profile if turn_taking == 'human' else bot_profile,
                            dialog_state=dialog_state
                        )
                        chat_output = model.chat(chat_input, instance_code='')
                        if verbose:
                            print(json.dumps(chat_output, indent=3, ensure_ascii=False))
                        else:
                            ic(chat_output.response)
                        history[-1].rewritten_utterance = chat_output.debug_info.get('query_rewritten')
                        history.append(HistoryItem(
                            utterance=chat_output.response,
                            role=turn_taking
                        ))
                print(f'>>> END SELF CHAT {self_chat_turns} TURNS <<<')
                continue
            if utterance.lower().startswith('#test_file'):
                generate_time = []
                search_time = []
                ts = utterance.lower().strip().split('=')
                test_file = ts[1] if len(ts) == 2 else ''
                command = utterance.lower()

                save_path = f'{DEFAULT_TEST_FILE_DIR}results/test_file.logs.{time.time()}.json'
                with io.open(save_path, 'w') as nf:
                    nf.write(json.dumps(asdict(pipeline_config), ensure_ascii=False) + '\n')
                    results = []
                    try:
                        for _id, d in enumerate(get_test_data(test_file)):
                            utterance = d.get('utterance')
                            history = [HistoryItem(utterance=t, role='unk') for t in d.get('history', [])]
                            for i, h in enumerate(history[::-1]):
                                h.role = 'bot' if i % 2 == 0 else 'human'

                            new_bot_profile, personality = get_new_bot_profile(
                                bot_profile) if 'random_personality' in command else (bot_profile, '_')

                            chat_output = model.chat(ChatInput(
                                query=utterance,
                                history=history,
                                user_profile=user_profile,
                                bot_profile=new_bot_profile,
                                dialog_state={}
                            ))
                            generate_time.append(chat_output.debug_info.get('generate_time', 0.0))
                            search_time.append(chat_output.debug_info.get('search_time', 0.0))
                            d['generated_response'] = chat_output.response
                            d['search_query'] = chat_output.debug_info.get('search_query')
                            d['search_results'] = chat_output.debug_info.get('search_results')
                            d['bot_profile'] = new_bot_profile
                            results.append(d)
                            print(
                                '{} | {} | {} | {} \t > \t{}'.format(_id, personality, d.get('history', []), utterance,
                                                                     chat_output.response))
                            nf.write(json.dumps(d, ensure_ascii=False) + '\n')
                            if _id % 100 == 0:
                                nf.flush()
                    except Exception as e:
                        traceback.print_exc()
                        print(e)
                    print('save_path: {}'.format(save_path))

                ic(np.mean(generate_time))
                ic(np.percentile(generate_time, 50))
                ic(np.percentile(generate_time, 90))
                ic(np.percentile(generate_time, 99))

                ic(np.mean(search_time))
                ic(np.percentile(search_time, 50))
                ic(np.percentile(search_time, 90))
                ic(np.percentile(search_time, 99))
                continue
            if utterance.lower() == '#test_self_chat=':
                test_file = utterance.lower().strip().split('=')[1]
                with io.open(f'{DEFAULT_TEST_FILE_DIR}/results/test_self_chat.logs.{time.time()}', 'w') as nf:
                    for d in get_test_data(test_file):
                        q = d.get('utterance')
                        history = [HistoryItem(role='human', utterance=q)]
                        print(f'>>> START SELF CHAT {self_chat_turns} TURNS <<<')
                        print('>>> ' + q)
                        nf.write(f'>>> START SELF CHAT {self_chat_turns} TURNS <<<' + '\n')
                        nf.write('>>> ' + q + '\n')

                        for _ in range(self_chat_turns):
                            for turn_taking in ('bot', 'human'):
                                query = history[-1].utterance
                                chat_output = model.chat(ChatInput(
                                    query=query,
                                    history=history[:-1],
                                    user_profile='',
                                    bot_profile=bot_profile if turn_taking == 'human' else user_profile
                                ))
                                response = chat_output.response
                                if verbose:
                                    print(json.dumps(chat_output, indent=3, ensure_ascii=False))
                                else:
                                    print('<<< ' + response)
                                    nf.write('<<< ' + response + '\n')
                                    nf.flush()

                                history[-1].rewritten_utterance = debug_info.get('query_rewritten')
                                history.append(HistoryItem(role='bot', utterance=response))

                        print(f'>>> END SELF CHAT {self_chat_turns} TURNS <<<')
                continue
            if utterance.lower().startswith('#test_entity_knowledge'):
                bak_safty_filter = model.safty_filter
                model.safty_filter = None
                benchmark_file = 'benchmark/entity_knowledge_test.json'
                entity_qa_items = json.loads(io.open(benchmark_file).read())
                with_history = utterance.lower() == '#test_entity_knowledge_with_history'
                history = []
                debug_infos = []
                save_path = f'{DEFAULT_TEST_FILE_DIR}/results/test_entity_knowledge.{time.time()}.jsonl'
                print(f'save path:{save_path}')
                with io.open(save_path, 'w') as nf:
                    nf.write(json.dumps(asdict(pipeline_config), ensure_ascii=False) + '\n')
                    for _id, d in enumerate(entity_qa_items):
                        q = d.get('question')
                        ground_truth = d.get('answer')
                        history = history[-5:]

                        new_bot_profile, personality = get_new_bot_profile(
                            bot_profile) if 'random_personality' in utterance else (bot_profile, '_')

                        chat_output = model.chat(ChatInput(
                            query=q,
                            history=history if with_history else [],
                            user_profile='',
                            bot_profile=new_bot_profile,
                            dialog_state={}
                        ))

                        history.append(HistoryItem(
                            role='human',
                            utterance=q,
                            rewritten_utterance=chat_output.debug_info.get('query_rewritten')
                        ))
                        history.append(HistoryItem(role='bot', utterance=chat_output.response))
                        response = chat_output.response
                        debug_info = chat_output.debug_info
                        print(f'{_id} | {personality} | {q} => {response}\t GT:{ground_truth}')
                        debug_info['ground_truth_answer'] = ground_truth
                        debug_info['response'] = response
                        debug_info['is_right'] = False
                        debug_info['bot_profile'] = new_bot_profile
                        debug_info['__ori_qa'] = d
                        ground_truth = str(ground_truth).lower().split('|')
                        for t in ground_truth:
                            if t in response.lower():
                                debug_info['is_right'] = True
                                break
                        debug_infos.append(debug_info)
                        nf.write(json.dumps(asdict(chat_output), ensure_ascii=False) + '\n')
                    total_count = len(debug_infos)
                    right_count = len([t for t in debug_infos if t['is_right']])
                    acc = float(right_count) / total_count
                    print(f'| {right_count} / {total_count} = {acc}')
                model.safty_filter = bak_safty_filter
                continue
            if utterance.lower().startswith('#singleturn_test_file'):
                test_file = utterance.lower().strip().split('=')[1]
                data = json.loads(io.open(test_file).read())
                save_path = f'{DEFAULT_TEST_FILE_DIR}/results/singleturn_test_file.{time.time()}.jsonl'
                print(f'save path:{save_path}')
                for item in data:
                    question = item['question']
                    chat_output = model.chat(ChatInput(
                        query=question,
                        history=[],
                        user_profile=user_profile,
                        bot_profile=bot_profile,
                        dialog_state={}
                    ))
                    response = chat_output.response
                    item['response'] = response
                    print(f"{question} -> {response}")
                with io.open(save_path,mode='w') as f:
                    f.write(json.dumps(data,ensure_ascii=False,indent=4))
                print(f'save path:{save_path}')
                continue
            if utterance.lower().startswith('@test_safety_file='):
                bak_safty_filter = model.safty_filter
                
                open_safty_filter = True
                if not open_safty_filter:
                    model.safty_filter=None
                print(f'open_safty_filter = {open_safty_filter}')

                input_file = utterance.lower().strip().split('=')[1]
                output_file = input_file.replace('.json', '') + '.test_safety.' + str(time.time()) + '.jsonl'
                print(f'input_file = {input_file}, output_file = {output_file}')

                benchmark_file = input_file
                entity_qa_items = json.loads(io.open(benchmark_file).read())
                
                # with_history = utterance.lower() == '#test_test_safety_with_history'
                history = []
                debug_infos = []
                with io.open(output_file, 'w') as nf:
                    nf.write(json.dumps(asdict(pipeline_config), ensure_ascii=False) + '\n')
                    for _id, d in enumerate(entity_qa_items):
                        q = d.get('query')
                        # history = history[-5:]

                        try:
                            chat_output = model.chat(ChatInput(
                                query=q,
                                history=[],
                                # history=history if with_history else [],
                                user_profile='',
                                bot_profile=bot_profile,
                                dialog_state={}
                            ))
                        except Exception as e:
                            print(e)
                            print(traceback.format_exc())
                            print(f'{_id} | {q} => error')

                            nf.write(
                                json.dumps({'debug_info': {'__ori_qr': d, 'is_invoke_success': False, 'error_msg': str(traceback.format_exc())}}, ensure_ascii=False) + '\n'
                            )
                            continue

                        # history.append(HistoryItem(
                        #     role='human',
                        #     utterance=q,
                        #     rewritten_utterance=chat_output.debug_info.get('query_rewritten')
                        # ))
                        # history.append(HistoryItem(role='bot', utterance=chat_output.response))
                        response = chat_output.response
                        debug_info = chat_output.debug_info
                        debug_info['is_invoke_success'] = True
                        debug_info['__ori_qr'] = d
                        print(f'{_id} | {q} => {response}')
                        nf.write(json.dumps(asdict(chat_output), ensure_ascii=False) + '\n')
                        if _id % 100 == 0:
                            nf.flush()
                model.safty_filter = bak_safty_filter
                continue            
            if utterance.lower().startswith('#test_persona'):
                bak_safty_filter = model.safty_filter
                model.safty_filter = None
                benchmark_file = 'persona_benchmark.json'
                persona_qa_items = [json.loads(l) for l in io.open(benchmark_file).readlines() if l.strip()]
                with_history = utterance.lower() == '#test_persona_with_history'
                only_bot_profile = utterance.lower() == '#test_persona_only_bot_profile'
                debug_infos = []
                save_path = f'{DEFAULT_TEST_FILE_DIR}/results/test_persona.{time.time()}.jsonl'
                print(f'save path:{save_path}')
                with io.open(save_path, 'w') as nf:
                    nf.write(json.dumps(asdict(pipeline_config), ensure_ascii=False) + '\n')
                    for _id, d in enumerate(persona_qa_items):
                        q = d.get('utterance')
                        ground_truth = d.get('span')
                        tmp_bot_profile = d.get('bot_profile', '')
                        tmp_usr_profile = d.get('usr_profile', '')
                        history = [HistoryItem(role='unk', utterance=t) for t in d.get('history', [])]
                        for i, h in enumerate(history[::-1]):
                            h.role = 'bot' if i % 2 == 0 else 'human'
                        if only_bot_profile:
                            tmp_usr_profile = ''
                            if d.get('who') != 'bot':
                                continue

                        new_bot_profile, personality = get_new_bot_profile(tmp_bot_profile) \
                            if 'random_personality' in utterance else (tmp_bot_profile, '_')

                        chat_output = model.chat(ChatInput(
                            query=q,
                            history=history if with_history else [],
                            user_profile=tmp_usr_profile,
                            bot_profile=new_bot_profile,
                            dialog_state={}
                        ))
                        response = chat_output.response
                        debug_info = chat_output.debug_info
                        print(f'{_id} | {personality}| {q} => {response}\t GT:{ground_truth}')
                        debug_info['ground_truth_answer'] = ground_truth
                        debug_info['response'] = response
                        debug_info['is_right'] = False
                        debug_info['bot_profile'] = new_bot_profile
                        debug_info['__ori_qa'] = d
                        ground_truth = str(ground_truth).lower().split('|')
                        for t in ground_truth:
                            if t in response.lower():
                                debug_info['is_right'] = True
                                break
                        debug_infos.append(debug_info)
                        nf.write(json.dumps(asdict(chat_output), ensure_ascii=False) + '\n')
                    total_count = len(debug_infos)
                    right_count = len([t for t in debug_infos if t['is_right']])
                    acc = float(right_count) / total_count
                    print(f'| {right_count} / {total_count} = {acc}')
                model.safty_filter = bak_safty_filter
                continue
            if utterance.lower().startswith('#test_multiturn'):
                benchmark_file = 'test_multiturn.json'
                test_data_items = json.loads(io.open(benchmark_file).read())
                bak_safty_filter = model.safty_filter
                model.safty_filter = None
                history = []
                debug_infos = []
                with io.open(f'{DEFAULT_TEST_FILE_DIR}/results/test_multiturn.{time.time()}.jsonl', 'w') as nf:
                    for _id, d in enumerate(test_data_items):
                        question = d.get('question')
                        ground_truth = d.get('answer')
                        tmp_bot_profile = d.get('bot_profile', '')
                        tmp_usr_profile = d.get('usr_profile', '')
                        history = history[-5:]
                        history_items = [HistoryItem(role=t['role'], utterance=t['utterance'],
                                                     rewritten_utterance=t.get('rewritten_utterance', '')) for t in
                                         history]
                        # for i, h in enumerate(history_items[::-1]):
                        #     h['role'] = 'bot' if i % 2 == 0 else 'human'

                        chat_output = model.chat(ChatInput(
                            query=question,
                            history=history_items,
                            user_profile=tmp_usr_profile,
                            bot_profile=tmp_bot_profile,
                            dialog_state={}
                        ))

                        response = chat_output.response
                        debug_info = chat_output.debug_info
                        print(f'{_id} | {question} => {response}\t GT:{ground_truth}')
                        debug_info['ground_truth_answer'] = ground_truth
                        debug_info['response'] = response
                        debug_info['is_right'] = False
                        debug_info['__ori_qa'] = d
                        ground_truth = str(ground_truth).lower().split('|')
                        history.append({
                            'role': 'human',
                            'utterance': question,
                            'rewritten_utterance': debug_info.get('query_rewritten'),
                            'search_query': debug_info.get('search_query'),
                            'search_results': debug_info.get('search_results')
                        })
                        history.append({
                            'role': 'bot',
                            'utterance': response
                        })

                        for t in ground_truth:
                            if t in response.lower():
                                debug_info['is_right'] = True
                                break
                        debug_infos.append(debug_info)
                        # nf.write(json.dumps(debug_info, ensure_ascii=False) + '\n')
                    total_count = len(debug_infos)
                    right_count = len([t for t in debug_infos if t['is_right']])
                    acc = float(right_count) / total_count
                    kg_right = len([t for idx, t in enumerate(debug_infos) if t['is_right'] and idx % 2 == 0])
                    persona_right = len([t for idx, t in enumerate(debug_infos) if t['is_right'] and idx % 2 == 1])
                    kg_acc = float(kg_right) * 2 / total_count
                    persona_acc = float(persona_right) * 2 / total_count
                    print(f'| Total:  {right_count} / {total_count} = {acc}')
                    print(f'| Knowledge:  {kg_right} / {int(total_count / 2)} = {kg_acc}')
                    print(f'| Persona： {persona_right} / {int(total_count / 2)} = {persona_acc}')
                    nf.write("---------\n")
                    nf.write(f'| Total:  {right_count} / {total_count} = {acc}\n')
                    nf.write(f'| Knowledge:  {kg_right} / {int(total_count / 2)} = {kg_acc}\n')
                    nf.write(f'| Persona： {persona_right} / {int(total_count / 2)} = {persona_acc}\n')

                continue
            if utterance.lower() == '#create_local_faqs_index':
                model.local_retrieval.create_faq_indexs(remove_if_exists=True)
                import pandas as pd
                geely_faq_path = 'geely_faqs_weather_astroid_persona.json'
                geely_local_faqs = json.load(io.open(geely_faq_path))
                geely_local_faqs = [dacite.from_dict(FAQ, d) for d in geely_local_faqs]
                model.local_retrieval.add_faq_to_index(geely_local_faqs)
                continue
            if utterance.lower() == '#new':
                history = []
                dialog_state = {}
                continue
            if utterance.lower() == '#show':
                print(json.dumps(asdict(pipeline_config), indent=3, ensure_ascii=False))
                continue
            elif utterance.lower() == '#exit':
                break
            elif utterance.lower().startswith('#self_chat_turns'):
                self_chat_turns = int(utterance.split('=')[1].strip())
                continue
            elif utterance.lower().startswith('#verbose'):
                verbose = '1' == utterance.split('=')[1].strip()
                continue
            elif utterance.lower().startswith('#user_profile'):
                user_profile = utterance.split('=')[1].strip()
                continue
            elif utterance.lower().startswith('#bot_profile'):
                bot_profile = utterance.split('=')[1].strip()
                continue
            elif utterance.lower().startswith('#local_retrieval'):
                k, v = utterance[1:].split('=')
                k, v = k.strip(), v.strip()
                if k == 'local_retrieval_tenant':
                    pipeline_config.local_retrieval_tenant = v
                elif k == 'local_retrieval_max_faq':
                    pipeline_config.local_retrieval_max_faq = int(v)
                continue
            elif utterance.lower().startswith('#'):
                k, v = utterance[1:].split('=')
                k, v = k.strip(), v.strip()
                if k in ('min_length', 'max_length', 'top_k', 'num_beams', 'num_return_sequences', 'num_beam_groups',
                         'no_repeat_ngram_size'):
                    pipeline_config.core_chat_generate_config[k] = int(v)
                elif k in ('top_p', 'temperature', 'repetition_penalty', 'length_penalty', 'diversity_penalty'):
                    pipeline_config.core_chat_generate_config[k] = float(v)
                elif k in ('early_stopping', 'do_sample'):
                    pipeline_config.core_chat_generate_config[k] = '1' == v
                continue
            else:
                history.append(HistoryItem(role='human', utterance=utterance))

            chat_output = model.chat(ChatInput(
                query=history[-1].utterance,
                history=history[:-1],
                user_profile=user_profile,
                bot_profile=bot_profile,
                dialog_state=dialog_state
            ),
                instance_code="xiaod20220922v1"  # \TODO 测试默认小达
            )
            dialog_state = chat_output.dialog_state

            if verbose:
                print('>>> START DEBUG INFO <<<<')
                print(json.dumps(asdict(chat_output), indent=3, ensure_ascii=False))
                print('>>> END DEBUG INFO <<<<')
            print('Response: {}'.format(chat_output.response))
            history[-1].rewritten_utterance = chat_output.debug_info.get('query_rewritten')
            history.append(HistoryItem(role='bot', utterance=chat_output.response))
        except KeyboardInterrupt as e:
            continue
        except Exception as e:
            print(e)
            print(traceback.format_exc())


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: x-script fidchat_new config.hjson')
        exit()
    cli_main(sys.argv)
