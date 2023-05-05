import sys
import traceback
import torch

import os
from xdpx.options import Options, Argument, Arg
from xdpx.utils import io, parse_model_path
from xdpx.bootstrap import bootstrap
import json
from xdpx.utils.chat_serve import Model, RewriteModel, KnowledgeIntervention, NER, QueryClassifier, QARankModel, \
    MrcModel
import numpy as np
from icecream import ic
from xdpx.utils import distributed_utils
import onnxruntime as ort
import time
import random

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('knowledge_path',
                 default=None), 
        Argument('ner_url', default=None),  # 默认关闭，当填写正确NER http请求地址时打开
        Argument('query_classify_model_dir', default=None),  # 默认关闭，当填写正确oss分类模型地址时打开
        Argument('qa_rerank_model_dir', default=None),
        Argument('rewrite_model_dir', required=True),

        Argument('rewrite_is_onnx', default=False),
        Argument('chat_is_onnx', default=False),
        Argument('provider', required=True, default='cuda',
                 validate=lambda value: (value in ['cpu', 'cuda', 'tensorrt'], f'Unknown provider {value}'), ),
        Argument('save_dir', required=True),
        Argument('checkpoint', doc='Full path is needed. If not provided, use the last checkpoint in save_dirs',
                 post_process=parse_model_path, type=str, default='<last>',
                 validate=lambda val: io.exists(val)),
        Argument('pretrained_version', default='google/mt5-base'),
        Argument('quantized', default=False),
        Argument('allspark_gpu_speed_up', default=False),
        Argument('command', default=None),
        Argument('use_mrc_model', default=False),
        Argument('search_cahce_json_path', default='search_cache.json')
    )
    bootstrap(options, main, __file__, argv)


DEFAULT_TEST_FILE_DIR = 'benchmark/'
DEFAULT_TEST_FILE = 'pangu.test.json'

personality_group100_file = 'personality_100groups.json'
personality_group100 = json.load(io.open(personality_group100_file))
PERSONALITY_GROUP100 = personality_group100['positive'] + personality_group100['neural'] + personality_group100[
    'negative']


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


def get_new_bot_profile(bot_profile):
    new_bot_profile = '{};我是个{}人;'.format(bot_profile, ';'.join(random.choice(
        PERSONALITY_GROUP100)))
    return new_bot_profile


def main(cli_args: Arg):
    print(f'| PyTorch version: {torch.__version__}')
    distributed_utils.show_dist_info()
    print(f'| onnxruntime version: {ort.__version__} ')
    print(f'| onnxruntime device: {ort.get_device()} ')
    print(f'| onnxruntime available providers: {ort.get_available_providers()} ')
    rewrite_model = RewriteModel(cli_args.rewrite_model_dir,
                                 is_onnx=cli_args.rewrite_is_onnx,
                                 provider=cli_args.provider if cli_args.rewrite_is_onnx else None)

    ## configs
    generate_config = {
        'num_beams': 3,
        'num_return_sequences': 1,
        # 'num_beam_groups': 1,
        # 'diversity_penalty': 1.2,
        'temperature': 0.8,
        'do_sample': False,
        'early_stopping': True,
        'top_k': 50,
        'top_p': 0.8,
        'repetition_penalty': 1.2,
        'length_penalty': 1.2,
        'min_length': 10,
        'max_length': 80,
        'no_repeat_ngram_size': 4
    }
    model = Model(cli_args.save_dir, checkpoint=cli_args.checkpoint if not cli_args.chat_is_onnx else None,
                  is_onnx=cli_args.chat_is_onnx,
                  pretrained_version=cli_args.pretrained_version,
                  quantized=cli_args.quantized,
                  provider=cli_args.provider if cli_args.chat_is_onnx else None,
                  allspark_gpu_speed_up=cli_args.allspark_gpu_speed_up,
                  allspark_gen_cfg=generate_config
                  )
    model.rewrite_model = rewrite_model
    model.knowledge_model = KnowledgeIntervention(cli_args.knowledge_path) if cli_args.knowledge_path else None
    model.ner_tool = NER(cli_args.ner_url) if cli_args.ner_url else None
    model.query_classifier = QueryClassifier(
        cli_args.query_classify_model_dir) if cli_args.query_classify_model_dir else None
    model.rerank_model = QARankModel(cli_args.qa_rerank_model_dir) if cli_args.qa_rerank_model_dir else None
    model.search_cache = json.load(io.open(cli_args.search_cahce_json_path)) if cli_args.search_cahce_json_path else {}
    model.mrc_model = MrcModel() if cli_args.use_mrc_model else None

    verbose = False

    bot_profile = '我是男的；我叫莫方; 我今年28岁;我是金牛座; 我在阿里巴巴工作; 我是个工程师；我是贵州人；我喜欢运动; 我比较擅长足球和篮球;我喜欢科比；我喜欢梅西；我从武汉大学计算机专业毕业; 我现在杭州阿里巴巴工作；我喜欢拍照，出版社还买过我的照片；我有一只小狗；我的小狗是泰迪犬；我养了一只泰迪;'
    user_profile = '我叫小婵;我21岁了; 我是天蝎座; 我是个小女生;我现在单身还没有男朋友；我现在上海复旦大学在读;我还没毕业；我还没工作；我的专业是工商管理；我平时喜欢唱歌和跳舞; 我从小学就开始学习中国舞；我曾经在学校的舞蹈比赛中获得过第一名；我的爸爸是当地工厂的技术员; 我的妈妈是小学语文老师；我喜欢吃水果；我喜欢和闺蜜一起逛商场购物;我的闺蜜是我同学;我喜欢吃巧克力；'

    history = []  # { utterance,rewritten_utterance,search_query, search_results}
    self_chat_turns = 6
    invoked_command = False
    response_rerank_model = None

    while True:
        try:
            if invoked_command:
                break
            if cli_args.command:
                utterance = cli_args.command
                model.search_engine = 'shenma_cache'
                invoked_command = True
            else:
                utterance = input('INPUT:  ')
            if utterance.lower() == '#':
                print(f'>>> START SELF CHAT {self_chat_turns} TURNS <<<')
                for _ in range(self_chat_turns):
                    query = history[-1].get('utterance')
                    utterance, debug_info = model.chat(query, history[:-1], bot_profile, user_profile, generate_config)
                    if verbose:
                        print(json.dumps(debug_info, indent=3, ensure_ascii=False))
                    else:
                        ic(utterance)
                    history[-1].update({
                        'rewritten_utterance': debug_info.get('query_rewritten'),
                        'search_query': debug_info.get('search_query'),
                        'search_results': debug_info.get('search_results')
                    })
                    history.append({
                        'role': 'human',
                        'utterance': utterance,
                    })

                    query = utterance
                    response, debug_info = model.chat(query, history[:-1], user_profile, bot_profile, generate_config)

                    if verbose:
                        print(json.dumps(debug_info, indent=3, ensure_ascii=False))
                    else:
                        ic(response)
                    history[-1].update({
                        'rewritten_utterance': debug_info.get('query_rewritten'),
                        'search_query': debug_info.get('search_query'),
                        'search_results': debug_info.get('search_results')
                    })
                    history.append({
                        'role': 'bot',
                        'utterance': response
                    })
                print(f'>>> END SELF CHAT {self_chat_turns} TURNS <<<')
                continue
            if utterance.lower().startswith('#test_file='):
                generate_time = []
                search_time = []
                test_file = utterance.lower().strip().split('=')[1]
                save_path = f'{DEFAULT_TEST_FILE_DIR}/results/test_file.logs.{time.time()}.json'
                with io.open(save_path, 'w') as nf:
                    config_json = {
                        'generate_config': generate_config,
                        'cli_args': cli_args.__dict__
                    }
                    nf.write(json.dumps(config_json, ensure_ascii=False) + '\n')

                    results = []
                    try:
                        for _id, d in enumerate(get_test_data(test_file)):
                            utterance = d.get('utterance')
                            history = [{'utterance': t} for t in d.get('history', [])]
                            for i, h in enumerate(history[::-1]):
                                h['role'] = 'bot' if i % 2 == 0 else 'human'

                            new_bot_profile = get_new_bot_profile(
                                bot_profile) if 'random_personality' in utterance else bot_profile
                            response, debug_info = model.chat(utterance, history, user_profile, new_bot_profile,
                                                              generate_config)
                            generate_time.append(debug_info['generate_time'])
                            search_time.append(debug_info.get('search_time', 0.0))
                            d['generated_response'] = response
                            d['search_query'] = debug_info.get('search_query')
                            d['search_results'] = debug_info.get('search_results')
                            d['bot_profile'] = new_bot_profile
                            results.append(d)
                            print('{} | {} | {} \t > \t{}'.format(_id, d.get('history', []), utterance, response))
                            nf.write(json.dumps(d, ensure_ascii=False) + '\n')
                            if _id % 100 == 0:
                                nf.flush()
                    except Exception as e:
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
                        history = [{
                            'role': 'human',
                            'utterance': q
                        }]
                        print(f'>>> START SELF CHAT {self_chat_turns} TURNS <<<')
                        print('>>> ' + q)
                        nf.write(f'>>> START SELF CHAT {self_chat_turns} TURNS <<<' + '\n')
                        nf.write('>>> ' + q + '\n')

                        for _ in range(self_chat_turns):
                            query = history[-1].get('utterance')
                            response, debug_info = model.chat(query, history[:-1], '', bot_profile,
                                                              generate_config)

                            if verbose:
                                print(json.dumps(debug_info, indent=3, ensure_ascii=False))
                            else:
                                print('<<< ' + response)
                                nf.write('<<< ' + response + '\n')

                            history[-1].update({
                                'rewritten_utterance': debug_info.get('query_rewritten'),
                                'search_query': debug_info.get('search_query'),
                                'search_results': debug_info.get('search_results')
                            })
                            history.append({
                                'role': 'bot',
                                'utterance': response
                            })

                            utterance, debug_info = model.chat(response, history[:-1], '', user_profile,
                                                               generate_config)

                            if verbose:
                                print(json.dumps(debug_info, indent=3, ensure_ascii=False))
                            else:
                                print('>>> ' + utterance)
                                nf.write('>>> ' + utterance + '\n')
                            nf.flush()
                            history[-1].update({
                                'rewritten_utterance': debug_info.get('query_rewritten'),
                                'search_query': debug_info.get('search_query'),
                                'search_results': debug_info.get('search_results')
                            })
                            history.append({
                                'role': 'human',
                                'utterance': utterance,
                            })

                        print(f'>>> END SELF CHAT {self_chat_turns} TURNS <<<')
                continue
            if utterance.lower().startswith('#test_entity_knowledge'):
                benchmark_file = 'entity_knowledge_test.json'
                entity_qa_items = json.loads(io.open(benchmark_file).read())
                with_history = utterance.lower() == '#test_entity_knowledge_with_history'

                history = []
                debug_infos = []
                with io.open(f'{DEFAULT_TEST_FILE_DIR}/results/test_entity_knowledge.{time.time()}.jsonl', 'w') as nf:
                    config_json = {
                        'generate_config': generate_config,
                        'cli_args': cli_args.__dict__
                    }
                    nf.write(json.dumps(config_json, ensure_ascii=False) + '\n')
                    for _id, d in enumerate(entity_qa_items):
                        q = d.get('question')
                        ground_truth = d.get('answer')
                        history = history[-5:]

                        new_bot_profile = get_new_bot_profile(
                            bot_profile) if 'random_personality' in utterance else bot_profile

                        if with_history:
                            response, debug_info = model.chat(q, history, '', new_bot_profile, generate_config)
                        else:
                            response, debug_info = model.chat(q, [], '', new_bot_profile, generate_config)

                        history.append({
                            'role': 'human',
                            'utterance': q,
                            'rewritten_utterance': debug_info.get('query_rewritten'),
                            'search_query': debug_info.get('search_query'),
                            'search_results': debug_info.get('search_results')
                        })
                        history.append({
                            'role': 'bot',
                            'utterance': response
                        })
                        print(f'{_id} | {q} => {response}\t GT:{ground_truth}')
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
                        nf.write(json.dumps(debug_info, ensure_ascii=False) + '\n')
                    total_count = len(debug_infos)
                    right_count = len([t for t in debug_infos if t['is_right']])
                    acc = float(right_count) / total_count
                    print(f'| {right_count} / {total_count} = {acc}')
                continue
            if utterance.lower().startswith('#test_persona'):
                benchmark_file = 'persona_benchmark.json'
                persona_qa_items = [json.loads(l) for l in io.open(benchmark_file).readlines() if l.strip()]
                with_history = utterance.lower() == '#test_persona_with_history'
                only_bot_profile = utterance.lower() == '#test_persona_only_bot_profile'
                debug_infos = []
                with io.open(f'{DEFAULT_TEST_FILE_DIR}/results/test_persona.{time.time()}.jsonl', 'w') as nf:
                    config_json = {
                        'generate_config': generate_config,
                        'cli_args': cli_args.__dict__
                    }
                    nf.write(json.dumps(config_json, ensure_ascii=False) + '\n')
                    for _id, d in enumerate(persona_qa_items):
                        q = d.get('utterance')
                        ground_truth = d.get('span')
                        tmp_bot_profile = d.get('bot_profile', '')
                        tmp_usr_profile = d.get('usr_profile', '')
                        history = [{'utterance': t} for t in d.get('history', [])]
                        for i, h in enumerate(history[::-1]):
                            h['role'] = 'bot' if i % 2 == 0 else 'human'

                        if only_bot_profile:
                            tmp_usr_profile = ''
                            if d.get('who') != 'bot':
                                continue

                        tmp_bot_profile = get_new_bot_profile(
                            bot_profile) if 'random_personality' in utterance else tmp_bot_profile

                        if with_history:
                            response, debug_info = model.chat(q, history, tmp_usr_profile, tmp_bot_profile,
                                                              generate_config)
                        else:
                            response, debug_info = model.chat(q, [], tmp_usr_profile, tmp_bot_profile, generate_config)

                        print(f'{_id} | {q} => {response}\t GT:{ground_truth}')
                        debug_info['ground_truth_answer'] = ground_truth
                        debug_info['response'] = response
                        debug_info['is_right'] = False
                        debug_info['bot_profile'] = tmp_bot_profile
                        debug_info['__ori_qa'] = d
                        ground_truth = str(ground_truth).lower().split('|')
                        for t in ground_truth:
                            if t in response.lower():
                                debug_info['is_right'] = True
                                break
                        debug_infos.append(debug_info)
                        nf.write(json.dumps(debug_info, ensure_ascii=False) + '\n')
                    total_count = len(debug_infos)
                    right_count = len([t for t in debug_infos if t['is_right']])
                    acc = float(right_count) / total_count
                    print(f'| {right_count} / {total_count} = {acc}')
                continue
            if utterance.lower().startswith('#test_multiturn'):
                benchmark_file = 'test_multiturn.json'
                test_data_items = json.loads(io.open(benchmark_file).read())
                history = []
                debug_infos = []
                with io.open(f'{DEFAULT_TEST_FILE_DIR}/results/test_multiturn.{time.time()}.jsonl', 'w') as nf:
                    for _id, d in enumerate(test_data_items):
                        question = d.get('question')
                        ground_truth = d.get('answer')
                        tmp_bot_profile = d.get('bot_profile', '')
                        tmp_usr_profile = d.get('usr_profile', '')
                        history = history[-5:]
                        response, debug_info = model.chat(question, history, tmp_usr_profile, tmp_bot_profile,
                                                          generate_config)

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
                        print(f'{_id} | {question} => {response}\t GT:{ground_truth}')
                        debug_info['ground_truth_answer'] = ground_truth
                        debug_info['response'] = response
                        debug_info['is_right'] = False
                        debug_info['__ori_qa'] = d
                        ground_truth = str(ground_truth).lower().split('|')
                        for t in ground_truth:
                            if t in response.lower():
                                debug_info['is_right'] = True
                                break
                        debug_infos.append(debug_info)
                        nf.write(json.dumps(debug_info, ensure_ascii=False) + '\n')
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

            if utterance.lower() == '#new':
                history = []
                continue
            if utterance.lower() == '#show':
                ic(verbose)
                ic(generate_config)
                ic(model.version)
                ic(model.search_engine)
                ic(model.CONCAT_CONTEXT_TURNS)
                ic(model.NO_REPEAT_SESSION_SIZE)
                ic(model.NO_REPEAT_NGRAM_SIZE)
                ic(model.NO_REPEAT_NGRAM_SIZE_FOR_Q)
                ic(model.CONCAT_HISTORY_INTO_PASSAGE)
                ic(user_profile)
                ic(bot_profile)
                continue
            elif utterance.lower() == '#exit':
                break
            elif utterance.lower().startswith('#concat_context_turns'):
                model.CONCAT_CONTEXT_TURNS = int(utterance.split('=')[1].strip())
                continue
            elif utterance.lower().startswith('#concat_history_into_passage'):
                model.CONCAT_HISTORY_INTO_PASSAGE = int(utterance.split('=')[1].strip())
                continue
            elif utterance.lower().startswith('#no_repeat_session_size'):
                model.NO_REPEAT_SESSION_SIZE = int(utterance.split('=')[1].strip())
                continue
            elif utterance.lower().startswith('#decode_no_repeat_ngram_size'):
                generate_config['no_repeat_ngram_size'] = int(utterance.split('=')[1].strip())
                continue
            elif utterance.lower().startswith('#no_repeat_ngram_size'):
                model.NO_REPEAT_NGRAM_SIZE = int(utterance.split('=')[1].strip())
                continue
            elif utterance.lower().startswith('#no_repeat_ngram_size_for_q'):
                model.NO_REPEAT_NGRAM_SIZE_FOR_Q = int(utterance.split('=')[1].strip())
                continue
            elif utterance.lower().startswith('#version'):
                model.version = int(utterance.split('=')[1].strip())
                continue
            elif utterance.lower().startswith('#self_chat_turns'):
                self_chat_turns = int(utterance.split('=')[1].strip())
                continue
            elif utterance.lower().startswith('#search_engine'):
                model.search_engine = utterance.split('=')[1].strip()
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
            elif utterance.lower().startswith('#'):
                k, v = utterance[1:].split('=')
                k, v = k.strip(), v.strip()
                if k in ('min_length', 'max_length', 'top_k', 'num_beams', 'num_return_sequences', 'num_beam_groups'):
                    generate_config[k] = int(v)
                elif k in ('top_p', 'temperature', 'repetition_penalty', 'length_penalty', 'diversity_penalty'):
                    generate_config[k] = float(v)

                elif k in ('early_stopping', 'do_sample'):
                    generate_config[k] = '1' == v
                continue
            else:
                history.append({
                    'role': 'human',
                    'utterance': utterance
                })

            response, debug_info = model.chat(history[-1].get('utterance'), history[:-1], user_profile, bot_profile,
                                              generate_config)

            if verbose:
                print('>>> START DEBUG INFO <<<<')
                print(json.dumps(debug_info, indent=3, ensure_ascii=False))
                print('>>> END DEBUG INFO <<<<')
            print('Response: {}'.format(response))
            history[-1].update({
                'rewritten_utterance': debug_info.get('query_rewritten'),
                'search_query': debug_info.get('search_query'),
                'search_results': debug_info.get('search_results')
            })
            history.append({
                'role': 'bot',
                'utterance': response
            })
        except KeyboardInterrupt as e:
            continue
        except Exception as e:
            print(e)
            print(traceback.format_exc())


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: x-script fid_t5chat config.hjson')
        exit()
    cli_main(sys.argv)
