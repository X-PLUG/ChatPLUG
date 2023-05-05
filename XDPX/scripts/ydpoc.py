import sys
import traceback
import torch

import os
from xdpx.options import Options, Argument, Arg
from xdpx.utils import io, parse_model_path
from xdpx.bootstrap import bootstrap
import json
from xdpx.utils.chat_serve import Model, RewriteModel
import numpy as np
from icecream import ic
from xdpx.utils import distributed_utils
import onnxruntime as ort
import time
import random
from xdpx.utils.timi_search import FewShotModel, search_timi, QARankModel

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('rewrite_is_onnx', default=False),
        Argument('rewrite_model_dir', required=True),

        Argument('chat_is_onnx', default=False),
        Argument('provider', required=True, default='cuda',
                 validate=lambda value: (value in ['cpu', 'cuda', 'tensorrt'], f'Unknown provider {value}'), ),
        Argument('save_dir', required=True),
        Argument('checkpoint', doc='Full path is needed. If not provided, use the last checkpoint in save_dirs',
                 post_process=parse_model_path, type=str, default='<last>',
                 validate=lambda val: io.exists(val)),
        Argument('pretrained_version', default='google/mt5-base'),
        Argument('quantized', default=False)
    )
    bootstrap(options, main, __file__, argv)


RERANK_MODEL = FewShotModel(
    'oss://xdp-expriment/gaoxing.gx/ydpoc/fewshot/training/v0.1/N64-K4_20000_40_6e-05_protonet_rdropout_True_kl1_proj768_cl0_infonce/')

QA_RERANK_MODEL = QARankModel(
    'oss://xdp-expriment/gaoxing.gx/ydpoc/dpr/training/v0.1/'
)
test_data = []
test_file = 'oss://xdp-expriment/gaoxing.gx/ydpoc/fewshot/dataset/for_test/tmp_065612_single_turn_test'
for l in io.open(test_file):
    ts = l.strip().split(',,,')
    if len(ts) == 7:
        test_data.append({
            'seller_nick': ts[0],
            'question': ts[1],
            'solution': ts[2],
            'package': ts[3],
            'scene_key': ts[4],
            'item_id': ts[5],
            'category': ts[6]
        })
ic(len(test_data))


def main(cli_args: Arg):
    print(f'| PyTorch version: {torch.__version__}')
    distributed_utils.show_dist_info()
    print(f'| onnxruntime version: {ort.__version__} ')
    print(f'| onnxruntime device: {ort.get_device()} ')
    print(f'| onnxruntime available providers: {ort.get_available_providers()} ')
    rewrite_model = RewriteModel(cli_args.rewrite_model_dir,
                                 is_onnx=cli_args.rewrite_is_onnx,
                                 provider=cli_args.provider if cli_args.rewrite_is_onnx else None)
    model = Model(cli_args.save_dir, checkpoint=cli_args.checkpoint if not cli_args.chat_is_onnx else None,
                  is_onnx=cli_args.chat_is_onnx,
                  pretrained_version=cli_args.pretrained_version,
                  quantized=cli_args.quantized,
                  provider=cli_args.provider if cli_args.chat_is_onnx else None
                  )
    model.rewrite_model = rewrite_model
    model.search_engine = 'timi'
    model.CONCAT_CONTEXT_TURNS = 1
    model.CONCAT_HISTORY = False

    ## configs
    generate_config = {
        'num_beams': 3,
        'num_return_sequences': 1,
        'temperature': 0.8,
        'do_sample': False,
        'early_stopping': True,
        'top_k': 50,
        'top_p': 0.8,
        'repetition_penalty': 1,
        'length_penalty': 1.2,
        'min_length': 15,
        'max_length': 150,
        'no_repeat_ngram_size': 4
    }

    search_config = {
        'item_id': '0',
        'package_names': ['美妆行业', '行业通用'],
        'seller_nick': '欧莱雅官方旗舰店',
        'size': 3,
        'qq_rerank_model': RERANK_MODEL,
        'qa_rerank_model':QA_RERANK_MODEL,
        'min_rerank_score': 0.4
    }

    verbose = False

    user_profile = ''
    bot_profile = '我叫欧小蜜；我是欧莱雅官方旗舰店的店员；我喜欢用欧莱雅的化妆品；我也推荐你用我们家的产品；'

    session = []
    rewritten_session = []
    while True:
        try:
            utterance = input('INPUT:  ')
            if utterance.lower() == '#test_find_knowledge':

                oulaiya_samples = [t for t in test_data if t['seller_nick'] == '欧莱雅官方旗舰店']
                random.shuffle(oulaiya_samples)
                test_oulaiya_samples = oulaiya_samples[:100]
                test_oulaiya_samples = sorted(test_oulaiya_samples, key=lambda x: x['question'])

                oulaiya_search_config = {
                    **search_config,
                    'item_id': '0',
                    'package_names': ['美妆行业', '行业通用'],
                    'seller_nick': '欧莱雅官方旗舰店',
                }
                diff = 0
                for sample in test_oulaiya_samples:
                    q = sample['question']
                    questions, solutions = search_timi(question=q, **oulaiya_search_config)
                    if questions:
                        first = questions[0]
                        if 'scene_key' in first:
                            newkey = first['scene_key']
                        else:
                            newkey = first['question_id']
                    else:
                        newkey = 'noScene'
                    sample['newkey'] = newkey
                    sample['knowledges'] = questions
                    oldkey = sample['scene_key']
                    if newkey != oldkey:
                        print(f'diff: {q}\t{oldkey}\t{newkey}')
                        diff += 1
                    else:
                        print(f'same: {q}\t{oldkey}\t{newkey}')

                ic(diff)
                with open(f'test_find_knowledge_oulaiya_diff.json.{time.time()}', 'w') as nf:
                    nf.write(json.dumps(test_oulaiya_samples, indent=3, ensure_ascii=False))

                continue
            elif utterance.lower() == '#test_chat_oulaiya':
                ori_search_engine = model.search_engine
                oulaiya_samples = [t for t in test_data if t['seller_nick'] == '欧莱雅官方旗舰店']
                random.shuffle(oulaiya_samples)
                test_oulaiya_samples = oulaiya_samples[:100]
                test_oulaiya_samples = sorted(test_oulaiya_samples, key=lambda x:x['question'])

                model.search_engine = 'timi'

                for sample in test_oulaiya_samples:
                    q = sample['question']
                    oulaiya_search_config = {
                        **search_config,
                        'item_id': sample['item_id'] if sample['item_id'] else '0',
                        'package_names': ['美妆行业', '行业通用'],
                        'seller_nick': '欧莱雅官方旗舰店'
                    }
                    response, debug_info = model.chat([q], [q], '', '',
                                                      generate_config,
                                                      oulaiya_search_config)

                    sample['new_response'] = response
                    sample['debug_info'] = debug_info
                    print(f'{q}\t>\t{response}\told:{sample["solution"]}\t{sample["item_id"]}')

                with open(f'test_chat_oulaiya.json.{time.time()}', 'w') as nf:
                    nf.write(json.dumps(test_oulaiya_samples, indent=3, ensure_ascii=False))
                model.search_engine = ori_search_engine
                continue
            if utterance.lower() == '#new':
                session = []
                rewritten_session = []
                continue
            if utterance.lower() == '#show':
                ic(verbose)
                ic(generate_config)
                ic(search_config)
                ic(model.version)
                ic(model.search_engine)
                ic(model.CONCAT_CONTEXT_TURNS)
                ic(model.NO_REPEAT_SESSION_SIZE)
                ic(model.NO_REPEAT_NGRAM_SIZE)
                ic(user_profile)
                ic(bot_profile)
                continue
            elif utterance.lower() == '#exit':
                break
            elif utterance.lower().startswith('#concat_context_turns'):
                model.CONCAT_CONTEXT_TURNS = int(utterance.split('=')[1].strip())
                continue
            elif utterance.lower().startswith('#no_repeat_session_size'):
                model.NO_REPEAT_SESSION_SIZE = int(utterance.split('=')[1].strip())
                continue
            elif utterance.lower().startswith('#no_repeat_ngram_size'):
                model.NO_REPEAT_NGRAM_SIZE = int(utterance.split('=')[1].strip())
                continue
            elif utterance.lower().startswith('#version'):
                model.version = int(utterance.split('=')[1].strip())
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
                if k in ('min_length', 'max_length', 'top_k', 'num_beams', 'num_return_sequences'):
                    generate_config[k] = int(v)
                elif k in ('top_p', 'temperature', 'repetition_penalty', 'length_penalty'):
                    generate_config[k] = float(v)

                elif k in ('early_stopping', 'do_sample'):
                    generate_config[k] = '1' == v
                elif k in ('item_id', 'seller_nick'):
                    search_config[k] = v
                elif k in ('package_names'):
                    search_config[k] = v.split(',')
                elif k in ('min_rerank_score'):
                    search_config[k] = float(v)
                elif k in ('size'):
                    search_config[k] = int(v)
                continue
            else:
                session.append(utterance)
                rewritten_session.append(utterance)

            response, debug_info = model.chat(session, rewritten_session, user_profile, bot_profile, generate_config,
                                              search_config)

            if verbose:
                print('>>> START DEBUG INFO <<<<')
                print(json.dumps(debug_info, indent=3, ensure_ascii=False))
                print('>>> END DEBUG INFO <<<<')
            print('Response: {}'.format(response))
            session.append(response)
            rewritten_session[-1] = debug_info.get('query_rewritten', rewritten_session[-1])
            rewritten_session.append(response)
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
