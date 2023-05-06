import sys
from xdpx.utils import io, parse_model_path
import os
import json
from flask import Flask, request, make_response
import traceback
from xdpx.options import Options, Argument, Arg
from xdpx.bootstrap import bootstrap
from xdpx.utils.chat_serve import Model, RewriteModel, KnowledgeIntervention, NER, QueryClassifier, QARankModel
import copy
from xdpx.utils import distributed_utils
import torch
import onnxruntime as ort
from waitress import serve
from datetime import datetime

## configs
generate_config = {
    'num_beams': 3,
    'num_return_sequences': 1,
    'temperature': 0.8,
    'do_sample': False,
    'early_stopping': True,
    'top_k': 50,
    'top_p': 0.8,
    'repetition_penalty': 1.2,
    'length_penalty': 1.2,
    'min_length': 10,
    'max_length': 80,
    'no_repeat_ngram_size': 3
}

CHAT_MODEL = None

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def process_handle_func():
    data = request.get_data().decode('utf-8')
    body = json.loads(data)
    res = process(body)
    response = make_response(res)
    response.status_code = 200
    return response


def process(input):
    """
       input: {
          'request_id': "235jil30d" 唯一请求输入标志
          'query':'天空为什么是蓝色',//required， 用户输入
          'audio':[], //optional, ndarray(float)
          'image':null, //optional, base64 string
          'history':[ //required, 按时间排序，最近10条
            {
               'utterance':'你好',
               'audio':[],
               'image':null,
               'role':'human'
            },
            {
                'utterance':'你好，很高兴认识你',
                'audio':[],
                'image':null,
                'role':'bot'
            }
           ],
          '_generate_config':{
            'num_beams':3
          }
          'user_profile':{
            'id':1344, //required
            'persona':['淘宝十年会员；喜欢时尚；爱好口红;'],
            'memory':{
            }
          },
          'bot_profile':{
            'id':1234, //required
            'persona':[ //required
              '我叫小莫,今年18岁，现在浙江大学，就读计算机专业大学二年级'
              '我平时在紫金港校区',
              '我喜欢听毛不易的歌',
              '我最喜欢的食物是火锅，可以和朋友一起吃',
              '我会开车,我有一辆特斯拉',
              '我喜欢购物，逛商场
              ],
            'memory':{
            }
          }
        }
        result:
        {
          'status':200, // status code, 200-success
          'response':'因为大气对太阳光的散射作用,使我们看到的天空呈现蓝色',//机器人回应
          'image':null,
          'video':null,
          'dialog_state':{ //optioal,
            'dialog_act':'statement',
            'emotion':'none',
            ....
          },
          'user_profile':{
            'update':false, //是否有更新
          },
          'bot_profile':{
            'id':1234,
            'update':true, //是否有更新
            'persona':[
              '我叫小莫,今年18岁，现在浙江大学，就读计算机专业大学二年级'
              '我平时在紫金港校区',
              '我喜欢听毛不易的歌',
              '我最喜欢的食物是火锅，可以和朋友一起吃',
              '我会开车,我有一辆特斯拉',
              '我喜欢购物，逛商场'
             ],
            'memory':{
              'xxxxx'
            }
          }
        }
    """

    result = {
        'response': '',
        'image': None,
        'video': None,
        'debug_info': {
        },
        'user_profile': {
            'update': False
        },
        'bot_profile': {
            'update': False
        }
    }
    debug_info = result['debug_info']
    request_id = input.get('request_id', '')
    debug_info["request_id"] = request_id

    try:
        utterance = input.get('query', '').strip()
        history = input.get('history', [])
        history = [] if history is None else history
        history_utterances = [h.get('utterance') for h in history]
        if not utterance:
            debug_info['exception'] = 'query is empty.'
            return result

        user_profile = ';'.join(input.get('user_profile', {}).get('persona', []))
        bot_profile = ';'.join(input.get('bot_profile', {}).get('persona', []))

        search_engine = input.get('___search_engine', '')  #
        if search_engine:
            CHAT_MODEL.search_engine = search_engine

        generate_config2 = copy.deepcopy(generate_config)
        tmp_generate_config = input.get('___generate_config', {})  #
        if tmp_generate_config:
            generate_config2.update(tmp_generate_config)

        response, _debug_info = CHAT_MODEL.chat(utterance, history, user_profile, bot_profile, generate_config2)
        debug_info.update(_debug_info)

        result['response'] = response
        print('----------')
        print(f'| {datetime.today()} {history_utterances} || {utterance}\t> \t{response}', flush=True)
        print(json.dumps(debug_info, indent=3, ensure_ascii=False))

    except Exception as e:
        debug_info['exception'] = traceback.format_exc()
        print(e)
    return result


def cli_main(argv=sys.argv):
    options = Options()
    options.register(
        Argument('knowledge_path',
                 default=None), 
        Argument('ner_url', default=None),  # 默认关闭，当填写正确NER http请求地址时打开
        Argument('query_classify_model_dir', default=None),  # 默认关闭，当填写正确oss分类模型地址时打开
        Argument('rewrite_is_onnx', default=False),
        Argument('rewrite_model_dir', required=True),
        Argument('qa_rerank_model_dir', default=None),

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
        Argument('search_cache_json_path', default='search_cache.json')
    )
    bootstrap(options, main, __file__, argv)


def main(cli_args: Arg):
    print(f'| PyTorch version: {torch.__version__}')
    distributed_utils.show_dist_info()
    print(f'| onnxruntime version: {ort.__version__} ')
    print(f'| onnxruntime device: {ort.get_device()} ')
    print(f'| onnxruntime available providers: {ort.get_available_providers()} ')

    try:
        rewrite_model = RewriteModel(cli_args.rewrite_model_dir,
                                     is_onnx=cli_args.rewrite_is_onnx,
                                     provider=cli_args.provider if cli_args.rewrite_is_onnx else None)
        print(f'rewrite_model created.')
        model = Model(cli_args.save_dir, checkpoint=cli_args.checkpoint if not cli_args.chat_is_onnx else None,
                      is_onnx=cli_args.chat_is_onnx,
                      pretrained_version=cli_args.pretrained_version,
                      quantized=cli_args.quantized,
                      provider=cli_args.provider if cli_args.chat_is_onnx else None,
                      allspark_gpu_speed_up=cli_args.allspark_gpu_speed_up,
                      allspark_gen_cfg=generate_config
                      )
        print(f'model created.')
        model.rewrite_model = rewrite_model
        model.knowledge_model = KnowledgeIntervention(cli_args.knowledge_path) if cli_args.knowledge_path else None
        print(f'knowledge_model created.')
        model.ner_tool = NER(cli_args.ner_url) if cli_args.ner_url else None
        print(f'ner_tool created.')
        model.query_classifier = QueryClassifier(
            cli_args.query_classify_model_dir) if cli_args.query_classify_model_dir else None
        print(f'query_classifier created.')
        model.rerank_model = QARankModel(cli_args.qa_rerank_model_dir) if cli_args.qa_rerank_model_dir else None
        print(f'rerank_model created.')
        model.NO_REPEAT_NGRAM_SIZE = 4

        global CHAT_MODEL
        CHAT_MODEL = model

        print(f'| load model success')

        example = {
            'query': '天空为什么是蓝色',
            'history': [],
            'user_profile': {

            },
            'bot_profile': {
                'persona': [
                    '我叫小婵，今年21岁，是个温暖热情的小仙女，现在单身还没有男朋友；我现在上海复旦大学在读，我的专业是工商管理；我平时喜欢唱歌和跳舞， 从小学就开始学习中国舞；我大学参加了学校的歌唱比赛，得了第一名 ；我的爸爸是当地工厂的技术员， 我的妈妈是小学语文老师；我喜欢吃苹果和葡萄， 甜甜的；我喝咖啡会容易失眠，但是我又不敢喝奶茶，怕长胖'
                ]
            }
        }

        first_blood = process(example)
        print(f'| first_blood:{json.dumps(first_blood, indent=3, ensure_ascii=False)}')
    except Exception:
        print(traceback.format_exc())
    except RuntimeError:
        print(traceback.format_exc())

    print('start flask app')
    serve(app, host='0.0.0.0', port=8000)


if __name__ == '__main__':
    cli_main(sys.argv)
