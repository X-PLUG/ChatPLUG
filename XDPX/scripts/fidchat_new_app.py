import sys
import dataclasses
import dacite
import json
import torch
import traceback
from flask import Flask, request, make_response

from xdpx.options import Options
from xdpx.utils.chat.base import ChatInput, HistoryItem, ChatOutput
from xdpx.utils.chat.pipeline import ChatPipeline, PipelineConfig
from xdpx.utils import distributed_utils

import onnxruntime as ort
from waitress import serve
from datetime import datetime
from dataclasses import asdict

CHAT_MODEL = None
NO_DEBUG = False
INSTANCE_CODE = None

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def process_handle_func():
    data = request.get_data().decode('utf-8')
    body = json.loads(data)
    res = process(body)
    response = make_response(res)
    response.status_code = 200
    return response


@app.route('/refresh_rule_control', methods=['GET'])
def refresh_rule_control():
    global CHAT_MODEL
    if CHAT_MODEL is not None:
        CHAT_MODEL.refresh_rule_control()
    response = make_response({'success': True})
    response.status_code = 200
    return response


def process(input):
    chat_output = ChatOutput(debug_info={'exception': ''})
    # 请求信息
    request_id = input.get('request_id', '')
    chat_output.debug_info["request_id"] = request_id
    instance_code = input.get('instance_code', '')
    if not instance_code:
        instance_code = INSTANCE_CODE
    chat_output.debug_info["instance_code"] = instance_code
    try:
        utterance = input.get('query', '').strip()
        history = input.get('history', [])
        history = [] if history is None else history
        history_utterances = [h.get('utterance') for h in history]
        if not utterance:
            chat_output.debug_info['exception'] = 'utterance is empty.'
            return asdict(chat_output)

        user_profile = ';'.join(input.get('user_profile', {}).get('persona', []))
        bot_profile = ';'.join(input.get('bot_profile', {}).get('persona', []))

        chat_output = CHAT_MODEL.chat(ChatInput(
            query=utterance,
            history=[HistoryItem(
                utterance=h.get('utterance'),
                role=h.get('role'),
                rewritten_utterance=h.get('rewritten_utterance')
            ) for h in history],
            dialog_state=input.get('dialog_state', {}),
            user_profile=user_profile,
            bot_profile=bot_profile
        ),
            instance_code=instance_code
        )
        response = chat_output.response
        response = response.replace("\n", "")
        chat_output.debug_info["request_id"] = request_id
        chat_output.debug_info["instance_code"] = instance_code
        print('----------')
        print(f'| {datetime.today()} {history_utterances} || {utterance}\t> \t{response}', flush=True)
    except Exception as e:
        print(e)
        chat_output.debug_info['exception'] = traceback.format_exc()
    result = asdict(chat_output)
    print(json.dumps(result, indent=3, ensure_ascii=False), flush=True)
    if NO_DEBUG:
        result['debug_info'] = {}
    return result


def cli_main(argv=sys.argv):
    print(f'| PyTorch version: {torch.__version__}')
    distributed_utils.show_dist_info()
    print(f'| onnxruntime version: {ort.__version__} ')
    print(f'| onnxruntime device: {ort.get_device()} ')
    print(f'| onnxruntime available providers: {ort.get_available_providers()} ')

    try:
        config_file = argv[1]
        pipeline_config = None
        try:
            pipeline_config = dacite.from_dict(data_class=PipelineConfig, data=Options.load_hjson(config_file))
            print(json.dumps(dataclasses.asdict(pipeline_config), indent=3, ensure_ascii=False))
        except Exception as e:
            print(e)
            exit(1)

        model = ChatPipeline(pipeline_config)
        global CHAT_MODEL
        CHAT_MODEL = model
        global NO_DEBUG
        global INSTANCE_CODE
        if len(argv) > 3:
            INSTANCE_CODE = argv[3]
        if len(argv) > 2:  # 指定port端口的时候， 不返回详细debug_info, for geely/猫精等
            port = argv[2]
            NO_DEBUG = True
        else:
            NO_DEBUG = False
            port = 8000

        print(f'| load model success')

        example = {
            'query': '世界第一等是谁作的词',
            'history': [],
            'user_profile': {

            },
            'bot_profile': {
                'persona': [
                    '我叫轻舞飞扬'
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
    serve(app, host='0.0.0.0', port=port)


if __name__ == '__main__':
    cli_main(sys.argv)
