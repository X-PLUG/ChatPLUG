# -*- coding: utf-8 -*-
import os
from chatplug.fid_dialogue_pipeline import FidDialoguePipeline

from fastapi import FastAPI, Request
import uvicorn, json, datetime


# base
# model_dir = "model_hub/240m"
# pipeline = FidDialoguePipeline(model_dir=model_dir, size='base')

# xl
model_dir = "model_hub/3.7b"
pipeline = FidDialoguePipeline(model_dir=model_dir, size='xl')

# xxl
# model_dir = "model_hub/13b"
# pipeline = FidDialoguePipeline(model_dir=model_dir, size='xxl')


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global pipeline

    preprocess_params = {
        'max_encoder_length': 380,
        'context_turn': 3
    }

    forward_params = {
        'min_length': 10,
        'max_length': 512,
        'num_beams': 3,
        'temperature': 0.8,
        'do_sample': False,
        'early_stopping': True,
        'top_k': 50,
        'top_p': 0.8,
        'repetition_penalty': 1.2,
        'length_penalty': 1.2,
        'no_repeat_ngram_size': 6
    }

    kwargs = {
        'preprocess_params': preprocess_params,
        'forward_params': forward_params
    }
    
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('query')                                             # str
    history = json_post_list.get('history')                                         # list of str
    knowledge = json_post_list.get('knowledge', '')                                 # list of str
    bot_profile = json_post_list.get('bot_profile', '我是娜娜;我是女生;我是单身')       # str

    forward_params['min_length'] = json_post_list.get('min_length', 10)
    forward_params['max_length'] = json_post_list.get('max_length', 512)
    forward_params['top_p'] = json_post_list.get('top_p', 0.8)
    forward_params['temperature'] = json_post_list.get('temperature', 0.8)

    history.append(query)
    inputs = {
        "history": "[SEP]".join(history),
        "bot_profile": bot_profile,
        "knowledge": "[SEP]".join(knowledge)
    }
    result = pipeline(inputs, **kwargs)
    response = result['text']
    history.append(response)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", query:"' + query + '", response:"' + repr(response) + '"'
    print(f"inputs = {inputs}")
    print(log)
    return answer


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
