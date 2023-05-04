# -*- coding: utf-8 -*-
import os
from chatplug.fid_dialogue_pipeline import FidDialoguePipeline


# base
# model_dir = "model_hub/240m"
# pipeline = FidDialoguePipeline(model_dir=model_dir, size='base')

# xl
model_dir = "model_hub/3.7b"
pipeline = FidDialoguePipeline(model_dir=model_dir, size='xl')

# xxl
# model_dir = "model_hub/13b"
# pipeline = FidDialoguePipeline(model_dir=model_dir, size='xxl')


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


def print_dialog(history):
    content = "欢迎使用ChatPLUG模型进行对话，clear 清空对话历史，stop 终止程序"
    for i, x in enumerate(history):
        if i % 2 == 0:
            content += f"\nUser: {x}"
        else:
            content += f"\nBot: {x}"
    return content
        

if __name__ == '__main__':
    history = []
    print("欢迎使用ChatPLUG模型进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\nUser: ")
        if query.strip() == "stop":
            break
        if query.strip() == 'clear':
            history = []
            os.system("clear")
            print("欢迎使用ChatPLUG模型进行对话，clear 清空对话历史，stop 终止程序")
            continue
        
        history.append(query)
        inputs = {
            "history": "[SEP]".join(history),
            "bot_profile": "我是娜娜;我是女生;我是单身",
            "knowledge": ""
        }

        result = pipeline(inputs, **kwargs)
        response = result['text']
        history.append(response)

        os.system("clear")
        print(print_dialog(history), flush=True)

