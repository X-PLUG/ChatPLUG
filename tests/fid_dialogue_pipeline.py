# -*- coding: utf-8 -*- 
"""
@Time : 2023-04-07 13:37 
@Author : zhimiao.chh 
@Desc : 
"""
from chatplug.fid_dialogue_pipeline import FidDialoguePipeline


if __name__ == '__main__':
    ""
    model_dir = "model_hub/0.3b"
    pipeline = FidDialoguePipeline(model_dir=model_dir, size='base')

    know_list = [
        "李白（701年—762年），字太白，号青莲居士，又号“谪仙人”。是唐代伟大的浪漫主义诗人，被后人誉为“诗仙”。与杜甫并称为“李杜”，为了与另两位诗人李商隐与杜牧即“小李杜”区别，杜甫与",
        "李白（701年2月28日－762），字太白，号青莲居士，唐朝诗人，有“诗仙”之称，最伟大的浪漫主义诗人。汉族，出生于西域碎叶城（今吉尔吉斯斯坦托克马克），5岁随父迁至剑南道之绵州（巴西郡）",
        "李白（701─762），字太白，号青莲居士，祖籍陇西成纪（今甘肃省天水县附近）。先世于隋末流徙中亚。李白即生于中亚的碎叶城（今吉尔吉斯斯坦境内）。五岁时随其父迁居绵州彰明县（今四川省江油"
    ]
    input = {
        "history": "你好[SEP]你好，我是娜娜，很高兴认识你！[SEP]李白是谁",
        "bot_profile": "我是娜娜;我是女生;我是单身",
        "knowledge": "[SEP]".join(know_list),
        "user_profile": "你是小明"
    }

    preprocess_params = {
        'max_encoder_length': 300,
        'context_turn': 3
    }
    forward_params = {
        'min_length': 10,
        'max_length': 512
    }
    kwargs = {
        'preprocess_params': preprocess_params,
        'forward_params': forward_params
    }
    result = pipeline(input,**kwargs)

    print(result)