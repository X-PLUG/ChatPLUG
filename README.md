# ChatPLUG: Chinese Personalized Large Language Model

[![](assets/Demo-ModelScope-brightgreen.svg)](https://www.modelscope.cn/studios/damo/role_play_chat/summary)
[![](assets/Paper-Arxiv-orange.svg)](https://arxiv.org/abs/2304.07849)
![Hex.pm](https://img.shields.io/hexpm/l/plug)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://chatplug.readthedocs.io/zh_CN/latest/)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FX-PLUG%2FChatP&count_bg=%23E97EBA&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)



This is the repo for the [ChatPLUG](./ChatPLUG.md) project, which aims to build and share a Chinese open-domain dialogue system.

<hr>

| 爱用emoji的萌妹子小婉  |  富有智慧的得道高僧 | 会说古文的的三国NPC关羽 |
:-------------------------:|:-------------------------:|:-------------------------:
<img src="assets/xiaowan.gif"  width="80%" /> | <img src="assets/gaoseng.gif"  width="90%" /> | <img src="assets/guanyu.gif"   width="80%" /> 

## News
- [2023/05/10] 👏👏👏 Add training code.
- [2023/04/26] Try our [Role-Play-Chat Online Demo](https://modelscope.cn/studios/damo/role_play_chat/summary) in ModelScope Now!
- [2023/04/19] Add content including spotlights, results and limitations. Upload models to [modelscope](https://modelscope.cn/my/overview). 
- [2023/04/16] Initialize project.


## Online Demo
[Role-Play-Chat](https://www.modelscope.cn/studios/damo/role_play_chat/summary)


## Spotlights

<img src="assets/spotlights.jpg" alt="spotlights" width="80%" />

Compared with existed open-source models, we highlight three feaures of ChatPLUG as follows:

1. **Knowledge Augmentation**

  > It's flexible to integrate external knowledge during inference, and this is an optional input. You can utilize a `search engine` to acquire up-to-date information or use a local knowledge base to obtain domain knowledge. 

2. **Personalization**

  > It's easy to customize the style of conversations and characters by setting `bot profiles` or using `role-paly instructions`.

3. **Multi Skills** 

  > It exhibits its proficiency in open-domain dialogue through mulit-turn conversation, while also displaying impressive `multi-task abilities` on a wide range of NLP tasks. 


## How to run
We offer three methods to use or continue developing ChatPLUG as follows:

|  | Getting Started | Inference | Train | Deploy |
|---|---|---|---|---|
| ModelScope | Easy | :heavy_check_mark: Cli | :x:  Not Ready | :x:  Not Ready |
| HuggingFace | Medium | :heavy_check_mark:Cli  (Coming soon) | :x: Not Ready | :x:  Not Ready |
| XDPX | Hard | :heavy_check_mark: Cli | :heavy_check_mark:    Support (Coming soon) | :heavy_check_mark: Serving |

### ModelScope
You can download and use ChatPLUG models from ModelScope.

| Model Name    | URL                                                          |
| ------------- | ------------------------------------------------------------ |
| ChatPLUG-240M | [ChatPLUG-开放域对话模型-240M](https://modelscope.cn/models/damo/ChatPLUG-240M/summary) |
| ChatPLUG-3.7B | [ChatPLUG-开放域对话模型-3.7B](https://modelscope.cn/models/damo/ChatPLUG-3.7B/summary) |


### HuggingFace
Coming soon.

### XDPX

XDPX is an easy-to-use library, that allows researchers and developers to train custom models and build own chatbots in a streamlined manner. Its all-in-one functionality allows for a one-stop solution that simplifies complex processes. [quick start](https://chatplug.readthedocs.io/zh_CN/latest/quick_start.html)

#### One-Click Inference

```bash
# Requirement
# in the dir of XDPX
cd XDPX
pip install -e .

# Download checkpoints
# in the same dir as the download.sh
cd ..
sh download.sh

# Inference
# in the dir of XDPX
cd XDPX
CUDA_VISIBLE_DEVICES=0 x-script fidchat_new chat_pipeline/chatplug_3.7B_sftv2.6.0_instruction.hjson
# input `#exit` and exit the terminal
```

#### One-Click Train

```bash
# 1. Download dataset from belle
# in ChatPLUG/data/belle dir
cd data/belle
git lfs install
git clone https://huggingface.co/datasets/BelleGroup/train_0.5M_CN

python process_belle_0.5M.py 
# $ls data/belle 
# train_0.jsonl dev.jsonl ...

# 2. Preprocess Data 
# in XDPX dir
x-prepro chat_pipeline/chatplug_prepro_sft_instruction.hjson
# $ls data/dialogue/sft/chatplug/belle_instruction 
# train_0.pt dev.pt

# 3. Training
# in XDPX dir
x-train chat_pipeline/chatplug_3.7B_train_sftv2.6.0_instruction.hjson
```


#### One-Click Deploy 
Coming soon.


## Installation

Please refer to [Installation](https://chatplug.readthedocs.io/zh_CN/latest/quick_start.html) for installation instructions.

For detailed user guides, please refer to our [documentation](https://chatplug.readthedocs.io/zh_CN/latest):

- User Guides

  - [Intro](https://chatplug.readthedocs.io/zh_CN/latest/get_started.html)
  - [Training](https://chatplug.readthedocs.io/zh_CN/latest/nlu_training.html#)
  - [Develop](https://chatplug.readthedocs.io/zh_CN/latest/develop.html)
  - [F.A.Q](https://chatplug.readthedocs.io/zh_CN/latest/faq.html)

## Citations

If you find our project useful in your work, please cite:

```
  @misc{tian2023chatplug,
        title={ChatPLUG: Open-Domain Generative Dialogue System with Internet-Augmented Instruction Tuning for Digital Human}, 
        author={Junfeng Tian and Hehong Chen and Guohai Xu and Ming Yan and Xing Gao and Jianhai Zhang and Chenliang Li and Jiayi Liu and Wenshen Xu and Haiyang Xu and Qi Qian and Wei Wang and Qinghao Ye and Jiejing Zhang and Ji Zhang and Fei Huang and Jingren Zhou},
        year={2023},
        eprint={2304.07849},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
  }
```

```
@misc{plug2021,
  title = {{PLUG: Pre-training for Language Understanding and Generation}},
  author={ModelScope},
  publisher = {ModelScope},
  journal = {ModelScope repository},
  year = {2021},
  howpublished = {\url{https://modelscope.cn/models/damo/nlp_plug_text-generation_27B/summary}},
}
```

License
This code is licensed under the [Apache License (Version 2.0). ](./LICENSE)