# Quick Start

## Requirement
```bash
# in the dir of XDPX
cd XDPX
pip install -e .
```

## Download checkpoints
```bash
# in the same dir as the download.sh
cd ..
sh download.sh
```


## Run cli
```bash
# in the dir of XDPX
cd XDPX
CUDA_VISIBLE_DEVICES=0 x-script fidchat_new chat_pipeline/chatplug_3.7B_sftv2.6.0_instruction.hjson
```


### Cli Usage
| command | action               |
|---------|----------------------|
| query   | response             |
| `#exit` | terminate            |
| `#show` | show reponse details |
| `#new`  | create a new session |


## Training

### 1. Downloading Dataset from Belle

```
# in the root dir
cd data/belle

git lfs install
git clone https://huggingface.co/datasets/BelleGroup/train_0.5M_CN

python process_belle_0.5M.py 
# $ls data/belle 
# train_0.jsonl dev.jsonl ...
```

### 2. Preprocessing Data

Convert data from str to ids.



Input data format

```json
{
    "context": "我有点烦怎么办", 
    "response": "您好，您有烦是很正常的，可以尝试放松自己，多多参加户外活动以及与朋友的交流，这样可以让自己的情绪得到释放。如果您有负面情绪，也可以咨询专业的心理咨询师，以释放压力。", 
    "passages": "bot_profile:你发微博图开心;你沉迷锻炼;你坚持健康饮食;你想让别人觉得你很酷;你也是;;;knowledge: 方式一：找亲朋好友倾诉。把自己的不顺心的事情和自己的亲朋好友诉说之后，可以起到发泄排解的作用，这样自己的心里会好受很多。方式二;;;knowledge: 方式一：找亲朋好友倾诉。把自己的不顺心的事情和自己的亲朋好友诉说之后，可以起到发泄排解的作用，这样自己的心里会好受很多。方式二：找方法解决掉烦心事。如果这个烦心事可以通过的自己的努力去得以解决，那么自己或者找同伴帮忙一起去解决他，方式三：如果暂时无法解决可以选择性的忘记它或者转移自己的注意力，去做自己喜欢的事情，让自己尽快摆脱不好的情绪。以上是小编提供的几种方式，希望更够帮助小伙伴们。;;;knowledge: 每个人都会有每个人的烦恼，我们的生活经历不同，遇见的事情人物也不同，所以会碰到不一样的烦恼。但是有烦恼也是很正常的，只要看我们如何去对待。那么，下面小编为大家;;;knowledge: 觉得很烦的原因有很多种，或是由于工作还有生活的压力导致，有事情没有和朋友和家人倾述，长时间下去有可能会得抑郁。根据你的情况，建议要乐观点，凡事不要钻牛角尖，应当;;;knowledge: 问题描述： 你好，心烦,有时候是人多想了.总是想到一些将来会发生的事情,这对一个人的心理就形成了压力。缓解压力的方法：大声吼1到2分钟将心里的怨气发泄出来；;;;knowledge: 特别烦，首先应注意对情绪进行调节。保持情绪平稳并舒缓心理压力，可以找朋友或家人倾诉心声，将内心不愉快的事情吐露出来，有利于缓解心理压力改善情绪，进而缓解心烦。", 
    "meta_info": {
        "data_type": "dialogue_emotion", 
        "readme": "v1.1.3 emotion数据", 
        "from": "emotion", 
        "from_file": "empathy.jsonl", 
        "line_num": 1496}
    }
```


Process tokens into ids

```
x-prepro chat_pipeline/chatplug_prepro_sft_instruction.hjson
# $ls data/dialogue/sft/chatplug/belle_instruction 
# train_0.pt dev.pt
```


### 3. Training

3.1 Runing training script.

```
x-train chat_pipeline/chatplug_3.7B_train_sftv2.6.0_instruction.hjson
```

Here: `global_batch_size = batch_size * update_freq`, and `batch_size = GPUs * batch_per_GPU`. 


3.2 Visualize the training curve

The training curve plots are in `{save_dir}/plots`.

3.3 Eval using the training checkpoint

- create a new  config file `chatplug_3.7B_sft_belle.hjson` from `chatplug_3.7B_sftv2.6.0_instruction.hjson`:
- edit `core_chat_save_dir` and `core_chat_checkpoint` to the corresponding path.

```hjson
core_chat_save_dir: ../checkpoints/sft/chatplug/3.7B/v2.6.0_epoch20_lr1e-4_bs512/
core_chat_checkpoint: ../checkpoints/sft/chatplug/3.7B/v2.6.0_epoch20_lr1e-4_bs512/checkpoint-6000.pt
```
