# ChatPLUG

[![](assets/Demo-ModelScope-brightgreen.svg)](https://www.modelscope.cn/studios/damo/role_play_chat/summary)
[![](assets/Paper-Arxiv-orange.svg)](https://arxiv.org/abs/2304.07849)
![Hex.pm](https://img.shields.io/hexpm/l/plug)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FX-PLUG%2FChatP&count_bg=%23E97EBA&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)


This is the repo for the ChatPLUG project, which aims to build and share a Chinese open-domain dialogue system.

<hr>

| 爱用emoji的萌妹子小婉  |  富有智慧的得道高僧 | 会说古文的的三国NPC关羽 |
:-------------------------:|:-------------------------:|:-------------------------:
<img src="assets/xiaowan.gif"  width="80%" /> | <img src="assets/gaoseng.gif"  width="90%" /> | <img src="assets/guanyu.gif"   width="80%" /> 

## News
- [2023/04/26] 👏👏👏 Try our [Role-Play-Chat Online Demo](https://modelscope.cn/studios/damo/role_play_chat/summary) in ModelScope Now!
- [2023/04/19] Add content including spotlights, results and limitations. Upload models to [modelscope](https://modelscope.cn/my/overview). 
- [2023/04/16] Initialize project.

## Overview

We present ChatPLUG, a Chinese open-domain dialogue system for digital human applications that instruction finetunes on a wide range of dialogue tasks in a unified internet-augmented format. Different from other open-domain dialogue models that focus on large-scale pre-training and scaling up model size or dialogue corpus, we aim to build a powerful and practical dialogue system for digital human with diverse skills and good multi-task generalization by internet-augmented instruction tuning. 

To this end, we first conduct large-scale pre-training on both common document corpus and dialogue data with curriculum learning, so as to inject various world knowledge and dialogue abilities into ChatPLUG. Then, we collect a wide range of dialogue tasks spanning diverse features of knowledge, personality, multi-turn memory and empathy, on which we further instruction tune ChatPLUG via unified natural language instruction templates. External knowledge from an internet search is also used during instruction finetuning for alleviating the problem of knowledge hallucinations. 

We show that ChatPLUG outperforms state-of-the-art Chinese dialogue systems on both automatic and human evaluation, and demenstrates strong multi-task generalization on a variety of text understanding and generation tasks. Our demo and model are made publicly available on [ModelScope](https://modelscope.cn/models/damo/ChatPLUG-3.7B/summary).  

<img src="assets/ChatPLUG.jpg" alt="ChatPLUG"  />

Please read our paper for more detains about ChatPLUG.

- ChatPLUG: Open-Domain Generative Dialogue System with Internet-Augmented Instruction Tuning for Digital Human. [https://arxiv.org/abs/2304.07849]( https://arxiv.org/abs/2304.07849)


## Online Demo
[Role-Play-Chat](https://www.modelscope.cn/studios/damo/role_play_chat/summary)


## Spotlights

<img src="assets/spotlights.jpg" alt="spotlights" style="zoom: 50%;" />

Compared with existed open-source models, we highlight three feaures of ChatPLUG as follows:

1. **Knowledge Augmentation**

  > It's flexible to integrate external knowledge during inference, and this is an optional input. You can utilize a search engine to acquire up-to-date information or use a local knowledge base to obtain domain knowledge. 

2. **Personalization**

  > It's easy to customize the style of conversations and characters by setting bot profiles or using role-paly instructions.

3. **Multi Skills** 

  > It exhibits its proficiency in open-domain dialogue through mulit-turn conversation, while also displaying impressive multi-task abilities on a wide range of NLP tasks. 

 

## Evaluation and Examples

### 1. Knowledge Augmentation

With external knowledge from a search engine, the problem of knowledge hallucinations is alleviated. Besides, it enables ChatPLUG to generate informative responses and can answer correctly when encountering real-time questions.

**Human evaluation of knowledge hallucination**

> With knowledge augmentation, ChatPLUG achieves better performance in terms of knowledge hallucination.

<img src="assets/knowledge_hallucination.jpg" style="zoom: 15%;" alt="knowledge_hallucination"/>


<details><summary><b>Examples of real-time questions (Click to view👇)</b></summary>
<img src="assets/knowledge_example.jpg" alt="knowledge_example" style="zoom: 67%;" />
<summary>Access up-to-date information from Internet enables ChatPLUG to provide accurate real-time answers to questions.  </summary>
</details> 


### 2. Personalization

It's flexible to customize dialogue style and characters by setting bot profiles through our FiD architecture or simply using the appropriate prompt. 

<details><summary><b>Examples of dialogue-style customization (Click to view👇)</b></summary>
<img src="assets/dialogue_style.jpg" alt="dialogue-style" style="zoom: 67%;" />
</details>  

<details><summary><b>Examples of character customization (Click to view👇)</b></summary>
<img src="assets/character_customization.jpg" alt="character_customization" style="zoom: 67%;" />
</details>  


### 3. Multi Skills

ChatPLUG can not only generate coherent and engaging responses in an open and multi-turn conversation scenario, but also demenstrate strong multi-task generalization on a variety of text understanding and generation tasks. We will compare its performance with other recent open-source models, and provide examples as below. 

**Human evaluation of multi-task generalization**

> We compare our model with open-source Chinese LLMs including <a href="https://github.com/LianjiaTech/BELLE">BELLE-7M-2B</a> and <a href="https://github.com/THUDM/ChatGLM-6B">ChatGLM-6B</a> following the four-level rating evaluation (A>B>C>D). First, all the models are able to follow the given instructions (very small quantity of RATING-D). Second, our model ChatPLUG-3.7B achieves better performance (more quantity of RATING-A and fewer quantity of RATING-C) than BELLE-7B-2M with fewer model parameters and is comparable to ChatGLM-6B. It demonstrates the strong multi-task generalization of ChatPLUG. Lastly, by scaling up the model size to 13B, our model ChatPLUG-13B obtains the best performance. 

<img src="assets/evaluation_of_multi_task.jpg" style="zoom: 20%;" alt="evaluation_of_multi_task"/>


<details><summary><b>Examples of multi-task generalization (Click to view👇)</b></summary>
<img src="assets/multitask_case_1.jpg" alt="multitask_case_1" style="zoom: 67%;" />
<img src="assets/multitask_case_2.jpg" alt="multitask_case_2" style="zoom: 67%;" />
</details>  


## How to run

👏👏👏You can download and use ChatPLUG models from modelscope.

| Model Name    | URL                                                          |
| ------------- | ------------------------------------------------------------ |
| ChatPLUG-240M | [ChatPLUG-开放域对话模型-240M](https://modelscope.cn/models/damo/ChatPLUG-240M/summary) |
| ChatPLUG-3.7B | [ChatPLUG-开放域对话模型-3.7B](https://modelscope.cn/models/damo/ChatPLUG-3.7B/summary) |



## Limitations

Based on the real-world dialogue capabilities of digital human, we mainly focus on mulit-turn conversation with three fundamental abilities including **knowledge augmentation**, **personalization** and **multi skills.** Please refer to our paper for futher details.

At the launch of the ChatPLUG project, we don't intentionally bulid a all-powerful large-scale model. Therefore, we find some following limitations.

- Dialogue style generation: ChatPLUG more often tends to generate short and concise responses, which are more suitable for conversational scenarios.
- Weak reasoning ability: it's not good at solving problems that require reasoning skills, such as math calculation.
- Weak code ability: it's not pre-trained with code, so it may perform poorly on coding task.
- May generate harmful content: hamrful or biased content may still be generated if induced.
- Limited multilingual ability: we focus on building a Chinese open-domain dialogue system and not intentinally to improve multilingual ability. 

We leave above shortcomings for future work.


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

- PLUG: Pre-training for Language Understanding and Generation [Link](https://modelscope.cn/models/damo/nlp_plug_text-generation_27B/summary) 

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






