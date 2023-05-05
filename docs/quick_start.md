

# Quick Start (ChatPLUG)


## Install 

```bash
# git clone
git clone .git

# 安装
cd XDPX
pip install -e .


# [可选]申请OSS账号
# 配置 user/modules/oss_credentials.py文件

```python
from xdpx.utils import OSS,io,os

# 授权ODPS
os.environ['PROJNAME']='<project>'
os.environ['ACCESS_ID']='<access_key_id>'
os.environ['ACCESS_KEY']='<access_key_secret>'
os.environ['ODPS_ENDPOINT']='http://service-corp.odps.aliyun-inc.com/api'

# 授权 OSS
access_key_id='<access_key_id>'
access_key_secret='<access_key_secret>'
region_bucket=[
    ['cn-hangzhou','<bucket_name>'],
    ['cn-beijing','<bucket_name>'],
]
oss=OSS(access_key_id,access_key_secret,region_bucket)
io.set_io(oss)
```

填上access_key_id、access_key_secret，并添加所有你需要访问的oss bucket的名字以及它们所在的地区名到region_bucket。比如链接为[http://pretrain-lm.oss-cn-hangzhou.aliyuncs.com/](http://pretrain-lm.oss-cn-hangzhou.aliyuncs.com/) 的话，bucket就是“pretrain-lm”，地区名就是“oss-”后面的那部分，即“cn-hangzhou”，传入的值为['cn-hangzhou','pretrain-lm']。之后程序运行便会自动加载这个配置并且支持访问oss路径。

## Inference

### Download Checkpoints 

```bash
sh download.sh
```


### CLI Inference

```bash
# 命令行运行测试
CUDA_VISIBLE_DEVICES=0 x-script fidchat_new oss://xdp-expriment/gaoxing.gx/chat/configs/new_test_chat/mt5_v1.1.3.0804_78000_qc_aliganyu621_newrewrite_ner_newskill.hjson
# 把local_search 改为 ''
# "local_retrieval_host": "",


# terminal运行后的操作
>>> #test_self_chat # self_chat
>>> #test_entity_knowledge # 测试知识
>>> #test_persona # 测试人设
# 测试多轮
>>> #test_file=cconv.test.json
x-script eval_open_dialog <test_file_path> 
>>> #exit # 退出
>>> #show # 展示中间结果

```
| 模型 | config 地址 | remark |
| --- | --- | --- |
| 0.3B | oss://xdp-expriment/zhimiao.chh/dialogue_config/chat_pipline/server/mt5_ctr.hjson | #test_entity_knowledge |
| 3.7B | oss://xdp-expriment/zhimiao.chh/dialogue_config/chat_pipline/server/mt5_xl_v1.2.1_v1.1.3_allspark_tmp.hjson |  |
| 3.7B | oss://xdp-expriment/gaoxing.gx/chat/backup/mt5_xl_v1.2.1_v1.1.3.zhimiao.rolechat3.hjson | 

#test_file=test_guodegang.txt
#bot_profile=郭德纲@@@我是郭德纲 |

版本更新到加速为止：[https://yuque.antfin.com/tjf141457/deqr3f/dhz6e9irwplsfc36](https://yuque.antfin.com/tjf141457/deqr3f/dhz6e9irwplsfc36)


## Training
```bash
# preprocess
PYTHONPATH=./ x-prepro oss://xdp-expriment/gaoxing.gx/chat/configs/finetune_chat/mt5_prepro_0917.hjson
# 产出prepro后的数据
->oss://xdp-expriment/gaoxing.gx/chat/prepro/mt5/0917/v1.1.3.2.ctr/

# training
PYTHONPATH=./ x-train oss://xdp-expriment/gaoxing.gx/chat/configs/finetune_chat/mt5_train_0917.hjson
# 产出预训练后的模型
-> oss://xdp-expriment/gaoxing.gx/chat/training/mt5_finetune/0917
```



## 代码结构
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2023/png/55526/1675150241268-ec241931-7714-45b4-9da7-61cd39f3bf8d.png#clientId=uab106b88-d556-4&from=paste&height=611&id=u2a3f8427&originHeight=1582&originWidth=1338&originalType=binary&ratio=1&rotation=0&showTitle=false&size=261295&status=done&style=none&taskId=ue9b0cf70-2939-4c46-bf80-91a353a2b08&title=&width=517)

## F.A.Q
### ~~1. cro 错误~~
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/55526/1663591097062-4a690e0c-9ebc-47b1-af46-22d868e918a1.png#clientId=u177eefa8-e6e1-4&errorMessage=unknown%20error&from=paste&height=457&id=u93ea348f&originHeight=914&originWidth=1782&originalType=binary&ratio=1&rotation=0&showTitle=false&size=191507&status=error&style=none&taskId=ud3588266-18be-4c03-9e27-5b905155527&title=&width=891)
~~xdpx/utils/chat/safety_filter.py~~
