

# Quick Start (ChatPLUG)


## Install

```bash
# install
cd XDPX
pip install -e .

# download checkpoints from modelscope
sh download.sh

# run cli
CUDA_VISIBLE_DEVICES=0 x-script fidchat_new chat_pipeline/chatplug_xl_sftv2.6.0_instruction.hjson
```

### Cli Usage
| command | action               |
|---------|----------------------|
| query   | response             |
| `#exit` | terminate            |
| `#show` | show reponse details |
| `#new`  | create a new session |



## [可选] 申请OSS账号 

```bash
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

