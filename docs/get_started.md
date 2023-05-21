# XDPX

项目地址：[https://github.com/X-PLUG/ChatPLUG](http://https://github.com/X-PLUG/ChatPLUG)
该项目主要支持**用于NLU的PyTorch代码（包括PAI在内）的跨环境训练**。有以下特点：

- 支持所有环境中，所有路径直接配置OSS路径，和本地路径一起无差别使用，大文件传输显示进度条；
- 支持本地/PAI、单卡/多卡/分布式训练的无缝切换，支持自动开启docker-fusion等PAI高级选项；
- 支持稳定的混合精度训练、virtual batch size、对抗训练、AutoML等多种高级训练方法；
- 内置多种Bert模型及其在pretrain、finetune上的改进方法和训练技巧，以及多种高效的非Transformer分类/匹配模型可用于线上部署；
- 兼容Tensorflow模型，可以直接读入/导出tf checkpoint/savedmodel，内置Huggingface的Bert格式，也可以移植已有的tf模型定义并快速实现双向转换；
- 可扩展性强，方便移植各种pytorch模型和训练方法，可以扩展到多语言；
- 可以直接在配置文件内定义超参数搜索、交叉验证等批量训练配置；
- 提交到PAI之前自动检查参数配置错误，包括路径是否存在、不同参数之间的冲突等，避免浪费提交和排队的时间；
- 单元测试全覆盖，方便debug。

常见的问题见下方“[FAQ](faq)”章节。
单独使用oss/本地文件混合操作能力参考[IO能力](oss)章节。



## 安装配置
### 本地安装

- 安装python>=3.6和pytorch
   - CUDA >=9: `pip install torch`
   - CUDA 8: [pytorch 1.0.1 for CUDA 8/python 3.6](https://download.pytorch.org/whl/cu80/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl)
- git clone该项目，进入项目根目录后使用`pip install -e .`
### 配置PAI训练
#### 本地安装XDPX
提交PAI训练之前本地同样需要安装XDPX：安装python>=3.6和pytorch（CPU版本即可），并在项目根目录运行`pip install -e .`
#### ODPS客户端
下载并配置[ODPS客户端](http://help.aliyun-inc.com/internaldoc/detail/27971.html?spm=a2c1f.8259796.3.4.141496d5bAtzKH)；随后将bin/添加到系统路径，以便于通过odpscmd命令直接启动客户端。
#### OSS授权
参考[官方教程](https://yuque.antfin-inc.com/pai-user/manual/tf_oss-role-arn-application#sbd9nw)，OSS授权ODPS访问。
在项目根目录下建立`user/modules/oss_credentials.py`文件：
```python
from xdpx.utils import OSS, io
access_key_id = '<access_key_id>'
access_key_secret = '<access_key_secret>'
region_bucket = [
    ['cn-hangzhou', '<bucket_name>'],
    ['cn-beijing', '<bucket_name>'],
]
oss = OSS(access_key_id, access_key_secret, region_bucket)
io.set_io(oss)
```
填上access_key_id、access_key_secret，并添加所有你需要访问的oss bucket的名字以及它们所在的地区名到region_bucket。比如链接为[http://pretrain-lm.oss-cn-hangzhou.aliyuncs.com/](http://pretrain-lm.oss-cn-hangzhou.aliyuncs.com/) 的话，bucket就是“pretrain-lm”，地区名就是“oss-”后面的那部分，即“cn-hangzhou”，传入的值为['cn-hangzhou','pretrain-lm']。之后程序运行便会自动加载这个配置并且支持访问oss路径。

## 使用
这一部分介绍XDPX的基本使用方法。已经实现的模型、训练方法等使用案例参见[内置的模型和训练方法](nlu_training)。想了解如何提交到PAI执行请直达[提交到PAI执行](#Giyl2)。想要了解参数配置的用法可以参考[参数配置指南](nlu_training)。

XDPX命令的基本组成为`<命令> <配置文件>`，如`x-train config.hjson`。下面的部分主要介绍如何完成配置文件。
## 数据预处理
命令：`x-prepro config.hjson`。这一步将原始文本转化为二进制文件，供下一步训练使用。
预处理的配置项：

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
| data_source | str | (required) | 数据源路径 |
| data_files | Union[Dict[str, str], List[str], str] | *.txt | data_source下被视为数据文件的相对文件路径，可以传入通配符、路径列表，或分片名到文件路径的映射。 |
| data_dir | str | (required) | 预处理后数据保存的路径 |
| vocab_file | str | None | predefined vocab file |
| target_map_file | str | None | predefined target map file |
| pretrained_embeddings | str | None |  |
| log_file | Optional[str] | log.txt | log filename under "data_dir" |
| workers | int | 1 | workers for parallel processing |
| seed | int | 1 | seed for non-pretrained embedding initialization, etc. |
| processor | str | (required) | See xdpx/processors/ |
| loader | str | (required) | See xdpx/loaders/ |
| tokenizer | str | (required) | See xdpx/tokenizers/ |
| parser | str | csv | See xdpx/loaders/parsers |
| max_len | int | (required) |  |
| min_len | int | 1 |  |
| pad_word | str | [PAD] |  |
| unk_word | str | [UNK] |  |
| lower | bool | TRUE | 输入统一转化为小写 |
| remove_duplicate | bool | FALSE | 去除数据集中的重复行 |
| skip_bad_lines | bool | FALSE | 遇见格式错误的行是否跳过，否则报错 |
| start_line | int | 0 | 数据文件从第几行开始 |

适配新的数据格式可以参考下文[开发指南](develop)。

当vocab_file没有指定时，将从数据中动态构建，有如下参数：

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
| threshold | int | -1 | 出现频率超过这个值才加入进词表 |
| ignore_in_emb | bool | TRUE | 如果一个词在预训练embedding中出现，则忽略threshold统一加入词表 |
| nwords | int | -1 | 词表的最大词数，-1为不限制 |


当target_map_file没有指定时，将从数据中动态构建，有如下参数：

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
| target_type | str | text | text: 标签位置上是标签名，target_map按出现频率排序；
index: 标签位置上是序号0, 1, ….target_map按数字从小到大排序 |
| special_targets | List[str] | [] | 这些标签总是在最顶端，比如在计算多分类F1时需要将负类放在最前面，此项应该配置为['unknown'] |


在数据量很大时，最好直接提供预定义好的vocab_file和target_map_file，否则预处理时需要分别将数据再全部遍历一遍以动态计算这些值。

预处理后的数据路径如果已经存在，再次预处理会覆盖此次指定的data_files，之前预处理过的部分会保留；如果分批预处理，最好指定预定义好的target_map_file和vocab_file，否则会不断重新构建。

预处理后data_dir内的文件及含义如下表所示：

| **文件名** | **含义** |
| --- | --- |
| args.py | 分组的完整预处理参数，带*表示和默认值不同 |
| train.pt | 预处理后的数据文件 |
| dev.pt |  |
| log.txt | 日志文件 |
| meta.hjson | 供训练时继承的预处理阶段的配置 |
| target_map.txt | 标签名称列表 |
| vocab.txt | 词表 |


## 训练
命令：`x-train config.hjson`。
一些Tips：

- 训练时默认会用上所有可见的GPU，如果需要限制请配置CUDA_VISIBLE_DEVICES的环境变量；
- 配置参数后，可以在本地命令后加上--dry预览当前配置下的完整参数，比如`x-train config.hjson --dry

`

训练目录中文件的含义如下表所示：

| **文件名** | **含义** |
| --- | --- |
| args.py | 分组的完整训练参数，带*表示和默认值不同 |
| checkpoint-20000.pt |  |
| log.txt | 日志文件 |
| plots | 绘图文件，类似tensorboard中的内容 |
| snapshot.zip | xdpx/目录的备份，包含当前使用的代码，可以在未来稳定地复现本次训练 |
| starter_config.hjson | 包含非默认参数，编辑后可以用于启动其他类似的训练 |
| train.log.tsv |  |
| valid.log.tsv |  |

### 基本训练配置
训练配置首先要继承预处理的配置`__parent__: ${data_dir}/meta`，示例如下：
```json
{
  __parent__: ${data_dir}/meta
	data_dir: "数据路径"
  save_dir: "训练保存的路径"
  // 其他训练配置项
  // ....
}
```
由于训练涉及的参数较多，这里先介绍最基本的训练配置，高级训练配置放在后面专题介绍。

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
|  | 路径相关的参数 |  |  |
| data_dir | str | (required) | x-prepro得到的数据路径 |
| save_dir | str | (required) |  |
| overwrite | bool | FALSE | whether to overwrite save_dir if exists. |
| auto_suffix | bool | FALSE | whether to use new save_dir with auto suffix if save_dir exists. Exclusive with "overwrite". |
| train_subset | Union[str, List[str]] | train | supports Unix filename pattern matching for multiple files |
|  | 训练相关的参数 |  |  |
| seed | int | 1 | to disable random seed, use None; |
| max_epoch | int | None | 两者同时设定时，取最小的步数作为total steps |
| max_update | int | None |  |
| batch_size | int | (required) |  |
| learning_rate | float | (required) |  |
| clip_norm | float | 5.0 | gradient norm clipping |
|  | 日志/验证相关的参数 |  |  |
| log_interval | int | 10 |  |
| log_file | Optional[str] | log.txt | log filename under "save_dir" |
| eval_interval | int | (required) | -1 means just eval at the end of training |
| valid_subset | str | dev |  |
| major_metric | str | (required) | major metric for early stopping and display |
| ascending_metric | bool | TRUE | whether the major metric is the higher the better |
| tolerance | int | None | If not None, do early stopping. |
| eval_interval_warmup | int | 0 | eval (& maybe save) n times less frequently in earlier steps before step K. K="eval_interval_warmup" n="eval_interval_warmup_mutiplier" |
| eval_interval_warmup_mutiplier | int | 6 |  |
|  | 模型保存相关的参数 |  |  |
| save | bool | TRUE |  |
| save_best_only | bool | FALSE |  |
| save_above_score | float | None | only save when major_metric is better than this score |
| save_full_checkpoint | bool | FALSE | save full checkpoint to support resumed training in the future. |
| save_last_only | bool | FALSE | if save_best_only is also true, save best & last |
|  | 各个模块的配置（配置项对应每个类上方`@register('...')`中的名字） |  |  |
| task | str | default | See xdpx/tasks/ |
| model | str | (required) | See xdpx/models/ |
| loss | str | (required) | See xdpx/lossess/ |
| optimizer | str | adam | See xdpx/optimizers/ |
| lr_scheduler | str | constant | See xdpx/lr_schedulers/ |


下面介绍一些高级的训练设定。
### 训练数据分片处理
当数据特别大的时候，由于预处理会在开头把数据都加载进内存，在多进程预处理中可能会把内存撑爆；大于5G的训练数据也会因为大于oss python-sdk的处理上限导致存储失败。此时可以将训练数据分片处理：
```bash
# 以切10个分片为例，最好先全局shuffle再分割；在MacOS上需要使用gsplit命令
shuf $file | split -a1 -d -l $(( ( $(wc -l <$file) + 9 ) / 10 )) - train
# 如果已经预先shuffle好，也可以使用
split -a1 -d -n l/10 $file train
# 批量加上.txt后缀
rename 's/(train\d+)/$1.txt/' train*
```
预处理时配置需要处理的多个分片，比如`data_files: ['train0.txt', 'train1.txt', 'dev.txt']`或者使用通配符`data_files: *.txt`。
分批处理的范例如下（以语言模型预训练-文档数据预训练为例）：
```json
{
    __parent__: [
        examples/pretrain/prepro/corpus
    ]
		data_files: {
      '相对于data_source的分片路径1': '处理后的分片名称1'
      '相对于data_source的分片路径2': '处理后的分片名称2'
      // 继续补充更多
    }
    // 以下两个资源文件必须给出，否则会加载所有分片先统计词表/预测标签
    vocab: '词表文件路径，必须给出'
    target_map: '预测标签名文件路径，必须给出'
    
    data_source: '数据源路径'
    data_dir: '预处理后的数据路径'
    break_mode: complete
    mask_whole_words: true
    loader: corpus_mix
    max_len: 128
    workers: 32
}
```
分片处理后，需要在训练配置中加入：`train_subset: ['分片名1', '分片名2', '分片名3']`
，或者使用通配符`train_subset: train*`。训练时，有两种模式：

- `lazy_load: false` 会一次性加载所有的分片并组合成一个完整的训练集，对于过大的数据集可能会在训练时有内存问题。
- `lazy_load: true` 会按顺序每次只加载一个分片，并只在分片内部shuffle，并且从第二个epoch开始shuffle分片读取顺序。这样的设定需要在分片之前就提前shuffle好数据集。这个模式下，只要第一个分片预处理完成（并且预处理速度快于训练速度），即可直接启动训练，程序会自动更新后续处理的分片信息。
### 混合精度训练
相关参数：

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
| fp16 | bool | FALSE | 配置为${__apex__}可以在环境满足时自动开启混合精度训练 |
| fp16_backend | str | fairseq |  |
| min_loss_scale | float | 0.1 | minimum FP16 loss scale, after which training is stopped |

混合精度训练的环境满足指有GPU且GPU的compute capability>=7、已安装apex相关依赖。

当backend为apex时，参数为

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
| opt_level | str | O1 | APEX opt_level, see [https://nvidia.github.io/apex/amp.html#opt-levels](https://nvidia.github.io/apex/amp.html#opt-levels) |


当backend为fairseq时，过程大致相当于apex O2，有更多的参数可以控制细节，并且混合精度的具体过程暴露在代码中，如果出现NaN等问题可以激活NaNDetector，方便debug。实际训练中结果和apex O2几乎一致；参数为

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
| memory_efficient_fp16 | bool | FALSE |  |
| fp16_init_scale | int | 128 |  |
| fp16_scale_tolerance | float | 0.0 | pct of updates that can overflow before decreasing the loss scale |
| threshold_loss_scale | int | None | threshold FP16 loss scale from below |


注意混合精度训练如果要有最佳的速度，所有Tensor的长度需要是8的倍数。目前已经自动检查batch_size，max_len和vocab_size，并且自动padding到8的倍数的序列长度，但其他模型中的参数仍然需要自行检查。
### 导数累积
通过导数累积来支持virtual batch size，例如配置 `update_freq: 2` 则两次前馈后计算一次导数，降低显存占用的同时不会明显增加训练时间，如果`batch_size: 64` ，此时实际batch size为128。训练时日志显示的步数为实际batch_size对应的步数（即优化器更新一次算一步）。
### 自动导数累积
配置 `auto_ga: true`可以在OOM的时候自动开启导数累积，这样就不用提前手动预估update_freq的数值。在PAI上训练时，由于默认情况下分配到的GPU型号不确定，单卡的显存可能在10G～32G之间，开启自动导数累积并设置初始的update_freq为1，可以尽可能的利用当前分配到的GPU显存。
> 注意：目前仅支持单GPU使用，多GPU时当一个worker OOM了，另一个worker没有的时候会出错。稳定的多GPU自动导数累积功能正在开发中。

### BMUF分布式优化
多卡/分布式训练时，配置`use_bmuf: true`可以使用[BMUF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/0005880.pdf)（分块更新滤波）算法进行分布式优化。实际经验是大数据量的任务效果会更好一些，小数据finetune时结果可能会不稳定。
### 分阶段训练/可恢复训练
普通的分阶段训练，后面的训练中的pretrained_model配置为前一步保存的模型即可。
如果要沿用前一阶段的learning rate schedule、恢复adam的动量信息和随机性等完整的训练状态，那么需要做一些额外的设置。在一些分布式长时间训练的场景下，训练可能因为OSS连接错误、多worker的通信错误而中断，此时也需要能随时恢复中断的训练。

前一阶段训练/可恢复的训练 需要的额外设置：
```json
{
    save: true
    save_full_checkpoint: true
    save_above_score: null
    // 设置需要训练的步数。注意只有max_update和max_epoch会影响lr schedule的计算。
    // train_steps不会影响。不设置这个值会直接运行到最后一步。
    // 如果只是为了确保当前训练在被中断后可恢复，则不需要设置train_steps
    train_steps: 5000  
}
```
当前配置会在每次eval后保存模型。对于分阶段训练来说，如果只需要保存最后一步，不在乎中间结果，可以设置save_last_only: true。如果设置了save_best_only为true，也需要同时设置save_last_only: true确保最后一步的模型可以保存下来。

后一阶段的训练/中断后恢复训练 需要的额外设置：
```json
{
    resume: true
    save_dir: "保存的路径" // 如果之前开过auto_suffix，需要指定具体的后缀
    overwrite: false
    auto_suffix: false
    // 其他需要覆盖的参数
}
```
注意这里不应该覆盖模型相关参数。如果覆盖了训练数据或者batch_size，需要注意相关的步数和schedule的计算会受到显著影响。
### AutoML
命令：`x-tune tune.hjson`
目前实现了基于[HyperBand](https://arxiv.org/abs/1603.06560)算法的AutoML。
![overview.png](https://intranetproxy.alipay.com/skylark/lark/0/2020/png/125074/1589795421613-2a20153d-4741-4d7f-bb00-b030a10bbd0b.png?x-oss-process=image%2Fwatermark%2Ctype_d3F5LW1pY3JvaGVp%2Csize_31%2Ctext_5bCP6Jyc%2Ccolor_FFFFFF%2Cshadow_50%2Ct_80%2Cg_se%2Cx_10%2Cy_10#height=367&id=Q2cNm&originHeight=837&originWidth=1081&originalType=binary&ratio=1&rotation=0&showTitle=false&size=114507&status=done&style=shadow&title=&width=474)
automl需要配置3个文件：

- train.hjson: 默认的训练配置
- space.hjson:指定搜索空间
- tune.hjson: 配置automl

参考的搜索空间配置space.hjson
```json
{
    learning_rate: {
        type: float
        values: [1e-5, 2e-4]
        log: true
    }
    seed: {
        type: int
        values: [1, 1000000]
    }
    hidden_size: {
        type: int
        values: [100, 300]
        step: 50
    }
    anneal_strategy: {
        type: categorical
        values: ['linear', 'cosine']
    }
		output_dropout: {
        type: categorical
        values: [0.0, 0.1]
    }
}
```
参考的automl配置tune.hjson
```json
{
    space_config: ./space
    base_config: ./train
    save_dir: 'automl保存的路径'
    tuner: random
    assess_start_step: 200 // 从哪一步开始assess
    assess_interval: 100 // 开始assess后assess的间隔
    parallel_runs: 16 // 并行的试验数
    max_trials: 16 // 最大试验数，如果配成32会跑两个episode
    min_ref: 3 // 会执行到最后的试验数
    cull_keep_ratio: 0.8 // 每轮assess保留的比例
    eval_deviation: 40 // 允许的训练数据量对比误差
    save_above_score: 0.7 // 最后只保留major_metric大于该分数的模型
}
```

### 对抗训练
在标准训练配置的基础上配置`task: freelb`即可。目前采用的是[FreeLB](https://arxiv.org/abs/1909.11764)在embedding层进行对抗训练。对抗训练可以使Bert finetune的结果更稳定；如果用的是非标准bert模型，需要继承FreeLBTask并且覆盖bottom_predict和upper_predict方法，前者给出输入向量，加入对抗扰动后进入后者继续前馈。

## 提交到PAI执行
完成PAI的[相关配置](#Mc2H2)后，在项目根目录下使用`bash pai.sh <命令> <配置>`即可提交到PAI训练，例如`bash pai.sh x-train config.hjson`。可以串联多个<命令> <配置>，这些任务会依次串行执行，例如`bash pai.sh x-train config1.hjson x-train config2.hjson`。

执行后会有prompt引导配置PAI训练的相关参数；如果不希望prompt，可以在指令后面加上相关配置，比如`bash pai.sh ... -v 131 -w 1 -g 2 -c 4 -m 10`（分别是pytorch version/workers/gpus/cpus/memory(G)）。

提交过的命令、提交时间以及对应的logview地址会记录在command_history.txt中，方便后续追踪。

如果需要强制指定V100（比如启动混合精度训练时），可以在prompt中选择，或者加上命令行参数--v100 y。
超卖特性的设置目前维持默认值。
### docker fusion
选择workers总数大于1，且每个worker gpu数量为1时，会自动启动[DockerFusion](https://www.atatech.org/articles/171093?spm=ata.13261165.0.0.23f45617ejdlbV)。在日志中可以看到具体的分配情况，比如Tesla V100-SXM2-16GB*4代表1个物理机上有4块V100，而Tesla V100-SXM2-16GB*(2+1+1)代表节点分布在三台机器上，分别有2、1、1块V100。
> 目前开启docker fusion并且串行执行多个任务的时候（见“参数配置指南-循环”），从第二个任务开始会静默卡住或者报h04e04357:542:1085 [0] NCCL INFO Call to connect returned Connection refused, retrying，这个已经和PAI同学确认了是docker fusion里的bug，目前尚未修复，因此目前无法在开启docker fusion的同时串行多个训练任务。

## 验证/预测
### 加载模型
在验证/预测之前，需要先加载已有的模型。加载已有的模型有两种模式：

1. 从一个已完成的训练加载：
```json
{
    save_dir: '训练保存的路径'
}
```

2. 从一个训练配置加载模型初始值（例如要加载一个不是用XDPX训练的模型，就可以把它设为训练配置中的pretrained_model，并利用其“初始值”；或者可以用这个方法得到随机初始化模型的baseline）：
```json
{
    config: '训练配置.hjson'
}
```
这两种配置是互斥的，只能选择其中一种。
其他的和模型/配置加载相关的配置项:

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
| extra_config | Dict | {} | 这里配置的参数会覆盖原有的配置，比如在验证/预测时使用一个新的Task或者Loss等等。 |
| checkpoint | Optional[str] | None | Full path is needed if provided. Default is the best one in "save_dir" mode and the initial one in "config" mode |
| from_tf | bool | FALSE |  |
| batch_size | int | None | If provided, use a new batch size. |
| cuda | bool | torch.cuda.is_available() |  |
| seed | int | 1 |  |

其中checkpoint可以用一些alias指定特定的模型：

- checkpoint: ${save_dir}/<best>   dev集最优的模型
- checkpoint: ${save_dir}/<last>  保存下来的最新的模型
### 加载数据
读取的数据源（对应x-eval中的valid_subset，以及x-pred中的predict_file_map的key）可以是预处理前的文本文件、预处理后的二进制文件、ODPS表格。
当数据源是纯文本文件时，以下配置会生效：

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
| workers | int | None | num of workers for raw text data loading. |
| skip_bad_lines | bool | FALSE |  |

数据源为ODPS表格只能在PAI上生效；提交PAI任务时需要指定关联的表格，比如`bash pai.sh x-eval config.hjson --tables odps://project/tables/tableName`。
### 评分验证
命令：`x-eval config.hjson`。这一命令可以在给定的模型和数据集上计算loss和其他分数。
在以上模型配置和数据配置的基础上，专属于评分验证的配置：

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
| valid_subset | Union[List[str], str] | (required) | 验证用的数据源，可以是dev, train这样已有的分片名称，或者一个完整的文件路径 |
| save_to_file | str | None | save evaluation result to a .tsv file |
| save_mode | str | 'a' |  |
| max_eval_steps | int | None |  |


### 预测
命令：`x-pred config.hjson`
在以上模型配置和数据配置的基础上，专属于预测的配置：

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
| predict_file_map | Dict | (required) | 需要预测的文件->结果保存的文件，保存格式为TSV |
| distill | bool | FALSE | whether to distill logits only |
| binary | bool | FALSE | store predicted results in binary files. |
| max_predict | int | None |  |
| concat_origin | bool | TRUE | concat original data lines in output files |
| header | bool | TRUE | 结果TSV里是否要带每列的标题 |

配置样例：
```json
{
    save_dir: '训练保存的路径'
    predict_file_map: {
        '原始数据文件1': '预测后的文件1'
        '原始数据文件2': '预测后的文件2'
    }
}
```
其中原始数据文件可以带标签，也可以不带。

如果需要读入预测后的文件进行错误分析，可以使用以下代码：
```python
import csv
import pandas as pd
# 如果预测时header为true
data = pd.read_csv(f, sep='\t', header=0, quoting=csv.QUOTE_NONE)
# 如果预测时header为false
data = pd.read_csv(f, sep='\t', header=None, quoting=csv.QUOTE_NONE, 
                   names='col1 col2 ...'.split()) # 列名根据具体情况修改
```

## 统计分析
### 结果汇总到表格
```shell
x-script aggregate_results "训练根目录" -包含的关键词 > stats.tsv
```

- 可以传入多个根目录（关键词是共享的）
- 如果训练根目录下都是同一批验证指标，那么可以在原有命令后加上--full，可以汇总训练速度和超参数配置的对比，并自动按照分数排序
### 可视化绘图
训练后会自动绘图。如果想调整绘图的配置或者和其他训练比对，可以单独运行绘图命令。
绘图配置：
```json
{
    save_dir: '训练保存的路径'
    ref_dir: '参照的训练保存的路径（可省略）'
    figext: 'png'
    walltime: false // 是否以walltime作为横坐标
}
```
执行绘图：`x-viz config.hjson`，图片保存在训练保存的路径下的“plots/”文件夹。注意默认的横坐标是步数，ref_dir中显示的步数会比照save_dir进行同步，使得同一横坐标下模型读取过的训练样本数相同。

有多个ref_dir时，可以指定名称方便比较：
```json
{
    save_dir: '训练保存的路径'
    label: '当前训练的标签'
    ref_dir: {
  		baseline: '对比路径0'
  		变体1: '对比路径1'
  		变体2: '对比路径2'
		}
}
```
### ![cls_f1.png](https://intranetproxy.alipay.com/skylark/lark/0/2020/png/125074/1603941084570-25655be1-339b-4999-b165-f60fe49d1429.png?x-oss-process=image%2Fwatermark%2Ctype_d3F5LW1pY3JvaGVp%2Csize_18%2Ctext_5bCP6Jyc%2Ccolor_FFFFFF%2Cshadow_50%2Ct_80%2Cg_se%2Cx_10%2Cy_10#height=354&id=zYrFg&originHeight=480&originWidth=640&originalType=binary&ratio=1&rotation=0&showTitle=false&size=43161&status=done&style=none&title=&width=472)
图中一条红色竖线代表的是一个epoch（或lazy_load开启时分片训练中的一个分片），绿点代表当前dev最优值，横坐标代表训练步数，注意对比路径中的训练步数会和当前训练按训练数据量换算，比如如果当前训练batch_size为2048，对比训练batch_size 1024，那么对比训练中的20000步的数据对应图中10000步。
### 超参数比对
```shell
x-script diff_params "训练保存的路径1" "训练保存的路径2"
x-script diff_params "训练保存的路径" "本地hjson训练配置"
```

### 交互式调用
命令：`python xdpx/serve.py <save_dir>`
有时候交互式调用会比较方便debug模型中的问题。交互式调用时需要输入和预处理前的训练数据格式相同的数据，比如['有没有 优惠券', '有 优惠券 吗']，如果是输入只有单个文本，可以直接输入文本本身。
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2020/png/125074/1598431378520-6026c219-e51b-4ba4-af16-7965fe8f1db3.png?x-oss-process=image%2Fwatermark%2Ctype_d3F5LW1pY3JvaGVp%2Csize_24%2Ctext_5bCP6Jyc%2Ccolor_FFFFFF%2Cshadow_50%2Ct_80%2Cg_se%2Cx_10%2Cy_10#height=93&id=yMPnH&originHeight=93&originWidth=855&originalType=binary&ratio=1&rotation=0&showTitle=false&size=39238&status=done&style=none&title=&width=855)
## 部署/兼容性
### 导出TorchScript
命令：`x-script export_torchscript config.hjson`

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
| save_dir | str | (required) |  |
| checkpoint | str | <best> | Full path is needed. If not provided, use the best checkpoint in save_dir |
| out_dir | str | (required) |  |

注意导出时环境中的pytorch版本要和部署环境中的pytorch版本相符。
### 导出TF checkpoint/savedmodel
命令：`x-script export_tf config.hjson`

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
| save_dir | str | (required) | 训练保存的路径 |
| checkpoint | str | <best> | Full path is needed. If not provided, use the best checkpoint in save_dir |
| out_dir | str | (required) | 导出的路径 |
| export_format | str | savedmodel | 可选savedmodel或checkpoint |
| strict_size | bool | TRUE |  |


在export_format为savedmodel时，有以下参数：

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
| signature_def_key | str | serving_default |  |
| init_tables | bool | TRUE | savedmodel加载时是否要初始化tables |
| fix_len | bool | FALSE | 是否使用固定长度输入（而不是动态padding） |
| check_outputs | bool | FALSE | 是否执行和pytorch模型的自动结果比对 |


在export_format为checkpoint时，有以下参数：

| **name** | **type** | **default** | **doc** |
| --- | --- | --- | --- |
| out_name | str | bert_model.ckpt |  |

### 导入tf checkpoint

- 在训练中可以配置`from_tf: true`并给pretrained_model参数配置tf路径来直接导入tf checkpoint / savedmodel，在本地训练时需要环境中有tf依赖，PAI训练时需要通过`bash pai.sh x-train config.hjson --tensorflow`启动训练，注意指定`--tensorflow`后安装环境依赖的时间会从1分钟增长到约9分钟
- 如果要频繁导入某个tf checkpoint，可以先将其转换成pytorch模型（离线转换目前只支持Bert家族的模型）。命令为`x-script export_pt config.hjson`。配置如下：
```json
{
    tf_ckpt_path: 'tf模型路径'
    tf_config: 'bert_config.json'
    out_dir: '导出路径'
    num_classes: 2  // bert中的分类任务是几分类,比如structbert是3
}
```
## IO能力
可以在命令行单独使用IO能力，比如执行`x-io copytree $src_dir $tgt_dir`，可以跨oss和本地路径进行文件传输。可用的命令包括exists, move, copy, copytree, makedirs, remove, rmtree, listdir, isdir, isfile, last_modified, size, md5, is_writable. 详情参考xdpx/utils/io_utils.py

## 调试工具

### Profiling/测速
![20200518191809.jpg](https://intranetproxy.alipay.com/skylark/lark/0/2020/jpeg/125074/1589800745227-90197f6f-05b7-419d-bedf-ac855e16c65a.jpeg?x-oss-process=image%2Fwatermark%2Ctype_d3F5LW1pY3JvaGVp%2Csize_68%2Ctext_5bCP6Jyc%2Ccolor_FFFFFF%2Cshadow_50%2Ct_80%2Cg_se%2Cx_10%2Cy_10#height=352&id=qXeuU&originHeight=352&originWidth=2390&originalType=binary&ratio=1&rotation=0&showTitle=false&size=155506&status=done&style=none&title=&width=2390)
仿照scripts/profiling中的代码，实现你要profile的模型配置并注册一个名字，然后使用
`python scripts/run_profiling <profile_name> <trace_save_path>`
执行profiling。导出的chrome_trace可以在chrome://tracing中打开查看。

### Gradient Inspector
![10.png](https://intranetproxy.alipay.com/skylark/lark/0/2020/png/125074/1598436093664-a5faed67-f8cd-4084-883b-29e3c358a4b8.png?x-oss-process=image%2Fwatermark%2Ctype_d3F5LW1pY3JvaGVp%2Csize_27%2Ctext_5bCP6Jyc%2Ccolor_FFFFFF%2Cshadow_50%2Ct_80%2Cg_se%2Cx_10%2Cy_10#height=445&id=wA7qy&originHeight=680&originWidth=930&originalType=binary&ratio=1&rotation=0&showTitle=false&size=47197&status=done&style=none&title=&width=608)
通过gradient inspector可以诊断模型是否有梯度爆炸/梯度消失的问题，并快速定位问题产生的位置。训练时配置`inspect_gradient: true`即可开启这一功能，结果保存在${save_dir}/plots/gradients/里。如果模型参数很多，会生成一个高清长图。
### NaNDetector
如果模型/loss中出现NaN/Inf，不论是forward还是backward，NaNDector都会自动启动（在开启混合精度训练时，在loss_scale调整到最大限度后才会自动开启NaNDector）。相关报告可以帮助定位NaN/Inf首次产生的位置。
### 模型summary
类似tensorboard中的`add_summary()`的功能可以通过如下代码实现：
```python
from xdpx.logger import log
value = x.square().mean()
log.add_summary('name', value)
```
运行x-viz命令时，会同时可视化加入的summary。目前只支持添加标量值并绘制其折线图，不支持添加矩阵并绘制其分布。

如果需要测量多步累积的统计指标，比如记忆网络在整个dev集上累积的访问次数分布等，可以使用
```python
from xdpx.logger import log
if not self.training:
    log.add_accumulative_summary(
        name='name',
        values=(
            tensor1.detach().cpu(),
            tensor2.detach().cpu(),
        ),
        reduce_fn=reduce_fn,
    )
```
reduce_fn输入每步的values组成的list，输出一个dict[str, float]，内容为{指标名：指标值}。

### 多进程pdb
在多进程环境下pdb无法直接使用，这里介绍相应的调试工具。

在GPU多卡训练中，可以使用：
```python
from xdpx.utils import pdb, distributed_utils
if distributed_utils.is_master(args):
    pdb.set_trace()
```
来添加断点。

在多进程数据处理或者其他通过multiprocess包手动创建的子进程中，需要在进入子进程之前导入内置的pdb包并手动设置进程数：
```python
from xdpx.utils import pdb;
pdb.set_nprocs(8)  # 假设有8个子进程
```
并且在需要添加断点的地方通过调用`pdb.set_trace()`来添加断点。


### 其他调试tips
💡  大的pytorch向量直接print只会显示一部分值，如果要比对不同的运行中某个向量值是否发生改变，可以用
```python
import hashlib
print(hashlib.sha1(tensor.detach().numpy()).hexdigest())
```
来查看向量的hash值。注意不能直接用`hash(tensor)`，否则数值完全相同的两个向量仍然会返回不同的结果。

💡  如果在GPU环境下报错，并且发现报错的那行不应该出错/和错误类型对不上，那么可能是由于CUDA的异步执行导致报错的行数不准，参见[官方文档](https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)；解决方法是程序启动时设定CUDA_LAUNCH_BLOCKING=1，就会报错在正确的地方。

💡  在PyCharm中调试时，可以配置“Script path”为xdpx/run.py，“Working directory”为当前路径，之后在“Parameters”中传入`x-train config.hjson`等指令即可。
