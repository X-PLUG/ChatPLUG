# 内置的模型和训练方法
## 语言模型预训练
### 文档数据预训练
主分支已有RoBERTa和StructBert两种预训练方法。
#### 数据预处理
预训练数据格式：一行一个句子，文章与文章之间用一个空行隔开/一行一轮对话，session之间用一个空行隔开。
预处理分句脚本示例（假设原先是一行一篇文章）：
```python
import blingfire
with open(infile) as f, open(outfile, 'w') as fout:
    for line in f:
        fout.write(blingfire.text_to_sentences(line))
        fout.write('\n\n')
```

填充模式("break_mode")：

- 'none': break tokens into equally sized blocks (up to block_size)
- 'none_doc': similar to 'none' mode, but do not cross document boundaries
- 'complete': break tokens into blocks (up to block_size) such that blocks contains complete sentences. If a single sentences exceeds max length, it is truncated.
- 'complete_doc': similar to 'complete' mode, but do not cross document boundaries
- 'eos': each block contains one sentence (block_size is ignored)

不使用whole word mask预处理配置：
```json
{
    __parent__: [
        examples/pretrain/prepro/corpus // RoBERTa
        // examples/pretrain/prepro/structbert // StructBert
    ]
    data_source: '数据源路径'
    data_dir: '预处理后的数据路径'
    max_len: 128
    workers: 32 // 预处理并行的CPU进程数
  
    break_mode: complete
}
```
如果使用whole word mask，原始数据文件需要预先分词并将词用空格隔开。
使用whole word mask预处理配置：
```json
{
    __parent__: [
        examples/pretrain/prepro/corpus // RoBERTa
        // examples/pretrain/prepro/structbert // StructBert
    ]
    data_source: '数据源路径'
    data_dir: '预处理后的数据路径'
    max_len: 128
    workers: 32 // 预处理并行的CPU进程数
  
    break_mode: complete
    mask_whole_words: true
    loader: corpus_mix
}
```

执行预处理：`x-prepro config.hjson`

#### 训练配置
```json
{
    __parent__: [
        ${data_dir}/meta
        examples/pretrain/tasks/corpus
        examples/pretrain/params/default
        tests/sample_data/bert_config.json
    ]

    data_dir: '数据路径'
    save_dir: '保存路径'
    pretrained_model: '预训练模型路径'
    from_tf: false
}
```
Tips:

- 如果不希望载入预训练模型从头训练，需要显式指定`pretrained_model: null`
- bert_config默认是中文bert base，如果使用其他的BERT模型，需要注意更改bert_config.json的路径
- RoBERTa和StructBert训练部分的配置相同，程序通过继承预处理的配置来采用不同的训练模式
- 训练中有两种指定最大步数的方式，配置max_update或max_epoch，如果两者都配置了，会取其中的**最小步数**，如果需要以其中一个为准，可以指定另一个为null使其预设值失效。

启动训练：

- 本地训练：`x-train config.hjson`
- PAI训练：`bash pai.sh x-train config.hjson

`

### 弱监督数据预训练
这里的弱监督数据特指带弱监督标签的句子对。数据格式为"<text1>\t<text2>\t<label>\n"。

预处理配置（默认使用whole word mask，原始数据文件需要预先分词并用空格隔开）：
```json
{
    __parent__: [
        examples/pretrain/prepro/pair
    ]
    data_source: '数据源路径'
    data_dir: '预处理后的数据路径'
    max_len: 40
    workers: 32  // 预处理并行的CPU进程数
}
```
如果不使用whole word mask，则需要在配置中加入：
```json
    loader: pair
    mask_whole_words: false
```

执行预处理：`x-prepro config.hjson`

训练配置：
```json
{
    __parent__: [
        ${data_dir}/meta
        examples/pretrain/tasks/pair
        examples/pretrain/params/default
        tests/sample_data/bert_config.json
    ]

    data_dir: '数据路径'
    save_dir: '保存路径'
    pretrained_model: '预训练模型路径'
    from_tf: false
    major_metric: cls_auc
}
```

- 本地训练：`x-train config.hjson`
- PAI训练：`bash pai.sh x-train config.hjson`
## 微调
### 分类任务微调
预处理配置：
```json
{
    data_source: '数据源路径'
    data_dir: '预处理后的数据路径'
    workers: 4
    max_len: 40
  	
    data_files: ['train.txt', 'dev.txt']
    processor: bert_single
    loader: single
    target_type: text
    special_targets: ['unknown']
    tokenizer: bert
    lower: true
    vocab_file: tests/sample_data/bert_vocab.txt
}
```
执行预处理：`x-prepro config.hjson`

训练配置：
```json
{
    __parent__: [
        ${data_dir}/meta
        tests/sample_data/bert_config.json
    ]

    data_dir: '数据路径'
    save_dir: '保存路径'
    pretrained_model: '预训练模型路径'
    from_tf: false
    save_best_only: true
    
    loss: cross_entropy
    model: bert_classification    
    major_metric: f1
    learning_rate: 2e-5
    batch_size: 64
    max_epoch: 5
    
    optimizer: adam
        adam_eps: 1e-6
        clip_norm: 5.0
        weight_decay: 0.01
        lr_scheduler: one_cycle
        warmup_steps: 0.1
    
    output_dropout_prob: 0.1
    
    eval_interval: @{batch_size:64}/50
    log_interval: @{batch_size:64}/10
}
```

- 本地训练：`x-train config.hjson`
- PAI训练：`bash pai.sh x-train config.hjson`
### 匹配任务微调
预处理配置：
```json
{
    data_source: '数据源路径'
    data_dir: '预处理后的数据路径'
    workers: 4
    max_len: 40
  
    data_files: ['train.txt', 'dev.txt']
    tokenizer: bert
    lower: true
    processor: bert_pair
    loader: pair
    vocab_file: tests/sample_data/bert_vocab.txt
    target_map_file: tests/sample_data/binary_target_map.txt
}
```
执行预处理：`x-prepro config.hjson`

训练配置：除了major_metric可能需要根据任务设置成auc等二分类指标以外，其他和上面“分类任务微调”的训练配置大体相同。

### Bert高级微调配置

- layer_lr_decay：逐层学习率衰减，默认1.0（不衰减）
- output_dropout_prob：在用于分类的线性层之前加入dropout，默认0.1
- load_cls_weights：加载预训练模型中用于分类的线性层权重（默认为false，即随机初始化分类权重）
- top_lr_ratio：对bert顶端的层（在标准架构中是pooler和用于分类预测的线性层）采用更大的学习率的倍数
- bert_wd：对bert embedding和encoder部分设置单独的weight_decay    
### 混合粒度模型
> 注意：使用混合粒度模型之前需要对数据按词进行分词；目前混合粒度模型仅用于单句分类任务

预处理配置
```json
{
    data_source: '数据源路径'
    data_dir: '预处理后的数据路径'
    workers: 4
    max_len: 40

    data_files: ['train.txt', 'dev.txt']
    processor: bert_single_mix
    loader: single_mix
    with_words: true
    target_type: text
    special_targets: ['unknown']
    tokenizer: bert
    lower: true
    vocab_file: tests/sample_data/bert_vocab.txt
    pretrained_embeddings: '预训练词向量路径'
    min_word_count: 5
    max_word_vocab: 20000
}
```
执行预处理：`x-prepro config.hjson`

训练配置
```json
{
    __parent__: [
        ${data_dir}/meta
        tests/sample_data/bert_config.json
    ]
    data_dir: '数据路径'
    save_dir: '保存路径'
    pretrained_model: '预训练模型路径'
    major_metric: f1
    save_best_only: true

    loss: cross_entropy
    model: bert_mix
    task_type: classification
    bert_seq_layer: -1
    char_hidden_size: 1024
    concat_cls: true
    lr_scheduler: one_cycle
    warmup_steps: 0.1
    eval_interval: @{batch_size:64}/50
    log_interval: @{batch_size:64}/10
    
    // 由于混合粒度模型较不稳定，下面这些参数调整后结果可能有较大差别
    max_epoch: 5
    warmup_steps: 250
    batch_size: 32
    learning_rate: 5e-5
    layer_lr_decay: 0.95
    top_lr_ratio: 50
    weight_decay: 0.01
    bert_wd: 0.0
    output_dropout_prob: 0.0
}
```

- 本地训练：`x-train config.hjson`
- PAI训练：`bash pai.sh x-train config.hjson`
## 模型蒸馏
### 经典蒸馏
这里介绍静态蒸馏的流程。静态蒸馏常用在(1)只用到最后一层的KL loss;(2)蒸馏所用的数据量不大的场景，过程是先把教师模型的logits导出存下来，再读入logits对学生模型进行蒸馏训练。静态蒸馏常用在学生模型为非bert模型的场景（此时通常只能用最后一层loss）。

导出蒸馏logits：`x-pred config.hjson`
```json
{
    save_dir: 'teacher模型训练保存的路径'
    __def__: {
        data_source: '蒸馏用的原始数据文件路径'
        out_dir: '导出logits的数据路径'
    }
    predict_file_map: {
        '${data_source}/train.txt': '${out_dir}/train.txt'
        '${data_source}/dev.txt': '${out_dir}/dev.txt'
    }
    distill: true
}
```
对logits预处理 `x-prepro config.hjson`
```json
{
    data_source: '上一步的out_dir'
    data_dir: '预处理后的数据路径'

    processor: single_with_logits // 或者pair_with_logits
    loader: single_with_logits // 或者pair_with_logits

    // 其他通常预处理的配置
    // ...
}
```
蒸馏训练 `x-train config.hjson`
```json
{
    __parent__: [
        ${data_dir}/meta
    ]

    data_dir: '上一步的data_dir'
    save_dir: '训练保存的路径'

    loss: static_distill
    gold_loss_lambda: 0.5  // 真实标签的cross_entropy占比多少
    
    // 以cls_cnn为例
    model: cls_cnn
        hidden_size: 200
        encoder_layers: 3
    
    // 其他通常训练配置，如learning_rate等
    // ...
}
```
### TinyBert
```json
{
    __parent__: [
        ${data_dir}/meta
        examples/pretrain/params/default
        examples/pretrain/tasks/tinybert
    ]
    data_dir: '数据路径'
    save_dir: '保存路径'
    teacher_pretrained_model: 'teacher路径'
    teacher_config_file: 'tests/sample_data/teacher_config.json'
    student_config_file: 'tests/sample_data/student_config.json'
    learning_rate: 1e-4
}
```

- 本地训练：`x-train config.hjson`
- PAI训练：`bash pai.sh x-train config.hjson`

导出TinyBert模型：
```json
{
    "save_dir": "模型路径"
    "out_dir": "导出路径"
    "inject_env": true
    "output_eas_model": true
}
```
执行导出：`python scripts/export_pyrankmodel.py config.hjson`
### MiniLM
命令：`x-train config.hjson`
```json
{
    __parent__: [
        ${data_dir}/meta
        examples/pretrain/params/default
        examples/pretrain/tasks/minilm
        tests/sample_data/teacher_config.json >> teacher_
        tests/sample_data/student_config.json
    ]
    __ignore__: teacher_vocab_size

    data_dir: '数据路径'
    save_dir: '保存路径'
    teacher_pretrained_model: 'teacher路径'
    pretrained_model: null
    learning_rate: 1e-4
}

```

## 模型裁剪
命令：`x-script prune_heads config.hjson`
实现了[http://arxiv.org/abs/1905.10650](http://arxiv.org/abs/1905.10650)中对Transformer head的裁剪。
参考配置：
```json
{
  config: ./train // 训练配置的路径，.hjson的后缀可以省略
  train_subset: train  // 计算importance时使用的数据分片
  train_steps: 10  // 计算importance时采用的steps
  prune_config: {  // 这个示例中在第0层和最后一层都要裁掉两个head
    -1: 2
    0: 2
  }
  save_path: bert_pruned/model.pt
}
```
# 参数配置指南
通常的参数配置就是一个配置文件中包含一个参数名到参数值的映射Dict[str, Any]。以下内容只是为了配置提效，不是必须使用的，完全可以跳过而不影响使用。

关于Hjson配置文件的语法可以参考[官方文档](https://hjson.github.io/)，同时在IDE中安装hjson语法插件可以在没有额外学习成本的情况下直接看出绝大部份语法的问题（比如vscode/sublime/vim中的hjson插件；Pycharm目前没有hjson插件，建议用其他编辑器单独管理配置目录）。如果对Hjson的语法不熟悉不确定，也不想了解太多细节，那么把配置文件直接写成标准的JSON格式即可（hjson同样兼容标准的JSON格式）。所有配置文件也可以直接传入.json后缀的json文件。

如果想要在正式启动之前查看（预览）当前配置解析后的所有参数的情况，在任意本地命令之后加上`--dry`并执行即可，比如`x-train config.hjson --dry`。

下面介绍一些可以选用的高级特性。
### 循环
一个配置文件内如果是一个dict，则是一份配置；如果是一个list[dict]，则会依次串行执行list中的每一个配置。
### 引用和宏定义

- 可以用`${param_name}` 引用其他参数的值或宏定义的值，引用后统一变成str类型。
- 使用`__def__`创建临时的宏定义，例如`__def__: {task_name: 'cainiao'}`。
- 参数值的引用解析在最后发生，因此可以在任意层次定义或引用；只有一个例外：继承的**路径**中引用的名字必须在其上层先定义，否则无法解析。
- 如果要引用int/float值，需要在参数定义代码加入`post_process=xdpx.options.parse_relative`，并且用类似`eval_interval: @{batch_size:64}/50`的方式来引用，这段引用解析为“当batch_size为64时，eval_interval为50”，如果batch_size为32，则eval_interval为100，其他以此类推，具体可以参考`parse_relative`的实现。
### 继承
`__parent__: `后跟（一个或多个）配置文件路径或者配置本身，则可以“继承”这些配置。当前已定义的值会覆盖被继承的值。

案例一
```json
{
  __parent__: ./parent_config.hjson
}
```
继承了同一目录下的parent_config.hjson中的内容。基于当前文件的相对路径需要以./开头，否则跟随系统默认的相对路径。配置文件路径末尾的“.hjson”可以省略；如果继承的是.json文件，则不能省略后缀名。

案例二
```json
{
  __parent__: ./parent_config
  auto_suffix: true
  overwrite: false
  
  lr_scheduler: constant
  __ignore__: ['warmup_steps']
}
```

- 由于auto_suffix和overwrite是互斥的（参考上文"使用-训练"中的内容），假设parent_config中overwrite为true，这里需要显式将其设置成false才能让auto_suffix生效；
- 如果parent_config中lr_scheduler是warmup_constant，在当前被配置为constant，就失去了warmup_steps这个参数，因此使用__ignore__将其忽略。

案例三
```json
{
  __parent__: [
    ./config1
    oss://bucket/path/config
    ./config2.json
  ]
  learning_rate: 1e-4
}
```
继承了多个配置。

案例四
```json
{
	__parent__: [
    ./train_config
    [
    	{learning_rate: 1e-5}
      {learning_rate: 2e-5}
    ]
		[
      {batch_size: 32}
      {batch_size: 64}
		]
  ]
	save_dir: outputs/run.lr${learning_rate}.bs${batch_size}
}
```
如果继承的不是一个文件路径，而是list[dict]，则相当于**插入了一个循环**，会进行笛卡尔积扩展。这个案例产生了四个配置文件，对learning_rate和batch_size进行了超参数搜索。注意这里文件名也引用了参数值，使得每次训练有不同的文件名。循环执行的顺序是上层循环视作内层循环，比如这个例子里执行的顺序是{1e-5, 32}, {2e-5, 32}, {1e-5, 64}, {2e-5, 64}。

案例五
```json
{
  __parent__: [
    ./train_config
    [
      {__def__: {valid_id: 0}}
      {__def__: {valid_id: 1}}
      {__def__: {valid_id: 2}}
      {__def__: {valid_id: 3}}
    ]
  ]
  train_subset: split*
  valid_subset: split${valid_id}
  exclude_valid_from_train: true
  save_dir: outputs/run.${valid_id}
}
```
同样利用继承内插入循环进行交叉验证。可以参考[训练数据分片处理](#QmoDp)。

案例六
```json
{
  __parent__: [
    ./config1
    ./teacher_bert_config.json >> teacher_
    ./bert_config.json
  ]
}
```
teacher_bert_config.jso中的配置都会有一个teacher_的前缀。在蒸馏等场景中比较常用，给teacher和student独立的配置文件。

除了上述案例以外，用了大量高级特性的真实配置范例可以参考examples/alime_benchmark/prepro.hjson和examples/alime_benchmark/train.hjson。

