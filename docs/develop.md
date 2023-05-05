# 开发指南

## 加入新特性

### 新的模型结构

- 加入到xdpx/models/中，继承xdpx.models.Model类；
- 如果是基于bert的模型，继承xdpx.models.bert.Bert。
- 如果需要导入huggingface/transformers的相关内容，应当从模型中的离线版本导入，例如`from xdpx.modules.thirdparty.transformers.modeling_bert import BertModel, BertConfig`
- 如果模型包含预训练embeddings，或用于对抗训练任务，需要实现Model类中的get_embeddings方法
### 新的损失函数/指标

- 加入到xpdx/losses中，继承xdpx.losses.Loss类。
- forward返回：
   - loss 后续用来backward的Tensor
   - sample_size：loss的由多少个样本组成，分布式训练时用于各个节点的加权；注意如果loss直接是所有target求出的loss的均值，那么这个值需要设置为1；如果loss是总和，那么这个值等于target的数量
   - logging_output：python native dict，注意所有tensor必须要.tolist()之后再传入
- aggregate_logging_outputs
   - 输入
      - logging_outputs: List[Dict[str, Any]]，所有分布式节点forward方法返回的logging_output的列表
      - sample_size：int，所有分布式节点forward方法返回的sample_size的和
   - 返回Dict[str, Union[float, int]]，根据所有节点的输出计算出的整体的log指标，会进入日志文件；注意sample_size和ntokens是默认需要传出的项：
```python
def aggregate_logging_outputs(self, logging_outputs, sample_size, max_count=None):
    """Aggregate logging outputs from data parallel training."""

    agg_output = {
        'sample_size': sample_size,
    }
    if 'ntokens' in logging_outputs[0]:
        ntokens = sum(log['ntokens'] for log in logging_outputs)
        agg_output['ntokens'] = ntokens
    # 其他需要计算的指标
    ...
    return agg_output
```

### 新的数据结构/格式
XDPX中的数据链路为

- [预处理 Parser.parse_line->Loader.parse->(Loader.merge->)Processor.numerize] 
- -> [训练Processor.collate -> Model.forward->Loss.forward]

内部流通的数据格式示例：

- 原始数据 取出每条样本[1]-> Parser.parse_line[2] -> ['数据字段1', '数据字段2', '标签']
- Loader[4]
   - Loader.parse[3] -> {'id': 1234, 'key1': ['数据','字段','1'], 'key2': ['数据','字段','2']}
   - Loader.parse_target -> {'target': '标签'}
- Processor.numerize[5] -> {'id': 1234, 'key1': [10, 111, 65], 'key2': [10, 111, 66], 'target': 0}
- Processor.collate[6] -> ...

[1] 如果parser配置为json，则假设数据是一个json文件并且主体是一个list，每个元素是一条样本；否则假设数据是基于行的，每行一条样本。
[2] 现有的Parser选项(none/csv/json/jsonl）应当可以通过配置来适配主流的数据格式，其他格式可以继承xdpx.loaders.parsers.Parser类实现一个新的parser
[3] `def parse(cls, contents: List[str], _id) -> dict`，这一步如果workers>1会在多进程内执行，通过cls.tokenize(text)可以调用Tokenizer，输出分词后的样本；如果这条样本不是有效的样本，可以在这一步return None把它过滤掉
[4] Loader如果有变动，还需要实现的方法：

- 实现def length(self, sample: dict) -> int，以便统计数据中超出最大长度的比例
- 如果数据有中标签的字段，实现num_sections和num_targets，分别为总字段数和标签占用的字段数，这样在预测无标签样本时，如果发现没有标签字段，则不会调用parse_target
- 如果需要跨行整合，实现def merge(self, samples: List[dict]) -> List[dict]

[5] 把str类型的token、target等根据self.dictionary、self.target_map转为对应的int类型的id，并限制最大序列长度
[6] 输入一个batch: List[dict]（里面每一个dict就是numerize的输出），做padding后给出模型的输入，以PairProcessor的collate方法为例：
```python
def collate(self, samples: List[dict]):
    tokens1 = torch.LongTensor(self.pad([sample['tokens1'] for sample in samples]))
    tokens2 = torch.LongTensor(self.pad([sample['tokens2'] for sample in samples]))
    mask1 = torch.ne(tokens1, self.args.pad_index)
    mask2 = torch.ne(tokens2, self.args.pad_index)
    batch = {
        'id': [sample['id'] for sample in samples],
        'net_input': {
            'tokens1': tokens1,
            'tokens2': tokens2,
            'mask1': mask1,
            'mask2': mask2,
        },
        'ntokens': tokens1.numel() + tokens2.numel(),
    }
    try:
        target = torch.LongTensor([sample['target'] for sample in samples])
        batch.update({'target': target})
    except KeyError:
        ...
    return batch
```
其中'net_input'中的就是直接输入模型forward方法的数据：logits = `model(**batch['net_input'])`，ntokens会用于训练速度的计算；'id'一般仅在debug或数据处理时使用，因此不需要转换成向量。
### 新的训练方式

- 新的训练方式：继承xdpx.tasks.Task类，一般实现的是train_step和valid_step方法。
- 新的数据采样方式：继承xdpx.datasets中的torch.utils.data.Dataset或者torch.utils.data.Sampler

注意：基类中的register方法，如果是staticmethod，那么继承后也应该使用staticmethod；如果是classmethod，那么继承后也应该使用classmethod。请参考其他已经实现的子类，如果他们调用了基类的注册方法`super().register(options)`，那么你实现的子类也应该调用。

## 单元测试
运行所有单元测试：`python -m unittest discover tests/unit_tests`
运行单个单元测试：`python -m unittest tests.unit_tests.test_xxx`
运行全流程测试：`bash tests/test_pipeline.sh`
在PAI上运行全流程测试：`bash pai.sh test`
在PAI上运行benchmark测试：`bash pai.sh benchmark`
在本地执行代码覆盖率测试：
```shell
coverage erase
coverage run tests/test_legacy.py test
coverage combine
coverage html
```
报告导出在user/htmlcov/。
在PAI上执行代码覆盖率测试：`bash pai.sh coverage oss://.../cov.zip`

在新代码合并master之前，需要通过单元测试，并且在各个环境验证全流程测试是否通过、benchmark测试是否可以回归之前的结果。

## 新参数定义
TBD；这块文档写起来没有代码本身直观，可以参考已有的参数定义方式或参考xdpx.options中的源码。


## 已知的局限

- AutoML在PAI上运行时由于每一个训练Snapshot都要存到云端，当前速度较慢；未来结合本地缓存+定期同步会大大加速；
- Tensorflow仅支持1.x版本，暂不支持和2.x版本的互相转换。
