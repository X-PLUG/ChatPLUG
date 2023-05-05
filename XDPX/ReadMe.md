XDPX / 新地平线 / X-lab DeepQA Platform X

项目文档见：[语雀](https://yuque.antfin-inc.com/books/share/007ad4ad-d634-4013-a945-fe6e80d5f297?# 《XDPX训练框架》)

该项目主要支持**用于NLU的PyTorch代码（包括PAI在内）的跨环境训练**。有以下特点：

## 跨环境通用
- 支持所有环境中，所有路径直接配置OSS路径，和本地路径一起无差别使用，大文件传输显示进度条；
- 支持本地/PAI、单卡/多卡/分布式训练的无缝切换，支持自动开启docker-fusion等PAI高级选项；
- 兼容Tensorflow/Huggingface模型，可以直接读入/导出tf checkpoint/savedmodel。

## NLU前沿技术
- 支持稳定的混合精度训练、virtual batch size、对抗训练等多种高级训练方法和海量大文件处理能力；
- 集成语言模型预训练和微调上的多种改进方法和训练技巧；
- 集成多种轻量且强大的分类模型/匹配模型/蒸馏技术可用于模型线上部署。

## 高效率算法开发
- 可扩展性强，直接插拔各种pytorch模型和训练方法，可以扩展到多语言；
- 提交到PAI之前自动检查参数配置错误，包括路径是否存在、不同参数之间的冲突等，降低提交任务和排队的试错成本；
- 可以在一个配置文件内定义批量训练任务，一键启动超参数搜索、交叉验证等配置；
- 集成包括NaNDetector、Gradient Inspector在内的大量调试工具，测试全覆盖，方便debug。
