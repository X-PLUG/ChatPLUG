# FAQ

1. 如果不想加载预训练模型如何让它随机初始化？

需要显式指定`pretrained_model: null`

2. 本地运行出现莫名其妙的像是不兼容的硬错误/无法跑通本地测试/改动了代码但是没生效？

如果clone过多次XDPX项目并且放在不同的地址，那么类似`x-train`、`x-prepro`这样的命令很可能链接到另一个项目地址，有两种解决方案：

   1. 此时在当前路径下重新执行一遍安装`pip install -e .`，把命令重新链接到当前路径
   2. 如果要同时保留两个路径，需要针对每个不同路径创建对应的虚拟环境，并将各个路径下的代码安装到各自的虚拟环境中

3. 为什么可视化不用Tensorboard？
   1. （最主要的原因）在对比多个实验时，XDPX中的可视化把横坐标统一成了「训练的样本数」而不是步数，实现中是通过实际batch size对参考的实验的横坐标进行了重新计算，这样对比的才是模型真正的收敛速度；而tensorboard不支持这样的功能，它里面最接近的指标是训练时间，而这个受到训练配置的影响比较大（GPU的型号，用的卡数等等），无法准确地对比模型的收敛情况；
   2. PAI训练时，tensorboard需要单独起一个PAI任务才能看到，而XDPX里的可视化存成PNG，可以直接在OSS Browser中打开
   3. XDPX可视化中加入了一些标记，比如最优值小绿点和标记epoch的红色分界线，可以更好理解优化中的行为，比如进入一个新的epoch了以后训练集的指标可能会有一个阶跃式的提升等等。

4. 随机性是怎么控制的？
   1. 模型参数初始化：由seed决定。
   2. data shuffling: 根据seed, epoch组合成的独立种子shuffle，不受其他影响。
   3. MLM预训练中的[MASK]: 根据seed, epoch, sample_id组合成的种子逐条样本生成，具体某条样本的[MASK]分布只和seed和epoch有关。
   4. 训练步骤中的随机性，例如dropout/对抗训练噪声：根据seed和num_updates决定，导数累积、分布式训练都会影响随机性，如果需要完全复现实验需要有同样的导数累积和分布式训练设定。

注意绝大多数情况下如果随机种子固定，那么数据、代码、配置、环境相同的训练，重复多次以后结果和得到的模型应当完全相同。但如果GPU环境不同（如CUDA8/CUDA9），那么结果很可能会不一致。

5. 为什么训练时汇报的dev集分数和x-eval或者x-pred计算的不一样？可以从什么角度debug？
   1. 检查训练时是否配置了max_eval_steps，如果配置了那么在x-eval中也应当配置；
   2. 检查模型是否完全读取，log中是否有报出missing keys, mismatched keys等模型未完全加载的情况；
   3. 检查训练时用的valid_subset和eval时用的是否为同一个（尤其是训练时没有用"dev"作为验证集的场景）
   4. 检查模型的padding是否有问题，由于多卡训练中每张卡平分了batch size，验证时不会分割，如果padding有问题就会导致不一致，可以试试batch size=1和等于一个很大的数，结果是否一致；
   5. 检查训练时的代码和验证时是否有变化，可以在log中找到对应的git commit，如果有不在git中的改动，可以把<save_dir>/snapshot.zip解压，里面保存了训练时实际用的代码备份；

6. 分布式训练如果遇到NCCL INFO Call to connect returned Connection refused该如何处理？

     串联多个任务时需要init-destroy多次NCCL，而docker fusion目前不支持init多次，因此关闭docker fusion或者一次提交只执行一个任务即可。

7. 有什么方法可以知道哪一步模型是最优的模型？

    (1) 运行`x-script aggregate_results ${save_dir}`可以看到最优的步数
    (2) 配置时不配置具体步数，直接用 ${save_dir}/<best>指代（目前训练中的pretrained_model参数、验证/预测中的checkpoint参数都支持这么配置）
    (3) 在valid.log.tsv里可以手动找到最优的步数（在评测指标和训练时指定的major_metric不一致时，可以用这个手动方法）

8. 如果训练时没有开启save_best_only或者save_last_only，但训练后发现每一步都保存模型，占用的磁盘空间太大了，怎么办？

    可以使用`x-script clean_checkpoints ${save_dir}`来递归清除指定路径下的XDPX训练目录里除了dev集最优的checkpoint和最新的checkpoint以外的所有模型文件，释放存储空间。
