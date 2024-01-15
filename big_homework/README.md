## 2023年研究生机器学习课程项目

这部分主要包括 Transformer 模型 encoder+decoder 自回归预测模型。

### 训练模型
可以调整 `train.sh` 文件的变量`length`为96或336来控制训练短程模型或是长程模型。
```shell
sh train.sh
```

### 评估模型和绘制图像
注意不同模型需要简单调整模型保存位置的`exp`变量。
```shell
sh evaluate.sh
```

### 文件说明
`models\transformer_model.py`为 transformer 模型；

`cmd.py` 包括训练和评估的核心代码，包括loss计算和反向传播等；

`train.py` 和 `evaluate.py` 分别包括加载数据，训练和评估的代码。

`pre_process.py`是对数据进行预处理的代码；

`run.py`是设定参数，包括随机种子等参数的执行脚本。
