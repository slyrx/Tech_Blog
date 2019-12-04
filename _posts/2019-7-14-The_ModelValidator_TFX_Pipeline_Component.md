---
layout: post
title:  "The ModelValidator TFX Pipeline Component"
date:   2019-07-14 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第十八篇。
        本文档详细介绍了 ModelValidator 组件的作用。


## 目录
+ 使用模型验证组件

## 正文
ModelValidator TFX管道组件可帮助您验证导出的模型，以确保它们“足够好”可以投入生产。

ModelValidator将新模型与基线（例如当前服务的模型）进行比较，以确定它们相对于基线是否“足够好”。通过评估数据集上的两个模型（例如，保留数据或黄金数据集）并计算其在指标上的表现（例如，AUC，损失）来做到这一点。如果新模型的指标相对于基线模型满足用户指定的标准（例如，AUC不低于），则模型“有福”（标记为“好”），从而向Pusher表示可以将模型推向生产环境。

消耗：来自SchemaGen组件的架构和来自StatisticsGen组件的统计信息。
Emits：对TensorFlow元数据的验证结果
使用ModelValidator组件
由于所有工作都是由ModelValidator TFX组件完成的，因此ModelValidator管道组件通常非常易于部署并且几乎不需要自定义。典型的代码如下所示：

```
from tfx import components
import tensorflow_model_analysis as tfma

...

# For model validation
taxi_mv_spec = [tfma.SingleSliceSpec()]

model_validator = components.ModelValidator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'])
```