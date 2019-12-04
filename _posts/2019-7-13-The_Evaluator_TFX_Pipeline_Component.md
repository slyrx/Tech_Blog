---
layout: post
title:  "The Evaluator TFX Pipeline Component"
date:   2019-07-13 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第十六篇。
        本文档详细介绍了 Evaluator 组件的作用。

## 目录
+ 评估器和 TensorFlow 模型分析
+ 使用评估器组件

## 正文
Evaluator TFX管道组件对模型的训练结果进行深入分析，以帮助您了解模型如何对数据子集执行。

消耗：来自Trainer的EvalSavedModel
发射：对ML元数据的分析结果
评估器和TensorFlow模型分析
评估程序利用TensorFlow模型分析库执行分析，然后使用Apache Beam进行可伸缩处理。

使用评估器组件
评估程序管道组件通常非常易于部署，几乎不需要自定义，因为所有工作都是由评估程序TFX组件完成的。 典型的代码如下所示：

```
from tfx import components
import tensorflow_model_analysis as tfma

...

# For TFMA evaluation
taxi_eval_spec = [
    tfma.SingleSliceSpec(),
    tfma.SingleSliceSpec(columns=['trip_start_hour'])
]

model_analyzer = components.Evaluator(
      examples=examples_gen.outputs['examples'],
      feature_slicing_spec=taxi_eval_spec,
      model_exports=trainer.outputs['model'],
      fairness_indicator_thresholds = [0.25, 0.5, 0.75]
      )
```
