---
layout: post
title:  "The ExampleValidator TFX Pipeline Component"
date:   2019-07-10 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第十四篇。
        本文档详细介绍了 ExampleValidator 组件的作用。

## 目录
+ 例子验证器和 TensorFlow 数据验证
+ 使用例子验证器组件


ExampleValidator管道组件可识别训练和服务数据中的异常。它可以检测数据中不同类别的异常。例如，它可以：

通过将数据统计信息与编码用户期望的模式进行比较来执行有效性检查
通过比较培训和服务数据来检测培训服务偏斜。
通过查看一系列数据来检测数据漂移。
ExampleValidator管道组件通过将StatisticsGen管道组件所计算的数据统计信息与模式进行比较，来识别示例数据中的任何异常。推断的架构将输入数据期望满足的属性进行编码，并且开发人员可以对其进行修改。

消耗：来自SchemaGen组件的架构和来自StatisticsGen组件的统计信息。
排放：验证结果
ExampleValidator和TensorFlow数据验证
ExampleValidator广泛使用TensorFlow数据验证来验证您的输入数据。

使用ExampleValidator组件
ExampleValidator管道组件通常非常易于部署，几乎不需要自定义。典型的代码如下所示：

```
from tfx import components

...

validate_stats = components.ExampleValidator(
      statistics=compute_eval_stats.outputs['statistics'],
      schema=infer_schema.outputs['schema']
      )
```