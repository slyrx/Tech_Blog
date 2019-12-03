---
layout: post
title:  "The StatisticsGen TFX Pipeline Component"
date:   2019-07-08 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第十二篇。
        本文档详细介绍了 StatisticsGen 组件的作用。

## 目录
+ StatisticsGen 和 TensorFlow 
+ 使用 StatisticsGen 组件

StatisticsGen TFX管道组件会生成有关训练和服务数据的功能统计信息，其他管道组件可以使用这些统计信息。 StatisticsGen使用Beam缩放到大型数据集。

消耗：由ExampleGen管道组件创建的数据集。
发射：数据集统计。
StatisticsGen和TensorFlow数据验证
StatisticsGen广泛使用TensorFlow数据验证从数据集中生成统计信息。

使用StatsGen组件
StatisticsGen管道组件通常非常易于部署，几乎不需要自定义。 典型的代码如下所示：

```
from tfx import components

...

compute_eval_stats = components.StatisticsGen(
      examples=example_gen.outputs['examples'],
      name='compute-eval-stats'
      )
```