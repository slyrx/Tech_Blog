---
layout: post
title:  "The SchemaGen TFX Pipeline Component"
date:   2019-07-09 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第十三篇。
        本文档详细介绍了 SchemaGen 组件的作用。

## 目录
+ SchemaGen 和 TensorFlow 数据验证
+ 使用 SchemaGen 组件

## 正文

某些TFX组件使用对输入数据的描述（称为模式）。 模式是schema.proto的实例。 它可以指定要素值的数据类型，是否在所有示例中都必须存在要素，允许的值范围以及其他属性。 SchemaGen管道组件将通过从训练数据中推断类型，类别和范围来自动生成模式。

消耗：来自StatisticsGen组件的统计信息
发射：数据架构原型
这是模式原型的摘录：

```
...
feature {
  name: "age"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_fraction: 1
    min_count: 1
  }
}
feature {
  name: "capital-gain"
  value_count {
    min: 1
    max: 1
  }
  type: FLOAT
  presence {
    min_fraction: 1
    min_count: 1
  }
}
...
```

以下TFX库使用该架构：

TensorFlow数据验证
TensorFlow转换
TensorFlow模型分析
在典型的TFX管道中，SchemaGen会生成一个架构，其他架构组件会使用该架构。

注意：自动生成的架构是尽力而为的，仅尝试推断数据的基本属性。 期望开发人员根据需要进行审查和修改。
SchemaGen和TensorFlow数据验证
SchemaGen广泛使用TensorFlow数据验证来推断模式。

使用SchemaGen组件
SchemaGen管道组件通常非常易于部署，几乎不需要自定义。 典型的代码如下所示：

```
from tfx import components

...

infer_schema = components.SchemaGen(
    statistics=compute_training_stats.outputs['statistics'])
```
