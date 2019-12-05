---
layout: post
title:  "TensorFlow Transform"
date:   2019-07-24 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第二十八篇。
        本文档介绍转换库的情况。


## 目录
+ 安装
+ + 依赖库
+ 兼容版本
+ 问题

## 正文
TensorFlow Transform是一个用于使用TensorFlow预处理数据的库。 tf.Transform对于需要完全通过的数据很有用，例如：

通过平均值和标准偏差对输入值进行归一化。
通过在所有输入值上生成词汇表，将字符串转换为整数。
通过根据观察到的数据分布将浮点数分配给存储桶，将浮点数转换为整数。
TensorFlow内置了对单个示例或一批示例进行操作的支持。 tf.Transform扩展了这些功能，以支持对示例数据的全过程。

tf.Transform的输出被导出为TensorFlow图以用于训练和服务。在训练和服务过程中使用相同的图表可以防止偏斜，因为在两个阶段都应用了相同的转换。

有关tf.Transform的介绍，请参见TFX Dev Summit关于TFX（链接）的tf.Transform部分。

警告：tf.Transform在1.0版之前可能向后不兼容。
安装
tensorflow-transform PyPI软件包是推荐的安装tf.Transform的方法：

```
pip install tensorflow-transform
```

依存关系
tf.Transform需要TensorFlow但不依赖于tensorflow PyPI包。 有关说明，请参见TensorFlow安装指南。

运行分布式分析需要Apache Beam。 默认情况下，Apache Beam在本地模式下运行，但也可以使用Google Cloud Dataflow在分布式模式下运行。 tf.Transform设计为可扩展为其他Apache Beam运行器。

兼容版本
下表是彼此兼容的tf.Transform软件包版本。 这是由我们的测试框架确定的，但其他未经测试的组合也可能有效。


tensorflow-transform	tensorflow	apache-beam[gcp]
GitHub master	nightly (1.x/2.x)	2.16.0
0.15.0	1.15 / 2.0	2.16.0
0.14.0	1.14	2.14.0
0.13.0	1.13	2.11.0
0.12.0	1.12	2.10.0
0.11.0	1.11	2.8.0
0.9.0	1.9	2.6.0
0.8.0	1.8	2.5.0
0.6.0	1.6	2.4.0
0.5.0	1.5	2.3.0
0.4.0	1.4	2.2.0
0.3.1	1.3	2.1.1
0.3.0	1.3	2.1.1
0.1.10	1.0	2.0.0

问题
请使用tensorflow-transform标签指示有关使用tf.Transform到Stack Overflow的任何问题。