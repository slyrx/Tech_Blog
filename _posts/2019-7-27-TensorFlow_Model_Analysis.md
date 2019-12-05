---
layout: post
title:  "TensorFlow Model Analysis"
date:   2019-07-27 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第三十一篇。
        本文档介绍模型分析库的使用。

## 目录
+ 安装
+ + 依赖库
+ 开始操作
+ 兼容版本
+ 问题

## 正文
TensorFlow模型分析（TFMA）是用于评估TensorFlow模型的库。 它允许用户使用他们的培训器中定义的相同指标，以分布式方式评估他们在大量数据上的模型。 这些指标可以在不同的数据切片上计算并在Jupyter笔记本中可视化。

![png](https://www.tensorflow.org/tfx/model_analysis/images/tfma-slicing-metrics-browser.gif)

安装
推荐的安装TFMA的方法是使用PyPI软件包：

```
pip install tensorflow-model-analysis
```

当前，TFMA要求安装TensorFlow，但对TensorFlow PyPI软件包没有明确的依赖性。 有关说明，请参见TensorFlow安装指南。

要在Jupyter Notebook中启用TFMA可视化：

```
  jupyter nbextension enable --py widgetsnbextension
  jupyter nbextension install --py --symlink tensorflow_model_analysis
  jupyter nbextension enable --py tensorflow_model_analysis
```

注意：如果您的主目录中已经安装了Jupyter笔记本，请在这些命令中添加--user。 如果Jupyter作为root用户安装或使用虚拟环境安装，则可能需要参数--sys-prefix。
依存关系
运行分布式分析需要Apache Beam。 默认情况下，Apache Beam在本地模式下运行，但也可以使用Google Cloud Dataflow在分布式模式下运行。 TFMA设计为可扩展为其他Apache Beam运行器。

入门
有关使用TFMA的说明，请参阅入门指南。

兼容版本
下表是彼此兼容的TFMA软件包版本。 这是由我们的测试框架确定的，但其他未经测试的组合也可能有效。

tensorflow-model-analysis	tensorflow	apache-beam[gcp]
GitHub master	nightly (1.x/2.x)	2.16.0
0.15.4	1.15 / 2.0	2.16.0
0.15.3	1.15 / 2.0	2.16.0
0.15.2	1.15 / 2.0	2.16.0
0.15.1	1.15 / 2.0	2.16.0
0.15.0	1.15	2.16.0
0.14.0	1.14	2.14.0
0.13.1	1.13	2.11.0
0.13.0	1.13	2.11.0
0.12.1	1.12	2.10.0
0.12.0	1.12	2.10.0
0.11.0	1.11	2.8.0
0.9.2	1.9	2.6.0
0.9.1	1.10	2.6.0
0.9.0	1.9	2.5.0
0.6.0	1.6	2.4.0

问题
请使用tensorflow-model-analysis标签将有关使用TFMA的任何问题定向到Stack Overflow。
    

