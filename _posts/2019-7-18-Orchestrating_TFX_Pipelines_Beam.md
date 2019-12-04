---
layout: post
title:  "Orchestrating TFX Pipelines Beam"
date:   2019-07-18 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第二十二篇。
        本文档详细介绍协调 TFX Pipelines。

## 正文

一些TFX组件依赖Beam进行分布式数据处理。 另外，TFX可以使用Apache Beam来协调和执行管道DAG。 Beam Orchestrator使用的BeamRunner与用于组件数据处理的BeamRunner不同。 使用默认的DirectRunner设置，Beam Orchestrator可以用于本地调试，而不会产生额外的Airflow或Kubeflow依赖关系，从而简化了系统配置。

有关详细信息，请参见Beam上的TFX示例。