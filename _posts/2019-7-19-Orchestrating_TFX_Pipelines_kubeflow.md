---
layout: post
title:  "Orchestrating TFX Pipelines Kubeflow"
date:   2019-07-19 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第二十三篇。
        本文档详细介绍协调 TFX Pipelines。

## 正文
Kubeflow是一个开源ML平台，致力于使机器学习（ML）工作流在Kubernetes上的部署变得简单，可移植和可扩展。 Kubeflow Pipelines是Kubeflow平台的一部分，该平台支持在Kubeflow上组合和执行可重复的工作流，并结合了实验和基于笔记本的体验。 Kubernetes上的Kubeflow Pipelines服务包括托管的元数据存储，基于容器的编排引擎，笔记本服务器和UI，可帮助用户大规模开发，运行和管理复杂的ML管道。 Kubeflow Pipelines SDK允许以编程方式创建和共享组件，组成和管线。

有关大规模运行TFX的详细信息，请参见Kubeflow管道上的TFX示例。