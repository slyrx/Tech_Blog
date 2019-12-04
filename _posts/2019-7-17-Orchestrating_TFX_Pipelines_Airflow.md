---
layout: post
title:  "Orchestrating TFX Pipelines Airflow"
date:   2019-07-17 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第二十一篇。
        本文档详细介绍协调 TFX Pipelines。


## Apache Airflow
Apache Airflow是一个以编程方式编写，计划和监视工作流的平台。 TFX使用Airflow将工作流编写为任务的有向无环图（DAG）。 Airflow计划程序在遵循指定的依存关系的同时在一组工作线程上执行任务。 丰富的命令行实用程序使在DAG上执行复杂的手术变得轻而易举。 丰富的用户界面使查看生产中正在运行的管道，监视进度以及在需要时对问题进行故障排除变得容易。 将工作流定义为代码时，它们将变得更具可维护性，可版本化，可测试和协作性。

有关安装和使用Apache Airflow的详细信息，请参见Apache Airflow。