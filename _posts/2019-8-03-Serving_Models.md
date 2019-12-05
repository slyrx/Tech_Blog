---
layout: post
title:  "Serving Models"
date:   2019-08-03 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第三十八篇。
        本文档介绍使模型服务化。

## 目录
+ 介绍

## 正文

介绍
TensorFlow Serving是一个针对机器学习模型的灵活，高性能的服务系统，专为生产环境而设计。 使用TensorFlow Serving可以轻松部署新算法和实验，同时保持相同的服务器体系结构和API。 TensorFlow Serving提供与TensorFlow模型的现成集成，但可以轻松扩展以服务于其他类型的模型和数据。

提供有关TensorFlow Serving的详细开发人员文档：

架构概述
服务器API
REST客户端API