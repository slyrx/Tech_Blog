---
layout: post
title:  "The Pusher TFX Pipeline Component"
date:   2019-07-15 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第十九篇。
        本文档详细介绍了 Pusher 组件的作用。

## 目录
+ 使用 Pusher 组件

## 正文

Pusher组件用于在模型训练或再训练期间将经过验证的模型推入部署目标。 它依赖于ModelValidator组件来确保新模型“足够好”以投入生产。

消耗：SavedModel格式的训练模型
发出结果：相同的SavedModel，以及版本控制元数据
使用推杆组件
Pusher管道组件通常非常易于部署，几乎不需要自定义，因为所有工作都是由Pusher TFX组件完成的。 典型的代码如下所示：

```
from tfx import components

...

pusher = components.Pusher(
  model=trainer.outputs['model'],
  model_blessing=model_validator.outputs['blessing'],
  push_destination=pusher_pb2.PushDestination(
    filesystem=pusher_pb2.PushDestination.Filesystem(
        base_directory=serving_model_dir)
  )
)
```