---
layout: post
title:  "tensorflow MLP"
date:   2018-09-28 10:11:30
tags: [机器学习, 数据挖掘, tensorflow]
---

### 遇到的问题
+ tensorflow.python.framework.errors_impl.PermissionDeniedError: /MNIST-data; Permission denied
+ 答：路径写成了"/MNIST-data/",改成"MNIST-data/"，修复。

+ Can not squeeze dim[1], expected a dimension of 1, got 10 for 'sparse_softmax_cross_entropy_loss/remove_squeezable_dimensions/Squeeze' (op: 'Squeeze') with input shapes: [50,10].
+ squeeze 压缩
+ 输入数据不能是one_hot编码，保持原类型。
+ mnist = input_data.read_data_sets("MNIST-data/", one_hot=True) 改为 mnist = input_data.read_data_sets("MNIST-data/"), 修复。

+ 'Tensor' object has no attribute 'numpy'
+ 在import tensorflow之后，添加tf.enable_eager_execution()，修复。

+ tf.contrib.learn.datasets.load_dataset("mnist")可以网络，在下载到指定文件后，会优先检查本地文件情况，如果存在，则会停止上网下载。

+ Python version 2.7 does not support this syntax. super() should have arguments in Python 2
+ 解决方法：super().__init__()变为 super(CNN, self).__init__()，修复。

+ 'NoneType' object has no attribute 'numpy'
+ 原因是代码中有错误的地方，在predict函数中缺少return tf.argmax(logits, axis=-1)，导致返回的预测值shape与测试集答案不同。修复。

+ python 2.7 和 python 3在print上不兼容的处理方式
+ 在文件最开始加一句话 from __future__ import print_function，之后使用python 3的print规则，修复。

+ python 中的with语句是用来做异常处理的，作用等同于try/catch.





end
