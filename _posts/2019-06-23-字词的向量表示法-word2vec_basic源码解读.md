---
layout: post
title:  "字词的向量表示法-word2vec_basic源码解读"
date:   2019-06-23 10:11:30
tags: [word2vec, tensorflow, NLP]
---

最近接触到一个通过 CNN 和 RNN 的方式对文本进行分类的项目。其中在文本的输入部分用到了 word2vec 。有机会感受到这个强大的文本向量化模型的优势。查阅了 tensorflow 关于 word2vec 的官方文档，觉得其中讲述 word2vec 基础模型的代码文档写的非常有意思。因此，有了顺便对这个文档进行翻译的想法。也通过这个过程，进一步加深自己对 word2vec 的理解。

首先，从结构上对文档进行分解，文档总共分7部分：
1. 依赖库导入
2. 数据下载
3. 创建词典和移除罕见词
4. 生成训练批次
5. 构建 skip-gram 模型
6. 开始训练
7. 可视化


### 依赖库导入

```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
```

### 数据下载
这里，数据的来源是 `url = 'http://mattmahoney.net/dc/'` , 将下载数据的过程封装成了 `maybe_download` 函数。传入的参数是待下载的文件名`text8.zip`及该文件的大小`31344016`。函数内部，首先判断是否在检查位置存在该文件，如果不存在，则开始执行下载请求。如果已经存在该文件，则检查文件的大小和期望的大小是不是一致，如果一致则打印找到的消息。否则，抛出异常进行提醒。

```
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  local_filename = os.path.join(gettempdir(), filename)
  if not os.path.exists(local_filename):
    local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                   local_filename)
  statinfo = os.stat(local_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception('Failed to verify ' + local_filename +
                    '. Can you get to it with a browser?')
  return local_filename


filename = maybe_download('text8.zip', 31344016)

```

检查过文件是否存在了，下面开始数据的正式读取。此处，通过封装函数`read_data`将数据读出，具体的逻辑是：先对文件进行解压`zipfile.ZipFile(filename)`, 再通过`tf.compat.as_str`对数据进行读取。
+ tf.compat.as_str 的作用是
将字节或 Unicode 转换为 bytes,使用 UTF-8 编码文本.

### 创建字典并移除罕见词 
在创建字典时，对字典总数需要进行预置，此处预置为 50000. 构建逻辑在`build_dataset`函数中实现。在`build_dataset`中，首先对加载的总数据样本进行 Counter 统计，统计的结果放入到 count 列表中。接下来，对 count 中的内容进行遍历，以单词的内容做 key 值，以此时存放单词的字典长度做 value 值，进行字典键值配对。下面，再对传入的加载字典进行遍历，通过`dictionary.get(word, 0)`判断 word 的情况。
+ dictionary.get(word, 0)
函数返回指定键的值，如果值不在字典中返回默认值。右边的0即为默认值。<br>
将找到的 word 的索引记录进入 data 列表；如果，从字典中求出的 count 值是0，则将 unk 标记累加1。
**最终返回 data, count, dictionary, reverse_dictionary**。

为了节约内存，我们此时可以将加载的总样本数据 vocabulary 删除，只保留50000个样本量进行运算。

### 构建 skip-gram 模型
首先是对一系列前提值的设置

```
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
```

下面来正式开启对 skip-gram 模型的构建：第一步，定义图对象`graph = tf.Graph()`,本篇代码是基于 tensorflow r1.14 实现的，因此，语法规则还是按照 tf 1.0 的规则。模型的后续构建需要在该 graph 的上下文中进行

```
with graph.as_default():
```

#### 在构建过程中，主要分为6个部分：
- 首先是输入数据的占位：
```
  with tf.name_scope('inputs'):
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
```
这里，输入数据分为3个部分：
1. 输入特征：是一个形状为(128,)的一维数组
2. 输出标签：是一个形状为(128,1)的二维数组
3. 有效数据集：是一个形状为(100, 16)的二维常量数组，其中的值为`np.random.choice`随机函数随机构建。

    + (128,) 和 (128,1) 的区别在于
    + [0,1,...,127]
    + [[0,1,...,127]]

- 指定运算位置及模型定义
指定运算的位置
```
  with tf.device('/cpu:0'):
```

+ tf.name_scope('embeddings') 
为 TensorBoard 显示内容定义名称
```
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
```
定义变量 embeddings ，令其的内容为分配到形状为(50000, 128)上的[-1.0, 1.0]之间的正态分布值。

```
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)
```
在 embeddings 中查找索引为 train_inputs 的变量值，这里**最终取出来的是随机生成的正态分布值**。


接下来，需要确定**权重**和**偏差**。
```
    with tf.name_scope('weights'):
      nce_weights = tf.Variable(
          tf.truncated_normal(
              [vocabulary_size, embedding_size],
              stddev=1.0 / math.sqrt(embedding_size)))
```
这里，在正态分布中取形状为(50000, 128)的截面，作为后续的权重赋值。

```
    with tf.name_scope('biases'):
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
```
而偏差则选择形状为[50000]的0值。

- 指定损失函数
```
  with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,
            num_sampled=num_sampled,
            num_classes=vocabulary_size))
```
损失函数选择使用求噪声比的平均值来衡量。噪声比的函数为`tf.nn.nce_loss`,它的内部逻辑是 weights、biases 都是通过从正态分布中采样预先定义好的，inputs 可以看成是带好正态分布采样权重的输入, num_sampled 表示采样出多少个负样本，这里被指定为64个。num_classes 表示可能的类数，这里设定为50000。

```
tf.summary.scalar('loss', loss)
```
向 TensorBoard 提交损失函数的变化信息
+ tf.summary.scalar
用来显示标量信息

- 指定优化器
```
  with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
```
+ tf.train.GradientDescentOptimizer
实现了梯度下降算法的优化器，其中参数1.0表示学习率。minimize 以使 loss 的梯度最小化为目标进行优化。minimize 中包括两个步骤，计算梯度 compute_gradients() 和更新参数 apply_gradients()。


- 余弦相似度计算公式
```
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
```
余弦相似度公式中的开根号底部
```
  normalized_embeddings = embeddings / norm
```
一个完整的待计算向量
```
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                            valid_dataset)
```
按照常量 valid_dataset 为形状[100,16]的索引从 normalized_embeddings 中取出部分值来。
+ tf.constant
生成一个给定值的常量


将各个部分按照余弦相似度公式合并，最终参与计算的向量形状为 valid_embeddings [16, 128] 和 normalized_embeddings [50000, 128], 得到的相似度结果为 [16, 50000]
```
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
```

- 其它
+ tf.summary.merge_all
merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。
+ tf.global_variables_initializer
返回一个用来初始化计算图中所有global variable的op，这个op到底是啥，还不清楚。
+ tf.train.Saver()
将训练好的模型参数保存起来，以便以后进行验证或测试。

### 开始训练
+ tf.summary.FileWriter
指定一个文件用来保存图。

设定要对训练执行100001次迭代。每个迭代执行的内容如下：
```
    batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                skip_window)
```
生成一批处理需要的数据，生成的规则为规定一批的大小 batch_size 为128；忽略的数量 num_skips 为2；忽略的窗口 skip_window 为1。得到的准备作为输入的形状为 batch_inputs (128,); 训练标签值 batch_labels 形状为 (128, 1)。


为式子的占位符进行初始值赋值。
```
feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
```

RunMetadata 表示定义一个容器来获取元数据。
```
run_metadata = tf.RunMetadata()
```

将定义好的容器传入到运行环境，同时将输入参数、优化器、损失函数等都传入运行环境。
```
    _, summary, loss_val = session.run(
        [optimizer, merged, loss],
        feed_dict=feed_dict,
        run_metadata=run_metadata)

    average_loss += loss_val
```
最终，可以得到的运算结果包括概要信息和损失函数值的情况。

将当前的概要信息记录，供后续 TensorBoard 显示。
```
    writer.add_summary(summary, step)
    # Add metadata to visualize the graph for the last run.
    if step == (num_steps - 1):
      writer.add_run_metadata(run_metadata, 'step%d' % step)
```

```
    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0
```
每训练2000次，对损失函数值进行一次平均，然后输出。

+ sim = similarity.eval()
eval() 其实就是tf.Tensor的Session.run() 的另外一种写法

每10000次训练后，计算余弦相似度。 
```
sim = similarity.eval()
```

遍历有效的窗口大小 valid_size，
```
valid_word = reverse_dictionary[valid_examples[i]]
```
从反字典表中查找单词，这里找到了 “many”，
```
    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
```
sim 的大小为16个50000长度的字符串，选择1个50000的 sim，`numpy.argsort`函数的含义是数组值从小到大的索引值。因此，该句意为在50000个经过从小到大排列的索引序列中，选择从1到8个位置。这8个位置被认为是与 'many' 最相似的词。下面开始输出这8个词。通过输出的 log 可以看到，