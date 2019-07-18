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

RunMetadata 表示定义一个容器来获取元数据。本质是一个容器。
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
sim 的大小为16个50000长度的字符串，选择1个50000的 sim，`numpy.argsort`函数的含义是数组值从小到大的索引值。因此，该句意为在50000个经过从小到大排列的索引序列中，选择从1到8个位置。这8个位置被认为是与 'many' 最相似的词。下面开始输出这8个词`log_str = 'Nearest to %s:' % valid_word`。通过输出的 log 可以看到，除了输出了16个示例词的相似词
```
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
```
，还针对每2000次循环进行了一次损失函数情况的打印。可以看到，由信噪比表达的损失函数的值在逐渐提高，由初始值的305最终提高到了4.69.

最后，对形状为[5000, 128]的 normalized_embeddings 余弦向量进行计算：
```
final_embeddings = normalized_embeddings.eval()
```

至此，对于模型的计算步骤已经执行完毕。
接下来，将反向字典中的内容保存为 metadata.tsv 文件到 'examples/tutorials/word2vec/log'路径下。
```
  with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
    for i in xrange(vocabulary_size):
      f.write(reverse_dictionary[i] + '\n')
```
将模型保存为 model.ckpt ，供后续的重复调用。
```
saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))
```

对 TensorBoard 可视化的内容进行设置，首先提取 TensorBoard 对象，
```
config = projector.ProjectorConfig()
```

增加设置项，这里使用 config.embeddings.add 函数为配置增加一个设置项 embedding_conf。
```
  embedding_conf = config.embeddings.add()
```
+ config.embeddings.add()
为配置增加一个选项


设置项的内容为
```
  embedding_conf.tensor_name = embeddings.name
  embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
```
将可视化设置执行写入。
```
projector.visualize_embeddings(writer, config)
```
执行完所有的这些步骤，将写入对象执行关闭操作。

### 对嵌入结果进行可视化
可视化过程需要导入两个用于画图的包：
```
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
```

其中 TSNE 包是是目前来说效果最好的数据降维与可视化方法。但是它的缺点也很明显，比如：占内存大，运行时间长。当我们想要对高维数据进行分类，又不清楚这个数据集有没有很好的可分性（即同类之间间隔小，异类之间间隔大），可以通过t-SNE投影到2维或者3维的空间中观察一下。如果在低维空间中具有可分性，则数据是可分的；如果在高维空间中不具有可分性，可能是数据不可分，也可能仅仅是因为不能投影到低维空间。 

```
  tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
```

渲染特征及label标签，选择500个单词描绘它们的关系。
```
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
```
final_embeddings 的形状是50000个维度为128个特征的单词，这里选择500个单词的特征进行渲染。它们的标签值就用索引字典反查表中查出的单词来表示。

执行画图操作
```
  plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))
```

到此位置，该篇的源码解读全部完成。

## 总结
通过通篇的源码解读，现在我们对 word2vec 有了一个基本的了解，在这个例子里，选择加载规模为17005207的 vocabulary 词汇表，对这些数据进行统计，最终形成四种格式化的数据形式供后续备用：

|变量名称|变量内容|
|---|---|
|data|[5234, 3081, 12, 6, 195, 2, 3134, 46, 59, 156]|
|count|[['UNK', 418391], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764)]|
|dictionary|{'UNK': 0, 'the': 1, 'of': 2, 'and': 3, 'one': 4, 'in': 5, 'a': 6, 'to': 7, 'zero': 8}|
|reverse_dictionary|['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']|

接下来，构建一个 skip-gram 模型，及该模型需要的数据准备。输入的数据形式为[128],数据label形式为[128,1], 有效的示例窗口形状为[100, 16]. 综合对以上的形式进行汇总，是对128个int单词编码进行训练，目标值是
另外的128个独立的单词索引。generate_batch 是对训练按照什么样步骤进行训练的一种预设。通常依据的原则是能尽量高效的将CPU的内核利用起来。因此通常单个批次的大小和批次设定的规模都会设置为以CPU能最高效的利用起来为原则。接下来逐批次进行训练，将训练数据交给 session ，再由 session 调用前面定义好的 tf.nn.nce_loss 进行正式的模型训练。将训练后的结果损失函数值进行输出，同时给出距离当前词最近的16个单词，进行展示。

最终的输出，是一个形状为[50000，128]的矩阵，50000行表示从语料库中截取的前50000个语料，列表示每次截取选择一句话中的128个单词，其中的值表示这句话中每个词的向量分布情况。
当需要 tsne 进行输出时，需要对这些句子的情况进行 tsne 模型的拟合，并输出低纬度信息。

# 疑问？
```
 low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
```
选择的前500行的句子得出的低纬度的信息正好能和 reverse_dictionary 反转字典中可以匹配上吗？为什么会是这种顺序呢？

### 补充训练数据中特征值和标签值的生成过程
generate_batch 的内部逻辑是：batch_size=128, num_skips=2, skip_window=1)

基本的声明，data_index 声明为全局变量；批次规模大小对2取余的结果必须等于0；忽略的词数量必须小于滑窗的两倍。
```
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
```

初始变量的定义
```
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
```
batch 定义大小为128，类型为整型，具体值没有指定。

```
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
```
labels 定义为 (128, 1), 类型为整型，具体值也没有指定。

```
span = 2 * skip_window + 1
```
间隔定义为滑窗的2倍加1，此处为3.

```
  buffer = collections.deque(maxlen=span)
```
初始化定义一个队列，队列的大小为3，初始化为空

```
  if data_index + span > len(data):
    data_index = 0
```
指定，如果选择的数据索引加上跨距已经超出了语料的最大长度，那么将选择数据的索引重新初始化为0，这是为了避免越界，如果发生了越界的情况，就对索引进行重置，避免了越界。

```
buffer.extend(data[data_index:data_index + span])
```
在准备传入的 buffer 中追加填入数据，数据内容是字典的当前索引位置加间隔3. 也就是在buffer中追加了训练数据，单次追加的训练形式为`<class 'list'>: [8645, 1, 1517]`。
+ deque.extend
一次性从右端添加多个元素,如果设置了最大间隔`buffer = collections.deque(maxlen=span)`则有新元素填入后，旧元素会被删除。


```
for i in range(batch_size // num_skips):
```
单批次下，按照2为间隔进行检查

```
context_words = [w for w in range(span) if w != skip_window]
```
按照从`the quick brown fox jumped over the lazy dog`中以`([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...`的形式进行截取，将表示上下文的词组提取放入 context_words 。也就是提取出 `<class 'list'>: [0, 2] [the, brown]`，

```
words_to_use = random.sample(context_words, num_skips)
```
+ random.sample(context_words, 2)
从 context_words 中随机提取出2个元素，这里表示从上下文中随机提取出2个元素。

```
for j, context_word in enumerate(words_to_use):
```
遍历随机抽出的2个元素

```
batch[i * num_skips + j] = buffer[skip_window]
```
将 buffer 中准备做训练的词赋值给 batch

```
labels[i * num_skips + j, 0] = buffer[context_word]
```
将上下文的其中一个词置于预测标签列表中，形成了下面的数据对应格式

|num|batch|labels|
|---|---|---|
|0|1|1517|
|1|1|8645|

```
      buffer.append(data[data_index])
      data_index += 1
```
在未到达数据结尾时，向 buffer 中追加一个新的单词，此时 buffer 的变化情况如下：

|buffer 变化前|buffer 变化后|
|---|---|
|[8645,1,1517]|[1,1517,293]|
|[1,1517,293]|[1517,293,4412]|

可以看到，在128个长度范围内的句子中，以2个为间隔，进行文字内容的特征输入，及标签值整理过程，向下一组三个单词方向进行处理。

|nums|batch|labels|
|---|---|---|
|0|1|1517|
|1|1|8645|
|2|1517|293|
|3|1517|1|

<br>
<br>
<br>
<br>

### 附录：一个批次的训练数据及标签值的展示

<table>
   <tr>
      <td>batch</td>
      <td>labels</td>
   </tr>
   <tr>
      <td>1</td>
      <td>[[ 1517]</td>
   </tr>
   <tr>
      <td>1</td>
      <td> [ 8645]</td>
   </tr>
   <tr>
      <td>1517</td>
      <td> [  293]</td>
   </tr>
   <tr>
      <td>1517</td>
      <td> [    1]</td>
   </tr>
   <tr>
      <td>293</td>
      <td> [ 4412]</td>
   </tr>
   <tr>
      <td>293</td>
      <td> [ 1517]</td>
   </tr>
   <tr>
      <td>4412</td>
      <td> [  293]</td>
   </tr>
   <tr>
      <td>4412</td>
      <td> [  558]</td>
   </tr>
   <tr>
      <td>558</td>
      <td> [    2]</td>
   </tr>
   <tr>
      <td>558</td>
      <td> [ 4412]</td>
   </tr>
   <tr>
      <td>2</td>
      <td> [  558]</td>
   </tr>
   <tr>
      <td>2</td>
      <td> [16825]</td>
   </tr>
   <tr>
      <td>16825</td>
      <td> [  457]</td>
   </tr>
   <tr>
      <td>16825</td>
      <td> [    2]</td>
   </tr>
   <tr>
      <td>457</td>
      <td> [    3]</td>
   </tr>
   <tr>
      <td>457</td>
      <td> [16825]</td>
   </tr>
   <tr>
      <td>3</td>
      <td> [ 5809]</td>
   </tr>
   <tr>
      <td>3</td>
      <td> [  457]</td>
   </tr>
   <tr>
      <td>5809</td>
      <td> [    3]</td>
   </tr>
   <tr>
      <td>5809</td>
      <td> [  558]</td>
   </tr>
   <tr>
      <td>558</td>
      <td> [ 5809]</td>
   </tr>
   <tr>
      <td>558</td>
      <td> [    2]</td>
   </tr>
   <tr>
      <td>2</td>
      <td> [ 1151]</td>
   </tr>
   <tr>
      <td>2</td>
      <td> [  558]</td>
   </tr>
   <tr>
      <td>1151</td>
      <td> [    2]</td>
   </tr>
   <tr>
      <td>1151</td>
      <td> [20854]</td>
   </tr>
   <tr>
      <td>20854</td>
      <td> [ 1151]</td>
   </tr>
   <tr>
      <td>20854</td>
      <td> [   25]</td>
   </tr>
   <tr>
      <td>25</td>
      <td> [20854]</td>
   </tr>
   <tr>
      <td>25</td>
      <td> [16827]</td>
   </tr>
   <tr>
      <td>16827</td>
      <td> [   25]</td>
   </tr>
   <tr>
      <td>16827</td>
      <td> [ 3213]</td>
   </tr>
   <tr>
      <td>3213</td>
      <td> [16827]</td>
   </tr>
   <tr>
      <td>3213</td>
      <td> [   47]</td>
   </tr>
   <tr>
      <td>47</td>
      <td> [ 3213]</td>
   </tr>
   <tr>
      <td>47</td>
      <td> [  199]</td>
   </tr>
   <tr>
      <td>199</td>
      <td> [   20]</td>
   </tr>
   <tr>
      <td>199</td>
      <td> [   47]</td>
   </tr>
   <tr>
      <td>20</td>
      <td> [  199]</td>
   </tr>
   <tr>
      <td>20</td>
      <td> [   58]</td>
   </tr>
   <tr>
      <td>58</td>
      <td> [   20]</td>
   </tr>
   <tr>
      <td>58</td>
      <td> [ 3213]</td>
   </tr>
   <tr>
      <td>3213</td>
      <td> [    5]</td>
   </tr>
   <tr>
      <td>3213</td>
      <td> [   58]</td>
   </tr>
   <tr>
      <td>5</td>
      <td> [ 3213]</td>
   </tr>
   <tr>
      <td>5</td>
      <td> [ 2924]</td>
   </tr>
   <tr>
      <td>2924</td>
      <td> [    3]</td>
   </tr>
   <tr>
      <td>2924</td>
      <td> [    5]</td>
   </tr>
   <tr>
      <td>3</td>
      <td> [    0]</td>
   </tr>
   <tr>
      <td>3</td>
      <td> [ 2924]</td>
   </tr>
   <tr>
      <td>0</td>
      <td> [    3]</td>
   </tr>
   <tr>
      <td>0</td>
      <td> [  435]</td>
   </tr>
   <tr>
      <td>435</td>
      <td> [    0]</td>
   </tr>
   <tr>
      <td>435</td>
      <td> [ 5191]</td>
   </tr>
   <tr>
      <td>5191</td>
      <td> [  435]</td>
   </tr>
   <tr>
      <td>5191</td>
      <td> [    7]</td>
   </tr>
   <tr>
      <td>7</td>
      <td> [ 4558]</td>
   </tr>
   <tr>
      <td>7</td>
      <td> [ 5191]</td>
   </tr>
   <tr>
      <td>4558</td>
      <td> [   58]</td>
   </tr>
   <tr>
      <td>4558</td>
      <td> [    7]</td>
   </tr>
   <tr>
      <td>58</td>
      <td> [ 3213]</td>
   </tr>
   <tr>
      <td>58</td>
      <td> [ 4558]</td>
   </tr>
   <tr>
      <td>3213</td>
      <td> [    5]</td>
   </tr>
   <tr>
      <td>3213</td>
      <td> [   58]</td>
   </tr>
   <tr>
      <td>5</td>
      <td> [ 3213]</td>
   </tr>
   <tr>
      <td>5</td>
      <td> [  158]</td>
   </tr>
   <tr>
      <td>158</td>
      <td> [15948]</td>
   </tr>
   <tr>
      <td>158</td>
      <td> [    5]</td>
   </tr>
   <tr>
      <td>15948</td>
      <td> [  112]</td>
   </tr>
   <tr>
      <td>15948</td>
      <td> [  158]</td>
   </tr>
   <tr>
      <td>112</td>
      <td> [15948]</td>
   </tr>
   <tr>
      <td>112</td>
      <td> [  150]</td>
   </tr>
   <tr>
      <td>150</td>
      <td> [ 9772]</td>
   </tr>
   <tr>
      <td>150</td>
      <td> [  112]</td>
   </tr>
   <tr>
      <td>9772</td>
      <td> [   40]</td>
   </tr>
   <tr>
      <td>9772</td>
      <td> [  150]</td>
   </tr>
   <tr>
      <td>40</td>
      <td> [ 3420]</td>
   </tr>
   <tr>
      <td>40</td>
      <td> [ 9772]</td>
   </tr>
   <tr>
      <td>3420</td>
      <td> [   40]</td>
   </tr>
   <tr>
      <td>3420</td>
      <td> [   29]</td>
   </tr>
   <tr>
      <td>29</td>
      <td> [  828]</td>
   </tr>
   <tr>
      <td>29</td>
      <td> [ 3420]</td>
   </tr>
   <tr>
      <td>828</td>
      <td> [   29]</td>
   </tr>
   <tr>
      <td>828</td>
      <td> [ 4412]</td>
   </tr>
   <tr>
      <td>4412</td>
      <td> [  828]</td>
   </tr>
   <tr>
      <td>4412</td>
      <td> [ 3035]</td>
   </tr>
   <tr>
      <td>3035</td>
      <td> [ 4412]</td>
   </tr>
   <tr>
      <td>3035</td>
      <td> [ 3035]</td>
   </tr>
   <tr>
      <td>3035</td>
      <td> [    0]</td>
   </tr>
   <tr>
      <td>3035</td>
      <td> [ 3035]</td>
   </tr>
   <tr>
      <td>0</td>
      <td> [   53]</td>
   </tr>
   <tr>
      <td>0</td>
      <td> [ 3035]</td>
   </tr>
   <tr>
      <td>53</td>
      <td> [   32]</td>
   </tr>
   <tr>
      <td>53</td>
      <td> [    0]</td>
   </tr>
   <tr>
      <td>32</td>
      <td> [   53]</td>
   </tr>
   <tr>
      <td>32</td>
      <td> [   12]</td>
   </tr>
   <tr>
      <td>12</td>
      <td> [  158]</td>
   </tr>
   <tr>
      <td>12</td>
      <td> [   32]</td>
   </tr>
   <tr>
      <td>158</td>
      <td> [   12]</td>
   </tr>
   <tr>
      <td>158</td>
      <td> [   12]</td>
   </tr>
   <tr>
      <td>12</td>
      <td> [  158]</td>
   </tr>
   <tr>
      <td>12</td>
      <td> [    9]</td>
   </tr>
   <tr>
      <td>9</td>
      <td> [    8]</td>
   </tr>
   <tr>
      <td>9</td>
      <td> [   12]</td>
   </tr>
   <tr>
      <td>8</td>
      <td> [    9]</td>
   </tr>
   <tr>
      <td>8</td>
      <td> [   33]</td>
   </tr>
   <tr>
      <td>33</td>
      <td> [   11]</td>
   </tr>
   <tr>
      <td>33</td>
      <td> [    8]</td>
   </tr>
   <tr>
      <td>11</td>
      <td> [   33]</td>
   </tr>
   <tr>
      <td>11</td>
      <td> [   14]</td>
   </tr>
   <tr>
      <td>14</td>
      <td> [   11]</td>
   </tr>
   <tr>
      <td>14</td>
      <td> [    1]</td>
   </tr>
   <tr>
      <td>1</td>
      <td> [   14]</td>
   </tr>
   <tr>
      <td>1</td>
      <td> [ 2968]</td>
   </tr>
   <tr>
      <td>2968</td>
      <td> [    1]</td>
   </tr>
   <tr>
      <td>2968</td>
      <td> [  136]</td>
   </tr>
   <tr>
      <td>136</td>
      <td> [ 2968]</td>
   </tr>
   <tr>
      <td>136</td>
      <td> [   77]</td>
   </tr>
   <tr>
      <td>77</td>
      <td> [ 3666]</td>
   </tr>
   <tr>
      <td>77</td>
      <td> [  136]</td>
   </tr>
   <tr>
      <td>3666</td>
      <td> [   77]</td>
   </tr>
   <tr>
      <td>3666</td>
      <td> [ 1462]</td>
   </tr>
   <tr>
      <td>1462</td>
      <td> [ 3035]</td>
   </tr>
   <tr>
      <td>1462</td>
      <td> [ 3666]</td>
   </tr>
   <tr>
      <td>3035</td>
      <td> [ 1462]</td>
   </tr>
   <tr>
      <td>3035</td>
      <td> [   80]</td>
   </tr>
   <tr>
      <td>80</td>
      <td> [    6]</td>
   </tr>
   <tr>
      <td>80</td>
      <td> [ 3035]]</td>
   </tr>
</table>
 
