---
layout: post
title:  "Get Started with TensorFlow Transform"
date:   2019-07-25 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第二十九篇。
        本文档介绍tf.Transform的基本概念以及如何使用它们。

## 目录
+ 定义一个预处理函数
+ + 预处理函数例子
+ + 批处理
+ Apache Beam 实现
+ 数据格式和模式
+ Apache Beam 的输入和输出
+ + 下载统计数据集
+ 整合 TensorFlow 训练

## 正文

本指南介绍了tf.Transform的基本概念以及如何使用它们。它会：

定义预处理功能，即对管线的逻辑描述，该管线将原始数据转换为用于训练机器学习模型的数据。
通过将预处理功能转换为Beam管道，展示用于转换数据的Apache Beam实现。
显示其他用法示例。
定义预处理功能
预处理功能是tf.Transform最重要的概念。预处理功能是数据集转换的逻辑描述。预处理函数接受并返回张量字典，其中张量表示Tensor或SparseTensor。有两种用于定义预处理功能的功能：

接受并返回张量的任何函数。这些将TensorFlow操作添加到图形，从而将原始数据转换为转换后的数据。
tf.Transform提供的任何分析器。分析器还接受并返回张量，但与TensorFlow函数不同，它们不向图形添加操作。而是由分析器使tf.Transform在TensorFlow之外计算全通运算。他们使用整个数据集中的输入张量值来生成恒定张量，并将其作为输出返回。例如，tft.min计算数据集中的张量的最小值。 tf.Transform提供了一组固定的分析器，但是将在以后的版本中进行扩展。
预处理功能示例
通过将分析器和常规的TensorFlow功能相结合，用户可以创建用于转换数据的灵活管道。以下预处理功能以不同方式转换了三个功能中的每一个，并结合了两个功能：

```
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

def preprocessing_fn(inputs):
  x = inputs['x']
  y = inputs['y']
  s = inputs['s']
  x_centered = x - tft.mean(x)
  y_normalized = tft.scale_to_0_1(y)
  s_integerized = tft.compute_and_apply_vocabulary(s)
  x_centered_times_y_normalized = x_centered * y_normalized
  return {
      'x_centered': x_centered,
      'y_normalized': y_normalized,
      'x_centered_times_y_normalized': x_centered_times_y_normalized,
      's_integerized': s_integerized
  }
```

在此，x，y和s是代表输入要素的张量。通过将tft.mean应用于x并从x中减去来创建第一个创建的新张量x_centered。 tft.mean（x）返回一个表示张量x平均值的张量。 x_centered是张量x减去均值。

以相似的方式创建第二个新张量y_normalized，但使用便利方法tft.scale_to_0_1。此方法执行的操作类似于计算x_centered，即计算最大值和最小值，并使用它们来缩放y。

张量s_integerized显示了字符串操作的示例。在这种情况下，我们采用字符串并将其映射为整数。这使用了便利功能tft.compute_and_apply_vocabulary。此函数使用分析器来计算输入字符串采用的唯一值，然后使用TensorFlow操作将输入字符串转换为唯一值表中的索引。

最后一列显示可以通过组合张量来使用TensorFlow操作创建新特征。

预处理功能定义了数据集上的一系列操作。为了应用管道，我们依赖于tf.Transform API的具体实现。 Apache Beam实现提供了PTransform，该函数将用户的预处理功能应用于数据。 tf.Transform用户的典型工作流程将构造一个预处理功能，然后将其合并到更大的Beam管道中，以创建用于训练的数据。

批处理
批处理是TensorFlow的重要组成部分。由于tf.Transform的目标之一是为预处理提供一个TensorFlow图，可以将其合并到服务图（以及可选的训练图）中，因此批处理也是tf.Transform中的重要概念。

尽管在上面的示例中并不明显，但用户定义的预处理功能是传递给张量的张量，而不是单个实例，这是在训练和使用TensorFlow服务期间发生的。另一方面，分析器对整个数据集执行计算，该计算返回单个值而不是一批值。 x是具有（batch_size，）形状的张量，而tft.mean（x）是具有（）形状的张量。减去x-tft.mean（x）进行广播，其中从x表示的批次的每个元素中减去tft.mean（x）的值。

Apache Beam实施
预处理功能旨在作为对在多个数据处理框架上实现的预处理管道的逻辑描述，而tf.Transform提供了在Apache Beam上使用的规范实现。此实现演示了实现所需的功能。此功能没有正式的API，因此每个实现都可以使用针对其特定数据处理框架的惯用API。

Apache Beam实现提供了两个PTransform，用于处理预处理功能的数据。下面显示了复合PTransform AnalyzeAndTransformDataset的用法：

```
raw_data = [
    {'x': 1, 'y': 1, 's': 'hello'},
    {'x': 2, 'y': 2, 's': 'world'},
    {'x': 3, 'y': 3, 's': 'hello'}
]

raw_data_metadata = ...
transformed_dataset, transform_fn = (
    (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(
        preprocessing_fn))
transformed_data, transformed_metadata = transformed_dataset
```

下面显示了transformd_data内容，其中包含与原始数据格式相同的转换列。 特别是，s_integerized的值是[0，1，0]，这些值取决于确定性，如何将单词hello和world映射到整数。 对于x_centered列，我们减去了平均值，因此x的值[1.0，2.0，3.0]变为[-1.0，0.0，1.0]。 同样，其余的列与它们的期望值匹配。


```
[{u's_integerized': 0,
  u'x_centered': -1.0,
  u'x_centered_times_y_normalized': -0.0,
  u'y_normalized': 0.0},
 {u's_integerized': 1,
  u'x_centered': 0.0,
  u'x_centered_times_y_normalized': 0.0,
  u'y_normalized': 0.5},
 {u's_integerized': 0,
  u'x_centered': 1.0,
  u'x_centered_times_y_normalized': 1.0,
  u'y_normalized': 1.0}]
```

raw_data和transformd_data都是数据集。 接下来的两节显示Beam实现如何表示数据集以及如何向磁盘读取和写入数据。 另一个返回值transform_fn表示应用于数据的转换，下面将详细介绍。

AnalyzeAndTransformDataset是实现AnalyzeDataset和TransformDataset提供的两个基本转换的组成。 因此，以下两个代码段是等效的：

```
transformed_data, transform_fn = (
    my_data | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
```

```
transform_fn = my_data | tft_beam.AnalyzeDataset(preprocessing_fn)
transformed_data = (my_data, transform_fn) | tft_beam.TransformDataset()
```

transform_fn是一个纯函数，表示应用于数据集每一行的操作。特别是，分析器值已被计算并视为常量。在示例中，transform_fn包含x列的平均值，y列的最小值和最大值以及用于将字符串映射为整数的词汇表作为常量。

tf.Transform的一个重要功能是transform_fn表示行上的映射-它是分别应用于每行的纯函数。汇总行的所有计算都在AnalyzeDataset中完成。此外，transform_fn表示为可以嵌入到服务图中的TensorFlow图。

在这种特殊情况下，提供AnalyzeAndTransformDataset进行优化。这与scikit-learn中使用的模式相同，提供了fit，transform和fit_transform方法。

数据格式和架构
在前面的代码示例中，省略了定义raw_data_metadata的代码。元数据包含定义数据布局的架构，因此可以从多种格式读取和写入数据。即使上一节中显示的内存中格式也不是自描述的，而是需要使用架构才能将其解释为张量。

这是示例数据的模式定义：

```
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

raw_data_metadata = dataset_metadata.DatasetMetadata(
    dataset_schema.from_feature_spec({
        's': tf.FixedLenFeature([], tf.string),
        'y': tf.FixedLenFeature([], tf.float32),
        'x': tf.FixedLenFeature([], tf.float32),
    }))
```

Dataset_schema.Schema类包含将数据从其磁盘上或内存中的格式解析为张量所需的信息。它通常通过调用带有dict映射功能键的tf.FixedLenFeature，tf.VarLenFeature和tf.SparseFeature值的dataset_schema.from_feature_spec来构造。有关更多详细信息，请参见tf.parse_example的文档。

在上面，我们使用tf.FixedLenFeature来指示每个功能都包含固定数量的值，在这种情况下为单个标量值。由于tf.Transform批处理实例，因此表示要素的实际Tensor将具有形状（None，），其中未知尺寸为批尺寸。

用Apache Beam输入和输出
到目前为止，示例的数据格式已使用字典列表。这是一种简化，它依赖于Apache Beam处理列表以及其主要数据表示形式PCollection的功能。 PCollection是形成Beam管道一部分的数据表示形式。通过应用各种PTransform（包括AnalyzeDataset和TransformDataset）并运行管道来形成Beam管道。 PCollection不在主二进制文件的内存中创建，而是分配在工作进程之间（尽管本节使用内存中执行模式）。

以下示例需要在磁盘上读取和写入数据，并需要将数据表示为PCollection（而不是列表），请参阅：census_example.py。下面我们展示了如何下载数据并运行此示例。 “人口普查收入”数据集由UCI机器学习存储库提供。该数据集包含分类和数字数据。

数据为CSV格式，这是前两行：

```
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
```

数据集的列是分类的或数字的。由于有许多列，因此通过遍历每种类型的所有列来生成模式（类似于前面的示例）。该数据集描述了一个分类问题：预测个人年收入高于或低于50K的最后一列。但是，从tf.Transform的角度来看，此标签只是另一个分类列。

使用此架构从CSV文件读取数据。 ordered_columns常数按在CSV文件中出现的顺序包含所有列的列表，这是必需的，因为架构不包含此信息。由于从CSV文件读取时已经完成了一些额外的Beam转换，因此已将其删除。每个CSV行都以内存格式转换为实例。

在此示例中，我们允许缺少教育编号功能。这意味着它在feature_spec中表示为tf.VarLenFeature，在preprocessing_fn中表示为tf.SparseTensor。为了处理可能缺少的要素值，我们使用默认值（在本例中为0）填充缺失的实例。

```
converter = tft.coders.CsvCoder(ordered_columns, raw_data_schema)

raw_data = (
    p
    | 'ReadTrainData' >> textio.ReadFromText(train_data_file)
    | ...
    | 'DecodeTrainData' >> beam.Map(converter.decode))
```

预处理与前面的示例相似，除了预处理功能是通过编程方式生成的，而不是手动指定每一列。 在下面的预处理功能中，NUMERICAL_COLUMNS和CATEGORICAL_COLUMNS是包含数字和类别列名称的列表：

```
def preprocessing_fn(inputs):
  """Preprocess input columns into transformed columns."""
  # Since we are modifying some features and leaving others unchanged, we
  # start by setting `outputs` to a copy of `inputs.
  outputs = inputs.copy()

  # Scale numeric columns to have range [0, 1].
  for key in NUMERIC_FEATURE_KEYS:
    outputs[key] = tft.scale_to_0_1(outputs[key])

  for key in OPTIONAL_NUMERIC_FEATURE_KEYS:
    # This is a SparseTensor because it is optional. Here we fill in a default
    # value when it is missing.
    dense = tf.sparse_to_dense(outputs[key].indices,
                               [outputs[key].dense_shape[0], 1],
                               outputs[key].values, default_value=0.)
    # Reshaping from a batch of vectors of size 1 to a batch to scalars.
    dense = tf.squeeze(dense, axis=1)
    outputs[key] = tft.scale_to_0_1(dense)

  # For all categorical columns except the label column, we generate a
  # vocabulary but do not modify the feature.  This vocabulary is instead
  # used in the trainer, by means of a feature column, to convert the feature
  # from a string to an integer id.
  for key in CATEGORICAL_FEATURE_KEYS:
    tft.vocabulary(inputs[key], vocab_filename=key)

  # For the label column we provide the mapping from string to index.
  initializer = tf.lookup.KeyValueTensorInitializer(
      keys=['>50K', '<=50K'],
      values=tf.cast(tf.range(2), tf.int64),
      key_dtype=tf.string,
      value_dtype=tf.int64)
  table = tf.lookup.StaticHashTable(initializer, default_value=-1)

  outputs[LABEL_KEY] = table.lookup(outputs[LABEL_KEY])

  return outputs
```

与上一个示例的不同之处在于，label列手动指定了从字符串到索引的映射。因此，将'> 50'映射为0，将'<= 50K'映射为1，因为知道训练模型中的哪个索引对应于哪个标签非常有用。

raw_data变量表示一个PCollection，它使用相同的AnalyzeAndTransformDataset转换，以与列表raw_data相同的格式包含数据（来自上一个示例）。该模式在两个地方使用：从CSV文件读取数据，并作为AnalyzeAndTransformDataset的输入。 CSV格式和内存格式都必须与模式配对，才能将它们解释为张量。

最后阶段是将转换后的数据写入磁盘，并且具有与读取原始数据类似的形式。用于执行此操作的模式是AnalyzeAndTransformDataset输出的一部分，该推断可推断输出数据的模式。写入磁盘的代码如下所示。该架构是元数据的一部分，但在tf.Transform API中可以互换使用两者（即，将元数据传递给ExampleProtoCoder）。请注意，这会写入不同的格式。代替textio.WriteToText，使用Beam对TFRecord格式的内置支持，并使用编码器将数据编码为Example protos。这是用于培训的更好格式，如下一节所示。 transform_eval_data_base提供了写入的各个分片的基本文件名。

```
transformed_data | "WriteTrainData" >> tfrecordio.WriteToTFRecord(
    transformed_eval_data_base,
    coder=tft.coders.ExampleProtoCoder(transformed_metadata))
```

除训练数据外，transform_fn还与元数据一起写出：

```
_ = (
    transform_fn
    | 'WriteTransformFn' >> tft_beam.WriteTransformFn(working_dir))
transformed_metadata | 'WriteMetadata' >> tft_beam.WriteMetadata(
    transformed_metadata_file, pipeline=p)
```

使用p.run（）。wait_until_finish（）运行整个Beam管道。 到现在为止，Beam管道表示延迟的分布式计算。 它提供了将要执行的指令，但是指令尚未执行。 此最终调用执行指定的管道。

下载人口普查数据集
使用以下shell命令下载人口普查数据集：

```
  wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
  wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

运行census_example.py脚本时，将包含此数据的目录作为第一个参数传递。该脚本创建一个临时子目录以添加预处理的数据。

与TensorFlow培训集成
census_example.py的最后一部分显示了如何使用预处理的数据来训练模型。有关详细信息，请参见估算器文档。第一步是构造一个需要对预处理列进行描述的估计器。每个数字列都被描述为real_valued_column，它是具有固定大小（在此示例中为1）的密集向量的包装。每个分类列都描述为sparse_column_with_integerized_feature。这表明从字符串到整数的映射已经完成。提供存储桶大小，这是列中包含的最大索引。我们已经知道普查数据的值，但是最好使用tf.Transform来计算它们。将来的tf.Transform版本会将这些信息写出为元数据的一部分，然后可以在此处使用。

```
real_valued_columns = [feature_column.real_valued_column(key)
                       for key in NUMERIC_COLUMNS]

one_hot_columns = [
    feature_column.sparse_column_with_integerized_feature(
        key, bucket_size=bucket_size)
    for key, bucket_size in zip(CATEGORICAL_COLUMNS, BUCKET_SIZES)]

estimator = learn.LinearClassifier(real_valued_columns + one_hot_columns)
```

下一步是创建一个生成器，以生成用于训练和评估的输入函数。 这与tf.Learn所使用的培训有所不同，因为不需要功能说明来解析转换后的数据。 而是，将元数据用于转换后的数据以生成功能规范。

```
def _make_training_input_fn(tf_transform_output, transformed_examples,
                            batch_size):
  ...
  def input_fn():
    """Input function for training and eval."""
    dataset = tf.contrib.data.make_batched_features_dataset(
        ..., tf_transform_output.transformed_feature_spec(), ...)

    transformed_features = dataset.make_one_shot_iterator().get_next()
    ...

  return input_fn
```

其余代码与使用Estimator类相同。 该示例还包含以SavedModel格式导出模型的代码。 Tensorflow Serving或Cloud ML Engine可以使用导出的模型。