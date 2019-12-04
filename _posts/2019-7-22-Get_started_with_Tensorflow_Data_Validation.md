---
layout: post
title:  "Get started with Tensorflow Data Validation"
date:   2019-07-22 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第二十六篇。
        本文档详细介绍探索和验证机器学习数据。


## 目录
+ 计算描述性数据统计
+ + 在 google 云上运行
+ 根据数据推断模式
+ 检查数据中的错误
+ + 将数据集的统计信息与模式进行匹配
+ + 根据示例检查错误
+ 模式环境
+ 检查数据的歪斜和趋势
+ 编写自定义数据连接器


## 正文
Tensorflow数据验证（TFDV）可以分析训练和提供数据以：

计算描述性统计数据，

推断一个模式，

检测数据异常。

核心API支持每一项功能，其便捷方法建立在顶部，并且可以在笔记本的上下文中调用。

计算描述性数据统计
TFDV可以计算描述性统计信息，从而根据存在的功能及其值分布的形状快速概述数据。 Facets Overview之类的工具可以提供这些统计信息的简洁可视化，以便于浏览。

例如，假设路径指向TFRecord格式的文件（该文件包含tensorflow.Example类型的记录）。 以下代码段说明了使用TFDV进行统计信息的计算：

```
    stats = tfdv.generate_statistics_from_tfrecord(data_location=path)
```

返回的值是DatasetFeatureStatisticsList协议缓冲区。 示例笔记本包含使用Facets概述的统计信息的可视化：

```
    tfdv.visualize_statistics(stats)
```

![png](https://www.tensorflow.org/tfx/data_validation/images/stats.png)

前面的示例假定数据存储在TFRecord文件中。 TFDV还支持CSV输入格式，并具有其他常见格式的可扩展性。您可以在此处找到可用的数据解码器。此外，TFDV为具有以pandas DataFrame表示的内存中数据的用户提供tfdv.generate_statistics_from_dataframe实用程序功能。

除了计算默认的一组数据统计信息之外，TFDV还可以计算语义域（例如图像，文本）的统计信息。要启用语义域统计信息的计算，请将tfdv.generate_statistics_from_tfrecord的tfdv.StatsOptions对象的enable_semantic_domain_stats设置为True。

在Google Cloud上运行
在内部，TFDV使用Apache Beam的数据并行处理框架来扩展大型数据集的统计计算。对于希望与TFDV进行更深入集成的应用程序（例如，在数据生成管道的末尾附加统计信息生成，以自定义格式生成数据统计信息），API还公开了Beam PTransform用于统计信息生成。

要在Google Cloud上运行TFDV，必须下载TFDV wheel文件并将其提供给Dataflow Worker。将wheel文件下载到当前目录，如下所示：

```
pip download tensorflow_data_validation \
  --no-deps \
  --platform manylinux1_x86_64 \
  --only-binary=:all:

```

以下代码段显示了TFDV在Google Cloud上的用法示例：

```

import tensorflow_data_validation as tfdv
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, StandardOptions, SetupOptions

PROJECT_ID = ''
JOB_NAME = ''
GCS_STAGING_LOCATION = ''
GCS_TMP_LOCATION = ''
GCS_DATA_LOCATION = ''
# GCS_STATS_OUTPUT_PATH is the file path to which to output the data statistics
# result.
GCS_STATS_OUTPUT_PATH = ''

PATH_TO_WHL_FILE = ''


# Create and set your PipelineOptions.
options = PipelineOptions()

# For Cloud execution, set the Cloud Platform project, job_name,
# staging location, temp_location and specify DataflowRunner.
google_cloud_options = options.view_as(GoogleCloudOptions)
google_cloud_options.project = PROJECT_ID
google_cloud_options.job_name = JOB_NAME
google_cloud_options.staging_location = GCS_STAGING_LOCATION
google_cloud_options.temp_location = GCS_TMP_LOCATION
options.view_as(StandardOptions).runner = 'DataflowRunner'

setup_options = options.view_as(SetupOptions)
# PATH_TO_WHL_FILE should point to the downloaded tfdv wheel file.
setup_options.extra_packages = [PATH_TO_WHL_FILE]

tfdv.generate_statistics_from_tfrecord(GCS_DATA_LOCATION,
                                       output_path=GCS_STATS_OUTPUT_PATH,
                                       pipeline_options=options)

```

在这种情况下，生成的统计信息原型存储在写入GCS_STATS_OUTPUT_PATH的TFRecord文件中。

注意在Google Cloud上调用任何tfdv.generate_statistics _...函数（例如tfdv.generate_statistics_from_tfrecord）时，您必须提供output_path。指定“无”可能会导致错误。

根据数据推断模式
该架构描述了数据的预期属性。其中一些属性是：

预期会出现哪些功能
他们的类型
每个示例中要素值的数量
所有示例中每个功能的存在
功能的预期领域。
简而言之，该模式描述了对“正确”数据的期望，因此可用于检测数据中的错误（如下所述）。此外，可以使用相同的架构来设置Tensorflow Transform进行数据转换。注意，该模式应该是相当静态的，例如，多个数据集可以符合同一模式，而统计信息（如上所述）可以随每个数据集而变化。

由于编写模式可能是一项繁琐的任务，尤其是对于具有许多功能的数据集，TFDV提供了一种基于描述性统计信息生成模式初始版本的方法：

```
    schema = tfdv.infer_schema(stats)
```

通常，TFDV使用保守的启发式方法从统计信息中推断稳定的数据属性，以避免将架构过度拟合到特定数据集。 强烈建议您检查推断的架构并根据需要对其进行优化，以捕获有关TFDV启发式可能会丢失的数据的任何领域知识。

默认情况下，如果value_count.min等于该功能的value_count.max，则tfdv.infer_schema会推断每个所需功能的形状。 将infer_feature_shape参数设置为False可禁用形状推断。

模式本身存储为模式协议缓冲区，因此可以使用标准协议缓冲区API进行更新/编辑。 TFDV还提供了一些实用程序方法来简化这些更新。 例如，假设该架构包含以下节来描述需要一个字符串值的必需字符串功能payment_type：

```
feature {
  name: "payment_type"
  value_count {
    min: 1
    max: 1
  }
  type: BYTES
  domain: "payment_type"
  presence {
    min_fraction: 1.0
    min_count: 1
  }
}
```

要标记至少在50％的示例中填充该功能，请执行以下操作：

```
    tfdv.get_feature(schema, 'payment_type').presence.min_fraction = 0.5
```

示例笔记本包含一个简单的图表可视化形式，并列出了每个特征及其在特征中编码的主要特征。

![png](https://www.tensorflow.org/tfx/data_validation/images/schema.png)

检查数据是否有错误
给定一个架构，可以检查数据集是否符合该架构中设置的期望，或者是否存在任何数据异常。您可以（a）通过将数据集的统计信息与模式进行匹配来检查整个数据集中的错误，或者（b）通过基于示例检查错误来检查数据中的错误。

将数据集的统计信息与模式进行匹配
为了检查汇总中的错误，TFDV将数据集的统计信息与模式进行匹配，并标记任何差异。例如：

    ＃假设other_path指向另一个TFRecord文件
    other_stats = tfdv.generate_statistics_from_tfrecord（data_location = other_path）
    异常= tfdv.validate_statistics（统计信息= other_stats，架构=模式）

结果是异常协议缓冲区的一个实例，并描述了统计信息与架构不一致的所有错误。例如，假设other_path的数据包含示例，这些示例的特征付款模式类型的值超出架构中指定的域。

这会产生异常

```
   payment_type  Unexpected string values  Examples contain values missing from the schema: Prcard (<1%).
```

表示在统计信息中发现特征值<1％的域外值。

如果期望如此，则可以按以下方式更新架构：

```
   tfdv.get_domain(schema, 'payment_type').value.append('Prcard')
```

如果异常确实指示数据错误，则应在使用基础数据进行训练之前将其修复。

此处列出了此模块可以检测到的各种异常类型。

示例笔记本以表格的形式包含异常的简单可视化，列出了检测到错误的功能以及每个错误的简短描述。

![png](https://www.tensorflow.org/tfx/data_validation/images/anomaly.png)

根据示例检查错误
TFDV还提供了基于示例验证数据的选项，而不是将整个数据集的统计信息与该模式进行比较。 TFDV提供了一些功能，可根据每个示例验证数据，然后为发现的异常示例生成摘要统计信息。 例如：

```
   options = tfdv.StatsOptions(schema=schema)
   anomalous_example_stats = tfdv.validate_tfexamples_in_tfrecord(
       data_location=input, stats_options=options)

```

validate_tfexamples_in_tfrecord返回的异常_example_stats是一个DatasetFeatureStatisticsList协议缓冲区，其中每个数据集都包含表现出特定异常的示例集。 您可以使用它来确定数据集中显示给定异常的示例数量以及这些示例的特征。

TFDV还提供了validate_instance函数，用于识别与架构匹配时单个示例是否显示异常。 要使用此功能，示例必须是将特征名称映射到特征值的numpy数组的dict。 您可以使用TFExampleDecoder将序列化的tf.train.Examples解码为这种格式。 例如：

```
   decoder = tfdv.TFExampleDecoder()
   example = decoder.decode(serialized_tfexample)
   options = tfdv.StatsOptions(schema=schema)
   anomalies = tfdv.validate_instance(example, options)
```

与validate_statistics一样，结果是Anomalies协议缓冲区的一个实例，该实例描述了示例与指定架构不一致的所有错误。

模式环境
默认情况下，验证假定管道中的所有数据集都遵循单个架构。 在某些情况下，有必要引入轻微的模式变化，例如在训练过程中需要使用用作标签的功能（并应进行验证），但在投放过程中会丢失这些功能。

环境可以用来表达这种要求。 特别是，可以使用default_environment，in_environment和not_in_environment将架构中的功能与一组环境关联。

例如，如果技巧功能在训练中用作标签，但在投放数据中缺失。 如果未指定环境，它将显示为异常。

```
    serving_stats = tfdv.generate_statistics_from_tfrecord(data_location=serving_data_path)
    serving_anomalies = tfdv.validate_statistics(serving_stats, schema)
```

![png](https://www.tensorflow.org/tfx/data_validation/images/serving_anomaly.png)

要解决此问题，我们需要将所有功能的默认环境设置为“ TRAINING”和“ SERVING”，并从SERVING环境中排除“ tips”功能。

```
    # All features are by default in both TRAINING and SERVING environments.
    schema.default_environment.append('TRAINING')
    schema.default_environment.append('SERVING')

    # Specify that 'tips' feature is not in SERVING environment.
    tfdv.get_feature(schema, 'tips').not_in_environment.append('SERVING')

    serving_anomalies_with_env = tfdv.validate_statistics(
        serving_stats, schema, environment='SERVING')
```

检查数据偏斜和漂移
TFDV除了检查数据集是否符合在模式中设置的期望之外，还提供了以下功能：

训练数据和服务数据之间的偏差
在不同日期的训练数据之间漂移
TFDV通过基于模式中指定的漂移/偏斜比较器比较不同数据集的统计信息来执行此检查。 例如，要检查训练和服务数据集中的“ payment_type”功能之间是否存在任何偏差：

```
    # Assume we have already generated the statistics of training dataset, and
    # inferred a schema from it.
    serving_stats = tfdv.generate_statistics_from_tfrecord(data_location=serving_data_path)
    # Add a skew comparator to schema for 'payment_type' and set the threshold
    # of L-infinity norm for triggering skew anomaly to be 0.01.
    tfdv.get_feature(schema, 'payment_type').skew_comparator.infinity_norm.threshold = 0.01
    skew_anomalies = tfdv.validate_statistics(
        statistics=train_stats, schema=schema, serving_statistics=serving_stats)
```

与检查数据集是否符合在模式中设置的期望相同，结果也是Anomalies协议缓冲区的一个实例，并描述了训练和服务数据集之间的任何偏差。 例如，假设服务数据包含更多具有价值Cash的特征payement_type的示例，这会产生偏斜异常

```
   payment_type  High L-infinity distance between serving and training  The L-infinity distance between serving and training is 0.0435984 (up to six significant digits), above the threshold 0.01. The feature value with maximum difference is: Cash
```

如果异常确实表明训练和服务数据之间存在偏差，则有必要进行进一步调查，因为这可能会对模型性能产生直接影响。

示例笔记本以表格的形式显示了偏斜异常的简单可视化，列出了检测到偏斜的功能以及每个偏斜的简短描述。

![png](https://www.tensorflow.org/tfx/data_validation/images/skew_anomaly.png)

可以类似的方式检测不同天的训练数据之间的偏差

```
    # Assume we have already generated the statistics of training dataset for
    # day 2, and inferred a schema from it.
    train_day1_stats = tfdv.generate_statistics_from_tfrecord(data_location=train_day1_data_path)
    # Add a drift comparator to schema for 'payment_type' and set the threshold
    # of L-infinity norm for triggering drift anomaly to be 0.01.
    tfdv.get_feature(schema, 'payment_type').drift_comparator.infinity_norm.threshold = 0.01
    drift_anomalies = tfdv.validate_statistics(
        statistics=train_day2_stats, schema=schema, previous_statistics=train_day1_stats)
```

编写自定义数据连接器
为了计算数据统计信息，TFDV提供了几种方便的方法来处理各种格式的输入数据（例如TFRecord的tf.train.Example，CSV等）。 如果您的数据格式不在此列表中，则需要编写一个自定义数据连接器以读取输入数据，并将其与TFDV核心API连接以计算数据统计信息。

用于计算数据统计信息的TFDV核心API是Beam PTransform，它接收输入示例的PCollection（特征名称对特征值的numpy数组的指示），并输出包含单个DatasetFeatureStatisticsList协议缓冲区的PCollection。 例如，如果输入数据有两个示例：

	feature_a	feature_b
Example1	1, 2, 3	'a', 'b'
Example2	4, 5	NULL

然后，tfdv.GenerateStatistics PTransform的输入应该是字典的PCollection。

```
[
  # Example1
  {
    'feature_a': numpy.array([1, 2, 3]),
    'feature_b': numpy.array(['a', 'b'])
  },
  # Example2
  {
    'feature_a': numpy.array([4, 5])
  }
]
```

一旦实现了将输入示例转换为字典的自定义数据连接器，就需要将其与tfdv.GenerateStatistics API连接以计算数据统计信息。 以tf.train.Example的TFRecord为例。 我们提供了TFExampleDecoder数据连接器，下面是如何将其与tfdv.GenerateStatistics API连接的示例。

```
import tensorflow_data_validation as tfdv
import apache_beam as beam
from tensorflow_metadata.proto.v0 import statistics_pb2

DATA_LOCATION = ''
OUTPUT_LOCATION = ''

with beam.Pipeline() as p:
    _ = (
    p
    # 1. Read out the examples from input files.
    | 'ReadData' >> beam.io.ReadFromTFRecord(file_pattern=DATA_LOCATION)
    # 2. Convert each example to a dict of feature name to numpy array of feature values.
    | 'DecodeData' >> beam.Map(tfdv.TFExampleDecoder().decode)
    # 3. Invoke TFDV `GenerateStatistics` API to compute the data statistics.
    | 'GenerateStatistics' >> tfdv.GenerateStatistics()
    # 4. Materialize the generated data statistics.
    | 'WriteStatsOutput' >> beam.io.WriteToTFRecord(
        file_path_prefix = OUTPUT_LOCATION,
        shard_name_template='',
        coder=beam.coders.ProtoCoder(
            statistics_pb2.DatasetFeatureStatisticsList)))
```

