---
layout: post
title:  "Using Fairness Indicators"
date:   2019-07-31 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第三十五篇。
        本文档介绍如何使用公平指标。

## 目录
+ Tensorflow 模型
+ + 数据
+ + 模型
+ + 配置切片
+ + 计算公平标准
+ + 渲染公平指示器
+ 模型不可知评估
+ + 数据
+ + 模型
+ + 构建模型不可知抽取器
+ + 计算公平标准

## 正文

公平性指标旨在与更广泛的Tensorflow工具包合作，支持团队评估和改善公平性问题模型。目前，我们的许多产品都在内部积极使用该工具，BETA现在可以使用该工具来尝试您自己的用例。

公平指标使二元和多类分类器的常用标识公平度量易于计算。许多现有的评估公平性问题的工具在大规模数据集和模型上效果不佳。对于Google来说，拥有可以在十亿用户系统上运行的工具对我们很重要。公平指标将使您能够评估各种规模的用例。

Tensorflow模型
数据
要使用TFMA运行公平指标，请确保已为要分割的要素标记了评估数据集。如果您没有针对公平问题的确切切片功能，则可以尝试尝试找到具有此功能的评估集，或者考虑在功能集中考虑可能突出结果差异的代理功能。有关其他指导，请参见此处。

模型
您可以使用Tensorflow Estimator类来构建模型。 TFMA即将提供对Keras模型的支持。如果要在Keras模型上运行TFMA，请参见下面的“与模型无关的TFMA”部分。

评估估算器后，您将需要导出保存的模型以进行评估。要了解更多信息，请参阅TFMA指南。

配置切片
接下来，定义要评估的切片：

```
slice_spec = [
  tfma.slicer.SingleSliceSpec(columns=[‘fur color’])
]
```

如果要评估相交切片（例如，毛发颜色和高度），则可以设置以下内容：

```
slice_spec = [
  tfma.slicer.SingleSliceSpec(columns=[‘fur_color’, ‘height’])
]`
```

计算公平性指标
向“ metrics_callback”列表添加“公平指标”回调。 在回调中，您可以定义将在其中评估模型的阈值列表。

```
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators

# Build the fairness metrics. Besides the thresholds, you also can config the example_weight_key, labels_key here. For more details, please check the api.
metrics_callbacks = \
    [tfma.post_export_metrics.fairness_indicators(thresholds=[0.1, 0.3,
     0.5, 0.7, 0.9])]

eval_shared_model = tfma.default_eval_shared_model(
    eval_saved_model_path=tfma_export_dir,
    add_metrics_callbacks=metrics_callbacks)
```

在运行配置之前，请确定是否要启用置信区间的计算。 置信区间是使用Poisson自举计算的，需要重新计算20个样本。

```
compute_confidence_intervals = True
```

运行TFMA评估管道：

```
validate_dataset = tf.data.TFRecordDataset(filenames=[validate_tf_file])

# Run the fairness evaluation.
with beam.Pipeline() as pipeline:
  _ = (
      pipeline
      | beam.Create([v.numpy() for v in validate_dataset])
      | 'ExtractEvaluateAndWriteResults' >>
       tfma.ExtractEvaluateAndWriteResults(
                 eval_shared_model=eval_shared_model,
                 slice_spec=slice_spec,
                 compute_confidence_intervals=compute_confidence_intervals,
                 output_path=tfma_eval_result_path)
  )
eval_result = tfma.load_eval_result(output_path=tfma_eval_result_path)
```

渲染公平性指标

```
from tensorflow_model_analysis.addons.fairness.view import widget_view

widget_view.render_fairness_indicator(eval_result)
```

![png](https://www.tensorflow.org/tfx/guide/images/fairnessIndicators.png)

使用公平指标的提示：

选中要显示的指标，方法是选中左侧的框。每个指标的单独图表将按顺序显示在小部件中。
使用下拉选择器更改基线切片，即图形上的第一条。将使用此基准值计算增量。
使用下拉选择器选择阈值。您可以在同一图形上查看多个阈值。所选阈值将以粗体显示，您可以单击粗体阈值以取消选择它。
将鼠标悬停在栏上可以查看该切片的指标。
使用“差异基线”列确定与基线的差异，该列确定当前切片与基线之间的百分比差异。
使用假设工具深入研究切片的数据点。请参阅此处的示例。
模型不可知论评估
为了更好地支持具有不同模型和工作流程的客户，我们开发了一个评估库，该库与要评估的模型无关。

想要评估其机器学习系统的任何人都可以使用此功能，尤其是在您具有非基于TensorFlow的模型的情况下。使用Apache Beam Python SDK，您可以创建独立的TFMA评估二进制文件，然后运行它来分析模型。

数据
此步骤是提供您要运行评估的数据集。它应为tf.proto格式示例，其中包含您可能希望切片的标签，预测和其他功能。

```
tf.Example {
    features {
        feature {
          key: "fur_color" value { bytes_list { value: "gray" } }
        }
        feature {
          key: "height" value { bytes_list { value: "tall" } }
        }
        feature {
          key: "prediction" value { float_list { value: 0.9 } }
        }
        feature {
          key: "label" value { float_list { value: 1.0 } }
        }
    }
}
```

模型
无需指定模型，而是创建模型不可知的评估配置和提取器以解析并提供TFMA计算指标所需的数据。 ModelAgnosticConfig规范定义了从输入示例中使用的功能，预测和标签。

为此，使用代表所有特征的键（包括标签键和预测键）以及代表特征数据类型的值来创建特征图。

```
feature_map[label_key] = tf.FixedLenFeature([], tf.float32, default_value=[0])
```

使用标签键，预测键和功能图创建模型不可知的配置。

```
model_agnostic_config = model_agnostic_predict.ModelAgnosticConfig(
    label_keys=list(ground_truth_labels),
    prediction_keys=list(predition_labels),
    feature_spec=feature_map)
```

设置模型不可知提取器
Extractor用于使用模型不可知的配置从输入中提取特征，标签和预测。 而且，如果要切片数据，则还需要定义切片键规范，其中包含有关要切片的列的信息。

```
model_agnostic_extractors = [
    model_agnostic_extractor.ModelAgnosticExtractor(
        model_agnostic_config=model_agnostic_config, desired_batch_size=3),
    slice_key_extractor.SliceKeyExtractor([
        slicer.SingleSliceSpec(),
        slicer.SingleSliceSpec(columns=[‘height’]),
    ])
]
```


计算公平性指标
作为EvalSharedModel的一部分，您可以提供希望对模型进行评估的所有指标。 指标以指标回调的形式提供，例如post_export_metrics或fairness_indicators中定义的回调。

```
metrics_callbacks.append(
    post_export_metrics.fairness_indicators(
        thresholds=[0.5, 0.9],
        target_prediction_keys=[prediction_key],
        labels_key=label_key))
```

它还接受了一个construct_fn，用于创建张量流图以执行评估。

```
eval_shared_model = types.EvalSharedModel(
    add_metrics_callbacks=metrics_callbacks,
    construct_fn=model_agnostic_evaluate_graph.make_construct_fn(
        add_metrics_callbacks=metrics_callbacks,
        fpl_feed_config=model_agnostic_extractor
        .ModelAgnosticGetFPLFeedConfig(model_agnostic_config)))
```

一切设置完毕后，请使用model_eval_lib提供的ExtractEvaluate或ExtractEvaluateAndWriteResults函数之一来评估模型。

```
_ = (
    examples |
    'ExtractEvaluateAndWriteResults' >>
        model_eval_lib.ExtractEvaluateAndWriteResults(
        eval_shared_model=eval_shared_model,
        output_path=output_path,
        extractors=model_agnostic_extractors))

eval_result = tensorflow_model_analysis.load_eval_result(output_path=tfma_eval_result_path)
```

最后，使用上面“渲染公平性指标”部分中的说明来渲染公平性指标。
