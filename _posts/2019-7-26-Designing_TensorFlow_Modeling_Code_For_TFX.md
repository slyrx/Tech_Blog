---
layout: post
title:  "Designing TensorFlow Modeling Code For TFX"
date:   2019-07-26 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第三十篇。
        本文档介绍 TFX 的模型设计。



## 正文

在为TFX设计TensorFlow建模代码时，需要注意一些事项，包括选择建模API。

消耗：来自Transform的SavedModel和来自ExampleGen的数据
发射：以SavedModel格式训练的模型
注意：TFX同时支持TensorFlow 1.x和2.0。 但是，Trainer当前不支持Keras Model API。 使用估算器或通过model_to_estimator从Keras模型创建估算器。
模型的输入层应使用由Transform组件创建的SavedModel，并且应将Transform模型的层包含在模型中，以便在导出SavedModel和EvalSavedModel时，它们将包含由Transform创建的转换 零件。

TFX的典型TensorFlow模型设计如下所示：

```
def _build_estimator(tf_transform_dir,
                     config,
                     hidden_units=None,
                     warm_start_from=None):
  """Build an estimator for predicting the tipping behavior of taxi riders.

  Args:
    tf_transform_dir: directory in which the tf-transform model was written
      during the preprocessing step.
    config: tf.contrib.learn.RunConfig defining the runtime environment for the
      estimator (including model_dir).
    hidden_units: [int], the layer sizes of the DNN (input layer first)
    warm_start_from: Optional directory to warm start from.

  Returns:
    Resulting DNNLinearCombinedClassifier.
  """
  metadata_dir = os.path.join(tf_transform_dir,
                              transform_fn_io.TRANSFORMED_METADATA_DIR)
  transformed_metadata = metadata_io.read_metadata(metadata_dir)
  transformed_feature_spec = transformed_metadata.schema.as_feature_spec()

  transformed_feature_spec.pop(_transformed_name(_LABEL_KEY))

  real_valued_columns = [
      tf.feature_column.numeric_column(key, shape=())
      for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
  ]
  categorical_columns = [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=_VOCAB_SIZE + _OOV_SIZE, default_value=0)
      for key in _transformed_names(_VOCAB_FEATURE_KEYS)
  ]
  categorical_columns += [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=_FEATURE_BUCKET_COUNT, default_value=0)
      for key in _transformed_names(_BUCKET_FEATURE_KEYS)
  ]
  categorical_columns += [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=num_buckets, default_value=0)
      for key, num_buckets in zip(
          _transformed_names(_CATEGORICAL_FEATURE_KEYS),  #
          _MAX_CATEGORICAL_FEATURE_VALUES)
  ]
  return tf.estimator.DNNLinearCombinedClassifier(
      config=config,
      linear_feature_columns=categorical_columns,
      dnn_feature_columns=real_valued_columns,
      dnn_hidden_units=hidden_units or [100, 70, 50, 25],
      warm_start_from=warm_start_from)
```

