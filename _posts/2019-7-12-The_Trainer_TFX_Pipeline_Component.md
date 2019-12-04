---
layout: post
title:  "The Trainer TFX Pipeline Component"
date:   2019-07-12 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第十六篇。
        本文档详细介绍了 Trainer 组件的作用。


## 目录
+ 训练器和 TensorFlow
+ 配置一个训练器组件

## 正文

Trainer TFX管道组件训练TensorFlow模型。

培训师需要：

tf。用于培训和评估的示例。
用户提供的模块文件，用于定义训练器逻辑。
由SchemaGen管道组件创建的数据模式，开发人员可以选择更改数据模式。
训练参数和评估参数的原型定义。
由上游Transform组件生成的可选变换图。
用于诸如热启动之类的方案的可选基本模型。
Trainer发出：一个SavedModel和一个EvalSavedModel

培训师和TensorFlow
Trainer广泛使用Python TensorFlow API来训练模型。

注意：TFX同时支持TensorFlow 1.x和2.0。但是，Trainer当前不支持Keras Model API。使用估算器或通过model_to_estimator从Keras模型创建估算器。
配置培训师组件
由于所有工作都是由Trainer TFX组件完成的，因此Trainer管道组件通常非常易于开发并且几乎不需要自定义。但是，您的TensorFlow建模代码可能会非常复杂。

典型的代码如下所示：

```
from tfx import components

...

trainer = Trainer(
      module_file=module_file,
      transformed_examples=transform.outputs['transformed_examples'],
      schema=infer_schema.outputs['schema'],
      base_models=latest_model_resolver.outputs['latest_model'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))

```

培训师调用一个培训模块，该模块在module_file参数中指定。 一个典型的培训模块如下所示：

```
# TFX will call this function
def trainer_fn(trainer_fn_args, schema):
  """Build the estimator using the high level API.

  Args:
    trainer_fn_args: Holds args used to train the model as name/value pairs.
    schema: Holds the schema of the training examples.

  Returns:
    A dict of the following:
      - estimator: The estimator that will be used for training and eval.
      - train_spec: Spec for training.
      - eval_spec: Spec for eval.
      - eval_input_receiver_fn: Input function for eval.
  """
  # Number of nodes in the first layer of the DNN
  first_dnn_layer_size = 100
  num_dnn_layers = 4
  dnn_decay_factor = 0.7

  train_batch_size = 40
  eval_batch_size = 40

  tf_transform_output = tft.TFTransformOutput(trainer_fn_args.transform_output)

  train_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      trainer_fn_args.train_files,
      tf_transform_output,
      batch_size=train_batch_size)

  eval_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      trainer_fn_args.eval_files,
      tf_transform_output,
      batch_size=eval_batch_size)

  train_spec = tf.estimator.TrainSpec(  # pylint: disable=g-long-lambda
      train_input_fn,
      max_steps=trainer_fn_args.train_steps)

  serving_receiver_fn = lambda: _example_serving_receiver_fn(  # pylint: disable=g-long-lambda
      tf_transform_output, schema)

  exporter = tf.estimator.FinalExporter('chicago-taxi', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=trainer_fn_args.eval_steps,
      exporters=[exporter],
      name='chicago-taxi-eval')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)

  run_config = run_config.replace(model_dir=trainer_fn_args.serving_model_dir)
  warm_start_from = trainer_fn_args.base_models[
      0] if trainer_fn_args.base_models else None

  estimator = _build_estimator(
      # Construct layers sizes with exponetial decay
      hidden_units=[
          max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
          for i in range(num_dnn_layers)
      ],
      config=run_config,
      warm_start_from=warm_start_from)

  # Create an input receiver for TFMA processing
  receiver_fn = lambda: _eval_input_receiver_fn(  # pylint: disable=g-long-lambda
      tf_transform_output, schema)

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': receiver_fn
  }

  ```
  