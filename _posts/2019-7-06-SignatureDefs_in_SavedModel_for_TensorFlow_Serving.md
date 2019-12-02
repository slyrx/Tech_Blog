---
layout: post
title:  "SignatureDefs in SavedModel for TensorFlow Serving"
date:   2019-07-06 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第十篇。
        本文档提供了 SavedModel 中 SignatureDef 的预期用法示例，这些示例映射到 TensorFlow Serving 的 API 。


目的
本文档提供了SavedModel中SignatureDef的预期用法示例，这些示例映射到TensorFlow Serving的API。

总览
SignatureDef定义TensorFlow图中支持的计算的签名。 SignatureDefs旨在提供通用支持以标识函数的输入和输出，并且可以在构建SavedModel时指定。

背景
TF-Exporter和SessionBundle使用的签名在概念上相似，但是要求用户区分命名签名和默认签名，以便在加载时正确检索它们。对于以前使用TF-Exporter / SessionBundle的用户，TF-Exporter中的签名将由SavedModel中的SignatureDefs替换。

SignatureDef结构
SignatureDef要求指定以下内容：

输入为到TensorInfo的字符串映射。
输出为TensorInfo的字符串映射。
method_name（与加载工具/系统中支持的方法名称相对应）。
注意TensorInfo本身需要名称，dtype和张量形状的规范。虽然图中已经存在张量信息，但是明确定义TensorInfo作为SignatureDef的一部分很有用，因为工具随后可以执行签名验证等，而不必读取图形定义。

相关常量和实用程序
为了方便在工具和系统之间重用和共享，将在TensorFlow Serving中支持的与SignatureDef相关的常用常量定义为常量。特别：

Python中的签名常量。
C ++中的签名常量。
此外，SavedModel提供了一个实用程序来帮助构建签名定义。

样本结构
TensorFlow Serving提供用于执行推理的高级API。要启用这些API，模型必须包含一个或多个SignatureDef，它们定义要用于输入和输出的确切TensorFlow节点。 TensorFlow Serving支持每个API的特定SignatureDef的示例，请参见下文。

请注意，TensorFlow Serving取决于每个TensorInfo的键（在SignatureDef的输入和输出中）以及SignatureDef的method_name。 TensorInfo的实际内容特定于您的图形。

分类SignatureDef
分类SignatureDefs支持对TensorFlow Serving的分类API的结构化调用。这些规定必须有一个输入张量，并且有两个可选的输出张量：类和分数，其中至少必须存在一个。

```
signature_def: {
  key  : "my_classification_signature"
  value: {
    inputs: {
      key  : "inputs"
      value: {
        name: "tf_example:0"
        dtype: DT_STRING
        tensor_shape: ...
      }
    }
    outputs: {
      key  : "classes"
      value: {
        name: "index_to_string:0"
        dtype: DT_STRING
        tensor_shape: ...
      }
    }
    outputs: {
      key  : "scores"
      value: {
        name: "TopKV2:0"
        dtype: DT_FLOAT
        tensor_shape: ...
      }
    }
    method_name: "tensorflow/serving/classify"
  }
}
```

预测SignatureDef
Predict SignatureDefs支持对TensorFlow Serving的Predict API的调用。这些签名使您可以灵活地支持任意多个输入和输出张量。对于以下示例，签名my_prediction_signature具有单个逻辑输入Tensor图像，这些图像映射到图形x：0中的实际Tensor。

Predict SignatureDefs支持跨模型的可移植性。这意味着您可以交换不同的SavedModels，可能使用不同的基础Tensor名称（例如，代替x：0也许您有一个带有Tensor z：0的新替代模型），而您的客户可以保持在线状态，持续查询旧的和新的此模型的版本没有客户端更改。

Predict SignatureDefs还允许您向输出中添加可选的其他张量，您可以显式查询。假设除了分数的下面的输出键外，您还希望获取池层用于调试或其他目的。在这种情况下，您只需添加一个额外的Tensor，并使用键（例如pool和适当的值）。


```
signature_def: {
  key  : "my_prediction_signature"
  value: {
    inputs: {
      key  : "images"
      value: {
        name: "x:0"
        dtype: ...
        tensor_shape: ...
      }
    }
    outputs: {
      key  : "scores"
      value: {
        name: "y:0"
        dtype: ...
        tensor_shape: ...
      }
    }
    method_name: "tensorflow/serving/predict"
  }
}
```

回归签名Def
回归SignatureDefs支持对TensorFlow Serving的回归API的结构化调用。 这些规定必须完全有一个输入张量，一个输出张量。

```
signature_def: {
  key  : "my_regression_signature"
  value: {
    inputs: {
      key  : "inputs"
      value: {
        name: "x_input_examples_tensor_0"
        dtype: ...
        tensor_shape: ...
      }
    }
    outputs: {
      key  : "outputs"
      value: {
        name: "y_outputs_0"
        dtype: DT_FLOAT
        tensor_shape: ...
      }
    }
    method_name: "tensorflow/serving/regress"
  }
}
```