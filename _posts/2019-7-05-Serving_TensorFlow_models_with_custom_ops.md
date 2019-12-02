---
layout: post
title:  "Serving TensorFlow models with custom ops"
date:   2019-07-05 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第九篇。
        本文档说明了一些 tensorflow ops 库的相关内容。


## 目录
+ 通过 op 源复制到 Serving 项目中
+ 为 op 构建静态库
+ 使用链接的 op 构建 ModelServer
+ 服务包含您的自定义操作的模型
+ 发送推理请求以手动测试操作


TensorFlow预先构建了一个广泛的ops和op内核库（实现），这些库针对不同的硬件类型（CPU，GPU等）进行了微调。这些操作将自动链接到TensorFlow Serving ModelServer二进制文件中，而无需用户进行任何其他工作。但是，有两种用例要求用户将操作显式链接到ModelServer：

您已经编写了自己的自定义操作（例如，使用本指南）
您正在使用TensorFlow随附的已实现的操作
注意：从2.0版开始，TensorFlow不再分发contrib模块;如果您正在使用contrib ops服务TensorFlow程序，请使用本指南将这些ops明确链接到ModelServer。
无论是否实施了操作，为了为具有自定义操作的模型提供服务，都需要访问操作的源。本指南将引导您完成使用源代码制作可投放的自定义操作的步骤。有关实现自定义操作的指导，请参阅tensorflow / custom-op回购。

先决条件：安装Docker后，您已经克隆了TensorFlow Serving存储库，当前工作目录是该存储库的根目录。

通过op源复制到Serving项目中
为了使用您的自定义操作构建TensorFlow Serving，您首先需要将Op源代码复制到您的服务项目中。对于此示例，您将使用上述custom-op存储库中的tensorflow_zero_out。

在服务仓库中，创建一个custom_ops目录，该目录将容纳您所有的自定义操作。对于此示例，您将仅具有tensorflow_zero_out代码。

```
mkdir tensorflow_serving/custom_ops
cp -r <custom_ops_repo_root>/tensorflow_zero_out tensorflow_serving/custom_ops
```


为op构建静态库
在tensorflow_zero_out的BUILD文件中，您会看到一个目标生成一个共享对象文件（.so），将其加载到python中以创建和训练模型。 但是，TensorFlow Serving在构建时静态链接操作，并且需要一个.a文件。 因此，您将向该tensorflow_serving / custom_ops / tensorflow_zero_out / BUILD添加一个生成该文件的构建规则：

```
cc_library(
    name = 'zero_out_ops',
    srcs = [
        "cc/kernels/zero_out_kernels.cc",
        "cc/ops/zero_out_ops.cc",
    ],
    alwayslink = 1,
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ]
)
```

使用链接的op构建ModelServer
要为使用自定义op的模型提供服务，必须使用链接到该op的模型构建ModelServer二进制文件。具体来说，您将上面创建的zero_out_ops构建目标添加到ModelServer的BUILD文件中。

编辑tensorflow_serving / model_servers / BUILD以将您的自定义操作构建目标添加到server_lib目标中包含的SUPPORTED_TENSORFLOW_OPS中：

```
SUPPORTED_TENSORFLOW_OPS = [
    ...
    "//tensorflow_serving/custom_ops/tensorflow_zero_out:zero_out_ops"
]
```

然后使用Docker环境构建ModelServer：

```
tools/run_in_docker.sh bazel build tensorflow_serving/model_servers:tensorflow_model_server
```

服务包含您的自定义操作的模型
现在，您可以运行ModelServer二进制文件并开始提供包含此自定义操作的模型：

```
tools/run_in_docker.sh -o "-p 8501:8501" \
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
--rest_api_port=8501 --model_name=<model_name> --model_base_path=<model_base_path>
```

发送推理请求以手动测试操作
现在，您可以将推理请求发送到模型服务器以测试您的自定义操作：

```
curl http://localhost:8501/v1/models/<model_name>:predict -X POST \
-d '{"inputs": [[1,2], [3,4]]}'
```

该页面包含用于将REST请求发送到模型服务器的更完整的API。