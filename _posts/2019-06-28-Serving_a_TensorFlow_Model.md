---
layout: post
title:  "Serving a TensorFlow Model"
date:   2019-06-28 10:11:30
tags: [tensorflow]
---

    导语：
       本文是 tensorflow 手册翻译系列的第二篇。
       原文链接：https://www.tensorflow.org/tfx/serving/serving_basic
       本文讲述如何使一个 TensorFlow 模型作为一个服务启用。

## 目录
+ 训练和导出 Tensorflow 模型
+ 使用标准的模型服务加载导出模型
+ 测试服务

这本手册展示了怎么使用 Tensorflow 服务组件取导出一个 Tensorflow 模型，并使用一个标准的 Tensorflow 模型服务 tensorflow_model_server 来实现服务的功能。如果你已经熟悉 Tensorflow 服务了，希望知道更多服务内部的工作原理，可以参见[Tensorflow 服务高级手册](https://www.tensorflow.org/tfx/serving/serving_advanced)

这本手册使用了简单的 Softmax 回归模型，该模型在 MNIST data 手写数字图像分类的讲解中介绍过。如果你还不知道什么是 Tensorflow 或者 MNIST，参见[MNIST初识](http://www.tensorflow.org/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners)
        Softmax：
            该函数是元素指数和所有元素指数的比值。通常应用于多分类问题。

在本节的介绍中，代码分为以下两个部分：
+ mnist_saved_model.py, 用于训练和导出模型的 python 文件
+ 二进制模型服务文件，可以用于 apt 安装或者使用 C++ 文件进行编译(eg. main.cc)。Tensorflow 服务的模型服务会发现新导入的模型并将其作为一个 gRPC 服务运行它们。

在开始前，记得先安装 Docker。

训练并导出 Tensorflow 模型
正如在 mnist_saved_model.py 中看到的那样，训练以在“MNIST初识”同样的方式完成了。Tensorflow 会话 session 中启动了 Tensorflow 图，张量图片以x作为输入，Softmax 分数以y作为输出张量。

之后，使用 Tensorflow 的保存模型构建器模块 SavedModelBuilder 进行导出。保存模型构建器会保存一个训练模型的快照到可靠的存储器中，这样方便在后续的推理中加载。

关于保存模型格式的细节，参见文档[SavedModel README.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md)。

以下是从 mnist_saved_model.py 中截取的代码片段，解释了一般的保存模型到硬盘的过程。
```
export_path_base = sys.argv[-1]
export_path = os.path.join(
      compat.as_bytes(export_path_base),
      compat.as_bytes(str(FLAGS.model_version)))
print('Exporting trained model to', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
           'predict_images':
               prediction_signature,
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               classification_signature,
      },
      main_op=tf.tables_initializer())
builder.save()
```

SavedModelBuilder.__init__ 有以下参数：
+ export_path 导出文件夹的路径

如果不存在该路径，则 SavedModelBuilder 会创建新的文件夹。在例子中，我们将命令行参数和 FLAGS.model_version 参数联系起来，来获得导出路径。FLAGS.model_version 特指模型的版本号。关于相同模型的版本应该指定一个大的整数给新版本。每个版本的模型都会导出到路径下不同的子文件夹下。

使用  SavedModelBuilder.add_meta_graph_and_variables() 添加单元图和变量到构建器中，需要添加的参数有以下内容：
+ sess 是 Tensorflow 的会话，它含有准备导出的训练模型。
+ tags 是用来保存单元图的 tags 的集合。在本例中，因为我们打算用图做服务，所以我们使用从预定义的保存模型标签常量中选择 serve 标签。更多的细节，参见 [tag_constants.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/tag_constants.py) 和 [相关的 Tensorflow API 文档](https://www.tensorflow.org/api_docs/python/tf/saved_model/tag_constants)。
+ signature_def_map 指定了 tensorflow 的用户密钥映射:: 添加 SignatureDef 到单元图中。 签名指定什么类型的模型被导出，输入和输出张量也在运行推理时绑定好了。

特殊的签名密钥 serving_default 指定了默认的服务签名。默认的服务签名密钥，伴随着其他签名常量，被定义成为了保存模型里的一部分签名常量。更多的细节，参见 [signature_constants.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_constants.py)和[相关的 TensorFlow 1.0 API 文档](https://www.tensorflow.org/api_docs/python/tf/saved_model/signature_constants)。

进一步的，为了让构建签名定义更简单，保存模型 API 提供了[签名定义工具包](https://www.tensorflow.org/api_docs/python/tf/saved_model/signature_def_utils)。具体来说，在原始的 [mnist_saved_model.py](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_saved_model.py) 文件中，使用 signature_def_utils.build_signature_def() 来构建预测签名 predict_signature 和分类签名 classification_signature。

以下是关于 predict_signature 怎样被定义的例子，工具包选择了以下参数：
+ inputs={'images': tensor_info_x} 指定了输入张量的信息
+ outputs={'scores': tensor_info_y} 指定了输出张量的信息
+ method_name 是用作推理的方法名称。为了预测的需求，应该设置到 tensorflow/serving/predict 中。其他到名字参见 [signature_constants.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_constants.py)和[TensorFlow 1.0 API 相关文档](https://www.tensorflow.org/api_docs/python/tf/saved_model/signature_constants)


注意，tensor_info_x 和 tensor_info_y 有 tensorflow::TensorInfo 的结构协议定义在[这里](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto)。为了方便 TensorFlow 保存模型 API 的使用，同时还提供了 [utils.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/utils.py)和[TensorFlow 1.0 API 相关文档](https://www.tensorflow.org/api_docs/python/tf/saved_model/signature_constants)

同时，images 和 scores 是张量的别名。它们可以设定成任何你希望的名称，它们同时也会变成张量 x 和 y 的逻辑名称，方便之后在预测时向对应的模型发送张量的设定。

例如，如果 x 的张量名称设定为 'long_tensor_name_foo'，同时 y 设定为 'generated_tensor_name_bar'，构建器将保存张量的逻辑名称到真名的映射上，例如把 'images' -> 'long_tensor_name_foo'，把 'scores' -> 'generated_tensor_name_bar'。这允许用户当在推理阶段时，使用逻辑名称设定这些张量。


现在来执行一下：
首先，如果还没有将项目 clone 到本地，执行以下命令
```
git clone https://github.com/tensorflow/serving.git
cd serving
```

如果已经存在输出文件夹，删除掉
```
rm -rf /tmp/mnist
```

现在开始训练模型吧
```
tools/run_in_docker.sh python tensorflow_serving/example/mnist_saved_model.py \
  /tmp/mnist
```

这将会产生下面的结果输出：
```
Training model...

...

Done training!
Exporting trained model to models/mnist
Done exporting!
```

现在，来看一看输出文件夹
```
$ ls /tmp/mnist
1
```

就像上面提到的，一个模型会创建一个子文件夹。FLAGS.model_version 有默认值为1，因此对应的子文件夹1就被创建了。

```
$ ls /tmp/mnist/1
saved_model.pb variables
```

每个版本的子文件夹包括如下文件：
+ saved_model.pb 是序列化之后的 tensorflow 保存模型。它包括一个或多个图模型的定义，也包括模型的元数据，比如签名。
+ variables 是持有序列化图变量的文件

有了这些，你的张量模型就导出成功了，可以用作下次加载了。


## 使用标准的 TensorFlow 模型服务加载导出模型
使用 Docker 服务镜像可以简单的加载模型作为服务
```
docker run -p 8500:8500 \
--mount type=bind,source=/tmp/mnist,target=/models/mnist \
-e MODEL_NAME=mnist -t tensorflow/serving &
```

## 测试服务
我们可以使用提供的 mnist 客户端工具包来测试服务。客户端会下载 MNIST 测试数据，把它们当作请求发送到服务器，并计算出推理的错误率。
```
tools/run_in_docker.sh python tensorflow_serving/example/mnist_client.py \
  --num_tests=1000 --server=127.0.0.1:8500
```

这将产生以下输出结果：
```
    ...
    Inference error rate: 11.13%
```

我们期望 Softmax 模型有 90% 的正确率，但是，依据1000张测试图片的推理错误率我们得到 11%。这证明了服务器成功的加载并运行了模型。




