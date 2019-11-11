---
layout: post
title:  "Building Standard TensorFlow ModelServer"
date:   2019-07-01 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第五篇。
        本篇描述如何构建一个标准的 Tensorflow 模型服务。是实际生产中部署不可或缺的参考手册。

## 目录
+ 训练和导出 TensorFlow 模型
+ 服务内核
+ 批处理
+ 测试和运行服务


这篇内容将向你介绍怎么使用 Tensorflow 服务组件来构建标准的 Tensorflow 模型服务，以便于动态的发现和服务新版本的 Tensorflow 训练模型。如果你知识想要使用标准服务来运行你的模型，参见 [Tensorflow 服务基础手册](https://www.tensorflow.org/tfx/serving/serving_basic)。

这个手册使用了 Tensorflow 手册中用于手写图像（MNIST data）分类问题的简单的 Softmax 回归模型。如果你还不知道什么是 Tensorflow 或者 MNIST, 参见 [机器学习初学者 MNIST 手册](http://www.tensorflow.org/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners)。

这个部分的代码有两部分：
+ 名为 mnist_saved_model.py 的 Python 文件，可以训练和导出多个版本的模型
+ 名为 main.cc 的 C++ 文件，它是标准的 Tensorflow 模型服务，用以发现新导出的模型和运行 gRPC 服务这些模型。

这篇介绍将按照下面的任务执行顺序进行展开：
1. 训练和导出一个 Tensorflow 模型
2. 使用 Tensorflow 服务 ServerCore 管理模型版本。
3. 使用 SessionBundleSourceAdapterConfig 来配置批处理
4. 使用 TensorFlow 服务的 ServerCore 来服务请求
5. 运行和测试服务

在开始前，需要先安装好 Docker。

### 训练和导出 TensorFlow 模型
首先，如果你还没有克隆下面的库到本地，先执行下面的代码：

```
git clone https://github.com/tensorflow/serving.git
cd serving
```

如果已经存在导出的文件夹，需要先执行清除命令：

```
rm -rf /tmp/models
```

训练大概100个迭代，然后导出第一个版本的模型：

```
tools/run_in_docker.sh python tensorflow_serving/example/mnist_saved_model.py \
  --training_iteration=100 --model_version=1 /tmp/mnist
```

训练大概2000个迭代，然后导出第二个版本的模型：

```
tools/run_in_docker.sh python tensorflow_serving/example/mnist_saved_model.py \
  --training_iteration=2000 --model_version=2 /tmp/mnist
```

正如你在 mnist_saved_model.py 中看到的，训练和导出在[ Tensorflow 服务基础手册](https://www.tensorflow.org/tfx/serving/serving_basic)中采用了同样的方式。为了做演示，你可以有意地在第一次运行时调小训练的迭代次数，并将它导出为 v1，之后，正常对其进行第二次训练并导出它为 v2 到相同的父目录中。因为我们期望后者的成果物借由更多有意的训练，能较前者能有更高的准确率。你应该能在 `/tmp/mnist` 文件夹看到每次训练的训练数据。

```
$ ls /tmp/mnist
1  2
```

### 服务核心
现在，想象 v1 和 v2 模型可以动态的在运行时生成，同时新的算法也正在被实验，或者同时模型正在用新的数据集进行训练。在生产环境中，你可能希望构建一个服务可以支持逐渐发布，例如 v2 可以被发现，加载，实验，监控或者回滚，当 v1 还在服务时。或者，你也可能想要在引入 v2 前拆除 v1。Tensorfolw 服务支持这些选项：它们之一利于在过度时期保持可利用的状态，另一个则利于高效的使用资源，例如 RAM.

Tensorflow 服务 Manager 负责上面的过程。它掌握了 Tensorflow 模型在版本转换时的全生命周期，包括加载，服务和卸载等活动。在这个手册中，你将会在 Tensorflow 服务 ServerCore 的顶端构建自己的服务, ServerCore 的内部包括了一个 AspiredVersionsManager。

```
int main(int argc, char** argv) {
  ...

  ServerCore::Options options;
  options.model_server_config = model_server_config;
  options.servable_state_monitor_creator = &CreateServableStateMonitor;
  options.custom_model_config_loader = &LoadCustomModelConfig;

  ::google::protobuf::Any source_adapter_config;
  SavedModelBundleSourceAdapterConfig
      saved_model_bundle_source_adapter_config;
  source_adapter_config.PackFrom(saved_model_bundle_source_adapter_config);
  (*(*options.platform_config_map.mutable_platform_configs())
      [kTensorFlowModelPlatform].mutable_source_adapter_config()) =
      source_adapter_config;

  std::unique_ptr<ServerCore> core;
  TF_CHECK_OK(ServerCore::Create(options, &core));
  RunServer(port, std::move(core));

  return 0;
}
```

ServerCore::Create() 带有一个 ServerCore::Options 参数：
这里有一些通常使用的选项：
+ 指定要加载的模型的ModelServerConfig。 通过model_config_list（定义一个静态模型列表）或custom_model_config（定义一个自定义方式来声明可能在运行时更新的模型列表）来声明模型。从平台名称（例如tensorflow）映射到用来创建SourceAdapter的PlatformConfig的
+ PlatformConfigMap。 SourceAdapter使StoragePath（发现模型版本的路径）适应模型加载程序（从存储路径加载模型版本，并为Manager提供状态转换接口）。 如果PlatformConfig包含SavedModelBundleSourceAdapterConfig，将创建一个SavedModelBundleSourceAdapter，我们将在后面解释。

SavedModelBundle是TensorFlow Serving的关键组件。 它代表从给定路径加载的TensorFlow模型，并提供与TensorFlow相同的Session :: Run接口以运行推理。 SavedModelBundleSourceAdapter使存储路径适应Loader <SavedModelBundle>，以便可以由Manager管理模型寿命。 请注意，SavedModelBundle是不推荐使用的SessionBundle的后继者。 鼓励用户使用SavedModelBundle，因为对SessionBundle的支持将很快删除。

有了这些，ServerCore在内部执行以下操作：

+ 实例化一个FileSystemStoragePathSource，它监视在model_config_list中声明的模型导出路径。
+ 使用PlatformConfigMap与在model_config_list中声明的模型平台实例化SourceAdapter，并将FileSystemStoragePathSource连接到该适配器。这样，只要在导出路径下发现了新的模型版本，SavedModelBundleSourceAdapter就会将其适应于Loader <SavedModelBundle>。
+ 实例化一个称为AspiredVersionsManager的Manager的特定实现，该实现管理所有由SavedModelBundleSourceAdapter创建的此类Loader实例。 ServerCore通过将调用委派给AspiredVersionsManager来导出Manager界面。

只要有新版本可用，此AspiredVersionsManager就会加载新版本，并在其默认行为下卸载旧版本。如果要开始自定义，建议您了解它在内部创建的组件以及如何配置它们。

值得一提的是，TensorFlow Serving从头开始设计，非常灵活且可扩展。您可以构建各种插件来自定义系统行为，同时利用ServerCore和AspiredVersionsManager等通用核心组件。例如，您可以构建一个监视云存储而不是本地存储的数据源插件，或者可以构建以不同方式进行版本转换的版本策略插件-实际上，您甚至可以构建一个自定义模型插件来提供服务非TensorFlow模型。这些主题超出了本教程的范围。但是，您可以参考自定义源和自定义可服务教程以获取更多信息。

### 批处理
我们在生产环境中需要的另一个典型服务器功能是批处理。 当大批量运行推理请求时，用于进行机器学习推理的现代硬件加速器（GPU等）通常可实现最佳计算效率。

在创建SavedModelBundleSourceAdapter时，可以通过提供适当的SessionBundleConfig来打开批处理。 在这种情况下，我们为BatchingParameters设置了很多默认值。 可以通过设置自定义超时，batch_size等值来微调批处理。 有关详细信息，请参阅BatchingParameters。

```
SessionBundleConfig session_bundle_config;
// Batching config
if (enable_batching) {
  BatchingParameters* batching_parameters =
      session_bundle_config.mutable_batching_parameters();
  batching_parameters->mutable_thread_pool_name()->set_value(
      "model_server_batch_threads");
}
*saved_model_bundle_source_adapter_config.mutable_legacy_config() =
    session_bundle_config;
```

达到满批处理后，推理请求将在内部合并为一个大请求（张量），并调用tensorflow :: Session :: Run（）（这是从GPU上获得实际效率的地方）


### 使用管理器进行服务

如上所述，TensorFlow Serving Manager被设计为通用组件，可以处理由任意机器学习系统生成的模型的加载，服务，卸载和版本转换。其API围绕以下关键概念构建：

Servable：Servable是可用于服务客户端请求的任何不透明对象。 servable的大小和粒度是灵活的，因此单个servable可能包括从查找表的单个分片到单个机器学习的模型再到模型的元组的任何内容。 servable可以是任何类型和接口。

可服务版本：可服务版本已版本化，TensorFlow Serving Manager可以管理一个或多个可服务版本。版本控制允许同时加载多个可服务项目版本，从而支持逐步推出和试验。

可服务的流：可服务的流是可服务的版本的序列，版本号不断增加。

模型：机器学习的模型由一个或多个可服务对象表示。可使用的示例包括：

TensorFlow会话或它们周围的包装器，例如SavedModelBundle。
其他种类的机器学习模型。
词汇查询表。
嵌入查询表。
组合模型可以表示为多个独立的可服务项，也可以表示为单个组合的可服务项。可服务项也可能对应于模型的一部分，例如，具有在许多Manager实例之间分片的大型查找表。

要将所有这些内容放在本教程的上下文中：

TensorFlow模型由一种可服务的-SavedModelBundle表示。 SavedModelBundle内部由一个tensorflow：Session组成，该会话与一些元数据配对，这些元数据涉及将什么图形加载到会话中以及如何运行它以进行推断。

有一个文件系统目录，其中包含TensorFlow导出流，每个都在其自己的子目录中，该子目录的名称是版本号。外部目录可以被认为是所服务的TensorFlow模型的可服务流的序列化表示。每个导出对应一个可以加载的可食用项。

AspiredVersionsManager监视导出流，并动态管理所有SavedModelBundle可服务对象的生命周期。

TensorflowPredictImpl :: Predict然后只是：

从管理器（通过ServerCore）请求SavedModelBundle。
使用通用签名将PredictRequest中的逻辑张量名称映射到实际张量名称，并将值绑定到张量。
运行推理。

### 测试并运行服务器
将导出的第一个版本复制到受监视的文件夹中：

```
mkdir /tmp/monitored
cp -r /tmp/mnist/1 /tmp/monitored
```

然后启动服务器：

```
docker run -p 8500:8500 \
  --mount type=bind,source=/tmp/monitored,target=/models/mnist \
  -t --entrypoint=tensorflow_model_server tensorflow/serving --enable_batching \
  --port=8500 --model_name=mnist --model_base_path=/models/mnist &
```

服务器将每秒发送一次日志消息，上面写着“ Aspiring version for servable ...”，这表示它已找到导出文件，并正在跟踪其继续存在的状态。

让我们以--concurrency = 10运行客户端。这会将并发请求发送到服务器，从而触发您的批处理逻辑。


```
tools/run_in_docker.sh python tensorflow_serving/example/mnist_client.py \
  --num_tests=1000 --server=127.0.0.1:8500 --concurrency=10
```

结果如下所示：

```
...
Inference error rate: 13.1%
```

然后，将导出的第二个版本复制到受监视的文件夹中，然后重新运行测试：

```
cp -r /tmp/mnist/2 /tmp/monitored
tools/run_in_docker.sh python tensorflow_serving/example/mnist_client.py \
  --num_tests=1000 --server=127.0.0.1:8500 --concurrency=10
```

结果如下所示：

```
...
Inference error rate: 9.5%
```
这确认您的服务器会自动发现新版本并将其用于服务！





