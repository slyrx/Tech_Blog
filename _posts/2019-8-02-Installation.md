---
layout: post
title:  "Installation"
date:   2019-08-02 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第三十七篇。
        本文档介绍模型服务器的安装。

## 目录
+ 安装模型服务器
+ + 使用 Docker 进行安装
+ + 使用 APT 进行安装
+ 从源码进行构建
+ TensorFlow 服务 Python API PIP 包


## 正文

安装ModelServer
使用Docker安装
使用TensorFlow Serving的最简单，最直接的方法是使用Docker映像。除非您有在容器中运行无法满足的特定需求，否则我们强烈建议您使用此路线。

提示：这也是使TensorFlow Serving与GPU支持一起工作的最简单方法。

使用APT安装
可用的二进制文件
TensorFlow Serving ModelServer二进制文件有两个变体：

tensorflow-model-server：完全优化的服务器，使用某些平台特定的编译器优化，例如SSE4和AVX指令。对于大多数用户而言，这应该是首选选项，但在某些较旧的计算机上可能无法使用。

tensorflow-model-server-universal：进行了基本优化，但不包括平台特定的指令集，因此应该可以在大多数（即使不是全部）机器上使用。如果tensorflow-model-server对您不起作用，请使用此选项。请注意，两个软件包的二进制名称相同，因此如果您已经安装了tensorflow-model-server，则应首先使用卸载它

```
apt-get remove tensorflow-model-server
```

安装
将TensorFlow Serving发行版URI添加为包来源（一次性设置）

```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
```

Install and update TensorFlow ModelServer

```
apt-get update && apt-get install tensorflow-model-server
<!-- common_typos_enable -->
```

安装后，可以使用命令tensorflow_model_server调用二进制文件。

您可以使用以下命令升级到新版本的tensorflow-model-server：

```
apt-get upgrade tensorflow-model-server
```

注意：如果您的处理器不支持AVX指令，请在上述命令中将tensorflow-model-server替换为tensorflow-model-server-universal。
从源头建造
从源代码构建的推荐方法是使用Docker。 TensorFlow Serving Docker开发映像封装了构建自己的TensorFlow Serving版本所需的所有依赖项。

有关这些依赖项的列表，请参见TensorFlow服务开发Dockerfile [CPU，GPU]。

注意：目前，我们仅支持构建在Linux上运行的二进制文件。
安装Docker
常规安装说明在Docker站点上。

克隆构建脚本
安装Docker之后，我们需要获取要构建的源代码。 我们将使用Git克隆TensorFlow Serving的master分支：

```
git clone https://github.com/tensorflow/serving.git
cd serving
```

建立
为了在一个封闭的环境中构建并处理所有依赖项，我们将使用run_in_docker.sh脚本。 该脚本将构建命令传递到Docker容器。 默认情况下，该脚本将使用最新的夜间Docker开发映像进行构建。

TensorFlow Serving使用Bazel作为其构建工具。 您可以使用Bazel命令来构建单个目标或整个源代码树。

要构建整个树，执行：

```
tools/run_in_docker.sh bazel build -c opt tensorflow_serving/...
```

二进制文件放置在bazel-bin目录中，可以使用以下命令运行：

```
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
```

要测试您的构建，请执行：

```
tools/run_in_docker.sh bazel test -c opt tensorflow_serving/...
```

有关运行TensorFlow Serving的更多深入示例，请参阅基础教程和高级教程。

构建特定版本的TensorFlow Serving
如果要从特定分支（例如发行分支）进行构建，请将-b <branchname>传递给git clone命令。

我们还希望通过将run_in_docker.sh脚本传递给我们要使用的Docker开发映像来匹配该代码分支的构建环境。

例如，要构建TensorFlow Serving的1.10版本：

```
$ git clone -b r1.10 https://github.com/tensorflow/serving.git
...
$ cd serving
$ tools/run_in_docker.sh -d tensorflow/serving:1.10-devel \
  bazel build tensorflow_serving/...
...
```

优化的构建
如果您想应用普遍推荐的优化方法，包括为处理器使用平台特定的指令集，则可以在构建TensorFlow Serving时将--config = nativeopt添加到Bazel构建命令中。

例如：

```
tools/run_in_docker.sh bazel build --config=nativeopt tensorflow_serving/...
```

也可以使用特定的指令集（例如AVX）进行编译。 只要在文档中看到bazel构建的任何地方，只需添加相应的标志即可：

Instruction Set	Flags
AVX	--copt=-mavx
AVX2	--copt=-mavx2
FMA	--copt=-mfma
SSE 4.1	--copt=-msse4.1
SSE 4.2	--copt=-msse4.2
All supported by processor	--copt=-march=native

例如：

```
tools/run_in_docker.sh bazel build --copt=-mavx2 tensorflow_serving/...
```

注意：这些指令集并非在所有机器上都可用，尤其是在较旧的处理器上。 如有疑问，请使用默认的--config = nativeopt为您的处理器构建TensorFlow Serving的优化版本。
使用GPU支持构建
为了构建具有GPU支持的TensorFlow Serving的自定义版本，我们建议您使用提供的Docker映像进行构建，或者遵循GPU Dockerfile中的方法进行构建。

TensorFlow服务Python API PIP软件包
要运行Python客户端代码而不需要构建API，可以使用以下命令安装tensorflow-serving-api PIP软件包：

```
pip install tensorflow-serving-api
```



