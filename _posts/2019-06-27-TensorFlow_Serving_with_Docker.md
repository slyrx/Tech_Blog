---
layout: post
title:  "TensorFlow Serving with Docker"
date:   2019-06-27 10:11:30
tags: [tensorflow]
---

    导语：
       本文是 tensorflow 手册翻译系列的第一篇。
       原文链接：https://www.tensorflow.org/tfx/serving/docker
       完整的项目首先需要基本的环境搭建作为基础，很多人往往是卡在了基础环境框架的搭建上了，因此对环境搭建的学习也是非常关键的。

## 目录
+ 安装 Docker
+ 使用 Docker 开启服务
+ + 拉取一个服务镜像
+ + 运行一个服务镜像
+ + 创建自己的服务镜像
+ + 服务的例子
+ 使用 GPU 进行 Docker 服务
+ + 安装 nvidia-docker
+ + 运行一个 GPU 服务镜像
+ + GPU 服务的例子
+ 使用 Docker 进行开发

## 正文

使用 Docker 是最简单的开启 TensorFlow 服务的方式。

实践代码如下：
+ 下载 TensorFlow 服务的 Docker 镜像和 github 库

```
docker pull tensorflow/serving

git clone https://github.com/tensorflow/serving
```

+ 通过 export 设置模型所处的路径

```
TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"
```

+ 开启 Tensorflow 服务的容器，并打开 REST API 端口

```
docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving &
```

+ 使用模型的预测 API 进行查询

```
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict
```

+ 将会获得以下的返回结果：

```
Returns => { "predictions": [2.5, 3.0, 4.5] }
```

下面将对命令依次进行详解：

### Docker 安装
通常来说，Docker 的安装介绍是在 Docker 的官方网站上的，这里给出一些快捷链接：

+ [Docker for macOS](https://docs.docker.com/docker-for-mac/install/)
+ [Docker for Windows](https://docs.docker.com/docker-for-windows/install/) for Windows 10 Pro or later
+ 旧版的系统安装 Docker 使用 [Docker Toolbox](https://docs.docker.com/toolbox/)

### 使用 Docker 开启 Serving 服务
#### 拉取一个服务镜像
当已经安装好 Docker 之后，就可以拉取最新 TensorFlow Serving docker 镜像运行了。

```
docker pull tensorflow/serving
```

该命令会拉取最小的 TensorFlow Serving Docker 镜像。具体的镜像版本信息可以到 Docker hub 的  tensorflow/serving 库中查看。

#### 运行一个服务镜像
CPU 和 GPU 的服务镜像有如下属性：
+ gRPC 的端口号为 8500
+ REST API 的端口号为 8501
+ 可选的环境变量 MODEL_NAME ，默认值为 model
+ 可选的环境变量 MODEL_BASE_PATH ，默认值为 /models

当服务镜像用作模型服务时，代码如下：

```
tensorflow_model_server --port=8500 --rest_api_port=8501 \
  --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}
```

使用 Docker 做服务器，需要做如下设定：
+ 主机用于服务的开放端口
+ 用作服务的保存模型
+ 给模型起一个对应的名称

需要做的工作包括：
1. 运行 Docker 容器
2. 给容器分配主机端口
3. 挂载主机路径上保存的模型

完整的示例代码

```
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/my_model/,target=/models/my_model \
  -e MODEL_NAME=my_model -t tensorflow/serving
```

在上面的代码中，展示了开启一个 Docker 容器，分配了主机的8501端口给容器，使用了名为 my_model 的模型，模型的路径设定如下 ${MODEL_BASE_PATH}/${MODEL_NAME} = /models/my_model。最后，将 my_model 填入环境变量 MODEL_NAME，令 MODEL_BASE_PATH 保留它的默认值。

当想要开启 gRPC 端口时， 使用 -p 8500:8500。可以同时选择开启 gRPC 和 REST API 端口，或者开启任意一个，代码如下：

```
tensorflow_model_server --port=8500 --rest_api_port=8501 \
  --model_name=my_model --model_base_path=/models/my_model
```

#### 传递额外的参数
tensorflow_model_server 支持许多额外的参数向提供服务的 docker 容器进行传递。例如可以将模型的配置文件作为模型名称的替代进行传递。示例代码如下：

```
docker run -p 8500:8500 -p 8501:8501 \
  --mount type=bind,source=/path/to/my_model/,target=/models/my_model \
  --mount type=bind,source=/path/to/my/models.config,target=/models/models.config \
  -t tensorflow/serving --model_config_file=/models/models.config
```

该命令行的形式支持所有 tensorflow_model_server 支持的平台。

#### 创建自己的服务镜像
如果你想要一个拥有自己模型的服务镜像容器，你可以自己创建一个：
+ 首先运行示例镜像作为后台驻留程序

```
docker run -d --name serving_base tensorflow/serving
```

+ 接下来，拷贝你的保存模型到容器模型的文件夹下：

```
docker cp models/<my model> serving_base:/models/<my model>
```

+ 最后，提交带有自己模型的服务镜像，修改为自定义的名字。

```
docker commit --change "ENV MODEL_NAME <my model>" serving_base <my container>
```

+ 到这一阶段，就可以停止 serving_base 的运行了

```
docker kill serving_base
```

+ 此时，就会只剩一个自定义名称的容器，下面就可以对这个模型里的服务进行部署和在启动时使用了。

#### 服务的示例
这里，举一个完整的调用保存模型并使用 REST API 调用的例子
+ 首先，拉取一个镜像

```
docker pull tensorflow/serving
```

该命令会拉取最新的装有模型服务的 TensorFlow 服务镜像。

+ 接下来，我们会使用一个示例模型“Half Plus Two”，进行测试

        该模型的内容是，生成 `0.5 * x + 2` 公式的对应值。

获得该模型需要现将 TensorFlow 服务库进行 clone。

```
mkdir -p /tmp/tfserving
cd /tmp/tfserving
git clone https://github.com/tensorflow/serving
```

+ 之后，启动运行这个指向该模型的 TensorFlow 服务容器，并打开它的 REST API 端口 8501。

```
docker run -p 8501:8501 \
  --mount type=bind,\
  source=/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu,\
  target=/models/half_plus_two \
  -e MODEL_NAME=half_plus_two -t tensorflow/serving &
```

这个命令将会运行 docker 容器，并且启动 TensorFlow 模型服务，绑定 REST API 端口 8501，同时，映射好了我们希望在容器中使用的模型。这个过程中，我们还设置了模型的名字作为环境变量，以便于我们在后期查询该模型时能更方便。

+ 使用预测 API 进行查询时，可以执行如下命令:

```
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
  -X POST http://localhost:8501/v1/models/half_plus_two:predict
```

+ 正确的返回值应该为下面：

```
{ "predictions": [2.5, 3.0, 4.5] }
```

更详细的 RESTful API 使用情况见[这里](https://www.tensorflow.org/tfx/serving/api_rest).