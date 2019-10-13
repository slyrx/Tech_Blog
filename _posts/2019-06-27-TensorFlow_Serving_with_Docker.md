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
1. 下载 TensorFlow 服务的 Docker 镜像和 github 库
```
docker pull tensorflow/serving

git clone https://github.com/tensorflow/serving
```

2. 通过 export 设置模型所处的路径
```
TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"
```

3. 开启 Tensorflow 服务的容器，并打开 REST API 端口
```
docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving &
```

4. 使用模型的预测 API 进行查询
```
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict
```

5. 将会获得以下的返回结果：
```
# Returns => { "predictions": [2.5, 3.0, 4.5] }
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



