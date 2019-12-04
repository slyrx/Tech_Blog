---
layout: post
title:  "TensorFlow Data Validation"
date:   2019-07-21 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第二十五篇。
        本文档详细介绍探索和验证机器学习数据。

## 目录
+ 从 Pypi 进行安装
+ 从 Docker 进行构建
+ + 安装 Docker
+ + 克隆 TFDV 库
+ + 构建 pip 包
+ + 安装 pip 包
+ 从源进行构建
+ + 先决条件
+ + 克隆 TFDV 库
+ + 构建 pip 包
+ + 安装 pip 包
+ 支持的平台
+ 依赖库
+ 兼容的版本
+ 问题
+ 链接

## 正文

TensorFlow数据验证（TFDV）是用于探索和验证机器学习数据的库。它的设计具有很高的可扩展性，可以与TensorFlow和TensorFlow Extended（TFX）很好地配合使用。

TF数据验证包括：

培训和测试数据摘要统计的可扩展计算。
与查看器集成以进行数据分发和统计，以及对特征对（Facets）进行多方面比较
自动生成数据架构以描述对数据的期望，例如所需的值，范围和词汇
模式查看器，以帮助您检查模式。
异常检测可以识别异常，例如缺失的特征，超出范围的值或错误的特征类型等。
异常查看器，以便您可以查看哪些功能存在异常并了解更多信息以进行纠正。
有关使用TFDV的说明，请参阅入门指南并尝试使用示例笔记本。在SysML'19中发表的技术论文中描述了TFDV中实现的一些技术。

注意：TFDV在1.0版之前可能向后不兼容。
从PyPI安装
推荐的安装TFDV的方法是使用PyPI软件包：

```
pip install tensorflow-data-validation
```

使用Docker进行构建
这是在Linux下构建TFDV的推荐方法，并且已在Google上进行了持续测试。

1.安装Docker
请首先按照以下说明安装docker和docker-compose：docker; docker-compose。

2.克隆TFDV存储库

```
git clone https://github.com/tensorflow/data-validation
cd data-validation
```

请注意，这些说明将安装TensorFlow数据验证的最新主分支。 如果要安装特定的分支（例如发行分支），请将-b <branchname>传递给git clone命令。

在Python 2上构建时，请确保使用以下命令在源代码中剥离Python类型：

```
pip install strip-hints
python tensorflow_data_validation/tools/strip_type_hints.py tensorflow_data_validation/
```

3.构建点子包
然后，在项目根目录下运行以下命令：

```
sudo docker-compose build manylinux2010
sudo docker-compose run -e PYTHON_VERSION=${PYTHON_VERSION} manylinux2010
```

其中PYTHON_VERSION是{27，35，36，37}之一。

将在dist /下生成一个轮子。

4.安装pip包

```
pip install dist/*.whl
```

从源代码构建
1.先决条件
要编译和使用TFDV，您需要设置一些先决条件。

安装NumPy
如果您的系统上未安装NumPy，请按照以下说明立即安装。

安装挡板
如果您的系统上未安装Bazel，请按照以下说明立即安装。

2.克隆TFDV存储库

```
git clone https://github.com/tensorflow/data-validation
cd data-validation
```

请注意，这些说明将安装TensorFlow数据验证的最新主分支。 如果要安装特定的分支（例如发行分支），请将-b <branchname>传递给git clone命令。

在Python 2上构建时，请确保使用以下命令在源代码中剥离Python类型：

```
pip install strip-hints
python tensorflow_data_validation/tools/strip_type_hints.py tensorflow_data_validation/
```

3.构建点子包
TFDV使用Bazel从源代码构建pip包。 在调用以下命令之前，请确保$ PATH中的python是目标版本之一，并且已安装NumPy。

```
bazel run -c opt --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 tensorflow_data_validation:build_pip_package
```

请注意，我们在此假设相关软件包（例如PyArrow）是使用低于5.1的GCC构建的，并使用标志D_GLIBCXX_USE_CXX11_ABI = 0与旧的std :: string ABI兼容。

您可以在dist子目录中找到生成的.whl文件。

4.安装pip包

```
pip install dist/*.whl
```

支持平台
TFDV在以下64位操作系统上经过测试：

macOS 10.12.6（Sierra）或更高版本。
Ubuntu 16.04或更高版本。
Windows 7或更高版本。
依存关系
TFDV需要TensorFlow，但不依赖于tensorflow PyPI软件包。 有关如何开始使用TensorFlow的说明，请参阅TensorFlow安装指南。

需要Apache Beam； 这就是支持高效的分布式计算的方式。 默认情况下，Apache Beam在本地模式下运行，但也可以使用Google Cloud Dataflow在分布式模式下运行。 TFDV设计为可扩展为其他Apache Beam跑步者。

还需要Apache Arrow。 TFDV使用Arrow来在内部表示数据，以便利用矢量化numpy函数。

兼容版本
下表显示了彼此兼容的软件包版本。 这是由我们的测试框架确定的，但其他未经测试的组合也可能有效。

tensorflow-data-validation	tensorflow	apache-beam[gcp]	pyarrow
GitHub master	nightly (1.x/2.x)	2.16.0	0.14.0
0.15.0	1.15 / 2.0	2.16.0	0.14.0
0.14.1	1.14	2.14.0	0.14.0
0.14.0	1.14	2.14.0	0.14.0
0.13.1	1.13	2.11.0	n/a
0.13.0	1.13	2.11.0	n/a
0.12.0	1.12	2.10.0	n/a
0.11.0	1.11	2.8.0	n/a
0.9.0	1.9	2.6.0	n/a

问题
请使用tensorflow-data-validation标记引导有关将TF数据验证与堆栈溢出配合使用的任何问题。

链接
TensorFlow数据验证入门指南
TensorFlow数据验证笔记本
TensorFlow数据验证API文档
TensorFlow数据验证博客文章
TensorFlow数据验证PyPI
TensorFlow数据验证文件
TensorFlow数据验证幻灯片




