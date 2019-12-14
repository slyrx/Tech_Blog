---
layout: post
title:  "TFX Airflow Tutorial"
date:   2019-08-04 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第三十九篇。
        本文档介绍 TFX Airflow 的使用手册。


## 目录
+ 介绍
+ + 学习更多
+ 手把手
+ 先决条件
+ + 依赖包
+ + MacOS 环境
+ 手册资料
+ 你需要做的工作
+ + 添加每步需要的代码
+ + 芝加哥出租车数据集
+ + 模型目标 - 实现二分类
+ 步骤 1: 安装环境
+ 步骤 2: 创建初始的管道框架
+ + Hello World
+ + 在浏览器中
+ 步骤 3: 深入到数据中
+ + 组件
+ + 在编辑器中
+ + 在浏览器中
+ + 回到 Jupyter 中
+ + 指向更高级的例子
+ 步骤 4: 特征工程
+ + 组件
+ + 在编辑器中
+ + 在浏览器中
+ + 回到 Jupyter 中
+ + 指向更高级的例子
+ 步骤 5: 训练
+ + 组件
+ + 在编辑器中
+ + 在浏览器中
+ + 回到 Jupyter 中
+ + 指向更高级的例子
+ 步骤 6: 分析模型的性能
+ + 组件
+ + 在编辑器中
+ + 在浏览器中
+ + 回到 Jupyter 中
+ + 指向更高级的例子
+ 步骤 7: 准备发布
+ + 组件
+ + 在编辑器中
+ + 在浏览器中
+ 下一步

## 正文
本教程旨在介绍 TensorFlow Extended（TFX），并帮助您学习创建自己的机器学习管道。 它在本地运行，并显示了与 TFX 和 TensorBoard 的集成以及 Jupyter 笔记本中与 TFX 的交互。

关键术语：TFX 管道是有向非循环图或“ DAG ”。 我们通常将管道称为 DAG。

您将遵循典型的 ML 开发过程，首先要检查数据集，然后完成一个完整的工作流程。 在此过程中，您将探索调试和更新管道以及评估性能的方法。

学到更多
请参阅《 TFX用户指南》以了解更多信息。

手把手
您将按照典型的 ML 开发过程逐步进行操作，逐步创建管道。 步骤如下：

1. 设置环境
2. 提出初步的管道框架
3. 深入研究您的数据
4. 特征工程
5. 训练
6. 分析模型性能
7. 准备生产

先决条件
Linux / MacOS
虚拟环境
Python 3.5+
Git

所需的包
根据您的环境，您可能需要安装几个软件包：

```
sudo apt-get install \
    build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev \
    python3-pip git software-properties-common
```

如果您运行的是Python 3.6，则应安装python3.6-dev：

```
sudo apt-get install python3.6-dev
```

如果您运行的是Python 3.7，则应安装python3.7-dev：

```
sudo apt-get install python3.7-dev
```

另外，如果您的系统的GCC版本小于7，则应更新GCC。 否则，在运行 airflow Web服务器时会看到错误。 您可以使用以下方法检查当前版本：

```
gcc --version
```

如果您需要更新GCC，可以运行以下命令：

```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-7
sudo apt install g++-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
```

MacOS环境
如果尚未安装Python 3和git，则可以使用Homebrew软件包管理器进行安装：

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
brew update
brew install python
brew install git
```

在运行Airflow时，MacOS有时会在分叉线程时遇到问题，具体取决于配置。 为了避免这些问题，您应该编辑〜/ .bash_profile并将以下行添加到文件末尾：

```
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

辅导资料
本教程的代码可在以下位置获得：
https://github.com/tensorflow/tfx/tree/master/tfx/examples/airflow_workshop


该代码是按照您正在执行的步骤进行组织的，因此对于每个步骤，您都将拥有所需的代码以及如何使用它的说明。

教程文件包括练习和练习解决方案，以防万一。

练习：
+ taxi_pipeline.py
+ taxi_utils.py
+ taxi DAG

解决方法：
+ taxi_pipeline_solution.py
+ taxi_utils_solution.py
+ taxi_solution DAG

你在做什么
您正在学习如何使用TFX创建ML管道

+ 当您要部署生产ML应用程序时，TFX管道是合适的
+ 当数据集很大时，TFX管道是合适的
+ 当培训/服务一致性很重要时，TFX管道是合适的
+ 当版本管理的推理很重要时，TFX管道是合适的
+ Google使用TFX管道进行生产ML

你将执行典型的机器学习部署过程：
+ 读取、理解和清洗数据
+ 特征工程
+ 训练
+ 分析模型性能
+ 生成数据、清洗数据、重复训练过程
+ 准备部署

为每个步骤添加代码
本教程旨在使所有代码都包含在文件中，但是步骤3-7的所有代码都被注释掉并用内联注释标记。 内联注释标识代码行适用于哪个步骤。 例如，步骤3的代码标有注释＃步骤3。

您将为每个步骤添加的代码通常分为以下3个区域：
+ 依赖
+ DAG 管道配置
+ 从 create_pipeline（）调用返回的列表
+ taxi_utils.py 中的代码

在阅读本教程时，您将取消注释当前正在使用的本教程步骤的代码行。 这将添加该步骤的代码，并更新您的管道。 在执行此操作时，我们强烈建议您查看要取消注释的代码。

芝加哥出租车数据集
![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/taxi.jpg)
![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/chicago.png)

您正在使用芝加哥市发布的出租车行车数据集。

注意：此站点使用原始数据www.cityofchicago.org（芝加哥市的官方网站）中经过修改的数据来提供应用程序。 芝加哥市对本站点提供的任何数据的内容，准确性，及时性或完整性不做任何声明。 本网站提供的数据随时可能更改。 据了解，使用本网站提供的数据的风险自负。

您可以在Google BigQuery中了解有关数据集的更多信息。 在BigQuery UI中浏览完整的数据集。

模型目标-二分类
客户会给小费多还是少20％？

步骤1：设定环境
设置脚本（setup_demo.sh）将安装TFX和Airflow，并以一种易于使用的方式配置Airflow。

在外壳中：

```
cd
virtualenv -p python3 tfx-env
source ~/tfx-env/bin/activate

git clone https://github.com/tensorflow/tfx.git
cd ~/tfx
# Release 0.14 is the latest stable release
git checkout -f origin/r0.14
cd ~/tfx/tfx/examples/airflow_workshop/setup
./setup_demo.sh
```

您应该查看setup_demo.sh以查看其作用。

步骤2：启动初始管道框架
你好，世界
在外壳中：

```
# Open a new terminal window, and in that window ...
source ~/tfx-env/bin/activate
airflow webserver -p 8080

# Open another new terminal window, and in that window ...
source ~/tfx-env/bin/activate
airflow scheduler

# Open yet another new terminal window, and in that window ...
# Assuming that you've cloned the TFX repo into ~/tfx
source ~/tfx-env/bin/activate
cd ~/tfx/tfx/examples/airflow_workshop/notebooks
jupyter notebook
```

您在此步骤中启动了Jupyter Notebook。 稍后，您将在此文件夹中运行笔记本。

在浏览器中：
打开浏览器，然后转到http://127.0.0.1:8080

故障排除
如果在Web浏览器中加载Airflow控制台时遇到任何问题，或者在运行airflow Web服务器时出现任何错误，则可能在端口8080上运行了另一个应用程序。这是Airflow的默认端口，但是您可以更改它 到其他未使用的用户端口。 例如，要在端口7070上运行Airflow，可以运行：

```
airflow webserver -p 7070
```

DAG查看按钮
![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/airflow_dag_buttons.png)

使用左侧的按钮启用出租车DAG
进行更改时，使用右侧的按钮刷新出租车DAG
使用右侧的按钮触发出租车DAG
单击出租车转到出租车DAG的图形视图

![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/graph_refresh_button.png)


您还可以使用Airflow CLI
您还可以使用Airflow CLI启用和触发DAG

```
# enable/disable
airflow unpause <your DAG>
airflow pause <your DAG>

# trigger
airflow trigger_dag <your DAG>
```

等待管道完成
在DAGs视图中触发管道后，您可以观察管道完成处理。 随着每个组件的运行，DAG图中该组件的轮廓颜色将更改以显示其状态。 组件完成处理后，轮廓将变为深绿色，表明已完成。

注意：您需要使用右侧的图形刷新按钮或刷新页面以查看组件运行时的更新状态。

到目前为止，我们的管道中只有CsvExampleGen组件，因此您需要等待它变为深绿色（约1分钟）。

![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/step2.png)

第3步：深入研究数据
任何数据科学或ML项目的首要任务是理解和清理数据。

+ 了解每个功能的数据类型
+ 寻找异常和缺失值
+ 了解每个功能的分布

组件
![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/examplegen1.png)

![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/examplegen2.png)


+ ExampleGen 读取和分割输入数据集
+ StatisticsGen 计算数据集的统计情况
+ SchemaGen 检查数据统计情况，并创建数据模式
+ ExampleValidator 寻找数据集中的异常值和缺失值。

注：
+ 在编辑器中是指对 python 代码的编辑
+ 在浏览器中是指通过 airflow 图形化界面对处理过程对监控
+ 在 Jupyter 中的是指展示数据的统计信息部分

在编辑器中：
在〜/ airflow / dags中取消注释taxi_pipeline.py中标记为步骤3的行
花一点时间查看您未注释的代码

在浏览器中：
通过单击左上角的“ DAG”链接返回到Airflow中的DAG列表页面
单击出租车DAG右侧的刷新按钮
您应该看到“ DAG [出租车]现在像雏菊一样新鲜”
触发出租车
等待管道完成
全深绿色
使用右侧的刷新或刷新页面

![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/step3.png)

返回Jupyter：
之前，您运行过jupyter笔记本，该笔记本在浏览器选项卡中打开了Jupyter会话。 现在返回浏览器中的该选项卡。

打开step3.ipynb
跟随笔记本

更高级的例子
这里提供的示例实际上仅是为了让您入门。 有关更高级的示例，请参见TensorFlow数据验证合作实验室。

有关使用TFDV探索和验证数据集的更多信息，请参见tensorflow.org上的示例。

步骤4：特征工程
您可以使用特征工程来提高数据的预测质量和/或减小维数。

特征交叉
词汇量
嵌入
PCA
分类编码

使用TFX的好处之一是您只需编写一次转换代码，并且在培训和服务之间所产生的转换将是一致的。

组件
![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/transform.png

+ Transform 体现了数据集的特征工程

在编辑器中：
在〜/ airflow / dags中，取消注释taxi_pipeline.py和taxi_utils.py中标记为步骤4的行。
花一点时间查看您未注释的代码

在浏览器中：
返回气流中的DAG列表页面
单击出租车DAG右侧的刷新按钮
您应该看到“ DAG [出租车]现在像雏菊一样新鲜”
触发出租车
等待管道完成
全深绿色
使用右侧的刷新或刷新页面

![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/step4.png)

返回Jupyter：
返回浏览器中的Jupyter选项卡。

打开step4.ipynb
跟随笔记本

更高级的例子
这里提供的示例实际上仅是为了让您入门。 有关更高级的示例，请参见 TensorFlow Transform Colab。

步骤5：训练
使用您的漂亮，干净，转换后的数据训练TensorFlow模型。

包括步骤4中的转换，以便一致地应用它们
将结果另存为SavedModel以进行生产
使用TensorBoard可视化并探索培训过程
还保存一个EvalSavedModel来分析模型性能

组件
+ Trainer 使用TensorFlow Estimators训练模型

在编辑器中：
在〜/ airflow / dags中，取消注释taxi_pipeline.py和taxi_utils.py中标记为步骤5的行
花一点时间查看您未注释的代码

在浏览器中：
返回气流中的DAG列表页面
单击出租车DAG右侧的刷新按钮
您应该看到“ DAG [出租车]现在像雏菊一样新鲜”
触发出租车
等待管道完成
全深绿色
使用右侧的刷新或刷新页面

![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/step5.png)

+ Jupyter 的作用是把本步骤产生的成果模型，通过 Jupyter 调出 TensorBoard 来显示。
返回Jupyter：
返回浏览器中的Jupyter选项卡。

打开step5.ipynb
跟随笔记本

![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/step5tboard.png)

更高级的例子
这里提供的示例实际上仅是为了让您入门。 有关更高级的示例，请参见TensorBoard教程。

步骤6：分析模型性能
了解更多，而不仅仅是顶级指标。

用户仅对其查询体验模型性能
顶级指标可以掩盖数据切片上的性能不佳
模型公平很重要
用户或数据的关键子集通常非常重要，并且可能很小
在关键但异常条件下的性能
面向主要受众（例如网红）的表演

组件
+ Evaluator 训练结果表现的深度解析

在编辑器中：
在〜/ airflow / dags中，取消注释两个taxi_pipeline.py中标记为步骤6的行
花一点时间查看您未注释的代码

在浏览器中：
返回气流中的DAG列表页面
单击出租车DAG右侧的刷新按钮
您应该看到“ DAG [出租车]现在像雏菊一样新鲜”
触发出租车
等待管道完成
全深绿色
使用右侧的刷新或刷新页面

![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/step6.png)

返回Jupyter：
返回浏览器中的Jupyter选项卡。

打开step6.ipynb
跟随笔记本

![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/step6notebook.png)

更高级的例子
这里提供的示例实际上仅是为了让您入门。 有关更高级的示例，请参见《 TFMA芝加哥出租车指南》。


步骤7：准备生产
如果新模型已准备就绪，请这样做。

如果要替换当前正在生产的模型，请首先确保新模型更好
ModelValidator告诉Pusher组件模型是否正确
Pusher将SavedModels部署到知名位置
部署目标从知名位置接收新模型

TensorFlow服务
TensorFlow Lite
TensorFlow JS
TensorFlow集线器

组件
+ ModelValidator 确保模型足够好，符合推理的条件
+ Pusher 不是模型到基本的服务器上

在编辑器中：
在〜/ airflow / dags中，取消注释在两个taxi_pipeline.py中标记为步骤7的行
花一点时间查看您未注释的代码
在浏览器中：
返回气流中的DAG列表页面
单击出租车DAG右侧的刷新按钮
您应该看到“ DAG [出租车]现在像雏菊一样新鲜”
触发出租车
等待管道完成
全深绿色
使用右侧的刷新或刷新页面

![](https://www.tensorflow.org/tfx/tutorials/tfx/images/airflow_workshop/step7.png)

下一步
现在，您已经训练并验证了模型，并在〜/ airflow / saved_models / taxi目录下导出了SavedModel文件。 您的模型现在可以投入生产了。 您现在可以将模型部署到任何TensorFlow部署目标，包括：

TensorFlow Serving，用于在服务器或服务器场上服务于模型并处理REST和/或gRPC推理请求。
TensorFlow Lite，用于将模型包含在Android或iOS本机移动应用程序或Raspberry Pi，IoT或微控制器应用程序中。
TensorFlow.js，用于在Web浏览器或Node.JS应用程序中运行模型。
