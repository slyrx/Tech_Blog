---
layout: post
title:  "Improving Model Quality With TensorFlow Model Analysis"
date:   2019-07-30 10:11:30
tags: [tensorflow]
---

    导语：
        本文是 tensorflow 手册翻译系列的第三十四篇。
        本文档介绍如何通过 Tensorflow 模型分析来提高模型的质量。


## 目录
+ 介绍
+ 概述
+ 从你的模型中导出模型评估过的保存模型
+ 在 Jupyter notebook 中可视化
+ + 评价标签
+ + 切片指标选项卡


## 正文
介绍
在开发过程中调整模型时，需要检查所做的更改是否正在改善模型。仅检查准确性可能还不够。例如，如果您有一个问题的分类器，其中95％的实例是肯定的，则仅通过始终预测为肯定就可以提高准确性，但是您将没有非常强大的分类器。

总览
TensorFlow模型分析的目的是为TFX中的模型评估提供一种机制。 TensorFlow Model Analysis允许您在TFX管道中执行模型评估，并在Jupyter笔记本中查看结果度量和绘图。具体来说，它可以提供：

根据整个培训和坚持数据集计算的指标，以及次日评估
随时间跟踪指标
在不同特征切片上的模型质量性能
从模型中导出EvalSavedModel
为了在TFX管道中设置TensorFlow模型分析，需要在训练期间导出EvalSavedModel，这是一种特殊的SavedModel，其中包含模型中指标，功能，标签等的注释。 TensorFlow模型分析使用此EvalSavedModel计算指标。

作为此过程的一部分，您将必须提供一个特殊的eval_input_receiver_fn，类似于serving_input_receiver_fn，它将从输入数据中提取特征和标签。就像serving_input_receiver_fn一样，我们有实用程序功能可以帮助您。在大多数情况下，您将不得不添加少于20行的代码。

Jupyter笔记本中的可视化
评估结果在Jupyter笔记本中可视化。

评估标签
用户界面由三部分组成：

指标选择器

默认情况下，将显示所有计算的指标，并且按字母顺序对列进行排序。指标选择器允许用户添加/删除/重新排序指标。只需从下拉列表中选中/取消选中指标（按住Ctrl即可进行多选），或者直接在输入框中输入/重新排列它们。

![png](https://www.tensorflow.org/tfx/guide/images/metricsSelector.png)

时间序列图

时间序列图可轻松发现数据范围或模型运行中特定指标的趋势。 要为感兴趣的指标呈现图形，只需从下拉列表中单击它即可。 要关闭图形，请单击右上角的X。

![png](https://www.tensorflow.org/tfx/guide/images/modelDrivenTimeSeriesGraph.png)

将鼠标悬停在图形中的任何数据点上都会显示工具提示，指示模型运行，数据范围和度量标准值。

指标表

指标表汇总了指标选择器中选择的所有指标的结果。 可以通过单击度量标准名称对其进行排序。

切片指标选项卡
切片指标选项卡显示特定评估运行的不同切片如何执行。 请选择所需的配置（评估，功能等），然后单击刷新。

该URL将在刷新后更新，并包含对所选配置进行编码的深层链接。 可以共享。

用户界面由三部分组成：

指标选择器

往上看。

公制可视化

度量可视化旨在提供有关所选要素中的切片的直觉。 快速过滤可用于过滤加权样本数少的切片。

![png](https://www.tensorflow.org/tfx/guide/images/sliceOverviewAfterFiltering.png)

支持两种类型的可视化：

切片概述

在此视图中，为每个切片呈现选定度量的值，并且可以按切片名称或另一个度量的值对切片进行排序。

![png](https://www.tensorflow.org/tfx/guide/images/sliceOverview.png)

当切片数较少时，这是默认视图。

指标直方图

在此视图中，切片根据其指标值细分为存储桶。 每个存储桶中显示的值可以是存储桶中的切片数，也可以是存储桶中所有切片的总加权样本计数，或两者都有。

![png](https://www.tensorflow.org/tfx/guide/images/metricsHistogram.png)

单击齿轮图标，可以更改铲斗数，并且可以在设置菜单中应用对数刻度。

![png](https://www.tensorflow.org/tfx/guide/images/metricsHistogramSetting.png)

也可以在直方图视图中滤除异常值。 只需在直方图中拖动所需的范围即可，如下面的屏幕截图所示。

![png](https://www.tensorflow.org/tfx/guide/images/metricsHistogramFiltered.png)

当切片数很大时，这是默认视图。

指标表

将仅渲染未过滤出的切片。 可以通过单击列标题进行排序。