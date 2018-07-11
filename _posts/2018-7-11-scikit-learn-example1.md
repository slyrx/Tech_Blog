---
layout: post
title:  "scikit learn: Feature importances with forests of trees"
date:   2018-07-11 13:01:30
tags: [机器学习, 数据挖掘, scikit-learn, ensemble methods]
---

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreeClassifier

X, y = make_classification(n_samples=1000, n_features=10, n_informative=3, n_redundant=0, n_repeated=0, n_classes=2, random_state=0, shuffle=False) # 生成训练数据集 n_samples样本数，n_features总的特征数，n_informative信息特征数,即每个聚为一类的内容高度相关的特征数，n_redundant多余的无效冗余的特征数，n_repeated重复的特征数，n_classes分类问题里的类型数量,random_state随机种子，shuffle弄乱样本的特征，使他们不要呈有序排列，shuffle就是弄乱的意思.

forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

forest.fit(X,y)
importances = forest.feature_importances_ # 返回的是以特征为长度的数组，对应的值是特征的重要程度
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0) # np.std()在特殊轴上计算标准偏离，返回标准偏离。标准翻译：计算沿指定轴的标准偏差，返回这个标准偏差，即分布扩展的度量。

indices = np.argsort(importances)[::-1] # 表示对importances变量从小到大排列，再对排列得到的结果做降序排列

print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d.feature %d(%f)" % (f + 1, indices[f], importances[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center") # ***yerr*** 表示偏差std，图中蓝色工字型
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

### 背景介绍
这个例子示范了利用森林来评估人造分类任务的特征重要度。红色条显示了特征的重要度，和树间的变异情况。
本例的结论是，由3个特征是非常重要的，而其他的则不是，图像将特征以重要度排开。
ExtraTreeClassifier 额外树分类器

#### inter-trees variability含义
+ 此处理解的inter-trees variability表示这里本来是一个森林在做分类，那么森林里树之间的相关性如何。
+ The red bars are the feature importances of the forest, along with their inter-trees variability.
+ 树之间差异区分之一的特征重要性程度。

#### np function
+ np.std() 计算矩阵标准差
+ np.std(xxx, axis=0/1)
+ axis=0表示一列，axis=1表示一行；也就是计算的是一列或者一行的标准差
+ 如果没有指定axis，那么就是所有行和列的标准差
+ np.argsort() 返回数组值从小到大的索引值
+ 此例中因为在后面加了[::-1],因此再对从小到大的索引值做倒序排列。
+ np.argsort(-x),如果其中参数带负号-，则表示对数组x做降序排列
+ 需要明确argsort返回的是数组的索引值。
+ np.argsort(x, axis=0/1), 这里的axis和上面std里的axis含义是一样的，0表示一列，1表示一行。

#### 数学概念
+ 什么是标准差？
+ 标准差是方差的算术平方根
+ 所以说到标准差，本质上还是在说方差
+ 核心是偏离均值的程度。
+ 6月26日的例子有对标准差很全面的阐述。


#### python opertation
+ a = [1,2,3,4]
+ a[::-1] output: [4, 3, 2, 1]
+ a[:-1]  output: [1, 2, 3]
+ a[:1]   output: [1]
+ a[::1]  output: [1, 2, 3, 4]
+ a[::-2] output: [4, 2]
+ a[::2]  output: [1, 3]
+ 全语法含义是：seq[start:end:step]
+ -负号表示从尾部向头部循环；step表示单次的跨距；start和end不写默认从0开始，写了就从指定下标开始




end
