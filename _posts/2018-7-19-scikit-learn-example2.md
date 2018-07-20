---
layout: post
title:  "scikit learn: Plot different SVM classifiers in the iris dataset"
date:   2018-07-19 17:09:30
tags: [机器学习, 数据挖掘, scikit-learn, Support Vector Machines]
---

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, dataset

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    return xx, yy

def plot_contours(ax, clf, xx, yy, \*\*params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, \*\*params)
    return out

iris =  datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

C = 1.0
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C), # gamma隐含的决定了数据映射到新到特征空间后到分布，gamma越大，支持向量越少，gamma越小,支持向量越多。
          svm.SVC(kernel='poly', degree=3, C=C))

models = (clf.fit(X, y) for clf in models)

titles = ('SVC with linear kernel', 'LinearSVC (linear kernel)', 'SVC with RBF kernel', 'SVC with polynomial(degree 3) kernel')

fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()


### 背景介绍
该示例是在2D投影的iris数据集上使用不同的线性SVM分类器做比较。此例中只考虑数据集的头两个特征:萼片长度，萼片宽度。
图例展示了四种不同内核SVM分类器的决策平面。
线性模型LinearSVC()和SVC(kernel='linear')会产生略微不同的决策边界。
结果有如下不同：
LinearSVC最小化了平方铰链损失; SVC最小化了正则铰链损失
LinearSVC使用One-vs-All多类减少; SVC使用One-vs-One多类减少。

所有线性模型都有线性决策边界，也称为横断超平面。而所有的非线性内核的模型(即多项式模型或高斯径向模型)都有更灵活的非线性决策边界，且这个灵活的边界形状依赖于内核的种类和它对应的参数。

当遇到高维数据，本例的模型特征显示可能会不那么明显。

#### 名词解释
squared hinge loss
regular hinge loss 参考 https://blog.csdn.net/u010976453/article/details/78488279
multiclass reduction
intersecting hyperplanes 横断超平面

#### 英语生词
projection 投影
2D projection 2D投影
yield 产生
consequence 结果
hinge
intuitive
realistic 现实主义

end
