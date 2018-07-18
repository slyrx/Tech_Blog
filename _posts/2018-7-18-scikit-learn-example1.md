---
layout: post
title:  "scikit learn: SVM: Separating hyperplane for unbalanced classes"
date:   2018-07-18 16:19:30
tags: [机器学习, 数据挖掘, scikit-learn, Support Vector Machines]
---

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

rng = np.random.RandomState(0)
n_samples_1 = 1000
n_samples_2 = 100
X = np.r_[1.5 * rng.randn(n_samples_1, 2), 0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
y = [0] * (n_samples_1) + [1] * (n_samples_2)

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.legend()

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = clf.decision_function(xy).reshape(XX.shape)

a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linesyles=['-'])

Z = wclf.decision_function(xy).reshape(XX.shape)

b = ax.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5, linestyles=['-'])

plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"], loc="upper right")

plt.show()

### 背景介绍
使用SVC分类器对不平衡的分类找出最优的分离超平面。
首先用普通的SVC找到分离平面，之后用虚线对不平衡对类画出自带矫正的分离超平面。

#### np.array
np.array的特有性质，在后面+ [1,2]后，将是给对应的矩阵每一位相加。实现叠加的效果。

#### 英语单词
dashed 虚线








end
