---
layout: post
title:  "scikit learn: SVM: Weighted samples"
date:   2018-07-18 14:16:30
tags: [机器学习, 数据挖掘, scikit-learn, Support Vector Machines]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm

    def plot_decision_function(classifier, sample_weight, axis, title):
        xx, yy = np.meshgrid(np.linespace(-4, 5, 500), np.linspace(-4, 5, 500))

        Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
        axis.scatter(X[:, 0], X[:, 1], c=y, s=100 * sample_weight, alpha=0.9, cmap=plt.cm.bone, edgecolors='black')

        axis.axis('off')
        axis.set_title(title)


    np.random.seed(0)
    X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)] # X是一个20个数据的点
    y = [1] * 10 + [-1] * 10 # 这20个数据对应的分类
    sample_weight_last_ten = abs(np.random.randn(len(X))) # 这个的数量都是20
    sample_weight_constant = np.ones(len(X))

    sample_weight_last_ten[15:] \*= 5
    sample_weight_last_ten[9] \*= 15

    clf_weights = svm.SVC()
    clf_weights.fit(X, y, sample_weight=sample_weight_last_ten) # 在拟合的过程中增加了权重


    clf_no_weights = svm.SVC()
    clf_no_weights.fit(X, y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_decision_function(clf_no_weights, sample_weight_constant, axes[0], "Constant weights")

    plot_decision_function(clf_weights, sample_weight_last_ten, axes[1], "Modified weights")

    plt.show()

### 背景介绍
绘制加权数据集的决策函数，其中点的大小与它的权重成正比。


#### 英语生词
rescales 调整
subtle  微妙
outliers  异常值
deformation  变形






end
