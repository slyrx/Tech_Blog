---
layout: post
title:  "scikit learn: Plot the decision surface of a decision tree on the iris dataset"
date:   2018-07-22 18:16:30
tags: [机器学习, 数据挖掘, scikit-learn, Decision Trees]
---

    print(__doc__)
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    n_classes = 3
    plot_colors = "ryb"
    plot_step = 0.02

    iris = load_iris()

    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
        X = iris.data[:, pair]
        y = iris.target

        clf = DecisionTreeClassifier().fit(X, y) # 训练出模型

        plt.subplot(2, 3, pairidx + 1)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # 新生成数据，并用来预测
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu) # 将预测结果通过等高线展示，这里就是题目中所说的决策平面。也就是能较明显的看到对归类的划分。

        plt.xlabel(iris.feature_names[pair[0]])
        plt.ylabel(iris.feature_names[pair[1]])

        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color,  label=iris.target_names[i], cmap=plt.cm.RdYlBu, edgecolor='black', s=15) # 这里的点应该是训练模型用的啊，这里是想说，原来的数据X和新的数据xx有相同的特征，在图像分布上有相同的分布规律，红色都在红色的位置，黄色在黄色的划分，蓝色同理。

    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend(loc="lower right", borderpad=0, handletextpad=0)
    plt.axis("tight")
    plt.show()

### 背景介绍
画出由决策树作出的针对iris数据集的2维特征的决策平面


end
