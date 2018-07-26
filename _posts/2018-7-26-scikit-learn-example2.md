---
layout: post
title:  "scikit learn: Nearest Neighbors Classification"
date:   2018-07-26 16:26:30
tags: [机器学习, 数据挖掘, scikit-learn, Nearest Neighbors]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn import neighbors, datasets

    n_neighbors = 15

    iris = dataset.load_iris()

    X = iris.data[:, :2]
    y = iris.target

    h = .02

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    for weights in ['uniform', 'distance']:
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light) # 终于明白了这里等高线和底图的意思了，就是预测值，和上面的散点图对应，表示预测的情况和训练的情况非常相似。也就是抽取出了对应的特征。

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        plt.title("3-class classification (k = %i, weights = '%s')" % (n_neighbors, weights))

    plt.show()

### 背景介绍
最近邻居分类法示例介绍。总的来说，作用和前一示例讲的差不多。

end
