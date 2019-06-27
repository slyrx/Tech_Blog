---
layout: post
title:  "scikit learn: Nearest Neighbors Classification"
date:   2018-07-26 17:31:30
tags: [机器学习, 数据挖掘, scikit-learn, Nearest Neighbors]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.neighbors import NearestCentroid

    n_neighbors = 15

    iris = datasets.load_iris()

    X = iris.data[:, :2]
    y = iris.target

    h = .02

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    for shrinkage in [None, .2]: # shrinkage参数的含义用于确定移除较多特征时设置的阈值
        clf = NearestCentroid(shrink_threshold = shrinkage)
        clf.fit(X, y)
        y_pred = clf.predict(X)

        print(shringage, np.mean(y == y_pred))

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='b', s=20)
        plt.title("3-class classification (shrink_threshold=%r)" % shrinkage)
        plt.axis('tight')

    plt.show()


#### 英语生词
shrinking 收缩
centroids 形心


end
