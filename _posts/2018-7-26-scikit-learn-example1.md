---
layout: post
title:  "scikit learn: Nearest Neighbors regression"
date:   2018-07-26 15:19:30
tags: [机器学习, 数据挖掘, scikit-learn, Nearest Neighbors]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import neighbors

    np.random.seed(0)
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    T = np.linspace(0, 5, 500)[:, np.newaxis]
    y = np.sin(X).ravel()

    y[::5] += 1 * (0.5 - np.random.rand(8))

    n_neighbors = 5

    for i, weights in enumerate(['uniform', 'distance']): # 两种权重方式, uniform表示所有点的权重都设置成一个固定的值。distance表示相聚较近的点的权重大于相聚较远的点的权重。
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        y_ = knn.fit(X, y).predict(T)

        plt.subplot(2, 1, i + 1)
        plt.scatter(X, y, c='k', label='data')
        plt.plot(T, y_, c='g', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title('KNeighborsRegressor (k = %i, weights='%s')' % (n_neighbors, weights))

    plt.show()

### 背景介绍
k-Nearest Neighbor算法两种权重方式效果对比。

end
