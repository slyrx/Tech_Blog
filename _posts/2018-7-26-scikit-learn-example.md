---
layout: post
title:  "scikit learn: Anomaly detection with Local Outlier Factor(LOF)"
date:   2018-07-26 14:19:30
tags: [机器学习, 数据挖掘, scikit-learn, Nearest Neighbors]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import LocalOutlierFactor

    np.random.seed(42)

    X = 0.3 * np.random.randn(100, 2)
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    X = np.r_[X + 2, X - 2, X_outliers]

    clf = LocalOutlierFactor(n_neighbors=20)
    y_pred = clf.fit_predict(X)
    y_pred_outliers = y_pred[200:]

    xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    Z = clf.\_decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape) # 绘制的是整个区域的等高线

    plt.title("Local Outlier Factor(LOF)")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    a = plt.scatter(X[:200, 0], X[:200, 1], c='white', edgecolor='k', s=20)

    b = plt.scatter(X[200:, 0], X[200:, 1], c='red', edgecolor='k', s=20)

    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([a, b], ["normal observations", "abnormal observations"], loc="upper left")

    plt.show()

### 背景介绍
主要介绍的是局部异常因子LOF估计器。是一种无监督的异常值检测方法。
n_neighbors一般取用20，普遍有不错的效果。

#### 英语生词
Outlier 异常值
Local Outlier Factor 局部异常因子
Anomaly 异常
deviation 偏离
substantially 可观的

end
