---
layout: post
title:  "scikit learn: One-class SVM with non-linear kernel(RBF)"
date:   2018-07-19 06:50:30
tags: [机器学习, 数据挖掘, scikit-learn, Support Vector Machines]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.font_manager
    from sklearn import svm

    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

    X = 0.3 * np.random.randn(100, 2) # np.random.randn返回一些服从“正态分布”的样例
    X_train = np.r_[X + 2, X - 2]
    X = 0.3 * np.random.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) # 获取超平面使用的是xx，yy数据, 通过超平面计算得到的Z，主要是用在等高线上的显示， X和xx目测毫无关系，只是在作图方面可以大概的画出这么一个趋势，所以用来画图
    Z = Z.reshape(xx.shape) # 用在等高线上

    plt.title("Novelty Dectection")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu) # 就是有一条轴可以囊括显示(Z.min(), Z.max())，现在强制把(Z.min(),0)按照一种方式显示，(0, Z.max())按照另外一种方式显示。
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred') # 整个这个部分就是在做等高线的绘制

    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s, edgecolors='k')

    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([a.collections[0], b1, b2, c],
               ["learned frontier", "training observations", "new regular observations", "new abnormal observations"], loc="upper left", prop=matplotlib.font_manager.FontProperties(size=11))

    plt.xlabel("error train: %d/200 ; errors novel regular: %d/40 ; "
               "errors novel abnormal: %d/40" % (n_error_train, n_error_test, n_error_outliers))

    plt.show()


### 背景介绍
该示例为使用单类SVM进行新颖地检测。
单类SVM是一种无监督算法，它学习用于新颖行检测的决策函数。其核心是:将新数据分类为与训练集相似或不同的形式。

#### 英语生词
novelty 新奇的
frontier  边界

end
