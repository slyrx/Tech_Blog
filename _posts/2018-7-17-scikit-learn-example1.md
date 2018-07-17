---
layout: post
title:  "scikit learn: Restricted Boltzmann Machine features for classification"
date:   2018-07-16 10:31:30
tags: [机器学习, 数据挖掘, scikit-learn, Support Vector Machines]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=40, centers=2, random_state=6)

    clf = svm.SVC(kernel='linear', C=1000) # C表示错误项的惩罚参数
    clf.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    ax = plt.gca() # 获取当前figure中的Axes实例，如果没有参数就创建一个，如果有参数就按照参数指定找对应的
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)

    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--']) # levels表示一系列等级曲线，以升序排列, 此处是3条线，最小，最大和中间。

    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none') # 画出落在超平面最大边距上的点，这样的点在支持向量机svm里就称做支持向量support_vectors

    plt.show()




### 背景介绍
画出超平面的最大边界界限，数据集采用由线性内核支持向量机分析使用的二分类数据集


#### 英语生词
blobs 斑点
blob 一团，一小团


#### 名词解释
rbf Radial Based Function

end
