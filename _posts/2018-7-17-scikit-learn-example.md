---
layout: post
title:  "scikit learn: Restricted Boltzmann Machine features for classification"
date:   2018-07-16 10:31:30
tags: [机器学习, 数据挖掘, scikit-learn, Neural Networks]
---

    print(__doc__)

    import numpy as np
    import mathplotlib.pyplot as plt
    from sklearn import svm

    xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
    np.random.seed(0)
    X = np.random.randn(300, 2) # 生成300个随机值，供2列
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0) # 都大于0的归类为1，其他归类为0

    clf = svm.NuSVC() # 非线性svm分类器
    clf.fit(X, Y)

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) # Z通过正负进行二分类，通过值的大小表示归属的概率
    Z = Z.reshape(xx.shape # Z的结果没有和Y一样是True和False的，而是一个浮点值，对应于xx和yy画出来的坐标所带的可能的值(或者说可能的概率).

    plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto', origin='lower', cmap=plt.cm.PuOr_r) # 以xx和yy坐标所能涉及的区域画图，imshow表示利用matplotlib包对图片进行绘制，image要绘制对图像或数组，cmap颜色图谱，interpolation表示图片以什么方式展现，像素或高斯模糊等15种方式。

    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linetypes="--") # 根据得出的结果Z画出等高线

    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired, edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis([-3, 3, -3, 3])
    plt.show()



### 背景介绍
该例演示了使用以RBF为核心的非线性SVC进行二分类。目标是预测输入的XOR情况。
结果图中的颜色表示了该svc模型学习得到的超平面(the decision function)

#### 名词解释
XOR 逻辑异或
decision_function 在二分类的点群中，想象它们之间有一个超级平面，简称超平面，能够准确的把这两类点分开，那么在这个超平面的哪一面就说明了这个点归属于二分类的哪一个类；而得到的值的大小，则表示了属于这个平面可能性的程度。更大表示更近更可能是结论类，更小表示更远更不可能是结论类。








end
