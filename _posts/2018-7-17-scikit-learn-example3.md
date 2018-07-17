---
layout: post
title:  "scikit learn: SVM with custom kernel"
date:   2018-07-17 17:01:30
tags: [机器学习, 数据挖掘, scikit-learn, Support Vector Machines]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm, datasets

    iris = datasets.load_iris()
    X = iris.data[:, :2]
    Y = iris.target

    def my_kernel(X, Y):
        M = np.array([[2, 0], [0, 1.0]])
        return np.dot(np.dot(X, M), Y.T)

    h = .02 # 可以看成是step

    clf = svm.SVC(kernel=my_kernel) # 支持向量机设置自己的核函数
    clf.fit(X, Y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # 预测值

    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired) # pcolormesh画一个四边形的网，对坐标点着色，

    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
    plt.title('3-Class classification using Support Vector Machine with custom kernel')

    plt.axis('tight')
    plt.show()

### 背景介绍
简单的使用支持向量机分类的例子。这个例子将画出决策面(可以理解成超平面)和支持向量。

#### 总结
等高线就等于轮廓线

#### 问题
+ pcolormesh为什么画出来的图是3个颜色？
+ pcolormesh和pcolor作用一样，只是在实现的时候使用的数据结构不一样，pcolormesh在效率上有所提升
+ 通过理解pcolor理解pcolormesh
+ pcolor(X, Y, C)绘制指定颜色C和指定网格线间间距的伪彩色图。参数X和Y是指定网格间间距的向量或者矩阵。若X,Y为矩阵，则X和Y，C维数相同；若X,Y为向量，则X,Y的长度分别等于矩阵C的列数和行数。
+ 因此pcolor表示，画出一个网格，这个网格的横坐标是[1,2,3,4,5,6], 纵坐标是[2,4,8,10,12,14], 其中2和8之间因为缺少6，会形成一个面积明显大很多的网格。
+ 参考 https://blog.csdn.net/yangfengman/article/details/53097591


end
