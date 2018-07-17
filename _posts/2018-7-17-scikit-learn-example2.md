---
layout: post
title:  "scikit learn: Support Vector Regression(SVR) using linear and non-linear kernels"
date:   2018-07-17 11:13:30
tags: [机器学习, 数据挖掘, scikit-learn, Support Vector Machines]
---

    print(__doc__)

    import numpy as np
    from sklearn.svm import SVR
    import matplotlib.pyplot as plt

    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X).ravel()

    y[::5] += 3 * (0.5 - np.random.rand(8))

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) # rbf表示Radial Based Function, gamma表示核系数，该核系数只用于rbf, poly和sigmoid
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)

    y_rbf = svr_rbf.fit(X, y).predict(X)
    y_lin = svr_lin.fit(X, y).predict(X)
    y_poly = svr_poly.fit(X, y).predict(X)

    lw = 2
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model') # lw表示lineWidth线条宽度
    plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()


### 背景介绍
这个一个针对linear, polynomial and RBF作为内核模型而编写的简单示例

#### 结论 图像解读
通过结果图像可以看到，RBF内核的还原度最好，线性模型就是一条趋势线，多因子模型只是稍微比线性模型好一些，轻微的有一些波动，但是也很难完整的体现契合度。

#### 名词解释
+ rbf Radisal Based Function 径向基函数，也叫高斯径向基函数。
+ 径向基函数是一个它的值(y)只依赖于变量(x)距原点距离的函数，即Φ(X) = Φ(\|\|X\|\|)
+ exp()以自然常数e为底的指数函数。
+ gamma gamma是选择RBF函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。
+ grid search Grid Search是用在Libsvm中的参数搜索方法。很容易理解：就是在C,gamma组成的二维参数矩阵中，依次实验每一对参数的效果。
+ 使用grid Search虽然比较简单，而且看起来很naïve。但是他确实有两个优点：
+ 可以得到全局最优
+ (C,gamma)相互独立，便于并行化进行

#### 问题
+ 惩罚系数是越大越好？还是越小越好？
+ C是惩罚系数，即对误差的宽容度。c越高，说明越不能容忍出现误差,容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差. 说明也不能过大也不能过小，取一个平衡值。
+ 惩罚系数是适用与所有的模型吗？还是只针对支持向量机？似乎在神经网络也见到过

#### 英语生词
Radial 放射状的，光线的
polynomial 多项式




end
