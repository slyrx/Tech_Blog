---
layout: post
title:  "scikit learn: Multi-output Decision Tree Regression"
date:   2018-07-22 11:06:30
tags: [机器学习, 数据挖掘, scikit-learn, Decision Trees]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.RandomState(1)
    X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(20, 2))

    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_3 = DecisionTreeRegressor(max_depth=8)
    regr_1.fit(X, y)
    regr_2.fit(X, y)
    regr_3.fit(X, y)

    X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
    y_3 = regr_3.predict(X_test)

    plt.figure()
    s = 50
    s = 25
    plt.scatter(y[:, 0], y[:, 1], c="navy", s=s, edgecolor="black", label="data")
    plt.scatter(y_1[:, 0], y_1[:, 1], c="cornflowerblue", s=s, edgecolor="black", label="max_depth=2")
    plt.scatter(y_2[:, 0], y_2[:, 1], c="red", s=s, edgecolor="black", label="max_depth=5")
    plt.scatter(y_3[:, 0], y_3[:, 1], c="orange", s=s, edgecolor="black", label="max_depth=8")

    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.xlabel("target 1")
    plt.ylabel("target 2")
    plt.title("Multi-output Decision Tree Regression")
    plt.legend(loc="best")
    plt.show()

### 背景介绍
该示例介绍了使用决策树进行多路输出回归。

可以看到，树越深，决策树对细节学习越好，但有可能过度拟合。

这个例子和上一个例子没有什么区别，只是输入数据由sin曲线变成了圆。并且也没有说清楚再选择决策树深度上是通过什么样的原则进行的。

#### 英语单词
Multi-output 多路输出
simultaneously 同时地
underlying feature 基础特征

end
