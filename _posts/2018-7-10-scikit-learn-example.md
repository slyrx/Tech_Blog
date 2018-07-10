---
layout: post
title:  "scikit learn: Decision Tree Regression with AdaBoost"
date:   2018-06-29 09:26:30
tags: [机器学习, 数据挖掘, scikit-learn, ensemble methods]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeRegressor
    form sklearn.ensemble import AdaBoostRegressor

    rng = np.random.RandomState(1)  #生成随机种子
    X = np.linspace(0, 6, 100)[:, np.newaxis] #linspace表示指定间隔内返回均匀间隔的数字
    y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0]) #ravel将多维数组降为一维

    regr_1 = DecisionTreeRegressor(max_depth=4)
    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng) #random_state表示随机数种子，随机数种子与当前系统时间有关。为了进行可重复的训练，需要固定一个random_state,调参的时候是不需要调random_state的。如果设置成了none，则会根据机器的不同随机产生的数也不同，那么理解为是为了平衡各个电脑环境的差异，而作出的平衡措施。

    regr_1.fit(X, y)
    regr_2.fit(X, y)

    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)

    plt.figure()
    plt.scatter(X, y, c="k", label="training samples")
    plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
    plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Boosted Decision Tree Regression")
    plt.legend() #显示图例
    plt.show()


### 背景介绍
这节是介绍使用adaboost.R2做梯度提升。使用的数据集是一个含有少量高斯噪点的正弦函数。在做这个梯度提升的过程中，使用了299次提升，也就是共有300个决策树生成，与其产生对比的是单决策树的回归判断。这里认为，随着梯度提升的数目攀升，回归器能够发现更多的细节。

#### 感悟
这些算法都是针对数字比较有效的处理方式，具有客观性。所以不论你拿到什么样的数据，都会有相同的处理结果。

















end
