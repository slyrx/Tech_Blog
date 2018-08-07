---
layout: post
title:  "scikit learn: Prediction Intervals for Gradient Boosting Regression"
date:   2018-08-07 11:10:30
tags: [机器学习, 数据挖掘, scikit-learn, ensemble methods]
---

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.ensemble import GradientBoostingRegressor

    np.random.seed(1)

    def f(x):
        return x * np.sin(x)

    X = np.atleast_2d(np.random.uniform(0, 10.0, size=100)).T
    X = X.astype(np.float32)

    y = f(X).ravel()

    dy = 1.5 + 1.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise
    y = y.astype(np.float32)

    xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
    xx = xx.astype(np.float32)

    alpha = 0.95
    clf = GradientBoostingRegressor(loss='quantile', alpha=alpha, n_estimators=25-, max_depth=3, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)

    clf.fit(X, y)

    y_upper = clf.predict(xx) # 在损失函数是quantile时的预测值，这时的预测结果总体偏松，包容度较强，因此得到的值都是偏大的，也就是可以做上限。

    clf.set_params(alpha=1.0 - alpha) # 在修改了alpha值，alpha是正则化项参数，的情况下重新拟合出新的模型
    clf.fit(X, y)

    y_lower = clf.predict(xx) # 新模型在调低正则化参数alpha的情况下，得出了预测值的下限

    clf.set_params(loss='ls') # 将损失函数调整为ls后，拟合出新的模型
    clf.fit(X, y)

    y_pred = clf.predict(xx)

    fig = plt.figure()
    plt.plot(xx, f(xx), 'g:', label=u'$f(x) = x\\, \\sin(x)$') # f(xx)是严格的正确数据
    plt.plot(X, y, 'b.', markersize=10, label=u'Observations') # f(xx)增加了部分干扰数据
    plt.plot(xx, y_pred, 'r-', label=u'Prediction') #带干扰数据得到的预测结果
    plt.plot(xx, y_upper, 'k-') #带干扰数据得到的预测结果，上限
    plt.plot(xx, y_lower, 'k-') #带干扰数据得到的预测结果，下限
    plt.fill(np.concatenate([xx, xx[::-1]]), np.concatenate([y_upper, y_lower[::-1]]), alpha=.5, fc='b', ec='None', label='90% prediction interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$') # __$__ 在这里的作用？
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.show()

end
