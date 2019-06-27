---
layout: post
title:  "scikit learn: Path with L1-Logistic Regression"
date:   2018-07-28 21:32:30
tags: [机器学习, 数据挖掘, scikit-learn, Generalized Linear Models]
---

    print(__doc__)

    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import linear_model
    from sklearn import datasets
    from sklearn.svm import l1_min_c

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = X[y != 2]
    y = y[y != 2]

    X -= np.mean(X, 0)

    cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3) #


    print("Computing regularization path ...")
    start = datetime.now()
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    coefs_ = []
    for c in cs:
        clf.set_params(C=c)
        clf.fit(X, y)
        coefs_.append(clf.coef_.ravel().copy())
    print("This took ", datetime.now() - start)

    coefs_ = np.array(coefs_)
    plt.plot(np.log10(cs), coefs_)
    ymin, ymax = plt.ylim()
    plt.xlabel('log(C)')
    plt.ylabel('Coefficients')
    plt.title('Logistic Regression Path')
    plt.axis('tight')
    plt.show()

### 背景介绍
原来这个示例介绍的是参数C在变化过程中，导致的系数波动程度。
这里红色线的波动最小，可以说最好，绿色最差。

end
