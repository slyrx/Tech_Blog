---
layout: post
title:  "scikit learn: Lasso path using LARS"
date:   2018-07-27 15:56:30
tags: [机器学习, 数据挖掘, scikit-learn, Generalized Linear Models]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import linear_model
    from sklearn import datasets

    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    print("Computing regularization path using the LARS ...")
    alphas, \_, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True) # 这里是提取了线性模型的关键参数

    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]

    plt.plot(xx, coefs.T) # 是对模型系数的一种直观的观测
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle='dashed')
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.title('LASSO path')
    plt.axis('tight')
    plt.show()

### 背景介绍
这一系列的例子都是介绍广义模型的参数和评价标准的。

end
