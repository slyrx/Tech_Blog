---
layout: post
title:  "scikit learn: Plot Ridge coefficients as a function of the regularization"
date:   2018-07-28 10:16:30
tags: [机器学习, 数据挖掘, scikit-learn, Generalized Linear Models]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import linear_model

    # X is the 10x10 Hilbert matrix
    X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
    y = np.ones(10)

    # #############################################################################
    # Compute paths

    n_alphas = 200
    alphas = np.logspace(-10, -2, n_alphas) # logspace表示返回在对数刻度上均匀间隔的数字。

    coefs = []
    for a in alphas:
        ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
        ridge.fit(X, y)
        coefs.append(ridge.coef_)

    # #############################################################################
    # Display results

    ax = plt.gca()

    ax.plot(alphas, coefs) # 画出alphas和coefs的对应关系，
    ax.set_xscale('log') # 曲线特征如果过小，通过log进行放大
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()

#### 结论
当α非常大时，正则化效应强，并且系数趋于零。 当α趋于零，系数表现出大的振荡。 在实践中，有必要以这样的方式调整alpha，以便在两者之间保持平衡
这一点，通过输出的图表就能看出来。

#### 疑问
coefs是一个二维数组，如何与alphas在图像上对应的？
本质就是一个线性函数的计算过程。

end
