---
layout: post
title:  "scikit learn: SGD: convex loss functions"
date:   2018-07-28 21:32:30
tags: [机器学习, 数据挖掘, scikit-learn, Generalized Linear Models]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt


    def modified_huber_loss(y_true, y_pred):
        z = y_pred * y_true
        loss = -4 * z
        loss[z >= -1] = (1 - z[z >= -1]) ** 2
        loss[z >= 1.] = 0
        return loss


    xmin, xmax = -4, 4
    xx = np.linspace(xmin, xmax, 100)
    lw = 2
    plt.plot([xmin, 0, 0, xmax], [1, 1, 0, 0], color='gold', lw=lw,
             label="Zero-one loss")
    plt.plot(xx, np.where(xx < 1, 1 - xx, 0), color='teal', lw=lw,
             label="Hinge loss")
    plt.plot(xx, -np.minimum(xx, 0), color='yellowgreen', lw=lw,
             label="Perceptron loss")
    plt.plot(xx, np.log2(1 + np.exp(-xx)), color='cornflowerblue', lw=lw,
             label="Log loss")
    plt.plot(xx, np.where(xx < 1, 1 - xx, 0) ** 2, color='orange', lw=lw,
             label="Squared hinge loss")
    plt.plot(xx, modified_huber_loss(xx, 1), color='darkorchid', lw=lw,
             linestyle='--', label="Modified Huber loss")
    plt.ylim((0, 8))
    plt.legend(loc="upper right")
    plt.xlabel(r"Decision function $f(x)$")
    plt.ylabel("$L(y=1, f(x))$")
    plt.show()

### 总结
以上示例是为了介绍损失函数，这里的损失其实都是相对比较简单的。

#### 英语生词
convex 凸面的，凸函数前的定语，表示此处的损失函数都是凸函数性质的。
