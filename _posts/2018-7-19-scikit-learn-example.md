---
layout: post
title:  "scikit learn: SVM Margins Example"
date:   2018-07-19 05:09:30
tags: [机器学习, 数据挖掘, scikit-learn, Support Vector Machines]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm

    np.random.seed(0)
    X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
    Y = [0] * 20 + [1] * 20

    fignum = 1

    for name, penalty in (('unreg', 1), ('reg', 0.05)):

        clf = svm.SVC(kernel='linear', C=penalty)
        clf.fit(X, Y)

        w = clf.coef_[0]
        a = -w[0]/w[1]
        xx = np.linspace(-5, 5) # 以数据集X,Y为依据产生的斜率和截距
        yy = a * xx - (clf.intercept_[0])/w[1]

        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        yy_down = yy - np.sqrt(1 + a ** 2) * margin
        yy_up = yy + np.sqrt(1 + a ** 2) * margin

        plt.figure(fignum, figsize=(4, 3))
        plt.clf() # 全称clear the current figure
        plt.plot(xx, yy, 'k-')
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')

        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none', zorder=10, edgecolors='k')
        plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired, edgecolors='k')

        plt.axis('tight')
        x_min = -4.8
        x_max = 4.2
        y_min = -6
        y_max = 6

        XX, YY = np.mgrid[x_min:x_ma:200j, y_min:y_max:200j]
        Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

        Z = Z.reshape(XX.shape)
        plt.figure(fignum, figsize=(4, 3)) # 表示图像里图层的意思，因此也就能叠加
        plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.xticks(())
        plt.yticks(())
        fignum = fignum + 1

    plt.show()


### 背景介绍
这个示例图解了惩罚系数C的影响效果。
当采取较大的C值时，往往是意味着对于数据集的分布并没有什么太大的信心。那么就只会考虑更靠近分割线的点。
而一个较小的C值，则启用了更多甚至所有的观察值，这时，会在计算边界时使用上所有的数据区域

这里较小的C值使用所有数据计算边界值，往往因为其取一个平均数可以容忍，所以最后画出来的线就在点群的中央。而较大的C值因为容忍度底，所以使用的数据要求高，容忍度低，只接受边界上的值，所以画出来的线就在点群的边界上。

#### 英语生词
faith  信心
distribution 分布

end
