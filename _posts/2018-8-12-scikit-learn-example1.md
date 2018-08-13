---
layout: post
title:  "scikit learn: Two-class AdaBoost"
date:   2018-08-12 17:21:30
tags: [机器学习, 数据挖掘, scikit-learn, ensemble methods]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_gaussian_quantiles

    X1, y1 = make_gaussian_quantiles(cov=2., n_samples=200, n_features=2, n_classes=2, random_state=1)

    X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=300, n_features=2, n_classes=2, random_state=1)

    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, - y2 + 1))

    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", min_estimators=200)

    bdt.fit(X, y)

    plot_colors = "br"
    plot_step = 0.02
    class_names = "AB"

    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

    Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis("tight")

    for i, n, c in zip(range(2), class_names, plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=c, cmap=plt.cm.Paired, s=20, edgecolor='k', label="Class %s" % n)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Decision Boundary')

    twoclass_output = bdt.decision_function(X) # 得出涉及到到决策函数
    plot_range = (twoclass_output.min(), twoclass_output.max())
    plt.subplot(122)
    for i, n, c in zip(range(2), class_names, plot_colors):
        plt.hist(twoclass_output[y == i], bins=10, range=plot_range, facecolor=c, label='Class %s' % n, alpha=.5, edgecolor='k') # 画矩形柱状图，通过y==i的形式判断出，当前选择的是显示哪一个类型的；y==i表示出了一个true和false的数组，twoclass_output[true]就表示当前这个下标的值会被显示，反之twoclass_output[false]表示不会被显示，这里一个tip是还和y==i这个数组的下标有关联的，y==i得到的下标相当于是一个蒙版。

    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2*1.2))
    plt.legend(loc='upper right')
    plt.ylabel('Samples')
    plt.xlabel('Score')
    plt.title('Decision Scores')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    plt.show()

### 背景介绍
decision_function得到的值可以看成一种打分，其中，通过正负进行了分类，通过值的大小来区别区分的确定程度。

end
