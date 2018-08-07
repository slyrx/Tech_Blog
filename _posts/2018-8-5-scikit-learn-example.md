---
layout: post
title:  "scikit learn: Plot the decision boundaries of a VotingClassifier"
date:   2018-08-05 18:31:30
tags: [机器学习, 数据挖掘, scikit-learn, ensemble methods]
---

    print(__doc__)

    from itertools import product

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier

    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]
    y = iris.target

    clf1 = DecisionTreeClassifier(max_depth=4)
    clf2 = KNeighborsClassifier(n_neighbors=7)
    clf3 = SVC(kernel='rbf', probability=True)
    eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting="soft", weights=[2, 1, 2])

    clf1.fit(X, y)
    clf2.fit(X, y)
    clf3.fit(X, y)
    eclf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

    for idx, clf, tt in zip(product([0, 1], [0, 1]), [clf1, clf2, clf3, eclf], ['Decision Tree (depth=4)', 'KNN (k=7)', 'Kernel SVM', 'Soft Voting']): # product表示一个矩阵[0, 0],[0, 1],[1, 0],[1, 1]
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
        axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

        axarr[idx[0], idx[1]].set_title(tt)

    plt.show()

### 背景介绍
这个例子才算是真正意义上的多模型组合预测。

#### 英语生词
meshgrid 用于三维曲面的分格线座标；产生“格点”矩阵

end
