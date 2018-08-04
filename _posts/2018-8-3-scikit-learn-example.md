---
layout: post
title:  "scikit learn: IsolationForest example"
date:   2018-08-03 18:31:30
tags: [机器学习, 数据挖掘, scikit-learn, ensemble methods]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest

    rng = np.random.RandomState(42)
     X = 0.3 * rng.randn(100, 2)
     X_train = np.r_[X + 2, X - 2]
     X = 0.3 * rng.randn(20, 2)
     X_test = np.r_[X + 2, X - 2]
     X_outliers = rng.uniform(ow=-4, high=4, size=(20, 2))

     clf = IsolationForest(max_samples=100, random_state=rng)
     clf.fit(X_train) # 需要注意的特征是这里拟合不需要y的参与，因此这个模型更偏向于一个聚类的方法
     y_pred_train = clf.predict(X_train)
     y_pred_test = clf.predict(X_test)
     y_pred_outliers = clf.predict(X_outliers)

     xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50)) # 新的数据，是一个5x5的矩阵，共50x50=2500个点，可以看成是2500个像素。
     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) # 这里的clf分类器，将这个新的50x50的矩阵也赋予了训练集X_train的特征。因此在最后的图形表现上也出现了相近的分布特征。
     Z = Z.reshape(xx.shape)

     plt.title("IsolationForest")
     plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

     b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolor='k')

     b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green', s=20, edgecolor='k')

     c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k')

     plt.axis('tight')
     plt.xlim((-5, 5))
     plt.ylim((-5, 5))
     plt.legend([b1, b2, c], ["training observations", "new regular observations", "new abnormal observations"], loc="upper left")

     plt.show()






### 背景介绍
这是一个使用IsolationForest进行异常检测的示例。

IsolationForest通过随机选择一个特征然后随机选择所选特征的最大值和最小值之间的分割值来“隔离”观察结果。
也就是，一种在合理值域内，随机抽样的过程。
而抽样数量的确定方式是，当前树根节点到叶节点的高度。
这种随机森林的平均路径长度是衡量正态性和决策函数的标准。就是说，可以通过平均路径长度做为指标来对正态性和决策函数做出判断。

这种区分方式认为较短的路径可以看成是异常值，也就是说被认为是相关性较小。

end
