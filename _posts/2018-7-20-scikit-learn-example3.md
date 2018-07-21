---
layout: post
title:  "scikit learn: Scaling the regularization parameter for SVCs"
date:   2018-07-20 14:18:30
tags: [机器学习, 数据挖掘, scikit-learn, Support Vector Machines]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.svm import LinearSVC
    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import GridSearchCV
    from sklearn.utils import check_random_state
    from sklearn import datasets

    rnd = check_random_state(1)

    n_samples = 100
    n_features = 300

    X_1, y_1 = datasets.make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, random_state=1) # 生成数据集

    y_2 = np.sign(.5 - rnd.rand(n_samples)) # np.sign表示返回表达的正负性，如果结果偏负就是-1，偏正就是1，中间就是0
    X_2 = rnd.randn(n_samples, n_features // 5) + y_2[:, np.newaxis] # np.newaxis表示为多维数组增加一个轴，例如x = array[0, 1, 2], x[:, np.newaxis] ==> x就变成了array([[0], [1], [2]])
    X_2 += 5 * rnd.randn(n_samples, n_features // 5)

    clf_sets = [(LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=1e-3), np.logspace(-2.3, -1.3, 10), X_1, y_1), (LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=1e-4), ***np.logspace(-4.5, -2, 10)***, X_2, y_2)]

    colors = ['navy', 'cyan', 'darkorange']
    lw = 2

    for fignum, (clf, cs, X, y) in enumerate(clf_sets):
        plt.figure(fignum, figsize=(9, 10))

        for k, train_size in enumerate(np.linspace(0.3, 0.7, 3)[::-1]): # [::-1]表示倒序排列
            param_grid = dict(C=cs) # 网格搜索的参数设置，也对应横坐标轴，正则化的数量

            grid = GridSearchCV(clf, refit=False, param_grid=param_grid, cv=ShuffleSplit(train_size=train_size, n_splits=250, random_state=1)) # cv表示交叉验证，GridSearchCV表示为分类器全面的搜索具体的参数。

            grid.fit(X, y)
            scores = grid.cv_results_['mean_test_score'] # 对分类器交叉验证的结果进行打分

            scales = [(1, 'No scaling'), ((n_samples * train_size), '1/n_samples'),]

            for subplotnum, (scaler, name) in enumerate(scales):
                plt.subplot(2, 1, subplotnum + 1)
                plt.xlabel('C')
                plt.ylabel('CV Score')
                grid_cs = cs * float(scaler) # 图标中的横坐标,正则化的数量
                plt.semilogx(grid_cs, scores, label='fraction %.2f'% train_size, color=colors[k], lw=lw) # semilogx在x轴上绘制一个log对数缩放图。log在公式中的作用就可以被称为起对数据缩放的作用。简单来说是对x取log，同比如果是y则为semilogy, 同时对xy取对数则为loglog(); 这里的scores应该对应不同scaling所取得的分数，这个示例为了简化把不同的scaling对应了相同的scores，所以从分数上看不出什么
                plt.title('scaling=%s, penalty=%s, loss=%s'% (name, clf.penalty, clf.loss))

        plt.legend(loc="best")
    plt.show()


### 背景介绍
这个示例是介绍在使用支持向量机进行分类时缩放调整正则化参数时会产生什么样的效果。这个示例有些许问题。具体原因在上面的代码部分已经简述。

#### 结果解读
在数据不进行缩放时，l1正则下，10的-2次方的数据量三种


#### 英语生词
wise 聪明
Exhaustive 全面的
scaling 缩放


end
