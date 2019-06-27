---
layout: post
title:  "scikit learn: SVM-Anova: SVM with univariate feature selection"
date:   2018-07-19 05:09:30
tags: [机器学习, 数据挖掘, scikit-learn, Support Vector Machines]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm, datasets, feature_selection
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    digits = datasets.load_digits()
    y = digits.target

    y = y[:200]
    X = digits.data[:200]
    n_samples = len(y)
    X = X.reshape((n_samples, -1))

    X = np.hstack((X, 2 * np.random.random((n_samples, 200)))) # np.random.random((n_samples, 200)) 生成数据集， n_samples行，200列

    transform = feature_selection.SelectPercentile(feature_selection.f_classif)

    clf = Pipeline([('anova', transform), ('svc', svm.SVC(C=1.0))])

    score_means = list()
    score_stds = list()
    percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

    for percentile in percentiles:
        clf.set_params(anova__percentile=percentile)  # 分类器在使用之前设置了方差分析的，这里循环设置了11个方差等级，用来查看产生的不同的影响
        this_scores = cross_val_score(clf, X, y, n_jobs=1) # 通过交叉验证对分类器进行打分
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())

    plt.errorbar(percentiles, score_means, np.array(score_stds)) # 绘制y与x，其中x作为行带有附加错误栏标记，也就是在标出结果值y的同时，还算了这个y值的方差偏离是多少

    plt.title('Performance of the SVM-Anova varying the percentile of features selected')

    plt.xlabel('Percentile')
    plt.ylabel('Prediction rate')

    plt.axis('tight')
    plt.show()


### 背景介绍
这个示例显示了如何在运行SVC支持向量分类器之前执行单变量特征选择以提高分类分数。

#### 名词解释
anova 方差分析

#### 英语生词
percentile
prediction rate




end
