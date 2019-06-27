---
layout: post
title:  "scikit learn: Plot class probabilities calculated by the Voting Classifier"
date:   2018-08-07 17:28:30
tags: [机器学习, 数据挖掘, scikit-learn, ensemble methods]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier

    clf1 = LogisticRegression(random_state=123)
    clf2 = RandomForestClassifier(random_state123)
    clf3 = GaussianNB()
    X = np.array([[-1.0, -1.0], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
    y = np.array([1, 1, 2, 2])

    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[1, 1, 5]) # 原来这就是全套组合的处理过程, eclf前面的这个e就是全套组合的意思

    probas = [c.fit(X, y).predict_proba(X) for c in (clf1, clf2, clf3, eclf)] # 对二分类结果概率的计算

    class1_1 = [pr[0, 0] for pr in probas] # 各分类器判断结果为类1的正确概率
    class2_1 = [pr[0, 1] for pr in probas] # 各分类器判断结果为类2的正确概率

    N = 4
    ind = np.arange(N)
    width = 0.35

    fig, ax = plt.subplots()

    p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width, color='green', edgecolor='k')

    p2 = ax.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), width, color='lightgreen', edgecolor='k')

    p3 = ax.bar(ind, [0, 0, 0, class1_1[-1]], width, color='blue', edgecolor='k')

    p4 = ax.bar(ind + width, [0, 0, 0, class2_1[-1]], width, color='steelblue', edgecolor='k')

    plt.axvline(2.8, color='k', linestyle='dashed')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(['LogisticRegression\nweight 1', 'GaussianNB\nweight 1', 'RandomForestClassifier\nweight 5', 'VotingClassifer\n(average probabilities)'], rotation=40, ha='right')

    plt.ylim([0, 1])
    plt.title('Class probabilities for sample 1 by differnt classifiers')
    plt.legend([p1[0], p2[0]], ['class 1', 'class 2'], loc='upper left')
    plt.show()

### 背景介绍
对各分类器最终结果准确率的比较。

end
