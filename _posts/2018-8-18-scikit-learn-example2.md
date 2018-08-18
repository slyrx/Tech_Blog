---
layout: post
title:  "scikit learn: Gradient Boosting Out-of-Bag estimates"
date:   2018-08-18 00:07:30
tags: [机器学习, 数据挖掘, scikit-learn, ensemble methods]
---

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import ensemble
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split

    n_samples = 1000
    random_state = np.random.RandomState(13)
    x1 = random_state.uniform(size=n_samples)
    x2 = random_state.uniform(size=n_samples)
    x3 = random_state.randint(0, 4, size=n_samples)

    p = 1 / (1.0 + np.exp(-(np.sin(3 * x1) - 4 * x2 + x3)))
    y = random_state.binomial(1, p, size=n_samples)

    X = np.c_[x1, x2, x3]

    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=9)

    params = {'n_estimators':1200, 'max_depth':3, 'subsample':0.5, 'learning_rate':0.01, 'min_samples_leaf': 1, 'random_state':3}
    clf = ensemble.GradientBoostingClassifier(\**params)

    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print("Accuracy: {:.4f}".format(acc))

    n_estimators = params['n_estimators']
    x = np.arange(n_estimators) + 1

    def heldout_socre(clf, X_test, y_test):
        score = np.zeros((n_estimators,), dtype=np.float64)
        for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
            score[i] = clf.loss_(y_test, y_pred)
        return score

    def cv_estimate(n_splits=3):
        cv = KFold(n_splits=n_splits)
        cv_clf = ensemble.GrandientBoostingClassifier(\**params)
        val_scores = np.zeros((n_estimators,), dtype=np.float64)
        for train, test in cv.split(X_train, y_train):
            cv_clf.fit(X_train[train], y_train[train])
            val_scores += heldout_socre(cv_clf, X_train[test], y_train[test])
        val_scores /= n_splits
        return val_scores

    cv_score = cv_estimate(3) # 训练集交叉验证打分结果

    test_score = heldout_socre(clf, X_test, y_test) # 测试集的打分结果

    cumsum = -np.cumsum(clf.oob_improvement_) # cumsum表示装袋改进结果

    oob_best_iter = x[np.argmin(cumsum)]

    test_score -= test_score[0]
    test_best_iter = x[np.argmin(test_score)]

    cv_score -= cv_score[0]
    cv_best_iter = x[np.argmin(cv_score)]

    oob_color = list(map(lambda x: x / 256.0, (190, 174, 212)))
    test_color = list(map(lambda x: x / 256.0, (127, 201, 127)))
    cv_color = list(map(lambda x: x / 256.0, (253, 192, 134)))

    plt.plot(x, cumsum, label='OOB loss', color=oob_color)
    plt.plot(x, test_score, label='Test loss', color=test_color)
    plt.plot(x, cv_score, label='CV loss', color=cv_color)
    plt.axvline(x=oob_best_iter, color=oob_color)
    plt.axvline(x=test_best_iter, color=test_color)
    plt.axvline(x=cv_best_iter, color=cv_color)

    xticks = plt.xticks()
    xticks_pos = np.array(xticks[0].tolist() + [oob_best_iter, cv_best_iter, test_best_iter])
    xticks_label = np.array(list(map(lambda t: int(t), xticks[0])) + ['OOB', 'CV', 'Test'])
    ind = np.argsort(xticks_pos)
    xticks_pos = xticks_pos[ind]
    xticks_label = xticks_label[ind]
    plt.xticks(xticks_pos, xticks_label)

    plt.legend(loc='upper right')
    plt.ylabel('normalized loss')
    plt.xlabel('number of iterations')

    plt.show()    

end
