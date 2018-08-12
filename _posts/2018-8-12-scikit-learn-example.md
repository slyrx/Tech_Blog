---
layout: post
title:  "scikit learn: OOB Errors for Random Forests"
date:   2018-08-12 16:42:30
tags: [机器学习, 数据挖掘, scikit-learn, ensemble methods]
---

import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

print(__doc__)

RANDOM_STATE = 123

X, y = make_classification(n_samples=500, n_features=25, n_clusters_per_class=1, n_informative=15, random_state=RANDOM_STATE)

ensemble_clfs = [("RandomForestClassifier, max_features='sqrt'", RandomForestClassifier(warm_start=True, oob_score=True, max_features="sqrt", random_state=RANDOM_STATE)), ("RandomForestClassifier, max_features='log2'", RandomForestClassifier(warm_start=True, max_features='log2', oob_score=True, random_state=RANDOM_STATE)), ("RandomForestClassifier, max_features=None", RandomForestClassifier(warm_start=True, max_features=None, oob_score=True, random_state=RANDOM_STATE))] # max_features是什么？寻找到最佳分割比例的参考特征数目，If “auto”, then max_features=sqrt(n_features).If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).If “log2”, then max_features=log2(n_features).If None, then max_features=n_features.

error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

min_estimators = 15
max_estimators = 175

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

for label, clf_err in error_rate.items():
    xs, ys = zip(\*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()


### 背景介绍


end
