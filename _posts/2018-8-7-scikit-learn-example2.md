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

end
