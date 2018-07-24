---
layout: post
title:  "scikit learn: Understanding the decision tree structure"
date:   2018-07-23 08:27:30
tags: [机器学习, 数据挖掘, scikit-learn, Decision Trees]
---

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data # 数据有4个特征
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
estimator.fit(X_train, y_train)
\# 决策评估器有一个属性叫做tree_, 它保存着整个树的结构并且允许访问低层级的属性。二叉树的tree_属性表现形式是平行的数字数组。每个数组中的第i个元素保存这第i个节点的信息。Node 0表示树的树根。

n_nodes = estimator.tree_.node_count # 决策树中的节点个数，此处为5，这里的节点就是指属性
children_left = estimator.tree_.children_left # 树的左子树有2个节点
children_right = estimator.tree_.children_right # 树的右子树有2个节点，通过左右子树的运行结果，可以看到左右下的分裂特征采用的是一样的
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold

node_depth = np.zeros(shape=n_nodes, dtype= np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)] # 采用了广度优先相近的思路
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    if(children_left[node_id] != children_right[node_id]):
      stack.append((children_left[node_id], parent_depth + 1))
      stack.append((children_right[node_id], parent_depth + 1))
    else:
      is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has the following tree structure:" % n_nodes)

for i in range(n_nodes):
    if is_leaves[i]:
       print("%snode=%s leaf node." % (node_depth[i] * "\\t", i))
    else:
       print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to node %s." % (node_depth[i] * "\\t", i, children_left[i], feature[i], threshold[i], children_right[i],))

print()

node_indicator = estimator.decision_path(X_test)

leave_id =  estimator.apply(X_test)

sample_id = 0
node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]

print('Rules used to predict sample %s: ' % sample_id)

for node_id in node_index:
    if leave_id[sample_id] != node_id:
        continue

    if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
        threshold_sign = "<="
    else:
        threshold_sign = ">"

    print("decision id node %s :(X_test[%d, %s] (= %s) %s %s)" % (node_id, sample_id, feature[node_id], X_test[sample_id, feature[node_id]], threshold_sign, threshold[node_id]))

sample_ids = [0, 1]
common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids))

common_node_id = np.arange(n_nodes)[common_nodes]

print("\\nThe following samples %s share the node %s in the tree" % (sample_ids, common_node_id))

print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))

### 背景介绍
通过分析决策树的结构可以洞察特征和预测目标之间的关系。


#### 英语生词
retrieve  取回、检索

end
