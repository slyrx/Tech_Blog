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
    stack = [(0, -1)] # 采用了广度优先相近的思路，遍历树一遍把所有叶子节点都标记出来，非叶子节点用另外的方式标记出来。
    while len(stack) > 0:
        node_id, parent_depth = stack.pop() # node_id是按照栈的进入顺序来规定序号的
        node_depth[node_id] = parent_depth + 1

        if(children_left[node_id] != children_right[node_id]): # 子树的左右节点相等的情况只出现在当它是叶子节点的时候，此时叶子节点因为两边子树节点都是nul，所以可以区分出来。
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

    node_indicator = estimator.decision_path(X_test) # 返回一条记录的决策路径，如果是一串记录，则一条条的详尽依次记录每条的判断过程。比如：(0, 0)，(0, 2)，(0, 4)；(1, 0)，(1, 2)，(1, 3)，(2, 0)，(2, 1)，(3, 0)，(3, 2)，(3, 4)	...(29, 0)，(29, 1)...,表示第0条是通过0，2，4得到结论；第1条是通过0，2，3得到结论；第2条是通过0，1得到结论，后续类似

    leave_id =  estimator.apply(X_test) # 返回每个记录被预测到的叶子节点的序号，也就是结果。每条记录最终被判定为哪一类。

    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]] # 这里的sample_id + 1，所加的1是一个单位，即一条记录的完整路径，比如0的完整路径有3个元素，所以加1看到的结果总数就是加了3；而2的完整路径是2个元素，所以加1看到的结果就是加了2；可以把它想象成一个列表，这里的1是按行相加，而每行的元素个数是不一定的，并不完全相等，依赖的是叶节点到根节点的距离。indptr表示指向这个行的指针；indices表示把这行读出来后，对这行数字的解读，最终将解析成一个narray数组。

    print('Rules used to predict sample %s: ' % sample_id)

    for node_id in node_index:
        if leave_id[sample_id] != node_id: # 查找第0个记录的结论点在该记录中的决策路径中是否出现了。直到找到结论点，之后将其打印输出。
            continue

        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s :(X_test[%d, %s] (= %s) %s %s)" % (node_id, sample_id, feature[node_id], X_test[sample_id, feature[node_id]], threshold_sign, threshold[node_id])) # 这里的decision node就是最终分类得到的结论，1、3、4这3个节点在特征属性上是属于同一个属性的，只是结论是不同的类型，一个属性特征里的3个结果分类。

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
