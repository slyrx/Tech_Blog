---
layout: post
title:  "scikit learn: Label Propagation digits active learning"
date:   2018-07-29 10:45:30
tags: [机器学习, 数据挖掘, scikit-learn, Semi Supervised Classification]
---

    print(__doc__)

    # Authors: Clay Woolam <clay@woolam.org>
    # License: BSD

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    from sklearn import datasets
    from sklearn.semi_supervised import label_propagation
    from sklearn.metrics import classification_report, confusion_matrix

    digits = datasets.load_digits()
    rng = np.random.RandomState(0)
    indices = np.arange(len(digits.data))
    rng.shuffle(indices)

    X = digits.data[indices[:330]]
    y = digits.target[indices[:330]]
    images = digits.images[indices[:330]]

    n_total_samples = len(y)
    n_labeled_points = 10
    max_iterations = 5

    unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]
    f = plt.figure()

    for i in range(max_iterations):
        if len(unlabeled_indices) == 0:
            print("No unlabeled items left to label.")
            break
        y_train = np.copy(y)
        y_train[unlabeled_indices] = -1

        lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)
        lp_model.fit(X, y_train) # 利用现有训练集进行模型训练

        predicted_labels = lp_model.transduction_[unlabeled_indices] # 选择部分数据做测试集进行预测，并取得结果。
        true_labels = y[unlabeled_indices] # 相同的序号id，找出真实的测试集结果

        cm = confusion_matrix(true_labels, predicted_labels,
                              labels=lp_model.classes_)

        print("Iteration %i %s" % (i, 70 * "_"))
        print("Label Spreading model: %d labeled & %d unlabeled (%d total)"
              % (n_labeled_points, n_total_samples - n_labeled_points,
                 n_total_samples))

        print(classification_report(true_labels, predicted_labels))

        print("Confusion matrix")
        print(cm)

        # compute the entropies of transduced label distributions
        pred_entropies = stats.distributions.entropy(
            lp_model.label_distributions_.T)

        # select up to 5 digit examples that the classifier is most uncertain about
        uncertainty_index = np.argsort(pred_entropies)[::-1]  # 按照熵值排序，选择当前分类器认为最不确定的5个数字做下一步要进行预测的测试集
        uncertainty_index = uncertainty_index[
            np.in1d(uncertainty_index, unlabeled_indices)][:5] # 选择最不确定的5个数据记录的过程

        # keep track of indices that we get labels for
        delete_indices = np.array([])

        # for more than 5 iterations, visualize the gain only on the first 5
        if i < 5:
            f.text(.05, (1 - (i + 1) * .183),
                   "model %d\n\nfit with\n%d labels" %
                   ((i + 1), i * 5 + 10), size=10)
        for index, image_index in enumerate(uncertainty_index):
            image = images[image_index]

            # for more than 5 iterations, visualize the gain only on the first 5
            if i < 5:
                sub = f.add_subplot(5, 5, index + 1 + (5 * i))
                sub.imshow(image, cmap=plt.cm.gray_r, interpolation='none')
                sub.set_title("predict: %i\ntrue: %i" % (
                    lp_model.transduction_[image_index], y[image_index]), size=10)
                sub.axis('off')

            # labeling 5 points, remote from labeled set
            delete_index, = np.where(unlabeled_indices == image_index)
            delete_indices = np.concatenate((delete_indices, delete_index))

        unlabeled_indices = np.delete(unlabeled_indices, delete_indices) # 向训练集增加新数据过程，从原来的没有标注组中删除，加入到有标注组。
        n_labeled_points += len(uncertainty_index) # 同上

    f.suptitle("Active learning with Label Propagation.\nRows show 5 most "
               "uncertain labels to learn with the next model.", y=1.15)
    plt.subplots_adjust(left=0.2, bottom=0.03, right=0.9, top=0.9, wspace=0.2,
                        hspace=0.85)
    plt.show()


### 背景介绍
本例的主题是数字主动学习，通过标签逐渐扩大的方式。这里的Propagation就是指标签由确定的5个变成10个，再到15、20、25等这一系列过程。

所谓半监督就是指，在训练集的样本中添加了标注标签的记录只有很少的一部分，需要通过这少量的记录对取样的一少部分，此处为5条做预测，并将预测结果作为新的训练集应用到模型训练中。之后再取5个测试集进行预测，求得结果。这样一步步扩大被标注的数据集的范围。

这个例子也间接的介绍了半监督学习的方式的核心思想。

#### 英语生词
Propagation 传播

end
