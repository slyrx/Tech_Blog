---
layout: post
title:  "scikit learn: Restricted Boltzmann Machine features for classification"
date:   2018-07-16 10:31:30
tags: [机器学习, 数据挖掘, scikit-learn, Neural Networks]
---

    from __future__ import print_function

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.ndimage import convolve
    from sklearn import linear_model, datasets, metrics
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import BernoulliRBM
    from sklearn.pipeline import Pipeline

    def nudge_dadaset(X, Y):
        direction_vectors = [
            [[0, 1, 1],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [1, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 1],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 1, 0]]
        ]

        shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel() # convolve表示卷积，该函数表示返回两个1维序列的离散型卷积,此处引用的scipy库的函数， 参考 https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html

        X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) for vector in dirction_vectors]) # 将两个array序列拼接起来，拼接的方向默认是从竖轴方向拼的，也就是行数在不停的增多。apply_along_axis对数组X执行函数Shift，相当于一个匿名函数，X和其后的vector都是准备输入到shift函数中的参数，应该是可以很多的，是可扩展的，根据自己的需要，可以执行函数的函数，1表示轴方向axis

        Y = np.concatenate([Y for _ in range(5)], axis=0)
        return X,Y

    digits = datasets.load_digits()
    X = np.asarray(digits.data, 'float32')
    X,Y = nudge_dataset(X, digits.target)
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001) # 目的是把X归化到0到1的尺度间

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # train_test_split 训练集和测试集分割函数

    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)

    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)]) # 组合成管道模型，顺序执行steps序列里的评估器

    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    rbm.n_components = 100 # 数据集大小
    logistic.C = 6000.0

    classifier.fit(X_train, Y_train)

    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train, Y_train)

    print()
    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            classifier.predict(X_test)))) # metics.classification_report输出分类结果报告，功能是输入真实值和预测值进行比较，得出最终的命中率，每一行表示一个类型，竖列则表示对应标记出来的属性。

    print("Logistic regression using raw pixel features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            logistic_classifier.predict(X_test))))

    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(rbm.components_):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
                   interpolation='nearest') # 抽取的特征以图像形式表达出来
        plt.xticks(())
        plt.yticks(())

    plt.suptitle('100 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()


### 背景介绍
对于灰度图像数据，其中的像素值可以被解释为白色背景上的黑度，比如手写数字识别，Bernoulli Restricted Boltzmann machine model可以有效的对非线性特征进行提取。

为了在小数据集上有(学习)好的潜在表现，我们人工通过扰动的方式生成更多的标注数据，具体方法是在各个方向上偏移1个像素。

这个实例展示了怎么建立一个分类器工作流，通过BernoulliRBM特征提取器和逻辑回归分类器。这整个模型里的超参数将会通过网格搜索被优化。但是这个搜索过程因为运行时的约束所致，不在这里再现。超参数包括：学习率、隐藏层数量、正则化(也叫正规化)等

逻辑回归在原始数据上处理作为比较。这个示例表明BernoulliRBM模型可以帮助改进分类准确度。

#### 名词解释
+ recall

#### 英语生词
Restricted 受限制的
latent 潜在的
perturbing 扰动，烦扰
artificially 人工
hyperparameters 超参数
optimized 优化
grid search 网格搜索
reproduced 重现，再现
regularization 正则化、正归化
raw 原始的，初始的，生的
nudge 轻推
convolve 卷积
discrete 离散的，分离的
estimator 统计量，评价者


end
