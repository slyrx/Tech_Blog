---
layout: post
title:  "scikit learn: Visualization of MLP weights on MNIST"
date:   2018-07-11 13:01:30
tags: [机器学习, 数据挖掘, scikit-learn, Neural Networks]
---

    print(__doc__)

    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_mldata
    from sklearn.neural_network import MLPClassifier

    mnist = fetch_mldata("MNIST original")

    X,y = mnist.data / 255., mnist.target
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.1) # hidden_layer_sizes在第i个隐藏层上出现的i个神经元数目，这里表示有50个神经元在隐藏层中。 max_iter迭代的最大次数；alpha:L2惩罚(正则化)参数；solver权重解决方案，共有三种{lbfgs, sgd, adam},lbfgs表示quasi-Newton类中的，sgd表示随机梯度，adam表示某种优化方式优化后的随机梯度. lbfgs适用于小数据集，处理快效果较好。adam适用于大数据集，数量至少几千个。verbose指示是否将执行过程打印到标准输出窗口；tol表示贴近优化的容忍度，也就是容错度。random_state随机种子。learning_rate_init学习率，其实就是梯度提升时的提升跨度，因此也只能在设置成梯度提升的解决方法来使用，配合solver设置成sgd或adam来使用。

    mlp.fit(X_train, y_train)
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))

    fig, axes = plt.subplots(4,4)

    vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max() # 可以理解为系数矩阵，也即权重矩阵的第一行中最小和最大值。有一个疑问是，***为什么只取第一行？***

    for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()): # axes表示16个子图被压缩到一个维度上了，这里会把16个子图一个一个取出来，并将各个子图的x和y轴隐藏不要显示，现在就是比较疑惑为什么这里取的是coefs_[0],难道取第一个值就已经完全表示系数矩阵了？
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5*vmin, vmax=.5*vmax)
        ax.set_xticks(())
        ax.set_yticks(())

    plt.show()


### 背景介绍
输入数据是28x28像素的手写数字，所以每张图片有28*28=784个特征。而权重矩阵也会是28*28的，如果要压缩，也会被压缩为784的1维。
由于数据集下载网址出现一些问题，导致不能访问，所以脚本没有运行起来。

#### 名词解释
MNIST 手写数字字符识别数据集。mldata.org 是一个公开的机器学习数据 repository ,由 PASCAL network 负责支持。
the first layer :表示神经网络的第一层，神经网络可以有很多层，每一层都有对应的权重。
quasi-Newton

#### 英语单词
coefficient 系数
spatial 空间的
stochastic 随机的
proposed 提出
optimizer 优化



end
