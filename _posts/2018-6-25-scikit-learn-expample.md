---
layout: post
title:  "scikit learn: Normal and shringkage Linear Discriminant Analysis for classification"
date:   2018-06-25 07:58:30
tags: [机器学习, 数据挖掘, scikit-learn]
---

    from __future__ import division # 确保在2.1之前版本对python可以正常运行一些新的语言特性，必须放在文件的头部

    import numpy as np #导入numpy库,进行数据操作
    import matplotlib.pyplot as plt #导入绘图，进行可视化操作

    from sklearn.dataSet import make_blobs #sklearn的数据集，生成同向高斯斑点
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #一个带线性决策边界的分类器；由贝叶斯规则和类别条件密度拟合到数据生成。

    n_train = 20 #samples for training
    n_test = 200 #samples for testing
    n_averages = 50 #how often to repeat classification
    n_features_max = 75 #maximum number of n_features
    step = 4 # step size for the calculation

    def generate_data(n_samples, n_features):
      """Generate random blob-ish data with noisy features.
      This returns an array of input data with shape `(n_samples, n_features)` and an array of `n_samples` target labels.

      Only one feature contains discriminative information, the other features contain only noise.
      """
      X,y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-2], [2]]) # centers表示生成一个list，其元素是由两个list组成的。

      #add non-discriminative features
      if n_features > 1:
        X = np.hstack([X, np.random.randn(n_samples, n_features - 1)]) # hstack把两个数组水平的拼接起来，[1,2,3] hstack [4,5] = [1,2,3,4,5]
        # np.random.randn从正态分布里返回一组随机数据，如果参数是一个数组，则数组里都是随机数，即一个随机数组。
      return X,y

    acc_clf1,acc_clf2 = [],[] # 初始化两个数组
    n_features_range = range(1, n_features_max + 1, step) # range也是一个list
    for n_features in n_features_range:
      score_clf1, socre_clf2 = 0, 0
      for _ in range(n_averages):# \__xxx__ 系统定义名字  \__xxx 类中 的私有变量名 这里就可以看成是一个私有的临时变量，只做循环不做调用
      # 表示抽取50组数据，分别生成模型，看看50次抽样的打分均值如何，是否一直维持在一个稳定的水平
          X,y = generate_data(n_train, n_features)# 生成训练数据

          clf1 = **LinearDiscriminantAnalysis(solver='lsqr', shringkage='auto').fit(X,y)# 模型拟合，线性判别式，最普通都判别式**
          clf2 = ***LinearDiscriminantAnalysis***(solver='lsqr', shringkage=None).fit(X,y)# 一个收缩了，一个没有收缩

          X,y = generate_data(n_test, n_features)#生成测试数据
          score_clf1 += clf1.score(X,y)#对测试数据进行验证
          score_clf2 += clf2.score(X,y)
        acc_clf1.append(score_clf1 / n_averages)
        acc_clf2.append(score_clf2 / n_averages)

    features_sample_ratio = np.array(n_features_range) / relevant_train # 算是一种归一化处理，把原来features的维度用比较简单的方式表达出来，原来的双位数用单位数即表达，方便观测

    plt.plot(features_samples_ratio, acc_clf1, linewidth = 2, lable="Linear Disciminant Analysis with shringkage", color = 'navy') # 画图函数，参数依次是x轴，y轴，线条宽度，线条标签名称，颜色
    plt.plot(features_samples_ratio, acc_clf2, linewidth = 2, lable="Linear Discriminant Analysis", color = 'gold')

    plt.xlabel('n_features / n_samples')
    plt.ylabel('Classification accuracy')

    plt.legend(loc=1, prop={'size':12}) #图例函数，占地面积，大小
    plt.suptitle('Linear Discirminant Analysis vs. \ shringkage Linear Discriminant Analysis (1 discriminative feature)')
    plt.show()


Linear Discriminant Analysis
模型简介
Shrinkage是一个改善估计协方差矩阵的工具，用在当训练样本数小于特征数目当情况下。
这个模型的核心就是通过使用贝叶斯公式计算类别的概率，进而作出二分类。
> **Shrinkage参数可以手动设置为0和1之间。**
>  + 0 对应于没有Shringkage，将使用由经验总结得到的协方差矩阵
>  + 1 对应于完全Shringkage, 将使用 *对角的方差矩阵* 来估计 *协方差矩阵*    













































end
