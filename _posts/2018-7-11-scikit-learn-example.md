---
layout: post
title:  "scikit learn: Pixel importances with a parallel forest of trees"
date:   2018-07-11 10:36:30
tags: [机器学习, 数据挖掘, scikit-learn, ensemble methods]
---

    print(__doc__)

    from time import time
    import matplotlib.pyplot as plt

    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.ensemble import ExtraTreeClassifier

    n_jobs = 1 #准备用来做任务的处理器个数

    data = fetch_olivetti_faces() #人脸数据
    X = data.images.reshape((len(data.images), -1)) #reshape 列数处显示-1，理解其含义为把当前矩阵由3列压缩成2列，就是在原来的基础上减少1列。如果是行数，同理。
    y = data.target

    mask = y < 5 # 把X中的种类限制在小于5个类的情况下
    X = X[mask] # 序列中所有下标小于5的对应值都显示出来
    y = y[mask]

    print("Fitting ExtraTreesClassifier on faces data with %d cores..." % n_jobs)
    t0 = time()
    forest = ExtraTreesClassifier(n_estimators=1000, max_features=128, n_jobs=n_jobs, random_state=0) # n_estimators森林里树的数目, max_features用于切分最优树的最大特征数， n_jobs并行开启的任务数，random_state用于生成随机数的随机种子，目的是为了在不同的系统环境下生成出相同的随机数，其实是一种用来平衡误差的办法。

    forest.fit(X, y)
    print("done in %0.3fs" % (time() - t0)) # 显示运行时间
    importances = forest.feature_importances_ # 这里forest的类型还是ExtraTreesClassifier, 在拟合过后训练的数据，得到了这个模型的些属性，其中feature_importanes_即是这样的一个属性。该属性定义为一个函数，这里因为计算的是图片，所以它的特征就是每个像素点上的附加信息。
    importances = importances.reshape(data.images[0].shape)

    plt.matshow(importances, cmap=plt.cm.hot) # 在一个画布中把一个数组以矩阵显示出来
    plt.title("Pixel importances with forests of trees")
    plt.show()





### 背景介绍
这个例子介绍了使用森林来评估像素的重要性，对象是一个图片（脸部识别）分类任务，看起来白热化的像素，表示越重要。
这个例子还介绍了多任务中，平行预测的构造和计算。
应用范围可以是提取最优特征值。

#### mask功能详解
+ X.shape(400, 4096)
+ y.shape(400)
+ mask = y < 5, mask.shape(400)
+ 其中mask里True:50 False:350
+ X[mask]表示显示 ***mask为True的下标*** 对应的X的值
+ y[mask]同理

### 引入库的方式区别
+ from collections import Counter
+ import Counter
+ import的只能是类名，如果文件名和类名不一样，则就需要使用from

#### 数据结构
+ data shape (400, 4096)
+ data.images shape (400, 64, 64)
+ 训练集 X shape (50, 4096); y shape (50)
+ 训练集里每一条4096个特征，是由一条image里（64，64）被强制变成一维后成了(4096)
+ 测试集结果 y 自然是分类得到的结果，同时分类器还做了对特征值划分重要程度的工作。
+ 因为是图片数据，所以可以将结果形式以图片展示的方式显示出来。而展示的方式就是：
+ 把4096的维度再还原回（64，64），再以二维图片显示。
+ 图片数据的理解：算是图片数据比较特有的特征，可以通过压缩计算再还原的方式来处理。

#### ravel的理解

   \|<br>
   \|<br>
   \|__________
  /
 /
/

   \|<br>
   \|<br>
   \|__________

意义：把三维压到二维表示，信息没有删减，只是全部追加到二维后面了

end
